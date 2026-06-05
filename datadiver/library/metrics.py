# Copyright 2026 The datadiver Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for computing metrics on code models."""

import ast
import dataclasses
import re
import textwrap
import time  # from google3.pyglib import timer
from typing import Any
from absl import logging
from datadiver.library import halstead_visitor
import numpy as np

# TODO(cwendelken): Move these up -- they aren't needed in this module.
DEFAULT_SCORE_KEY = 'score'  # Default label for a single primary score
AVERAGE_SCORE_KEY = 'average'  # Default label for an average of primary scores


RUN_STATUS = 'RUN'
RUNTIME_ERROR_STATUS = 'RUNTIME_ERROR'


@dataclasses.dataclass
class MetricSpec:
  function: Any
  metric: str
  transform: str
  normalize: bool = False

  def apply(self, code: str) -> Any:
    return self.function(code, self.metric, self.transform, self.normalize)


@dataclasses.dataclass
class MetricTest:
  status: str
  stderr: str | None
  parsed_output: Any
  score: float | None
  metadata: dict[str, float]
  analysis_time_s: float


TRANSFORMS = {
    'identity': lambda x: x,
    'inverse': lambda x: 1.0 / (x + 1e-9),
    'negative': lambda x: -x,
}

# These are not set in stone. Update as needed, but be careful when comparing
# experiments run with different ranges!
EMPIRICAL_RANGES = {
    'volume': (1, 50_000),
    'effort': (1, 7_500_000),
    'difficulty': (1, 150),
    'num_lines': (1, 1_500),
    'num_comments': (1, 500),
}


def normalize_transformed_score(
    score: float,
    metric: str,
    transform: str,
) -> float:
  """Normalize the transformed score using the empirical range.

  Args:
    score: The score to normalize.
    metric: The metric to normalize.
    transform: The transform to apply to the score.

  Returns:
    The normalized score.
  """
  if metric not in EMPIRICAL_RANGES:
    raise ValueError(f'No empirical range found for metric {metric}.')
  min_value, max_value = EMPIRICAL_RANGES[metric]
  if transform == 'inverse':
    min_value, max_value = 1.0 / max_value, 1.0 / min_value
  elif transform == 'negative':
    min_value, max_value = -max_value, -min_value
  elif transform != 'identity':
    raise ValueError(f'Unrecognized transform {transform}')
  normalized_score = (score - min_value) / (max_value - min_value)
  if normalized_score < 0.0 or normalized_score > 1.0:
    # Don't fail, just issue a warning and clamp to the lowest/highest score.
    warning_message = (
        f'Normalized score {normalized_score} is not in [0, 1] for metric'
        f' {metric} and transform {transform}.\n'
        f'Original value: {score}.\n'
        f'Empirical range: ({min_value}, {max_value}).'
    )
    logging.warning(warning_message)
    normalized_score = np.clip(normalized_score, 0.0, 1.0)
  return normalized_score


# TODO(cwendelken): Merge this into compute_code_stats.
## RENAME as compute_halstead_metric_score
def compute_halstead(
    code: str, metric: str, transform: str, normalize: bool = False
) -> MetricTest:
  """Return the transform of the selected Halstead score as a test result.

  Args:
    code: The code to compute the metrics for.
    metric: The metric to compute.
    transform: The transform to apply to the metric.
    normalize: Whether to normalize the score.

  Returns:
    The test result.
  """
  # stopwatch = timer.TimerStart()
  # stopwatch.Start()
  time_start = time.time()
  try:
    assert transform in TRANSFORMS, f'Transform {transform} is not recognized.'
    halstead_metrics = get_halstead_metrics(code)
    assert metric in halstead_metrics, f'Metric {metric} is not recognized.'
  except BaseException as e:  # pylint: disable=broad-except
    score = None
    parsed_output = None
    status = RUNTIME_ERROR_STATUS
    stderr = str(e)
  else:
    selected_metric = halstead_metrics[metric]
    score = TRANSFORMS[transform](selected_metric)
    key = f'{transform}_halstead_{metric}'
    if normalize:
      score = normalize_transformed_score(score, metric, transform)
      key = 'normalized_' + key
    parsed_output = {key: score}
    status = RUN_STATUS
    stderr = None

  return MetricTest(
      status=status,
      stderr=stderr,
      score=score,
      parsed_output=parsed_output,
      metadata={},
      analysis_time_s=(time.time() - time_start) / 1000,  # stopwatch.Stop(),
  )


### RENAME as compute_code_metric_score
def compute_code_stats(
    code: str, metric: str, transform: str, normalize: bool = False
) -> MetricTest:
  """Return the transform of the selected code stat."""
  # stopwatch = timer.TimerStart()
  # stopwatch.Start()
  time_start = time.time()
  try:
    assert transform in TRANSFORMS, f'Transform {transform} is not recognized.'
    code_stats = get_code_stats(code)
    if metric in code_stats:
      stat = code_stats[metric]
    elif metric in code_stats['code_stats']:
      stat = code_stats['code_stats'][metric]
    else:
      raise ValueError(f'Unrecognized code stat {metric}')

  except BaseException as e:  # pylint: disable=broad-except
    score = None
    parsed_output = None
    status = RUNTIME_ERROR_STATUS
    stderr = str(e)
  else:
    score = TRANSFORMS[transform](stat)
    key = f'{transform}_{metric}'
    if normalize:
      score = normalize_transformed_score(score, metric, transform)
      key = 'normalized_' + key
    parsed_output = {key: score}
    status = RUN_STATUS
    stderr = None
  return MetricTest(
      status=status,
      stderr=stderr,
      score=score,
      parsed_output=parsed_output,
      metadata={},
      analysis_time_s=(time.time() - time_start) / 1000,  # stopwatch.Stop(),
  )


ALL_METRICS = {
    'negative_n_params': MetricSpec(
        function=compute_code_stats, metric='n_params', transform='negative'
    ),
    'inverse_halstead_volume': MetricSpec(
        function=compute_halstead,
        metric='volume',
        transform='inverse',
    ),
    'inverse_halstead_effort': MetricSpec(
        function=compute_halstead, metric='effort', transform='inverse'
    ),
    'inverse_halstead_difficulty': MetricSpec(
        function=compute_halstead, metric='difficulty', transform='inverse'
    ),
    'negative_halstead_volume': MetricSpec(
        function=compute_halstead, metric='volume', transform='negative'
    ),
    'negative_halstead_effort': MetricSpec(
        function=compute_halstead, metric='effort', transform='negative'
    ),
    'negative_halstead_difficulty': MetricSpec(
        function=compute_halstead, metric='difficulty', transform='negative'
    ),
    'normalized_inverse_halstead_volume': MetricSpec(
        function=compute_halstead,
        metric='volume',
        transform='inverse',
        normalize=True,
    ),
    'normalized_inverse_halstead_effort': MetricSpec(
        function=compute_halstead,
        metric='effort',
        transform='inverse',
        normalize=True,
    ),
    'normalized_inverse_halstead_difficulty': MetricSpec(
        function=compute_halstead,
        metric='difficulty',
        transform='inverse',
        normalize=True,
    ),
    'normalized_negative_halstead_volume': MetricSpec(
        function=compute_halstead,
        metric='volume',
        transform='negative',
        normalize=True,
    ),
    'normalized_negative_halstead_effort': MetricSpec(
        function=compute_halstead,
        metric='effort',
        transform='negative',
        normalize=True,
    ),
    'normalized_negative_halstead_difficulty': MetricSpec(
        function=compute_halstead,
        metric='difficulty',
        transform='negative',
        normalize=True,
    ),
    'inverse_num_lines': MetricSpec(
        function=compute_code_stats, metric='num_lines', transform='inverse'
    ),
    'inverse_num_comments': MetricSpec(
        function=compute_code_stats, metric='num_comments', transform='inverse'
    ),
    'inverse_total_imports': MetricSpec(
        function=compute_code_stats, metric='total_imports', transform='inverse'
    ),
    'inverse_num_variables': MetricSpec(
        function=compute_code_stats, metric='num_variables', transform='inverse'
    ),
    'negative_num_lines': MetricSpec(
        function=compute_code_stats, metric='num_lines', transform='negative'
    ),
    'negative_num_comments': MetricSpec(
        function=compute_code_stats, metric='num_comments', transform='negative'
    ),
    'negative_total_imports': MetricSpec(
        function=compute_code_stats,
        metric='total_imports',
        transform='negative',
    ),
    'negative_num_variables': MetricSpec(
        function=compute_code_stats,
        metric='num_variables',
        transform='negative',
    ),
    'normalized_inverse_num_lines': MetricSpec(
        function=compute_code_stats,
        metric='num_lines',
        transform='inverse',
        normalize=True,
    ),
    'normalized_inverse_num_comments': MetricSpec(
        function=compute_code_stats,
        metric='num_comments',
        transform='inverse',
        normalize=True,
    ),
    'normalized_negative_num_lines': MetricSpec(
        function=compute_code_stats,
        metric='num_lines',
        transform='negative',
        normalize=True,
    ),
    'normalized_negative_num_comments': MetricSpec(
        function=compute_code_stats,
        metric='num_comments',
        transform='negative',
        normalize=True,
    ),
}


def get_halstead_metrics(code: str) -> dict[str, int | float]:
  """Compute Halstead metrics for the given code."""
  try:
    tree = ast.parse(code)
  except SyntaxError:
    logging.exception(
        'Returning without code stats. Failed to parse code: %s', code
    )
    return {'invalid_code': -1}  # TODO(cwendelken): Raise error instead.

  visitor = halstead_visitor.HalsteadVisitor()
  visitor.visit(tree)
  visitor.compute_halstead_metrics()
  return visitor.halstead_metrics


### RENAME as get_code_metrics
def get_code_stats(code: str) -> dict[str, Any]:
  """Get code statistics."""
  stats = {}
  code_lines = code.split('\n')
  stats['num_lines'] = len(code_lines)
  num_comments = 0
  for line in code_lines:
    stripped_line = line.strip(' ')
    if stripped_line and stripped_line[0] == '#':
      num_comments += 1
  stats['num_comments'] = num_comments
  stats['code_stats'] = parse_abstract_syntax_tree(code)
  halstead_stats = get_halstead_metrics(code)
  stats['n_params'] = get_n_params_used(code)
  stats.update({f'halstead/{k}': v for k, v in halstead_stats.items()})
  return stats


def parse_abstract_syntax_tree(code: str) -> dict[str, Any]:
  """Parse AST and extract code statistics, return as JSON string."""
  code_stats = {
      'total_imports': 0,
      'num_variables': 0,
  }
  try:
    code_ast = ast.parse(code)
  except SyntaxError:
    logging.exception(
        'Returning without code stats. Failed to parse code: %s', code
    )
    return {'invalid_code': -1}  # TODO(cwendelken): Raise error instead.

  for node in ast.walk(code_ast):
    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
      code_stats['total_imports'] += len(node.names)
    if isinstance(node, ast.Name):
      code_stats['num_variables'] += 1
    regex = re.search("<class 'ast.(.+?)'>", str(type(node)))
    if regex is None:
      continue

    node_type = f'num_{regex.group(1)}'
    code_stats[node_type] = code_stats.get(node_type, 0) + 1
  return code_stats


def get_n_params_used(agent_fun_source: str) -> int:
  """Get the number of elements of params used by agent_fun.

  This function checks the source code of the agent_fun to see how many
  elements of params it accesses. It only checks for accesses using constant
  integers (it will not catch indexing that is dynamic and depends on the
  inputs to agent_fun).

  Args:
    agent_fun_source: The source code of the agent function to check.

  Returns:
    The number of parameters used by agent_fun.
  """
  source = textwrap.dedent(agent_fun_source)
  tree = ast.parse(source)
  visitor = _ParamCountVisitor()
  visitor.visit(tree)
  return visitor.max_index + 1


class _ParamCountVisitor(ast.NodeVisitor):
  """AST visitor to find the highest index used to index into 'params'."""

  def __init__(self):
    super().__init__()
    self.max_index = -1

  def visit_Assign(self, node: ast.Assign):  # pylint: disable=invalid-name
    """Check for unpacking assignment from 'params'."""
    if isinstance(node.value, ast.Name) and node.value.id == 'params':
      for target in node.targets:
        if isinstance(target, (ast.Tuple, ast.List)):
          self.max_index = max(self.max_index, len(target.elts) - 1)
    self.generic_visit(node)

  def visit_Subscript(self, node: ast.Subscript):  # pylint: disable=invalid-name
    """Check if the base of the subscript is the variable named 'params'."""
    if isinstance(node.value, ast.Name) and node.value.id == 'params':
      index_node = node.slice

      if isinstance(index_node, ast.Constant):
        # Case: params[k] (direct indexing with a constant integer k)
        if isinstance(index_node.value, int):
          self.max_index = max(self.max_index, index_node.value)
      elif isinstance(index_node, ast.Slice):
        # Case: params[a:b], params[a:], params[:b]

        # Check lower bound of the slice.
        if (
            index_node.lower is not None
            and isinstance(index_node.lower, ast.Constant)
            and isinstance(index_node.lower.value, int)
        ):
          self.max_index = max(self.max_index, index_node.lower.value)
        # Check upper bound of the slice.
        # (Slicing L[x:y] accesses elements up to index y-1.
        if (
            index_node.upper is not None
            and isinstance(index_node.upper, ast.Constant)
            and isinstance(index_node.upper.value, int)
        ):
          self.max_index = max(self.max_index, index_node.upper.value - 1)

    # Ensure we visit children nodes as well
    self.generic_visit(node)
