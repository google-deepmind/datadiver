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

"""Utilities for computing metrics on code models.

The functions here constitute the collection of datadiver skills for computing
code complexity metrics.

When used a skills, these functions should be copied directly verbatim into the
namespace where they are used.
"""

import ast
import textwrap
from typing import Any

import numpy as np


def get_complexity_metrics(
    code: str,
) -> dict[str, Any]:
  """Get raw code complexity metrics.

  Args:
    code: The code for which to compute the raw complexity metrics.

  Returns:
    A dictionary of raw complexity metrics for the given code. This includes:
    - num_lines: The number of lines in the code.
    - num_comments: The number of comments in the code.
    - total_imports: The number of imports in the code.
    - num_variables: The number of variables in the code.
    - halstead_vocab: The Halstead vocabulary size.
    - halstead_size: The Halstead total size.
    - halstead_volume: The Halstead volume.
    - halstead_difficulty: The Halstead difficulty.
    - halstead_effort: The Halstead effort.
  """
  stats = {}
  stats['num_lines'] = len(code.split('\n'))
  stats['num_comments'] = _get_num_comments(code)
  stats.update(_get_ast_metrics(code))
  stats.update(_get_halstead_metrics(code))
  return stats


def get_complexity_scores(code: str) -> dict[str, float]:
  """Get complexity scores for the given code.

  Complexity scores are normalized to [0, 1] and transformed to ensure that
  higher scores always indicate lower code complexity.

  Args:
    code: The code for which to compute the complexity scores.

  Returns:
    A dictionary of complexity scores for the given code. This includes:
    - normalized_negative_halstead_volume_score
    - normalized_negative_halstead_effort_score
    - normalized_negative_halstead_difficulty_score
    - normalized_negative_num_lines_score
    - normalized_negative_num_variables_score
  """
  metrics = get_complexity_metrics(code)
  return _get_complexity_scores_from_metrics(metrics)


TRANSFORM_FUNCTIONS = {
    'identity': lambda x: x,
    'inverse': lambda x: 1.0 / (x + 1e-9),
    'negative': lambda x: -x,
}

# Transforms to apply to metric values to produce score for maximizing
METRIC_TRANSFORMS = {
    'halstead_volume': 'negative',
    'halstead_effort': 'negative',
    'halstead_difficulty': 'negative',
    'num_lines': 'negative',
    'num_variables': 'negative',
}


# Empirical ranges for raw metrics to normalize scores to [0, 1].
METRIC_EMPIRICAL_RANGES = {
    'halstead_volume': (10, 50_000),
    'halstead_difficulty': (1, 150),
    'halstead_effort': (10, 7_500_000),  # volume * difficulty
    'num_lines': (1, 1_500),
    'num_variables': (1, 500),
    'total_imports': (0, 50),
}


def _transform_and_normalize_metric(
    metric: str,
    value: float,
    transform: str = 'identity',
) -> float:
  """Normalize the transformed score using the empirical range.

  Args:
    metric: The metric name.
    value: The raw value to transform and normalize.
    transform: The transform to apply to the score.

  Returns:
    The normalized score.
  """
  trans_func = TRANSFORM_FUNCTIONS[transform]
  trans_value = trans_func(value)
  if metric not in METRIC_EMPIRICAL_RANGES:
    raise ValueError(f'No empirical range found for metric {metric}.')
  min_value, max_value = METRIC_EMPIRICAL_RANGES[metric]
  if transform in ['inverse', 'negative']:
    trans_min_value = trans_func(max_value)
    trans_max_value = trans_func(min_value)
  elif transform in ['identity']:
    trans_min_value = trans_func(min_value)
    trans_max_value = trans_func(max_value)
  else:
    raise ValueError(f'Unrecognized transform {transform}')
  trans_range = trans_max_value - trans_min_value
  normalized_score = (trans_value - trans_min_value) / trans_range
  if normalized_score < 0.0 or normalized_score > 1.0:
    normalized_score = np.clip(normalized_score, 0.0, 1.0)
  return normalized_score


def normalize_metrics(raw_metrics: dict[str, Any]) -> dict[str, float]:
  """Normalize raw metrics to [0, 1] using empirical ranges.

  Args:
    raw_metrics: A dictionary of raw metrics.

  Returns:
    A dictionary of normalized metrics.
  """
  normalized = {}
  for k, v in raw_metrics.items():
    if k in METRIC_EMPIRICAL_RANGES:
      min_value, max_value = METRIC_EMPIRICAL_RANGES[k]
      norm_val = (v - min_value) / (max_value - min_value)
      normalized[k] = float(np.clip(norm_val, 0.0, 1.0))
  return normalized


def _get_complexity_scores_from_metrics(
    metrics: dict[str, Any],
) -> dict[str, float]:
  """Get complexity scores from metrics."""
  scores = {}
  for k, v in metrics.items():
    if k in METRIC_TRANSFORMS:
      transform = METRIC_TRANSFORMS[k]
      scores[f'normalized_{transform}_{k}_score'] = (
          _transform_and_normalize_metric(
              metric=k, value=v, transform=transform
          )
      )
  return scores


def _get_num_comments(code: str) -> int:
  """Get the number of comments in the given code."""
  code_lines = code.split('\n')
  num_comments = 0
  for line in code_lines:
    stripped_line = line.strip(' ')
    if stripped_line and stripped_line[0] == '#':
      num_comments += 1
  return num_comments


def _get_halstead_metrics(code: str) -> dict[str, int | float]:
  """Compute Halstead metrics for the given code.

  Args:
    code: The code to compute the Halstead metrics for.

  Returns:
    A dictionary of Halstead metrics for the given code.

  Raises:
    SyntaxError: If the code cannot be parsed.
  """
  tree = ast.parse(code)
  visitor = HalsteadVisitor()
  visitor.visit(tree)
  visitor.compute_halstead_metrics()
  stats = {f'halstead_{k}': v for k, v in visitor.halstead_metrics.items()}
  return stats


def _get_ast_metrics(code: str) -> dict[str, Any]:
  """Parse Abstract Syntax Tree (AST) and extract code statistics."""
  code_stats = {'total_imports': 0, 'num_variables': 0}
  try:
    code_ast = ast.parse(code)
  except SyntaxError:
    return {}
  for node in ast.walk(code_ast):
    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
      code_stats['total_imports'] += len(node.names)
    if isinstance(node, ast.Name):
      code_stats['num_variables'] += 1
    node_class_name = type(node).__name__
    node_type = f'num_{node_class_name}'
    code_stats[node_type] = code_stats.get(node_type, 0) + 1
  return code_stats


class HalsteadVisitor(ast.NodeVisitor):
  """NodeVisitor for computing Halstead complexity."""

  def __init__(self):
    super().__init__()
    self.unique_operators = set()
    self.unique_operands = set()
    self.num_operators = 0
    self.num_operands = 0
    self.halstead_metrics = {}

  def compute_halstead_metrics(self):
    """Compute the various Halstead metrics."""
    num_unique_operators = len(self.unique_operators)
    num_unique_operands = len(self.unique_operands)
    vocabulary_size = num_unique_operators + num_unique_operands
    total_size = self.num_operators + self.num_operands
    volume = total_size * np.log2(vocabulary_size)
    difficulty = (num_unique_operators / 2) * (
        self.num_operands / num_unique_operands
    )
    effort = difficulty * volume
    self.halstead_metrics = {}
    self.halstead_metrics['vocab'] = vocabulary_size
    self.halstead_metrics['size'] = total_size
    self.halstead_metrics['volume'] = volume
    self.halstead_metrics['difficulty'] = difficulty
    self.halstead_metrics['effort'] = effort

  def get_node_id(self, node):
    if isinstance(node, ast.Name):
      node_id = node.id
    elif isinstance(node, ast.Constant):
      node_id = str(node.value)
    elif isinstance(node, ast.Attribute):
      node_id = node.attr
    else:
      node_id = 'UNKNOWN_NODE'
    return node_id

  def generic_visit(self, node):
    if isinstance(node, ast.FunctionDef):
      for child in node.body:
        self.visit(child)
      return

    node_type = 'None'
    node_id = ''

    if isinstance(node, ast.Name):
      node_type = 'Operand'
      node_id = node.id
    elif isinstance(node, ast.Constant):
      node_type = 'Operand'
      node_id = str(node.value)
    elif isinstance(node, ast.BinOp) or isinstance(node, ast.UnaryOp):
      node_type = 'Operator'
      node_id = type(node.op).__name__

    elif (
        isinstance(node, ast.Subscript)
        or isinstance(node, ast.If)
        or isinstance(node, ast.While)
        or isinstance(node, ast.For)
        or isinstance(node, ast.Assign)
        or isinstance(node, ast.AugAssign)
    ):
      node_type = 'Operator'
      node_id = type(node).__name__

    elif isinstance(node, ast.Call):
      func = node.func
      node_type = 'Operator'
      node_id = self.get_node_id(func)

    if node_type == 'Operand':
      self.unique_operands.add(node_id)
      self.num_operands += 1
    elif node_type == 'Operator':
      self.unique_operators.add(node_id)
      self.num_operators += 1

    if isinstance(node, ast.Compare):
      node_type = 'Operator'
      for op in node.ops:
        node_id += type(op).__name__ + ','
        self.unique_operators.add(type(op).__name__)
        self.num_operators += 1

    ast.NodeVisitor.generic_visit(self, node)


def get_n_params_used(agent_fun: str) -> int:
  """Get the number of elements of params used by agent_fun.

  This function checks the source code of the agent_fun to see how many
  elements of params it accesses. It only checks for accesses using constant
  integers (it will not catch indexing that is dynamic and depends on the
  inputs to agent_fun).

  Args:
    agent_fun: The source code of the agent function to check.

  Returns:
    The number of parameters used by agent_fun.
  """
  source = textwrap.dedent(agent_fun)
  tree = ast.parse(source)
  visitor = _ParamCountVisitor()
  visitor.visit(tree)
  return visitor.max_index + 1


def verify_num_params(
    agent_fun: str,
    n_params: int,
) -> int:
  """Verify that the number of params used by agent_fun is at most n_params.

  This function should be called before fitting a model to check that the
  agent function is compatible with the specified number of parameters.

  Args:
    agent_fun: The source code of the agent function to check.
    n_params: The number of parameters available.

  Returns:
    The number of parameters used by agent_fun.

  Raises:
    ValueError: If the number of parameters used by agent_fun is greater than
    n_params.
  """
  n_params_used = get_n_params_used(agent_fun)
  if n_params_used > n_params:
    raise ValueError(
        f'Agent fun uses {n_params_used} params, but only {n_params} are '
        'available.'
    )
  return n_params_used


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
