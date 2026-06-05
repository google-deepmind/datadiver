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

"""Utilities to compute Halstead complexity of a piece of code."""

import ast
from absl import logging
import numpy as np


class HalsteadVisitor(ast.NodeVisitor):
  """NodeVisitor for computing Halstead complexity.

  This class will compute the following measures:
    vocabulary: (number of distinct operators) + (number of distinct operands)
    length: (total number of operators) + (total number of operands)
    volume: length x log(vocabulary)
    difficulty: (number of distinct operators) /
                (2 x (total number of operators / number of distinct operands))
    effort: difficulty x volume

  See https://en.wikipedia.org/wiki/Halstead_complexity_measures for more
  details.
  """

  def __init__(self):
    super().__init__()
    self.unique_operators = set()
    self.unique_operands = set()
    self.num_operators = 0
    self.num_operands = 0

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
      logging.warning('Unknown function type: %s', type(node).__name__)
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
