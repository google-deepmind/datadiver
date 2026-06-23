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

import ast
from absl.testing import absltest
from absl.testing import parameterized
from datadiver.library import code_metrics as metrics


class MetricsTest(absltest.TestCase):

  def test_get_complexity_metrics(self):
    code = """
# This is a comment
import os
from sys import exit

def my_function(a, b):
  c = a + b
  return c
"""
    stats = metrics.get_complexity_metrics(code)

    self.assertEqual(stats["num_lines"], 9)
    self.assertEqual(stats["num_comments"], 1)

    self.assertEqual(stats["total_imports"], 2)  # os, exit

    # Check some halstead stats exist
    self.assertIn("halstead_vocab", stats)
    self.assertIn("halstead_size", stats)
    self.assertIn("halstead_volume", stats)
    self.assertIn("halstead_difficulty", stats)
    self.assertIn("halstead_effort", stats)

  def test_normalize_metrics(self):
    raw_metrics = {
        "num_lines": 150.9,
        "total_imports": 10,
        "invalid_metric": 100,
    }
    normalized = metrics.normalize_metrics(raw_metrics)

    self.assertAlmostEqual(normalized["num_lines"], 0.1)
    self.assertAlmostEqual(normalized["total_imports"], 0.2)
    self.assertNotIn("invalid_metric", normalized)


class HalsteadVisitorTest(parameterized.TestCase):

  def test_simple_assignment(self):
    code = "a = 1 + 2"
    tree = ast.parse(code)
    visitor = metrics.HalsteadVisitor()
    visitor.visit(tree)
    visitor.compute_halstead_metrics()
    halstead_metrics = visitor.halstead_metrics

    self.assertEqual(visitor.unique_operators, {"Assign", "Add"})
    self.assertEqual(visitor.unique_operands, {"a", "1", "2"})
    self.assertEqual(visitor.num_operators, 2)
    self.assertEqual(visitor.num_operands, 3)

    self.assertEqual(halstead_metrics["vocab"], 5)
    self.assertEqual(halstead_metrics["size"], 5)
    self.assertAlmostEqual(halstead_metrics["volume"], 11.6096, places=3)
    self.assertAlmostEqual(halstead_metrics["difficulty"], 1.0)
    self.assertAlmostEqual(halstead_metrics["effort"], 11.6096, places=3)

  def test_if_statement(self):
    code = "if a > 0: a = 1"
    tree = ast.parse(code)
    visitor = metrics.HalsteadVisitor()
    visitor.visit(tree)
    visitor.compute_halstead_metrics()
    halstead_metrics = visitor.halstead_metrics

    self.assertEqual(visitor.unique_operators, {"If", "Gt", "Assign"})
    self.assertEqual(visitor.unique_operands, {"a", "0", "1"})
    self.assertEqual(visitor.num_operators, 3)
    self.assertEqual(visitor.num_operands, 4)

    self.assertEqual(halstead_metrics["vocab"], 6)
    self.assertEqual(halstead_metrics["size"], 7)
    self.assertAlmostEqual(halstead_metrics["volume"], 18.09475, places=3)
    self.assertAlmostEqual(halstead_metrics["difficulty"], 2.0)
    self.assertAlmostEqual(halstead_metrics["effort"], 36.1895, places=3)


if __name__ == "__main__":
  absltest.main()
