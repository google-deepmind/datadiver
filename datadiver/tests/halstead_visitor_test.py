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
from datadiver.library import halstead_visitor


class HalsteadVisitorTest(parameterized.TestCase):

  def test_simple_assignment(self):
    code = "a = 1 + 2"
    tree = ast.parse(code)
    visitor = halstead_visitor.HalsteadVisitor()
    visitor.visit(tree)
    visitor.compute_halstead_metrics()
    metrics = visitor.halstead_metrics

    self.assertEqual(visitor.unique_operators, {"Assign", "Add"})
    self.assertEqual(visitor.unique_operands, {"a", "1", "2"})
    self.assertEqual(visitor.num_operators, 2)
    self.assertEqual(visitor.num_operands, 3)

    self.assertEqual(metrics["vocab"], 5)
    self.assertEqual(metrics["size"], 5)
    self.assertAlmostEqual(metrics["volume"], 11.6096, places=3)
    self.assertAlmostEqual(metrics["difficulty"], 1.0)
    self.assertAlmostEqual(metrics["effort"], 11.6096, places=3)

  def test_if_statement(self):
    code = "if a > 0: a = 1"
    tree = ast.parse(code)
    visitor = halstead_visitor.HalsteadVisitor()
    visitor.visit(tree)
    visitor.compute_halstead_metrics()
    metrics = visitor.halstead_metrics

    self.assertEqual(visitor.unique_operators, {"If", "Gt", "Assign"})
    self.assertEqual(visitor.unique_operands, {"a", "0", "1"})
    self.assertEqual(visitor.num_operators, 3)
    self.assertEqual(visitor.num_operands, 4)

    self.assertEqual(metrics["vocab"], 6)
    self.assertEqual(metrics["size"], 7)
    self.assertAlmostEqual(metrics["volume"], 18.09475, places=3)
    self.assertAlmostEqual(metrics["difficulty"], 2.0)
    self.assertAlmostEqual(metrics["effort"], 36.1895, places=3)


if __name__ == "__main__":
  absltest.main()
