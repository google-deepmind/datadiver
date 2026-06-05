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

from absl.testing import absltest
from datadiver.library import dataset_types


class DatasetTypesTest(absltest.TestCase):

  def test_arg_spec_creation_basic(self):
    spec = dataset_types.ArgSpec(name="choice", dtype="int", n_values=2)
    self.assertEqual(spec.name, "choice")
    self.assertEqual(spec.dtype, "int")
    self.assertEqual(spec.n_values, 2)
    self.assertTrue(spec.should_shift)
    self.assertEqual(spec.noinfo_name, "choice")

  def test_arg_spec_creation_with_noinfo_override(self):
    spec = dataset_types.ArgSpec(
        name="choice",
        dtype="int",
        n_values=2,
        noinfo_name_override="x",
    )
    self.assertEqual(spec.name, "choice")
    self.assertEqual(spec.noinfo_name, "x")

  def test_arg_spec_n_values_fails_for_non_int(self):
    with self.assertRaises(AssertionError):
      dataset_types.ArgSpec(name="reward", dtype="float", n_values=2)


if __name__ == "__main__":
  absltest.main()
