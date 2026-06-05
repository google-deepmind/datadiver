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

"""Tests for functions in utils.py."""

from absl.testing import absltest
from absl.testing import parameterized
from datadiver.library import dataset_types
from datadiver.library import dataset_utils
import numpy as np


# from google3.learning.deepmind.research.neuroexp.aotm.datasets import types
# from google3.learning.deepmind.research.neuroexp.aotm.datasets import utils
# from google3.pyglib import gfile


class UtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_tmpdir = self.create_tempdir()

  ### ADDED as alternative to _create_test_json
  def _create_test_data(
      self, num_rows: int, num_trials: int, has_ignore: bool
  ) -> tuple[dataset_types.DataDict, dataset_types.NeuroDataset]:
    """Create a DataDict for testing."""
    data_dict = {x: [] for x in ['choice', 'reward']}  # , 'ignore']}
    if has_ignore:
      data_dict['ignore'] = []
    for row in range(num_rows):
      data_dict['choice'].append([(row + t) % 2 for t in range(num_trials)])
      data_dict['reward'].append([t * (row + 1) for t in range(num_trials)])
      if has_ignore:
        data_dict['ignore'].append([bool(t % 7) for t in range(num_trials)])
    data_dict = {x: np.array(y) for x, y in data_dict.items()}
    inputs, targets = dataset_utils.format_data_for_next_choice_prediction(
        data_dict, input_fields=['choice', 'reward'],
    )
    dataset = dataset_types.NeuroDataset.from_data_dict(
        data_dict, dataset_types.BanditTrial, inputs=inputs, targets=targets
    )
    return data_dict, dataset

  @parameterized.parameters([
      dict(split='even', n=10, expected_inds=[0, 2, 4, 6, 8]),
      dict(split='odd', n=10, expected_inds=[1, 3, 5, 7, 9]),
      dict(split='all', n=10, expected_inds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
      dict(split='5', n=10, expected_inds=[0, 1, 2, 3, 4]),
      dict(split='-5', n=10, expected_inds=[5, 6, 7, 8, 9]),
      dict(split='-2', n=10, expected_inds=[8, 9]),
  ])
  def testGetIndicesFromSplit(self, split, n, expected_inds):
    self.assertSequenceEqual(
        expected_inds, dataset_utils.get_indices_from_split(split, n)
    )

  @parameterized.parameters([
      dict(split='even', n=-4),
      dict(split='odd', n=0),
      dict(split='potatoes', n=10),
      dict(split='15', n=10),
      dict(split='-25', n=10),
  ])
  def testGetIndicesFromSplitErrors(self, split, n):
    self.assertRaises(
        ValueError, dataset_utils.get_indices_from_split, split, n
    )

  def testSplitAndPrepareDataForJitWithDefaults(self):
    num_rows = 2
    num_trials = 10
    # expected_dictionary = self._create_test_json(
    #     num_rows=num_rows, num_trials=num_trials, has_ignore=False)
    # dataset = dataset_utils.load_from_json(datafile_path=self._test_json)
    data_dict, dataset = self._create_test_data(
        num_rows=num_rows, num_trials=num_trials, has_ignore=False
    )
    even, odd = dataset_utils.split_and_prepare_data_for_jit(dataset)
    # `inputs` will include all-but-the-last element
    for field in ['choice', 'reward']:
      np.testing.assert_equal(even['inputs'][field], data_dict[field][::2, :-1])
      np.testing.assert_equal(odd['inputs'][field], data_dict[field][1::2, :-1])
    # `targets` will include all-but-the-first element
    np.testing.assert_equal(
        even['targets']['next_choice'], data_dict['choice'][::2, 1:]
    )
    # np.testing.assert_equal(
    #     odd['targets']['ignore'], data_dict['ignore'][1::2, 1:]
    # )

  def testSplitAndPrepareDataForJitWithNonDefaults(self):
    num_rows = 2
    num_trials = 10
    # expected_dictionary = self._create_test_json(
    #     num_rows=num_rows, num_trials=num_trials, has_ignore=False)
    # dataset = dataset_utils.load_from_json(datafile_path=self._test_json)
    data_dict, dataset = self._create_test_data(
        num_rows=num_rows, num_trials=num_trials, has_ignore=True
    )  ### ignore is not added via load here, so include it explicitly

    even, odd = dataset_utils.split_and_prepare_data_for_jit(
        dataset, split0_fn=lambda x: x[::4], split1_fn=lambda x: x[2::4]
    )
    # `inputs` will include all-but-the-last element
    for field in ['choice', 'reward']:
      np.testing.assert_equal(even['inputs'][field], data_dict[field][::4, :-1])
      np.testing.assert_equal(odd['inputs'][field], data_dict[field][2::4, :-1])
    # `targets` will include all-but-the-first element
    np.testing.assert_equal(
        even['targets']['next_choice'], data_dict['choice'][::4, 1:]
    )
    np.testing.assert_equal(
        odd['targets']['ignore'], data_dict['ignore'][2::4, 1:]
    )

  def testSplitAndPrepareDataForJitWithInvalidSplits(self):
    num_rows = 2
    num_trials = 10
    # _ = self._create_test_json(
    #     num_rows=num_rows, num_trials=num_trials, has_ignore=False)
    # dataset = dataset_utils.load_from_json(datafile_path=self._test_json)
    _, dataset = self._create_test_data(
        num_rows=num_rows, num_trials=num_trials, has_ignore=False
    )
    with self.assertRaises(AssertionError):
      _, _ = dataset_utils.split_and_prepare_data_for_jit(
          dataset, split0_fn=lambda x: x[::2], split1_fn=lambda x: x
      )

  def testDatasetSlice(self):
    num_rows = 4
    num_trials = 10
    # _ = self._create_test_json(
    #     num_rows=num_rows, num_trials=num_trials, has_ignore=False)
    # dataset = dataset_utils.load_from_json(datafile_path=self._test_json)
    _, dataset = self._create_test_data(
        num_rows=num_rows, num_trials=num_trials, has_ignore=False
    )
    even = dataset.slice(lambda x: x[::2])
    odd = dataset.slice(lambda x: x[1::2])
    zero_three = dataset.slice(lambda x: x[[0, 3]])
    np.testing.assert_equal(len(even.sessions), num_rows // 2)
    np.testing.assert_equal(len(odd.sessions), num_rows // 2)
    # `inputs` will include all-but-the-last element
    for field in ['choice', 'reward']:
      np.testing.assert_equal(
          even.dictionary[field], dataset.dictionary[field][::2]
      )
      np.testing.assert_equal(
          odd.dictionary[field], dataset.dictionary[field][1::2]
      )
      np.testing.assert_equal(
          zero_three.dictionary[field], dataset.dictionary[field][[0, 3]]
      )

  def testSplitDataset(self):
    num_rows = 4
    num_trials = 10
    # _ = self._create_test_json(
    #     num_rows=num_rows, num_trials=num_trials, has_ignore=False)
    # dataset = dataset_utils.load_from_json(datafile_path=self._test_json)
    _, dataset = self._create_test_data(
        num_rows=num_rows, num_trials=num_trials, has_ignore=False
    )
    dataset.dictionary['scalar_names'] = ['scalar1', 'scalar2']
    even, odd = dataset_utils.split_dataset(
        dataset, split0_fn=lambda x: x[::2], split1_fn=lambda x: x[1::2]
    )
    np.testing.assert_equal(len(even.sessions), num_rows // 2)
    np.testing.assert_equal(len(odd.sessions), num_rows // 2)

    # `inputs` will include all-but-the-last element
    for field in ['choice', 'reward']:
      np.testing.assert_equal(
          even.dictionary[field], dataset.dictionary[field][::2]
      )
      np.testing.assert_equal(
          odd.dictionary[field], dataset.dictionary[field][1::2]
      )
    self.assertIn('scalar_names', even.dictionary)
    self.assertIn('scalar_names', odd.dictionary)
    self.assertEqual(
        even.dictionary['scalar_names'], dataset.dictionary['scalar_names']
    )
    self.assertEqual(
        odd.dictionary['scalar_names'], dataset.dictionary['scalar_names']
    )

  def testSplitDatasetWithInvalidSplits(self):
    num_rows = 2
    num_trials = 10
    _, dataset = self._create_test_data(
        num_rows=num_rows, num_trials=num_trials, has_ignore=False
    )
    with self.assertRaises(AssertionError):
      _, _ = dataset_utils.split_dataset(
          dataset, split0_fn=lambda x: x[::2], split1_fn=lambda x: x
      )

  @parameterized.parameters([
      dict(list0=[1, 3], list1=[2, 4], expected=[1, 2, 3, 4]),
      dict(list0=[1, 3, 5], list1=[2, 4], expected=[1, 2, 3, 4, 5]),
      dict(list0=[1, 3], list1=[2, 4, 6], expected=[1, 2, 3, 4, 6]),
      dict(list0=[], list1=[2, 4], expected=[2, 4]),
      dict(list0=[1, 3], list1=[], expected=[1, 3]),
      dict(list0=[], list1=[], expected=[]),
      dict(list0=['a', 'c'], list1=['b', 'd'], expected=['a', 'b', 'c', 'd']),
  ])
  def testZipperMergeLists(self, list0, list1, expected):
    self.assertSequenceEqual(
        dataset_utils._zipper_merge_lists(list0, list1), expected
    )

  def testZipperMergeDatasets(self):
    # dataset 0
    d0_choices = np.array([[0, 1, 0], [1, 0, 1]])
    d0_rewards = np.array([[0, 1, 2], [0, 1, 2]])
    d0_ignore = np.array([[False, False, False], [False, False, False]])
    d0_sessions = [
        dataset_types.NeuroSession(
            trials=[
                dataset_types.BanditTrial(choice=0, reward=0, ignore=False),
                dataset_types.BanditTrial(choice=1, reward=1, ignore=False),
                dataset_types.BanditTrial(choice=0, reward=2, ignore=False),
            ]
        ),
        dataset_types.NeuroSession(
            trials=[
                dataset_types.BanditTrial(choice=1, reward=0, ignore=False),
                dataset_types.BanditTrial(choice=0, reward=1, ignore=False),
                dataset_types.BanditTrial(choice=1, reward=2, ignore=False),
            ]
        ),
    ]
    dataset0 = dataset_types.NeuroDataset(
        sessions=d0_sessions,
        dictionary={
            'choice': d0_choices,
            'reward': d0_rewards,
            'ignore': d0_ignore,
            'scalar_names': ['a', 'b'],
        },
    )

    # dataset 1
    d1_choices = np.array([[1, 0]])
    d1_rewards = np.array([[10, 11]])
    d1_ignore = np.array([[False, False]])
    d1_sessions = [
        dataset_types.NeuroSession(
            trials=[
                dataset_types.BanditTrial(choice=1, reward=10, ignore=False),
                dataset_types.BanditTrial(choice=0, reward=11, ignore=False),
            ]
        ),
    ]
    dataset1 = dataset_types.NeuroDataset(
        sessions=d1_sessions,
        dictionary={
            'choice': d1_choices,
            'reward': d1_rewards,
            'ignore': d1_ignore,
        },
    )

    merged = dataset_utils.zipper_merge_datasets(dataset0, dataset1)

    self.assertLen(merged.sessions, 3)
    self.assertIs(merged.sessions[0], d0_sessions[0])
    self.assertIs(merged.sessions[1], d1_sessions[0])
    self.assertIs(merged.sessions[2], d0_sessions[1])

    # Test dictionary merging and padding.
    # dataset0 has shape (2,3), dataset1 has shape (1,2).
    # max_shape along trial dim is 3.
    # d1_choices should be padded to length 3.
    # merged choices should come from d0s0, d1s0, d0s1
    expected_choices = np.array([
        [0, 1, 0],
        [1, 0, -1],  # padded from [1,0]
        [1, 0, 1],
    ])
    expected_rewards = np.array([
        [0, 1, 2],
        [10, 11, -1],  # padded from [10, 11]
        [0, 1, 2],
    ])
    expected_ignore = np.array([
        [False, False, False],
        [False, False, True],  # padded from [False, False]
        [False, False, False],
    ])

    np.testing.assert_array_equal(merged.dictionary['choice'], expected_choices)
    np.testing.assert_array_equal(merged.dictionary['reward'], expected_rewards)
    np.testing.assert_array_equal(merged.dictionary['ignore'], expected_ignore)
    self.assertIn('scalar_names', merged.dictionary)
    self.assertEqual(merged.dictionary['scalar_names'], ['a', 'b'])

    # Test with dataset1 being passed as first argument
    merged = dataset_utils.zipper_merge_datasets(dataset1, dataset0)
    self.assertLen(merged.sessions, 3)
    self.assertIs(merged.sessions[0], d1_sessions[0])
    self.assertIs(merged.sessions[1], d0_sessions[0])
    self.assertIs(merged.sessions[2], d0_sessions[1])
    expected_choices_10 = np.array([
        [1, 0, -1],
        [0, 1, 0],
        [1, 0, 1],
    ])
    np.testing.assert_array_equal(
        merged.dictionary['choice'], expected_choices_10
    )
    self.assertNotIn('scalar_names', merged.dictionary)

  def testMergeDatasets(self):
    # dataset 0
    d0_choices = np.array([[0, 1, 0], [1, 0, 1]])
    d0_rewards = np.array([[0, 1, 2], [0, 1, 2]])
    d0_ignore = np.array([[False, False, False], [False, False, False]])
    d0_sessions = [
        dataset_types.NeuroSession(
            trials=[
                dataset_types.BanditTrial(choice=0, reward=0, ignore=False),
                dataset_types.BanditTrial(choice=1, reward=1, ignore=False),
                dataset_types.BanditTrial(choice=0, reward=2, ignore=False),
            ]
        ),
        dataset_types.NeuroSession(
            trials=[
                dataset_types.BanditTrial(choice=1, reward=0, ignore=False),
                dataset_types.BanditTrial(choice=0, reward=1, ignore=False),
                dataset_types.BanditTrial(choice=1, reward=2, ignore=False),
            ]
        ),
    ]
    dataset0 = dataset_types.NeuroDataset(
        sessions=d0_sessions,
        dictionary={
            'choice': d0_choices,
            'reward': d0_rewards,
            'ignore': d0_ignore,
            'scalar_names': ['a', 'b'],
        },
    )

    # dataset 1
    d1_choices = np.array([[1, 0]])
    d1_rewards = np.array([[10, 11]])
    d1_ignore = np.array([[False, False]])
    d1_sessions = [
        dataset_types.NeuroSession(
            trials=[
                dataset_types.BanditTrial(choice=1, reward=10, ignore=False),
                dataset_types.BanditTrial(choice=0, reward=11, ignore=False),
            ]
        ),
    ]
    dataset1 = dataset_types.NeuroDataset(
        sessions=d1_sessions,
        dictionary={
            'choice': d1_choices,
            'reward': d1_rewards,
            'ignore': d1_ignore,
            'scalar_names': ['a', 'b'],
        },
    )

    merged = dataset_utils.merge_datasets([dataset0, dataset1])

    self.assertLen(merged.sessions, 3)
    self.assertIs(merged.sessions[0], d0_sessions[0])
    self.assertIs(merged.sessions[1], d0_sessions[1])
    self.assertIs(merged.sessions[2], d1_sessions[0])

    # Test dictionary merging and padding.
    # dataset0 has shape (2,3), dataset1 has shape (1,2).
    # max_shape along trial dim is 3.
    # d1 arrays should be padded to length 3.
    # merged choices should come from d0s0, d0s1, d1s0
    expected_choices = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [1, 0, -1],  # padded from [1,0]
    ])
    expected_rewards = np.array([
        [0, 1, 2],
        [0, 1, 2],
        [10, 11, -1],  # padded from [10, 11]
    ])
    expected_ignore = np.array([
        [False, False, False],
        [False, False, False],
        [False, False, True],  # padded from [False, False]
    ])

    np.testing.assert_array_equal(merged.dictionary['choice'], expected_choices)
    np.testing.assert_array_equal(merged.dictionary['reward'], expected_rewards)
    np.testing.assert_array_equal(merged.dictionary['ignore'], expected_ignore)
    self.assertIn('scalar_names', merged.dictionary)
    self.assertEqual(merged.dictionary['scalar_names'], ['a', 'b'])


if __name__ == '__main__':
  absltest.main()
