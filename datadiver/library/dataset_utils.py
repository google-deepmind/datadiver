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

"""Utility functions for loading and processing datasets."""

import itertools
from typing import Any, Callable

import chex
from datadiver.library import dataset_types
import jax
import jax.numpy as jnp
import numpy as np

DataDict = dataset_types.DataDict
NeuroDataset = dataset_types.NeuroDataset
TrainableDataset = dataset_types.TrainableDataset


PADDING_VALUES = {
    'choice': -1,
    'reward': -1,
    'instructed': False,
    'is_novel': -1,
    'state': -1,
    'ignore': True,
    'padding': True,
    'scalars': -1.0,
    'payouts': -1,
}


def get_indices_from_split(split: str | int, n: int) -> tuple[int, ...]:
  """Returns the indices to load based on the split.

  Args:
    split: The split to use. Can be'even', 'odd', 'all', 'every4th', or an
      integer with magnitude <= n. If split is a positive integer i, the first i
      items will be loaded. If it is a negative integer -i, the last i items
      will be loaded.
    n: The number of items to split.

  Returns:
    A range of indices to load.
  """
  # Remove quote marks from split string if present.
  # TODO(stachenfeld): Find out why FunSearch needs an extra set of quotes.
  split = str(split).replace('"', '').replace("'", '')

  if n <= 0:
    raise ValueError(f'Invalid number of items: {n}')

  if split == 'even':
    inds = range(0, n, 2)
  elif split == 'odd':
    inds = range(1, n, 2)
  elif split == 'all':
    inds = range(n)
  elif split == 'every4th':
    inds = range(0, n, 4)
  elif isinstance(split, str) and (
      split.isnumeric() or (split.startswith('-') and split[1:].isnumeric())
  ):
    split = int(split)
    if abs(split) > n:
      raise ValueError(f'Invalid split: {split} (must be <= {n})')
    if split >= 0:
      inds = range(split)
    else:
      inds = range(n + split, n)
  else:
    raise ValueError(f'Invalid split: {split}')
  return tuple(inds)


def _insert_dummy_trial_at_start(
    x: chex.Array, dummy_val: int = -1, keep_shape: bool = True
) -> chex.Array:
  """Insert dummy trial at the start of each input field.

  Args:
    x: array that is n_sessions x n_trials x ...
    dummy_val: The value to insert as a dummy trial for trial 0.
    keep_shape: If True, the shape of the input array will be preserved by
      truncating the last trial. Otherwise, the shape of the returned array will
      be n_sessions x (n_trials+1) x ...

  Returns:
    A new array with the dummy trial inserted at the start of each session.
  """
  x_with_dummy = jnp.concatenate(
      [dummy_val * jnp.ones_like(x[:, :1]), x], axis=1
  )
  if keep_shape:
    x_with_dummy = x_with_dummy[:, :-1]
  return x_with_dummy


def format_data_for_next_choice_prediction(
    data_dict: DataDict,
    input_fields: tuple[str, ...] | list[str],
    *,
    input_fields_to_shift: tuple[str, ...] | list[str] = (),
    choice_field: str = 'choice',
    add_previous_to_input_fields_name: bool = False,
    dummy_first_trial: bool = False,
    dummy_val: int = -1,
) -> tuple[DataDict, DataDict]:
  """Formats the dataset for training on next choice prediction.

  Args:
    data_dict: The dictionary of data, with fields that will be used to format
      input and targets. Must contain 'choice' and 'ignore', and will contain
      'reward' as well for n-armed bandit datasets. All fields in
      input_fields_prev_trial and input_fields_curr_trial must be present as
      keys in data_dict.
    input_fields: The name of fields to include in input.
    input_fields_to_shift: The name of fields to include in input that
      correspond to information about the next trial that inform the next choice
      prediction, and therefore should be shifted to match the indexing of the
      next choice. For example, this might include 'obs' if the prediction
      problem is to predict the next choice, based on the previous choice,
      previous reward, and observation of the current trial. Default=().
    choice_field: The name of the field that contains the choice. Defaults to
      'choice'.
    add_previous_to_input_fields_name: Whether to add a prefix to the input
      fields.
    dummy_first_trial: If True, a dummy "first trial" will be inserted at the
      start of each session in order to shift the inputs with respect to the
      targets. Otherwise, the first trial will be removed from the target in
      order to shift the data with respect to the input.
    dummy_val: The value to use as a dummy trial. Defaults to -1.

  Returns:
    inputs: A dictionary of input fields, with keys corresponding to the input
      fields specified in input_fields.
    targets: A dictionary of target fields containing 'next_choice' and
      'ignore'.
  """
  if choice_field in input_fields_to_shift:
    raise ValueError((
        'If choice_field is an input field, it must not be in'
        ' input_fields_to_shift. Otherwise, the model will predict current'
        ' choice from current choice, which is not a next choice prediction'
        ' problem.'
    ))

  if dummy_first_trial:
    # Current trials will have dummy trial at the start to shift forward with
    # respect to the next trial targets and next trial input features.
    keep_at_curr_trial = lambda x: _insert_dummy_trial_at_start(x, dummy_val)
    shift_to_next_trial = lambda x: x
  else:
    # First trial will be removed from next trial targets and next trial input
    # features to shift back with respect to current trial input features.
    keep_at_curr_trial = lambda x: x[:, :-1]
    shift_to_next_trial = lambda x: x[:, 1:]

  inputs = {}
  for k in input_fields:
    if k not in data_dict:
      raise ValueError(f'Input field {k} not found in data_dict.')

    if k in input_fields_to_shift:
      inputs[k] = shift_to_next_trial(data_dict[k])

    else:
      name = ('previous_' + k) if add_previous_to_input_fields_name else k
      inputs[name] = keep_at_curr_trial(data_dict[k])

  targets = {
      'next_choice': shift_to_next_trial(data_dict[choice_field]),
  }
  if 'ignore' in data_dict:
    targets['ignore'] = shift_to_next_trial(data_dict['ignore'])

  return inputs, targets


def split_dataset(
    dataset: NeuroDataset,
    split0_fn: Callable[[chex.Array], chex.Array] = lambda x: x[::2],
    split1_fn: Callable[[chex.Array], chex.Array] = lambda x: x[1::2],
) -> tuple[NeuroDataset, NeuroDataset]:
  """Split a dataset in two."""
  assert not set(split0_fn(np.arange(10))) & set(split1_fn(np.arange(10)))
  dataset0 = dataset.slice(split0_fn)
  dataset1 = dataset.slice(split1_fn)
  return dataset0, dataset1


def split_and_prepare_data_for_jit(
    dataset: NeuroDataset | TrainableDataset,
    split0_fn: Callable[[chex.Array], chex.Array] = lambda x: x[::2],
    split1_fn: Callable[[chex.Array], chex.Array] = lambda x: x[1::2],
) -> tuple[TrainableDataset, TrainableDataset]:
  """Convert a NeuroDataset into a trainable format and split in two."""
  # First verify that the two functions produce non-overlapping sets.
  assert not set(split0_fn(np.arange(10))) & set(split1_fn(np.arange(10)))

  if isinstance(dataset, NeuroDataset):
    dataset = dataset.as_trainable_dataset()

  dataset = jax.tree_util.tree_map(jnp.array, dataset)
  dataset_0 = jax.tree_util.tree_map(split0_fn, dataset)
  dataset_1 = jax.tree_util.tree_map(split1_fn, dataset)
  return (dataset_0, dataset_1)


def _get_pad_value(key: str | None = None) -> Any:
  """Get the pad value for a given key."""
  if key is not None:
    return PADDING_VALUES[key]
  else:
    return -1


def pad_and_concatenate_sessions(
    list_of_arrays: list[chex.Array], pad_key: str | None = None
) -> chex.Array:
  """Pads session data to match longest session, then concatenates.

  Args:
    list_of_arrays: A list of arrays, each representing a session.
    pad_key: The key to use to deternine padding value.

  Returns:
    A single array containing the padded and concatenated sessions.
  """
  if not list_of_arrays:
    raise ValueError('list_of_arrays must not be empty')
  max_session_length = np.max([x.shape[0] for x in list_of_arrays])

  first_array_shape = np.shape(list_of_arrays[0])
  first_array_dtype = list_of_arrays[0].dtype
  padding_shape_suffix = first_array_shape[1:]

  padded_list_of_arrays = []
  for x in list_of_arrays:
    padding_length = max_session_length - x.shape[0]
    if padding_length > 0:
      # Pad the session data with -1s to match max_session_length
      padding_shape = (padding_length,) + padding_shape_suffix
      pad_value = _get_pad_value(pad_key)  ### Use get_pad_value (was -1)
      padding = pad_value * np.ones(padding_shape, dtype=first_array_dtype)
      padded_list_of_arrays.append(np.concatenate((x, padding), axis=0))
    elif padding_length == 0:
      # This session does not need to be padded
      padded_list_of_arrays.append(x)
    else:
      raise ValueError('padding_length cannot be negative.')

  return np.stack(padded_list_of_arrays, axis=0)


def _pad_and_concatenate(
    arrays: list[np.ndarray], axis: int = 0, pad_value: int = -1
) -> np.ndarray:
  """Pad arrays to the same size along nonzero axes and concatenate them."""
  max_shape = np.max([arr.shape for arr in arrays], axis=0)
  new_arrays = []
  for arr in arrays:
    pad_width = [
        (0, max_shape[i] - arr.shape[i]) if i != axis else (0, 0)
        for i in range(len(arr.shape))
    ]
    new_arrays.append(
        np.pad(arr, pad_width, mode='constant', constant_values=pad_value)
    )
  return np.concatenate(new_arrays, axis=axis)


def _zipper_merge_lists(list0: list[Any], list1: list[Any]) -> list[Any]:
  """Merge two lists, zipper-style, into a single list."""
  zipped_list = []
  for item_0, item_1 in itertools.zip_longest(list0, list1):
    if item_0 is not None:
      zipped_list.append(item_0)
    if item_1 is not None:
      zipped_list.append(item_1)
  return zipped_list


def zipper_merge_datasets(
    dataset0: NeuroDataset, dataset1: NeuroDataset
) -> NeuroDataset:
  """Merge two datasets, zipper-style,  into a single dataset.

  Args:
    dataset0: The first dataset to merge.
    dataset1: The second dataset to merge.

  Returns:
    A NeuroDataset containing the merged data.
  """
  sessions = _zipper_merge_lists(dataset0.sessions, dataset1.sessions)
  dictionary = {}
  for key in dataset0.dictionary.keys():
    if key == 'scalar_names':
      dictionary[key] = dataset0.dictionary[key]
    else:
      dictionary[key] = _pad_and_concatenate(
          _zipper_merge_lists(
              [np.expand_dims(row, axis=0) for row in dataset0.dictionary[key]],
              [np.expand_dims(row, axis=0) for row in dataset1.dictionary[key]],
          ),
          axis=0,
          pad_value=_get_pad_value(key),
      )
  return NeuroDataset(
      sessions=sessions,
      dictionary=dictionary,
  )


def merge_datasets(datasets: list[NeuroDataset]) -> NeuroDataset:
  """Merge a list of per-subject datasets into a single multisubject dataset."""
  sessions = []
  for dataset in datasets:
    sessions.extend(dataset.sessions)

  merged_dictionary = {}
  if datasets:
    if not all(
        set(d.dictionary.keys()) == set(datasets[0].dictionary.keys())
        for d in datasets
    ):
      raise ValueError('All datasets must have the same dictionary keys.')
    for key in datasets[0].dictionary:
      if key == 'scalar_names':
        merged_dictionary[key] = datasets[0].dictionary[key]
      else:
        merged_dictionary[key] = _pad_and_concatenate(
            [d.dictionary[key] for d in datasets],
            pad_value=_get_pad_value(key),
            axis=0,
        )
  return NeuroDataset(sessions=sessions, dictionary=merged_dictionary)
