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

"""Datatypes for dataset loaders."""

import abc
import dataclasses
from typing import Callable, Dict, Optional, Self, TypedDict

import chex
import jax
import numpy as np


DataDict = Dict[str, chex.Array]


class TrainableDataset(TypedDict):
  """A dataset that can be used for training."""

  inputs: DataDict
  targets: DataDict


@dataclasses.dataclass
class NeuroTimestep:
  ...


@dataclasses.dataclass(kw_only=True)
class BanditTrial(NeuroTimestep):
  choice: int
  reward: int | float
  instructed: bool = False
  ignore: bool = False
  scalars: np.ndarray | None = None
  scalar_names: list[str] | None = None
  padding: bool = False


@dataclasses.dataclass
class NeuroSession:
  trials: list[NeuroTimestep]

  def get_trial_types(self) -> list[str]:
    if not self.trials:
      return []

    return list(self.trials[0].__dataclass_fields__.keys())


@dataclasses.dataclass
class NeuroDataset:
  """Class of datasets returned by loaders.

  This object will hold the data in two formats:
  - sessions uses the hierarchical structure defined above, which can make it
    easier for inspection of the data.
  - dictionary maps field names to numpy arrays containing the loaded data for
    each separate field; this is typically used for training.
  - inputs is a dictionary containing the data that will be passed to the agent
    function as keyword arguments.
  - targets is a dictionary containing the data that will be used as targets
    for the agent function.
  """

  sessions: list[NeuroSession]
  dictionary: DataDict
  inputs: Optional[dict[str, np.ndarray]] = None
  targets: Optional[dict[str, np.ndarray]] = None

  def slice(self, slice_fn: Callable[[chex.Array], chex.Array]) -> Self:
    """Select a subset of the sessions using the given slice function.

    Args:
      slice_fn: a function that selects a subset of the rows of an inputed
        array. For example, lambda x: x[::2] would select even numbered
        sessions.

    Returns:
      A NeuroDataset with the same structure as the input, but with
      the sessions, dictionary, inputs, and targets sliced according to the
      given slice function.
    """
    inds = np.arange(len(self.sessions))
    inds = slice_fn(inds)
    sessions = [self.sessions[i] for i in inds]

    def _slice_fn_arrays_only(x):
      if isinstance(x, chex.Array):
        return slice_fn(x)
      else:
        return x

    dictionary = jax.tree_util.tree_map(_slice_fn_arrays_only, self.dictionary)
    if self.inputs is None:
      inputs = None
    else:
      inputs = jax.tree_util.tree_map(slice_fn, self.inputs)

    if self.targets is None:
      targets = None
    else:
      targets = jax.tree_util.tree_map(slice_fn, self.targets)

    split_dataset = NeuroDataset(
        sessions=sessions, dictionary=dictionary, inputs=inputs, targets=targets
    )
    return split_dataset

  def as_trainable_dataset(self) -> TrainableDataset:
    """Returns the dataset as a TrainableDataset."""
    if self.inputs is None or self.targets is None:
      raise ValueError("Inputs or targets are None.")
    return TrainableDataset(inputs=self.inputs, targets=self.targets)

  @classmethod
  def from_data_dict(
      cls,
      data_dict: DataDict,
      trial_type: type[NeuroTimestep],
      *,
      inputs: Optional[DataDict] = None,
      targets: Optional[DataDict] = None,
  ):
    """Builds a NeuroDataset from a data dictionary."""
    example_arrays = next(iter(data_dict.values()))
    num_sessions = len(example_arrays)
    sessions = []
    for s in range(num_sessions):
      num_trials = len(example_arrays[s])
      trials = []
      for t in range(num_trials):
        trial_kwargs = {k: v[s, t] for k, v in data_dict.items()}
        trials.append(trial_type(**trial_kwargs))
      sessions.append(NeuroSession(trials=trials))
    return cls(
        sessions=sessions, dictionary=data_dict, inputs=inputs, targets=targets
    )


@dataclasses.dataclass
class ArgSpec:
  """Class for defining the signature of an agent function.

  Attributes:
    name: The name of the argument.
    dtype: The dtype of the argument.
    n_values: The number of values the argument can take. Not valid for
      non-integer arguments.
    should_shift: Whether the argument should be shifted by one trial. If True,
      the argument corresponds to a value from the previous trial.
    noinfo_name_override: If provided, the name of the argument in noinfo
      format.
    noinfo_name: The name of the argument in noinfo format (initialize this
      through noinfo_name_override if you want to override the default name).
  """

  name: str
  dtype: str
  n_values: int | None = None  # Only supported for integer arguments.
  should_shift: bool = True
  noinfo_name_override: dataclasses.InitVar[Optional[str]] = None
  noinfo_name: str = dataclasses.field(init=False)

  def __post_init__(self, noinfo_name_override: Optional[str]):
    if noinfo_name_override is None:
      self.noinfo_name = self.name
    else:
      self.noinfo_name = noinfo_name_override
    assert (
        self.n_values is None or self.dtype == "int"
    ), "n_values is only supported for integer arguments."


class DatasetLoader(abc.ABC):
  """Abstract base class for dataset loaders."""

  @classmethod
  @abc.abstractmethod
  def load(cls, **kwargs) -> Optional[NeuroDataset]:
    ...

  @classmethod
  @abc.abstractmethod
  def num_subjects(cls) -> int:
    """Returns the number of subjects in the dataset."""
    ...

  @classmethod
  def output_shape(cls) -> tuple[int, ...] | None:
    """Returns the required output shape of agent functions for this dataset."""
    return None

  @classmethod
  @abc.abstractmethod
  def identifier(cls) -> str:
    """Returns a human readable name for the dataset."""
    ...

  @classmethod
  def supports_multisubject(cls) -> bool:
    """Returns whether the dataset supports multisubject training."""
    return False

  @classmethod
  def supports_single_subject(cls) -> bool:
    """Returns whether the dataset supports single-subject training."""
    return False

  @classmethod
  def agent_signature(cls) -> list[ArgSpec]:
    """Returns the signature of the agent function."""
    return [
        ArgSpec(
            name="choice", noinfo_name_override="x", dtype="int", n_values=2
        ),
        ArgSpec(
            name="reward", noinfo_name_override="y", dtype="int", n_values=2
        ),
    ]

  @classmethod
  @abc.abstractmethod
  def default_file_path(cls) -> str:
    """Returns the default file path for the dataset."""
    ...
