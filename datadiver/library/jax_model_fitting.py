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

"""Utilities for fitting choice prediction models.

The types and functions defined here constitute the collection of datadiver
skills for fitting choice prediction models.

When used a skills, these functions should be copied directly verbatim into the
namespace where they are used.
"""

import functools
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

import chex
import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.typing
import numpy as np
import optax

LARGE_NUMBER = 1e10
VERY_LARGE_NUMBER = 1e15


class ChoicePredictionAgentFunction(Protocol):
  """Protocol defining the interface for agent functions used in datadiver.

  Use this protocol for interfacing with a ChoicePredictionDataset.

  Note that this requires all arguments to be passed as keyword args.
  """

  def __call__(
      self,
      *,
      params: chex.Array,
      agent_state: Any,
      **inputs: jax.typing.ArrayLike,
  ) -> tuple[chex.Array, Any]:
    """Computes choice logits and updates agent state.

    Args:
      params: Model parameters.
      agent_state: Current internal state of the agent.
      **inputs: Inputs to the agent. This should match the fields of
        dataset.inputs for the corresponding ChoicePredictionDataset.

    Returns:
      choice_logits: Array of logits for the next choice.
      next_agent_state: Updated internal state of the agent.
    """
    ...


@chex.dataclass
class ChoicePredictionDataset:
  """Structured dataset for choice prediction.

  Attributes:
    inputs: Dictionary of inputs to the agent.
    next_choice: Array representing the target choices to predict.
    next_ignore: Binary mask array indicating which trials to ignore in loss.
  """

  inputs: dict[str, chex.Array]
  next_choice: chex.Array
  next_ignore: chex.Array

  def __post_init__(self):
    def get_len(x) -> int:
      if hasattr(x, 'shape'):
        return x.shape[0] if x.ndim > 0 else 1
      try:
        return len(x)
      except TypeError:
        return 1

    lengths = {
        'next_choice': get_len(self.next_choice),
        'next_ignore': get_len(self.next_ignore),
    }
    for k, v in self.inputs.items():
      lengths[f'inputs[{k}]'] = get_len(v)
    if len(set(lengths.values())) > 1:
      raise ValueError(
          'All arrays in ChoicePredictionDataset must have the same length'
          f' (first dimension). Found lengths: {lengths}'
      )

  @property
  def targets(self) -> dict[str, chex.Array]:
    """Returns targets dictionary for model fitting/evaluation."""
    return {'next_choice': self.next_choice, 'next_ignore': self.next_ignore}


def make_choice_prediction_dataset(
    data_dict: dict[str, chex.Array],
    input_fields: Sequence[str],
    *,
    choice_field: str = 'choice',
    reward_field: str = 'reward',
    ignore_field: str = 'ignore',
) -> ChoicePredictionDataset:
  """Formats data dictionary to ChoicePredictionDataset format."""
  inputs = {k: data_dict[k] for k in input_fields}
  next_choice = data_dict[choice_field]
  next_ignore = data_dict[ignore_field]

  # Shift inputs by one trial so they represent past trial data.
  shifted_inputs = {}
  for k, v in inputs.items():
    if k == choice_field or k == reward_field:
      # Pad first trial with 0.
      padded_v = jnp.pad(
          v[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=0
      )
      shifted_inputs[k] = padded_v
    else:
      # Do not shift context fields (e.g. stimulus info).
      shifted_inputs[k] = v

  return ChoicePredictionDataset(
      inputs=shifted_inputs,
      next_choice=next_choice,
      next_ignore=next_ignore,
  )


def split_choice_prediction_dataset(
    dataset: ChoicePredictionDataset,
) -> tuple[ChoicePredictionDataset, ChoicePredictionDataset]:
  """Splits ChoicePredictionDataset into even and odd sessions."""
  dataset0 = ChoicePredictionDataset(
      inputs=jax.tree_util.tree_map(lambda x: x[::2], dataset.inputs),
      next_choice=dataset.next_choice[::2],
      next_ignore=dataset.next_ignore[::2],
  )
  dataset1 = ChoicePredictionDataset(
      inputs=jax.tree_util.tree_map(lambda x: x[1::2], dataset.inputs),
      next_choice=dataset.next_choice[1::2],
      next_ignore=dataset.next_ignore[1::2],
  )
  return dataset0, dataset1


def _process_one_trial_dummy(
    ll_state: tuple[float, Any],
    logits_shape: tuple[int, ...] | None = None,
    compute_predictions: bool = False,
) -> tuple[tuple[float, Any], dict[str, chex.Array]]:
  """Dummy trial step for padding/ignored trials after session end."""
  total_ll, agent_state = ll_state
  logits_shape = logits_shape if logits_shape is not None else (2,)
  if compute_predictions:
    predictions = {
        'choice_logits': jnp.zeros(logits_shape),
        'chosen_logprob': jnp.zeros(()),
        'agent_state': agent_state,
    }
  else:
    predictions = {'loss': jnp.array(total_ll)}
  return (total_ll, agent_state), predictions


def _process_one_trial_step(
    ll_state: Tuple[float, Any],
    dataset: ChoicePredictionDataset,
    *,
    agent_fun: ChoicePredictionAgentFunction,
    params: chex.Array | None,
    expected_output_shape: tuple[int, ...] | None,
    compute_predictions: bool,
    verbose: bool = True,
) -> tuple[tuple[chex.Array, Any], dict[str, chex.Array]]:
  """Get NLL and predictions for one trial."""
  # Unpack agent_state, choice, reward, ignore, and next_choice.
  total_ll, current_agent_state = ll_state

  next_choice = dataset.next_choice
  ignore_next_choice = dataset.next_ignore

  # Apply agent function using explicit arguments
  choice_logits, next_agent_state = agent_fun(
      params=params,
      agent_state=current_agent_state,
      **dataset.inputs,
  )
  if not isinstance(next_agent_state, chex.Array):
    raise ValueError(
        'The agent state returned by the agent function must be a Chex array,'
        f' not {next_agent_state}.'
    )
  choice_logprobs = choice_logits - jsp.special.logsumexp(choice_logits)

  # Check that output shape is as expected.
  if expected_output_shape is not None:
    if choice_logits.shape != expected_output_shape:
      total_ll -= LARGE_NUMBER
      if verbose:
        print('Bad shape -- adding large penalty to total_ll')

  # If choice probabilities do not sum to 1, penalize ll
  # This can happen if logits take extreme values and cause numerical issues.
  bad_logits = jnp.abs(jnp.sum(jnp.exp(choice_logprobs)) - 1.0) > 1e-3
  total_ll -= LARGE_NUMBER * bad_logits

  chosen_logprob = choice_logprobs[jnp.int32(next_choice)]

  # Penalizing whenever the choice_logprobs aren't big enough to contain the
  # correct choice. Otherwise, due to peculiarities of indexing in JAX, we'll
  # spuriously count some trials as correct.
  chosen_logprob -= LARGE_NUMBER * (next_choice >= len(choice_logprobs))

  # Increment ll unless this is an ignore trial
  total_ll += lax.cond(ignore_next_choice, lambda: 0.0, lambda: chosen_logprob)

  # Return predictions if requested.
  if compute_predictions:
    predictions = {
        'choice_logits': choice_logits,
        'chosen_logprob': chosen_logprob,
        'agent_state': next_agent_state,
    }
  else:
    predictions = {'loss': total_ll}

  ll_state = (total_ll, next_agent_state)
  return ll_state, predictions


def _process_one_trial(
    ll_state: tuple[float, Any],
    dataset: ChoicePredictionDataset,
    *,
    agent_fun: ChoicePredictionAgentFunction,
    params: chex.Array | None,
    expected_output_shape: tuple[int, ...] | None,
    compute_predictions: bool,
    logits_shape: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> tuple[tuple[chex.Array, Any], dict[str, chex.Array]]:
  """Get NLL and predictions for one trial."""

  session_over = dataset.next_choice == -1

  return lax.cond(
      session_over,
      lambda: _process_one_trial_dummy(
          ll_state=ll_state,
          logits_shape=logits_shape,
          compute_predictions=compute_predictions,
      ),
      lambda: _process_one_trial_step(
          ll_state=ll_state,
          dataset=dataset,
          agent_fun=agent_fun,
          params=params,
          expected_output_shape=expected_output_shape,
          compute_predictions=compute_predictions,
          verbose=verbose,
      ),
  )


def _concat_init_rest_dict(
    init_d: dict[str, chex.Array],
    rest_d: dict[str, chex.Array],
) -> dict[str, chex.Array]:
  """Concatenates the first trial results with later trials."""
  res = {}
  for k in init_d:
    res[k] = jnp.concatenate(
        [jnp.expand_dims(init_d[k], axis=0), rest_d[k]], axis=0
    )
  return res


def _process_one_session(
    total_ll: float | chex.Array,
    dataset: ChoicePredictionDataset,
    *,
    agent_fun: ChoicePredictionAgentFunction,
    params: chex.Array | None,
    expected_output_shape: tuple[int, ...] | None,
    compute_predictions: bool,
    verbose: bool = True,
) -> tuple[float, tuple[chex.Array, dict[str, chex.Array]]]:
  """Processes one session, incrementing ll."""
  # Initialize the log-likelihood and agent state.
  init_session_ll = 0.0
  init_agent_state = None  # Initialize agent_state to None.
  ll_state_0 = (init_session_ll, init_agent_state)

  # Process first trial.
  # This is special because agent_state is None on the first trial,
  # not an array.
  dataset_0 = jax.tree_util.tree_map(lambda x: x[0], dataset)
  first_trial_fun = functools.partial(
      _process_one_trial_step,
      agent_fun=agent_fun,
      params=params,
      expected_output_shape=expected_output_shape,
      compute_predictions=compute_predictions,
      verbose=verbose,
  )

  ll_state_1, pred_1 = first_trial_fun(ll_state_0, dataset_0)

  logits_shape = None
  if compute_predictions and 'choice_logits' in pred_1:
    logits_shape = pred_1['choice_logits'].shape

  # Process all later trials.
  dataset_after_first = jax.tree_util.tree_map(lambda x: x[1:], dataset)

  trial_fun_after_first = functools.partial(
      _process_one_trial,
      agent_fun=agent_fun,
      params=params,
      expected_output_shape=expected_output_shape,
      compute_predictions=compute_predictions,
      logits_shape=logits_shape,
      verbose=verbose,
  )

  (session_ll, _), pred_after_first = lax.scan(
      trial_fun_after_first, ll_state_1, dataset_after_first
  )

  predictions = _concat_init_rest_dict(pred_1, pred_after_first)

  total_ll += session_ll
  return total_ll, (session_ll, predictions)


def _compute_nll(
    params: Any,
    agent_fun: ChoicePredictionAgentFunction,
    *,
    dataset: ChoicePredictionDataset,
    expected_output_shape: tuple[int, ...] | None = None,
    compute_predictions: bool = False,
    verbose: bool = True,
) -> tuple[float, tuple[chex.Array, dict[str, chex.Array]]]:
  """Compute negative log-likelihood given parameters, model, and dataset."""

  sess_fun = functools.partial(
      _process_one_session,
      agent_fun=agent_fun,
      params=params,
      expected_output_shape=expected_output_shape,
      compute_predictions=compute_predictions,
      verbose=verbose,
  )

  total_ll, (sess_ll, predictions) = lax.scan(sess_fun, 0.0, dataset)

  total_nll = -total_ll
  sess_nll = -sess_ll
  return total_nll, (sess_nll, predictions)


_compute_nll_jitted = jax.jit(
    _compute_nll,
    static_argnames=[
        'agent_fun',
        'expected_output_shape',
        'compute_predictions',
        'verbose',
    ],
)


def _negloglik_to_normlik(
    nll: float | chex.Array, n_valid_trials: int | chex.Array
) -> float | chex.Array:
  """Convert negative log-likelihood to normalized likelihood."""
  assert (
      np.array([nll]).shape == np.array([n_valid_trials]).shape
  ), 'nll and n_valid_trials must have the same shape'
  return np.exp(-nll / n_valid_trials)


@functools.partial(
    jax.jit,
    static_argnames=[
        'agent_fun',
        'optimizer',
        'expected_output_shape',
        'verbose',
        'is_lbfgs',
    ],
)
def _opt_step(
    params: Any,
    opt_state: Any,
    *,
    dataset: ChoicePredictionDataset,
    agent_fun: ChoicePredictionAgentFunction,
    optimizer: optax.GradientTransformationExtraArgs,
    expected_output_shape: tuple[int, ...] | None,
    verbose: bool = True,
    is_lbfgs: bool = False,
) -> Tuple[Any, Any]:
  """Apply jitted step of an optimizer to a set of parameters."""
  if is_lbfgs:
    value_grad_fn = jax.value_and_grad(_compute_nll, has_aux=True)
    (loss_val, _), grads = value_grad_fn(
        params,
        dataset=dataset,
        agent_fun=agent_fun,
        expected_output_shape=expected_output_shape,
        verbose=verbose,
    )
    loss_fn_lbfgs = lambda p: _compute_nll(
        p,
        dataset=dataset,
        agent_fun=agent_fun,
        expected_output_shape=expected_output_shape,
        verbose=verbose,
    )[0]
    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        params,
        value=loss_val,
        grad=grads,
        value_fn=loss_fn_lbfgs,
    )
  else:
    grads, _ = jax.grad(_compute_nll, has_aux=True)(
        params,
        dataset=dataset,
        agent_fun=agent_fun,
        expected_output_shape=expected_output_shape,
        verbose=verbose,
    )
    updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return (params, opt_state)


def _score_match(score, best_score, score_match_eps):
  return np.abs(score - best_score) / best_score < score_match_eps


@functools.partial(
    jax.jit,
    static_argnames=[
        'agent_fun',
        'optimizer',
        'n_params',
        'n_opt_steps',
        'max_iterations',
        'expected_output_shape',
        'verbose',
        'is_lbfgs',
    ],
)
def _run_optimization_loop(
    rng: jax.Array,
    *,
    dataset: ChoicePredictionDataset,
    agent_fun: ChoicePredictionAgentFunction,
    optimizer: optax.GradientTransformation,
    n_params: int,
    n_opt_steps: int,
    max_iterations: int,
    score_percent_change_thresh: float,
    expected_output_shape: tuple[int, ...] | None,
    verbose: bool = True,
    is_lbfgs: bool = False,
) -> Tuple[chex.Array, float]:
  """JIT-compiled nested loop for training convergence."""
  params = jax.random.normal(rng, shape=(n_params,))
  opt_state = optimizer.init(params)

  nll_kwargs = {
      'agent_fun': agent_fun,
      'dataset': dataset,
      'expected_output_shape': expected_output_shape,
      'verbose': verbose,
  }

  initial_score, _ = _compute_nll(params, **nll_kwargs)

  # (params, opt_state, score, iterations_complete, continue_training)
  init_val = (
      params,
      opt_state,
      initial_score,
      jnp.array(0, dtype=jnp.int32),
      jnp.array(True, dtype=jnp.bool_),
  )

  def cond_fun(val) -> chex.Array:
    return val[4]

  def body_fun(val):
    params, opt_state, old_score, iterations, _ = val

    # Inner loop over n_opt_steps using lax.scan
    def step_fn(carry, _):
      p, s = carry
      new_p, new_s = _opt_step(
          p,
          s,
          optimizer=optimizer,
          is_lbfgs=is_lbfgs,
          **nll_kwargs,
      )
      return (new_p, new_s), None

    (new_params, new_opt_state), _ = lax.scan(
        step_fn, (params, opt_state), length=n_opt_steps
    )

    new_score, _ = _compute_nll(new_params, **nll_kwargs)

    # Check convergence criteria
    converged = (
        jnp.abs(new_score - old_score) / old_score < score_percent_change_thresh
    )
    max_iter_reached = iterations + 1 >= max_iterations
    nan_params = jnp.isnan(new_params).any()
    nan_score = jnp.isnan(new_score)

    continue_training = jnp.logical_not(
        jnp.logical_or(
            converged,
            jnp.logical_or(
                max_iter_reached, jnp.logical_or(nan_params, nan_score)
            ),
        )
    )

    return (
        new_params,
        new_opt_state,
        new_score,
        iterations + 1,
        continue_training,
    )

  final_val = lax.while_loop(cond_fun, body_fun, init_val)
  return final_val[0], final_val[2]


def _train_to_convergence(
    agent_fun: ChoicePredictionAgentFunction,
    dataset: ChoicePredictionDataset,
    rng: jax.Array,
    optimizer: optax.GradientTransformation = optax.adabelief(learning_rate=5e-2),  # pylint: disable=line-too-long
    n_params: int = 10,
    n_opt_steps: int = 100,
    max_iterations: int = 100,
    score_percent_change_thresh: float = 1e-4,
    expected_output_shape: tuple[int, ...] | None = None,
    verbose: bool = True,
    is_lbfgs: bool = False,
):
  """Optimize params of the agent function until the score converges."""
  params, score = _run_optimization_loop(
      rng,
      dataset=dataset,
      agent_fun=agent_fun,
      optimizer=optimizer,
      n_params=n_params,
      n_opt_steps=n_opt_steps,
      max_iterations=max_iterations,
      score_percent_change_thresh=score_percent_change_thresh,
      expected_output_shape=expected_output_shape,
      verbose=verbose,
      is_lbfgs=is_lbfgs,
  )
  return np.array(params), float(score)


def fit_choice_prediction_model(
    agent_fun: ChoicePredictionAgentFunction,
    dataset: ChoicePredictionDataset,
    rng: jax.Array | None = None,
    optimizer: optax.GradientTransformation = optax.adabelief(learning_rate=5e-2),  # pylint: disable=line-too-long
    n_params: int = 10,
    n_converged_to_best: int = 3,
    score_match_eps: float = 1e-3,
    max_attempts_grad_descent: int = 10,
    n_steps_per_epoch_grad_descent: int = 100,
    n_epochs_grad_descent: int = 100,
    expected_output_shape: tuple[int, ...] | None = None,
    convergence_thresh_grad_descent: float = 1e-4,
    verbose: bool = True,
    is_lbfgs: bool = False,
) -> Tuple[chex.Array, float]:
  """Finds best parameters for a choice prediction model given a dataset.

  This function fits a choice prediction model to a dataset using the given
  optimizer and number of optimization steps. It does this multiple times
  (max_attempts_grad_descent) to avoid getting stuck in a local minimum. It
  stops when the model has converged to the best score a certain number of times
  (n_converged_to_best).

  Args:
    agent_fun: The agent function.
    dataset: The dataset for the model.
    rng: The random number generator.
    optimizer: The optimizer to use.
    n_params: The number of parameters in the model.
    n_converged_to_best: The number of times the model must converge to
      the best score to be considered converged.
    score_match_eps: The epsilon value for score matching.
    max_attempts_grad_descent: The maximum number of times to try fitting the
      model.
    n_steps_per_epoch_grad_descent: The number of optimization steps per
      training loop.
    n_epochs_grad_descent: The maximum number of iterations for
      gradient descent.
    expected_output_shape: If present, add LARGE NUMBER to the error if the
      output shape is incorrect. This penalizes models that are violating
      assumptions about agent_fun without throwing an error.
    convergence_thresh_grad_descent: The percent change in score
      required for convergence.
    verbose: Whether to print verbose output.
    is_lbfgs: Whether the optimizer is L-BFGS.

  Returns:
    A tuple of the best parameters and the best score.
  """
  if rng is None:
    rng = jax.random.PRNGKey(0)

  n_converged = 0
  n_tried = 0

  best_score = VERY_LARGE_NUMBER
  best_params = None
  scores = []

  while (
      n_tried <= max_attempts_grad_descent
      and n_converged < n_converged_to_best
  ):
    if verbose:
      print(f'Fitting!: Attempt {n_tried}')
    n_tried += 1
    subrng, rng = jax.random.split(rng, 2)
    params, score = _train_to_convergence(
        agent_fun=agent_fun,
        dataset=dataset,
        rng=subrng,
        optimizer=optimizer,
        n_params=n_params,
        n_opt_steps=n_steps_per_epoch_grad_descent,
        max_iterations=n_epochs_grad_descent,
        score_percent_change_thresh=convergence_thresh_grad_descent,
        expected_output_shape=expected_output_shape,
        verbose=verbose,
        is_lbfgs=is_lbfgs,
    )

    scores.append(score)

    if score < best_score:
      best_score = score
      best_params = params

    n_converged = np.sum(
        [_score_match(s, best_score, score_match_eps) for s in scores]
    )

  if n_converged < n_converged_to_best:
    raise ValueError(
        f'Found only {n_converged} fits that converged to the best score. '
        f'This is less than the target of {n_converged_to_best}.  We have '
        f'reached the maximum number of tries {max_attempts_grad_descent}, so'
        f' stopping. Scores were: {np.sort(np.array(scores))}. Best params'
        f' were: {best_params}'
    )

  if best_params is None:
    raise ValueError(
        'Best params is None. Something went wrong during model fitting.'
    )
  return best_params, best_score


def choice_prediction_twofold_crossvalidation(
    agent_fun: ChoicePredictionAgentFunction,
    dataset: ChoicePredictionDataset,
    seed: int = 0,
    n_params: int = 10,
    n_steps_per_epoch_grad_descent: int = 100,
    n_epochs_grad_descent: int = 100,
    max_attempts_grad_descent: int = 10,
    n_converged_to_best: int = 3,
    score_match_eps: float = 1e-3,
    convergence_thresh_grad_descent: float = 1e-4,
    expected_output_shape: tuple[int, ...] | None = None,
    save_session_likelihoods: bool = False,
    verbose: bool = True,
    is_eval: bool = False,
    optimizer: optax.GradientTransformation = optax.adabelief(learning_rate=5e-2),  # pylint: disable=line-too-long
    is_lbfgs: bool = False,
) -> Tuple[Optional[float], Dict[str, Any]]:
  """Performs even-odd two-fold cross validation on a dataset.

  This function splits the dataset into two halves, fits a model to each half,
  and then evaluates each model on the other half. This is done twice, once
  with the first half as the training set and once with the second half as the
  training set. The final normalized likelihood is the average of the two
  evaluations.

  Args:
    agent_fun: The agent function.
    dataset: The choice prediction dataset.
    seed: The seed for the random number generator.
    n_params: The number of parameters in the model.
    n_steps_per_epoch_grad_descent: The number of optimization steps per
      training loop.
    n_epochs_grad_descent: The maximum number of iterations for
      gradient descent.
    max_attempts_grad_descent: The maximum number of times to try fitting the
      model.
    n_converged_to_best: The number of times the model must converge to
      the best score to be considered converged.
    score_match_eps: The epsilon value for score matching.
    convergence_thresh_grad_descent: The percent change in score
      required for convergence.
    expected_output_shape: If present, add LARGE NUMBER to the error if the
      output shape is incorrect. This penalizes models that are violating
      assumptions about agent_fun without throwing an error.
    save_session_likelihoods: Whether to save the session likelihoods.
    verbose: Whether to print verbose output.
    is_eval: Whether to evaluate the model.
    optimizer: The optimizer to use.
    is_lbfgs: Whether the optimizer is L-BFGS.

  Returns:
    A tuple of the normalized likelihood and a summary dictionary containing
    the parameters and scores for each fold, as well as the overall normalized
    likelihood.
  """
  rng = jax.random.PRNGKey(seed)

  dataset0, dataset1 = split_choice_prediction_dataset(dataset)

  subrng1, subrng2, _ = jax.random.split(rng, 3)
  params0, train_score0 = fit_choice_prediction_model(
      agent_fun,
      dataset0,
      subrng1,
      optimizer=optimizer,
      n_params=n_params,
      n_converged_to_best=n_converged_to_best,
      score_match_eps=score_match_eps,
      convergence_thresh_grad_descent=convergence_thresh_grad_descent,
      n_steps_per_epoch_grad_descent=n_steps_per_epoch_grad_descent,
      n_epochs_grad_descent=n_epochs_grad_descent,
      max_attempts_grad_descent=max_attempts_grad_descent,
      expected_output_shape=expected_output_shape,
      verbose=verbose,
      is_lbfgs=is_lbfgs,
  )

  params1, train_score1 = fit_choice_prediction_model(
      agent_fun,
      dataset1,
      subrng2,
      optimizer=optimizer,
      n_params=n_params,
      n_converged_to_best=n_converged_to_best,
      score_match_eps=score_match_eps,
      convergence_thresh_grad_descent=convergence_thresh_grad_descent,
      n_steps_per_epoch_grad_descent=n_steps_per_epoch_grad_descent,
      n_epochs_grad_descent=n_epochs_grad_descent,
      max_attempts_grad_descent=max_attempts_grad_descent,
      expected_output_shape=expected_output_shape,
      verbose=verbose,
      is_lbfgs=is_lbfgs,
  )

  if is_eval:
    nll_fn = functools.partial(_compute_nll, compute_predictions=True)
  else:
    nll_fn = _compute_nll_jitted

  nll_0, (eval_sess_nll_0, _) = nll_fn(
      params0,
      agent_fun=agent_fun,
      dataset=dataset1,
      expected_output_shape=expected_output_shape,
      verbose=verbose,
  )

  nll_1, (eval_sess_nll_1, _) = nll_fn(
      params1,
      agent_fun=agent_fun,
      dataset=dataset0,
      expected_output_shape=expected_output_shape,
      verbose=verbose,
  )
  nll_total = np.array(nll_0) + np.array(nll_1)
  if jnp.isnan(nll_total):
    raise ValueError(
        'Models fit successfully, but cross-validation resulted in NaN. Fold'
        f' scores: {nll_0}, {nll_1}'
    )

  n_trials_total = np.sum(
      np.logical_not(np.array(dataset0.next_ignore))
  ) + np.sum(np.logical_not(np.array(dataset1.next_ignore)))
  norm_ll = float(_negloglik_to_normlik(float(nll_total), n_trials_total))
  summary = {
      'params0': params0,
      'params1': params1,
      'train_score0': train_score0,
      'train_score1': train_score1,
      'xval_ll_0': nll_0,
      'xval_ll_1': nll_1,
      'total_xval_ll': nll_0 + nll_1,
      'n_valid_trials': n_trials_total,
      'normalized_likelihood': norm_ll,
  }

  if save_session_likelihoods:
    n_valid_trials_0_sess = np.sum(
        np.logical_not(np.array(dataset0.next_ignore)), axis=1
    )
    n_valid_trials_1_sess = np.sum(
        np.logical_not(np.array(dataset1.next_ignore)), axis=1
    )

    eval_norm_lik_0_sess = _negloglik_to_normlik(
        eval_sess_nll_0, n_valid_trials_1_sess
    )
    eval_norm_lik_1_sess = _negloglik_to_normlik(
        eval_sess_nll_1, n_valid_trials_0_sess
    )

    summary.update({
        'sess_nll_0': eval_sess_nll_0,
        'sess_nll_1': eval_sess_nll_1,
        'eval_norm_lik_0_sess': eval_norm_lik_0_sess,
        'eval_norm_lik_1_sess': eval_norm_lik_1_sess,
    })

  if (norm_ll < 0) or (norm_ll > 1):
    norm_ll = None
    if verbose:
      print((
          f'Normalized likelihood is outside of [0, 1] (got {norm_ll}).'
          'Returning None.'
      ))

  return norm_ll, summary


def evaluate_choice_prediction_model(
    agent_fun: ChoicePredictionAgentFunction,
    dataset: ChoicePredictionDataset,
    params: chex.Array | None,
    expected_output_shape: tuple[int, ...] | None = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
  """Evaluates a choice prediction agent on a dataset.

  Args:
    agent_fun: The agent function.
    dataset: The dataset to evaluate the agent on.
    params: The parameters of the agent function.
    expected_output_shape: If present, add LARGE NUMBER to the error if the
      output shape is incorrect. This penalizes models that are violating
      assumptions about agent_fun without throwing an error.

  Returns:
    A tuple of the normalized likelihood and a dictionary of predictions.
  """
  total_nll, (_, predictions) = _compute_nll(
      params=params,
      agent_fun=agent_fun,
      dataset=dataset,
      expected_output_shape=expected_output_shape,
      compute_predictions=True,
  )

  n_trials_total = np.sum(np.logical_not(np.array(dataset.next_ignore)))
  norm_ll = float(_negloglik_to_normlik(float(total_nll), n_trials_total))
  return norm_ll, predictions
