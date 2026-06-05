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

"""Utilities for fitting cognitive models using optax."""

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Union

import chex
from datadiver.library import dataset_types
from datadiver.library import metrics
import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax


TrainableDataset = dataset_types.TrainableDataset
NeuroDataset = dataset_types.NeuroDataset

LARGE_NUMBER = 1e10
VERY_LARGE_NUMBER = 1e15


def _concat_init_rest_dict(x_init_dict, x_rest_dict):
  return jax.tree_util.tree_map(
      lambda x, y: jnp.concatenate([x[None], y], axis=0),
      x_init_dict,
      x_rest_dict,
  )


def _process_one_trial_session_over(
    ll_state: Tuple[float, chex.Array],
    logits_shape: Optional[Tuple[int, ...]],
    compute_predictions: bool,
) -> Tuple[Tuple[chex.Array, chex.Array], dict[str, chex.Array]]:
  """Dummy method for computing NLL and predictions when session is over."""
  loss, agent_state = ll_state
  logits_shape = logits_shape if logits_shape is not None else (2,)
  if not isinstance(agent_state, chex.Array):
    raise ValueError(f'Agent state should be a JAX array, not {agent_state}.')
  if compute_predictions:
    predictions = {
        'choice_logits': jnp.zeros(logits_shape),
        'chosen_logprob': jnp.array(0.0),
        'agent_state': agent_state,
    }
  else:
    predictions = {'loss': jnp.array(loss)}
  return (jnp.array(loss), agent_state), predictions


def _process_one_trial(
    ll_state: Tuple[float, Any],
    inputs_targets: Any,
    *,
    agent_fun: Callable[..., Tuple[chex.Array, Any]],
    params: chex.Array | None,
    penalize_output_shape: Optional[Tuple[int, ...]],
    compute_predictions: bool,
    verbose: bool = True,
) -> tuple[tuple[chex.Array, Any], dict[str, chex.Array]]:
  """Get NLL and predictions for one trial."""
  # Unpack agent_state, choice, reward, ignore, and next_choice.
  total_ll, current_agent_state = ll_state
  input_kwargs = inputs_targets['inputs']
  input_kwargs['agent_state'] = current_agent_state

  next_choice = inputs_targets['targets']['next_choice']
  ignore_next_choice = inputs_targets['targets']['ignore']

  # Apply agent function and get choice logprobs.

  # TODO(stachenfeld): allow positional and keyword arguments.
  choice_logits, next_agent_state = agent_fun(params, **input_kwargs)
  if not isinstance(next_agent_state, chex.Array):
    raise ValueError(
        'The agent state returned by the agent function must be a Chex array,'
        f' not {next_agent_state}.'
    )
  choice_logprobs = choice_logits - jsp.special.logsumexp(choice_logits)

  # Check that output shape is as expected.
  if penalize_output_shape is not None:
    assert isinstance(
        penalize_output_shape, tuple
    ), 'penalize_output_shape must be a tuple.'
    if choice_logits.shape != penalize_output_shape:
      total_ll -= LARGE_NUMBER
      if verbose:
        print('Bad shape -- adding large penalty to total_ll')

  # If choice probabilities do not sum to 1, penalize ll
  # This can happen if logits take extreme values and cause numerical issues.
  bad_logits = jnp.abs(jnp.sum(jnp.exp(choice_logprobs)) - 1.0) > 1e-3
  total_ll -= LARGE_NUMBER * bad_logits

  chosen_logprob = choice_logprobs[jnp.int32(next_choice)]

  if penalize_output_shape is None:
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


def process_one_trial(
    ll_state: tuple[float, Any],
    inputs_targets: Any,
    *,
    agent_fun: Callable[..., Tuple[chex.Array, Any]],
    params: chex.Array | None,
    penalize_output_shape: Optional[Tuple[int, ...]],
    compute_predictions: bool,
    logits_shape: Optional[Tuple[int, ...]] = None,
    verbose: bool = True,
) -> tuple[tuple[chex.Array, Any], dict[str, chex.Array]]:
  """Get NLL and predictions for one trial.

  This function is meant to be used by an outer lax.scan over a full session. If
  the session is over, it will not alter the current log likelihood or state.
  Otherwise,it will increment log likelihood and state for the current trial.
  Args:
    ll_state: (total_ll, agent_state), where total_ll is the total
      log-likelihood of all trials before this one, and agent_state is the
      current agent state.
    inputs_targets: A dictionary with fields inputs and targets. inputs is a
      dictionary of inputs to the agent function, with keys equal to the names
      of the keyword arguments to the agent function. and targets is a
      dictionary of targets to the agent function.
    agent_fun: Function to compute the new logits and agent state.
    params: Parameters passed to agent_fun for computing choice_logits.
    penalize_output_shape: If not None, will compare with the shape of
      choice_logits and add a large penalty if they do not match.
    compute_predictions: Whether to include prediction outputs as a dictionary.
    logits_shape: The actual shape of choice_logits, computed from previous
      trials. Used only if the session is over and `compute_predictions` is
      True, to make sure the shape of the dummy predicted logits is correct.
    verbose: Whether to print messages.

  Returns:
    ll_state: (total_ll, agent_state), where total_ll is the total
      log-likelihood of all trials *including* this one,
      and agent_state is the new current agent state.
    predictions: dictionary containing predictions for choice_logits,
      chosen_logprob, and agent_state.
  """

  # Session is over if choice or reward are -1.
  # In this case do not update state.
  input_kwargs = inputs_targets['inputs']
  session_over = False
  if 'choice' in input_kwargs and 'reward' in input_kwargs:
    session_over = jnp.logical_or(
        inputs_targets['inputs']['choice'] == -1,
        inputs_targets['inputs']['reward'] == -1,
    )
  return lax.cond(
      session_over,
      lambda: _process_one_trial_session_over(
          ll_state=ll_state,
          logits_shape=logits_shape,
          compute_predictions=compute_predictions,
      ),
      lambda: _process_one_trial(
          ll_state=ll_state,
          inputs_targets=inputs_targets,
          agent_fun=agent_fun,
          params=params,
          penalize_output_shape=penalize_output_shape,
          compute_predictions=compute_predictions,
          verbose=verbose,
      ),
  )


def process_one_session(
    total_ll,
    inputs_targets: Any,
    *,
    agent_fun: Callable[..., Tuple[chex.Array, Any]],
    params: chex.Array | None,
    penalize_output_shape: Optional[Tuple[int, ...]],
    compute_predictions: bool,
    dummy_first_trial: bool = False,
    dummy_first_trial_fields: Tuple[str, ...] = ('choice', 'reward'),
    verbose: bool = True,
) -> Tuple[float, Tuple[chex.Array, dict[str, chex.Array]]]:
  """Processes one session, incrementing ll."""
  # Initialize the log-likelihood and agent state.
  init_session_ll = 0.0
  init_agent_state = None  # Initialize agent_state to None.
  ll_state_0 = (init_session_ll, init_agent_state)

  # Process first trial.
  # This is special because agent_state is None on the first trial,
  # not an array.
  inputs_targets_0 = jax.tree_util.tree_map(lambda x: x[0], inputs_targets)
  first_trial_fun = functools.partial(
      # We're assuming no trials of length 0, so we don't need to handle
      # the case where the session is over in the first trial.
      _process_one_trial,
      agent_fun=agent_fun,
      params=params,
      penalize_output_shape=penalize_output_shape,
      compute_predictions=compute_predictions,
      verbose=verbose,
  )

  if dummy_first_trial:
    inputs_targets_0['inputs'].update(
        {k: None for k in dummy_first_trial_fields}
    )

  ll_state_1, pred_1 = first_trial_fun(ll_state_0, inputs_targets_0)

  logits_shape = None
  if compute_predictions and 'choice_logits' in pred_1:
    logits_shape = pred_1['choice_logits'].shape

  # Process all later trials.
  inputs_targets_after_first = jax.tree_util.tree_map(
      lambda x: x[1:], inputs_targets
  )

  trial_fun_after_first = functools.partial(
      process_one_trial,
      agent_fun=agent_fun,
      params=params,
      penalize_output_shape=penalize_output_shape,
      compute_predictions=compute_predictions,
      logits_shape=logits_shape,
      verbose=verbose,
  )

  (session_ll, _), pred_after_first = lax.scan(
      trial_fun_after_first, ll_state_1, inputs_targets_after_first
  )

  predictions = _concat_init_rest_dict(pred_1, pred_after_first)

  total_ll += session_ll
  return total_ll, (session_ll, predictions)


def compute_nll(
    params: Any,
    agent_fun: Callable[..., Tuple[chex.Array, Any]],
    *,
    inputs_targets: TrainableDataset,
    penalize_output_shape: Optional[Tuple[int, ...]] = None,
    compute_predictions: bool = False,
    dummy_first_trial: bool = False,
    dummy_first_trial_fields: Tuple[str, ...] = ('choice', 'reward'),
    verbose: bool = True,
) -> Tuple[float, Tuple[chex.Array, dict[str, chex.Array]]]:
  """Compute negative log-likelihood given parameters, model, and dataset."""

  sess_fun = functools.partial(
      process_one_session,
      agent_fun=agent_fun,
      params=params,
      penalize_output_shape=penalize_output_shape,
      compute_predictions=compute_predictions,
      dummy_first_trial=dummy_first_trial,
      dummy_first_trial_fields=dummy_first_trial_fields,
      verbose=verbose,
  )

  total_ll, (sess_ll, predictions) = lax.scan(sess_fun, 0.0, inputs_targets)

  total_nll = -total_ll
  sess_nll = -sess_ll
  return total_nll, (sess_nll, predictions)


compute_nll_jitted = jax.jit(
    compute_nll,
    static_argnames=[
        'agent_fun',
        'penalize_output_shape',
        'compute_predictions',
        'dummy_first_trial',
        'dummy_first_trial_fields',
        'verbose',
    ],
)


def negloglik_to_normlik(
    nll: float | chex.Array, n_valid_trials: int | chex.Array
) -> float | chex.Array:
  """Convert negative log-likelihood to normalized likelihood.

  Args:
    nll: Negative log-likelihood(s) (if float, it will be NLL for one session or
      dataset; if array, it will be NLLs for multiple sessions or datasets).
    n_valid_trials: Number of valid trials in each dataset. Must have same shape
      as nll.

  Returns:
    Normalized likelihood(s).
  """
  assert (
      np.array([nll]).shape == np.array([n_valid_trials]).shape
  ), 'nll and n_valid_trials must have the same shape'
  return np.exp(-nll / n_valid_trials)


@functools.partial(
    jax.jit,
    static_argnames=[
        'agent_fun',
        'optimizer',
        'penalize_output_shape',
        'dummy_first_trial',
        'dummy_first_trial_fields',
        'verbose',
        'is_lbfgs',
    ],
)
def opt_step(
    params: Any,
    opt_state: Any,
    *,
    inputs_targets: TrainableDataset,
    agent_fun: Callable[..., Any],
    optimizer: optax.GradientTransformationExtraArgs,
    penalize_output_shape: Optional[Tuple[int, ...]],
    dummy_first_trial: bool = False,
    dummy_first_trial_fields: Tuple[str, ...] = ('choice', 'reward'),
    verbose: bool = True,
    is_lbfgs: bool = False,
) -> Tuple[Any, Any]:
  """Apply jitted step of an optimizer to a set of parameters."""
  if is_lbfgs:
    value_grad_fn = jax.value_and_grad(compute_nll_jitted, has_aux=True)
    (loss_val, _), grads = value_grad_fn(
        params,
        inputs_targets=inputs_targets,
        agent_fun=agent_fun,
        penalize_output_shape=penalize_output_shape,
        dummy_first_trial=dummy_first_trial,
        dummy_first_trial_fields=dummy_first_trial_fields,
        verbose=verbose,
    )
    loss_fn_lbfgs = lambda p: compute_nll_jitted(
        p,
        inputs_targets=inputs_targets,
        agent_fun=agent_fun,
        penalize_output_shape=penalize_output_shape,
        dummy_first_trial=dummy_first_trial,
        dummy_first_trial_fields=dummy_first_trial_fields,
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
    grads, _ = jax.grad(compute_nll_jitted, has_aux=True)(
        params,
        inputs_targets=inputs_targets,
        agent_fun=agent_fun,
        penalize_output_shape=penalize_output_shape,
        dummy_first_trial=dummy_first_trial,
        dummy_first_trial_fields=dummy_first_trial_fields,
        verbose=verbose,
    )
    updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return (params, opt_state)


def _score_match(score, best_score, score_match_eps):
  return np.abs(score - best_score) / best_score < score_match_eps


# TODO(kevinjmiller): Deprecate this when refactoring noinfo.
_FUNCTION_KWARGS_MAPPING = {
    'choice': 'x',
    'reward': 'y',
    'stimulus': 'z',
    'agent_state': 'state',
}


def _remap_kwargs(agent_fun_raw: Callable[..., Any]) -> Callable[..., Any]:
  """Wraps function to rename kwargs. This support the noinfo interface.

  Specifically, it looks for arguments 'x', 'reward', and 'state', and replaces
  these with 'choice','reward','agent_state'. This behavior is configured by
  the dictionary _FUNCTION_KWARGS_MAPPING.

  Args:
    agent_fun_raw: The function to wrap.

  Returns:
    The wrapped function.
  """
  # Get a list of all function kwarg names
  sig = inspect.signature(agent_fun_raw)
  func_params = set(sig.parameters)

  @functools.wraps(agent_fun_raw)
  def wrapper(*args, **kwargs):
    remapped_kwargs = kwargs.copy()
    for raw_name, remapped_name in _FUNCTION_KWARGS_MAPPING.items():
      if raw_name not in func_params and remapped_name in func_params:
        # e.g. if the function doesn't take 'choice', but it does take 'x',
        # and 'choice' is present in the current kwargs.
        if raw_name in remapped_kwargs:
          value = remapped_kwargs.pop(raw_name)
          remapped_kwargs[remapped_name] = value
    return agent_fun_raw(*args, **remapped_kwargs)

  return wrapper


def check_inputs_targets(
    inputs_targets: Union[TrainableDataset, NeuroDataset],
    agent_fun: Callable[..., Any],
) -> TrainableDataset:
  """Checks that inputs_targets has the correct keys and types for agent_fun."""
  if isinstance(inputs_targets, NeuroDataset):
    inputs_targets = inputs_targets.as_trainable_dataset()
  if 'inputs' not in inputs_targets:
    raise ValueError('inputs_targets must have an "inputs" key.')
  if 'targets' not in inputs_targets:
    raise ValueError('inputs_targets must have a "targets" key.')
  if not isinstance(inputs_targets['inputs'], dict):
    raise ValueError('inputs_targets["inputs"] must be a dictionary.')
  if not isinstance(inputs_targets['targets'], dict):
    raise ValueError('inputs_targets["targets"] must be a dictionary.')
  # Check that all inputs needed by agent_fun are present in inputs_targets.
  sig = inspect.signature(agent_fun)
  agent_params_set = set(sig.parameters)
  if 'params' not in agent_params_set:
    raise ValueError(
        'agent_fun must have an argument named "params". Instead args were'
        f' {list(sig.parameters)}'
    )

  # These arguments are handled internally, not from inputs_targets['inputs'].
  agent_params_to_check = agent_params_set.copy()
  agent_params_to_check.discard('params')
  agent_params_to_check.discard('agent_state')
  agent_params_to_check.discard('state')

  input_keys = set(inputs_targets['inputs'].keys())

  # Figure out what keys will be passed to agent_fun after remapping.
  # This logic mirrors _remap_kwargs.
  effective_input_keys = input_keys.copy()
  for raw_name, remapped_name in _FUNCTION_KWARGS_MAPPING.items():
    if (
        raw_name in input_keys
        and raw_name not in agent_params_set
        and remapped_name in agent_params_set
    ):
      effective_input_keys.discard(raw_name)
      effective_input_keys.add(remapped_name)

  missing = agent_params_to_check - effective_input_keys
  if missing:
    raise ValueError(
        'The following arguments required by agent_fun were not found in '
        f"inputs_targets['inputs']: {list(missing)}. "
        f'Agent function expects arguments: {list(agent_params_to_check)}. '
        f'Available inputs are: {list(input_keys)}. '
        'Note: some argument names can be remapped (e.g. choice -> x).'
    )
  return inputs_targets


def train_to_convergence(
    agent_fun: Callable[..., Tuple[chex.Array, Any]],
    inputs_targets: Union[TrainableDataset, NeuroDataset],
    rng: jax.Array,
    optimizer: optax.GradientTransformation = optax.adabelief(learning_rate=5e-2),  # pylint: disable=line-too-long
    n_params: int = 10,
    n_opt_steps: int = 100,
    max_iterations: int = 100,
    score_percent_change_thresh: float = 1e-4,
    penalize_output_shape: Optional[Tuple[int, ...]] = None,
    dummy_first_trial: bool = False,
    dummy_first_trial_fields: Tuple[str, ...] = ('choice', 'reward'),
    check_num_params_used: bool = True,
    verbose: bool = True,
    is_lbfgs: bool = False,
):
  """Train a set of params until they converge."""

  if check_num_params_used:
    # Check the agent_fun does not use more than n_params elements of params.
    agent_fun_source = inspect.getsource(agent_fun)
    n_params_used = metrics.get_n_params_used(agent_fun_source)
    if n_params_used > n_params:
      raise ValueError('Agent fun uses too many params.')
    n_params = n_params_used

  agent_fun = _remap_kwargs(agent_fun)
  inputs_targets = check_inputs_targets(inputs_targets, agent_fun)

  params = jax.random.normal(rng, shape=(n_params,))
  opt_state = optimizer.init(params)

  continue_training = True
  iterations_complete = 0

  nll_kwargs = {
      'agent_fun': agent_fun,
      'inputs_targets': inputs_targets,
      'penalize_output_shape': penalize_output_shape,
      'dummy_first_trial': dummy_first_trial,
      'dummy_first_trial_fields': dummy_first_trial_fields,
      'verbose': verbose,
  }

  score = float(compute_nll_jitted(params, **nll_kwargs)[0])

  # Each iteration trains for a fixed number of steps, then checks whether the
  # score has changed by less than a threshold. If so, we declare convergence.
  while continue_training:
    for _ in range(n_opt_steps):
      params, opt_state = opt_step(
          params,
          opt_state,
          optimizer=optimizer,
          is_lbfgs=is_lbfgs,
          **nll_kwargs,
      )
      if np.isnan(np.asarray(params)).any():
        break
    new_score = float(compute_nll_jitted(params, **nll_kwargs)[0])

    if _score_match(score, new_score, score_percent_change_thresh):
      continue_training = False
      if verbose:
        print('  converged!')

    iterations_complete += 1
    if iterations_complete >= max_iterations:
      continue_training = False
      if verbose:
        print('  max iterations reached...')

    if np.isnan(new_score):
      continue_training = False
      if verbose:
        print('  nan score...')

    score = new_score

  return params, score


def fit_model(
    agent_fun: Callable[..., Tuple[chex.Array, Any]],
    inputs_targets: Union[TrainableDataset, NeuroDataset],
    rng: jax.Array | None = None,
    optimizer: optax.GradientTransformation = optax.adabelief(learning_rate=5e-2),  # pylint: disable=line-too-long
    n_params: int = 10,
    n_converged_to_best_target: int = 3,
    score_match_eps: float = 1e-3,
    max_n_to_try: int = 10,
    n_opt_steps: int = 100,
    max_iterations_gradient_descent: int = 100,
    penalize_output_shape: Tuple[int, ...] | None = None,
    dummy_first_trial: bool = False,
    dummy_first_trial_fields: Tuple[str, ...] = ('choice', 'reward'),
    check_num_params_used: bool = True,
    score_percent_change_thresh_for_convergence: float = 1e-4,
    verbose: bool = True,
    is_lbfgs: bool = False,
) -> Tuple[chex.Array, float]:
  """Finds best parameters for a model given a dataset.

  This function repeatedly initializes model parameters and trains them to
  convergence using the provided optimizer. It keeps track of the best score
  (negative log-likelihood) found so far and stops after a certain number of
  training runs converge to a score close to the best score.

  Args:
    agent_fun: A function representing the agent model. It takes parameters and
      keyword arguments from `inputs_targets['inputs']` and returns choice
      logits and the next agent state.
    inputs_targets: A `types.NeuroDataset` or a dictionary containing the
      training data. The data is expected in a structure with two keys, 'inputs'
      and 'targets', where each contains a dictionary of arrays. -
      `inputs_targets['inputs']`: A dictionary where keys are argument names for
      `agent_fun` and values are `chex.Array`s of input data. -
      `inputs_targets['targets']`: A dictionary of target data for computing the
      loss. It should contain at least 'next_choice' and 'ignore' arrays. The
      arrays in these dictionaries typically have a shape of (n_sessions,
      n_trials, ...).
    rng: JAX random number generator key for parameter initialization.
    optimizer: An optax optimizer for training.
    n_params: The number of parameters to be fit. To check agent_fun does not
      use more than this number of params, set check_num_params_used to True.
      Note that JAX is happy to fit models with the wrong number of params,
      fitting extra params that do nothing or extending the params vector by
      copying the final element.
    n_converged_to_best_target: The number of training runs that need to
      converge to a score close to the best score before stopping.
    score_match_eps: Relative tolerance to consider a score as "close to" the
      best score.
    max_n_to_try: The maximum number of random parameter initializations to try.
    n_opt_steps: The number of optimization steps in each gradient descent
      training iteration.
    max_iterations_gradient_descent: The maximum number of iterations of
      gradient descent optimization runs to run before stopping.
    penalize_output_shape: If provided, the output of `agent_fun` is penalized
      if its shape does not match this tuple.
    dummy_first_trial: If True, the inputs for the first trial of each session
      are replaced with dummy values.
    dummy_first_trial_fields: The names of the input fields to be replaced with
      dummy values on the first trial.
    check_num_params_used: If True, statically analyze `agent_fun` to ensure it
      doesn't use more than `n_params`.
    score_percent_change_thresh_for_convergence: Relative tolerance for the
      score change between iterations to consider training converged.
    verbose: If True, print progress information during fitting.
    is_lbfgs: Whether the optimizer is an LBFGS optimizer, which requires
      special handling for value and gradient computation.

  Returns:
    A tuple containing:
      - The best parameters found.
      - The best score (negative log-likelihood) achieved.
  """

  if rng is None:
    rng = jax.random.PRNGKey(0)

  n_converged_to_best = 0
  n_tried = 0

  # Large value to start. We're minimizing, so this is a bad score.
  best_score = VERY_LARGE_NUMBER
  best_params = None
  scores = []
  # Each iteration fits one set of params to convergence. If score is close
  # enough to the best ever score, we count it as a success. n_converged_to_best
  # is the number of successes we need to see before we stop trying.
  while (
      n_tried <= max_n_to_try
      and n_converged_to_best < n_converged_to_best_target
  ):
    if verbose:
      print(f'Fitting!: Attempt {n_tried}')
    n_tried += 1
    subrng, rng = jax.random.split(rng, 2)
    params, score = train_to_convergence(
        agent_fun=agent_fun,
        inputs_targets=inputs_targets,
        rng=subrng,
        optimizer=optimizer,
        n_params=n_params,
        n_opt_steps=n_opt_steps,
        max_iterations=max_iterations_gradient_descent,
        score_percent_change_thresh=score_percent_change_thresh_for_convergence,
        penalize_output_shape=penalize_output_shape,
        dummy_first_trial=dummy_first_trial,
        dummy_first_trial_fields=dummy_first_trial_fields,
        check_num_params_used=check_num_params_used,
        verbose=verbose,
        is_lbfgs=is_lbfgs,
    )

    scores.append(score)

    if score < best_score:
      best_score = score
      best_params = params

    n_converged_to_best = np.sum(
        [_score_match(s, best_score, score_match_eps) for s in scores]
    )

  # After the while loop, check if we met the convergence target.
  if n_converged_to_best < n_converged_to_best_target:
    raise ValueError(
        f'Found only {n_converged_to_best} fits that converged to the best'
        f' score. This is less than the target of {n_converged_to_best_target}.'
        f' We have reached the maximum number of tries {max_n_to_try}, so'
        f' stopping. Scores were: {np.sort(np.array(scores))}. Best params'
        f' were: {best_params}'
    )

  if best_params is None:
    raise ValueError(
        'Best params is None. Something went wrong during model fitting.'
    )
  return best_params, best_score


def twofold_crossvalidation(
    dataset0: TrainableDataset,
    dataset1: TrainableDataset,
    agent_fun: Callable[..., Tuple[chex.Array, Any]],
    seed: int = 0,
    n_params: int = 10,
    n_opt_steps: int = 100,
    max_iterations_gradient_descent: int = 100,
    max_n_to_try: int = 10,
    n_converged_to_best_target: int = 3,
    score_match_eps: float = 1e-3,
    score_percent_change_thresh_for_convergence: float = 1e-4,
    penalize_output_shape: Optional[Tuple[int, ...]] = None,
    save_session_likelihoods: bool = False,
    dummy_first_trial: bool = False,
    dummy_first_trial_fields: Tuple[str, ...] = ('choice', 'reward'),
    check_num_params_used: bool = True,
    verbose: bool = True,
    is_eval: bool = False,
    optimizer: optax.GradientTransformation = optax.adabelief(learning_rate=5e-2),  # pylint: disable=line-too-long
    is_lbfgs: bool = False,
) -> Tuple[Optional[float], Dict[str, Any]]:
  """Performs even-odd cross validation on a dataset.

  Args:
    dataset0: The first dataset to train on.
    dataset1: The second dataset to train on.
    agent_fun: The function to compute the agent's next choice given the current
      agent state.
    seed: The seed for the random number generator.
    n_params: The number of parameters to use for the model.
    n_opt_steps: The number of optimization steps to run per iteration.
    max_iterations_gradient_descent: The maximum number of iterations of
      gradient descent to run.
    max_n_to_try: The maximum number of times to re-run the gradient descent
      process to find parameters that converge.
    n_converged_to_best_target: The number of times that the parameters must
      converge to the best score before stopping.
    score_match_eps: Relative tolerance to consider a score as "close to" the
      best score.
    score_percent_change_thresh_for_convergence: Relative tolerance for the
      score change between iterations to consider training converged.
    penalize_output_shape: The shape to penalize the output of the agent_fun.
    save_session_likelihoods: Whether to save the likelihoods for each session.
    dummy_first_trial: Whether to replace the first input trial of each session
      with a dummy value (-1).
    dummy_first_trial_fields: The fields for which the first entry's inputs are
      "dummy values" that should be replaced with Nones on the initial trial.
    check_num_params_used: Whether we should check the number of parameters used
      by agent_fun. This needs to be turned off when executing code in a
      sandbox.
    verbose: Whether to print verbose messages.
    is_eval: Whether we are evaluating the agent (rather than training it).
    optimizer: An optax optimizer for training.
    is_lbfgs: Whether the optimizer is an LBFGS optimizer, which requires
      special handling for value and gradient computation.

  Returns:
    The normalized likelihood of the agent on the datasets, and a dictionary of
    training and validation statistics.
  """
  agent_fun_processed = _remap_kwargs(agent_fun)
  rng = jax.random.PRNGKey(seed)

  # Fit parameters to even and odd datasets
  subrng1, subrng2, _ = jax.random.split(rng, 3)
  params0, train_score0 = fit_model(
      agent_fun_processed,
      dataset0,
      subrng1,
      optimizer=optimizer,
      n_params=n_params,
      n_converged_to_best_target=n_converged_to_best_target,
      score_match_eps=score_match_eps,
      score_percent_change_thresh_for_convergence=score_percent_change_thresh_for_convergence,
      n_opt_steps=n_opt_steps,
      max_iterations_gradient_descent=max_iterations_gradient_descent,
      max_n_to_try=max_n_to_try,
      penalize_output_shape=penalize_output_shape,
      dummy_first_trial=dummy_first_trial,
      dummy_first_trial_fields=dummy_first_trial_fields,
      check_num_params_used=check_num_params_used,
      verbose=verbose,
      is_lbfgs=is_lbfgs,
  )

  params1, train_score1 = fit_model(
      agent_fun_processed,
      dataset1,
      subrng2,
      optimizer=optimizer,
      n_params=n_params,
      n_converged_to_best_target=n_converged_to_best_target,
      score_match_eps=score_match_eps,
      score_percent_change_thresh_for_convergence=score_percent_change_thresh_for_convergence,
      n_opt_steps=n_opt_steps,
      max_iterations_gradient_descent=max_iterations_gradient_descent,
      max_n_to_try=max_n_to_try,
      penalize_output_shape=penalize_output_shape,
      dummy_first_trial=dummy_first_trial,
      dummy_first_trial_fields=dummy_first_trial_fields,
      check_num_params_used=check_num_params_used,
      verbose=verbose,
      is_lbfgs=is_lbfgs,
  )

  # If we're evaluating, computing the predictions (which requires non-jitted
  # NLL computation) improves the numerical stability of the logits, which
  # avoids spurious penalties for bad logits (which result in scores of zero).
  # If we're training, we'd rather avoid evolving programs with stability
  # issues, and the jitted computation is faster.
  if is_eval:
    nll_fn = functools.partial(compute_nll, compute_predictions=True)
  else:
    nll_fn = compute_nll_jitted

  # Compute likelihoods on held-out datasets
  nll_0, (eval_sess_nll_0, _) = nll_fn(
      params0,
      agent_fun=agent_fun_processed,
      inputs_targets=dataset1,
      penalize_output_shape=penalize_output_shape,
      dummy_first_trial=dummy_first_trial,
      dummy_first_trial_fields=dummy_first_trial_fields,
      verbose=verbose,
  )

  nll_1, (eval_sess_nll_1, _) = nll_fn(
      params1,
      agent_fun=agent_fun_processed,
      inputs_targets=dataset0,
      penalize_output_shape=penalize_output_shape,
      dummy_first_trial=dummy_first_trial,
      dummy_first_trial_fields=dummy_first_trial_fields,
      verbose=verbose,
  )
  nll_total = np.array(nll_0) + np.array(nll_1)
  if jnp.isnan(nll_total):
    raise ValueError(
        'Models fit successfully, but cross-validation resulted in NaN. Fold'
        f' scores: {nll_0}, {nll_1}'
    )

  n_trials_total = np.sum(
      np.logical_not(np.array(dataset0['targets']['ignore']))
  ) + np.sum(np.logical_not(np.array(dataset1['targets']['ignore'])))
  norm_ll = float(negloglik_to_normlik(float(nll_total), n_trials_total))
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
    # Compute normalized likelihood per session.
    # Get n trials per session.
    n_valid_trials_0_sess = np.sum(
        np.logical_not(np.array(dataset0['targets']['ignore'])), axis=1
    )
    n_valid_trials_1_sess = np.sum(
        np.logical_not(np.array(dataset1['targets']['ignore'])), axis=1
    )

    # Compute normalized likelihood per session.
    # eval_sess_nll_0 was trained on 0 and evaluated on 1, so we need to
    # normalize with n_trials from split 1 (and vice versa).
    eval_norm_lik_0_sess = negloglik_to_normlik(
        eval_sess_nll_0, n_valid_trials_1_sess
    )
    eval_norm_lik_1_sess = negloglik_to_normlik(
        eval_sess_nll_1, n_valid_trials_0_sess
    )

    summary.update({
        'sess_nll_0': eval_sess_nll_0,
        'sess_nll_1': eval_sess_nll_1,
        'eval_norm_lik_0_sess': eval_norm_lik_0_sess,
        'eval_norm_lik_1_sess': eval_norm_lik_1_sess,
    })

  # Normalized likelihood should never be outside of [0, 1].
  if (norm_ll < 0) or (norm_ll > 1):
    norm_ll = None
    if verbose:
      print((
          f'Normalized likelihood is outside of [0, 1] (got {norm_ll}).'
          'Returning None.'
      ))

  return norm_ll, summary


def evaluate_agent_on_dataset(
    dataset: TrainableDataset,
    agent_fun: Callable[..., Tuple[chex.Array, Any]],
    params: chex.Array | None,
    penalize_output_shape: Optional[Tuple[int, ...]] = None,
    dummy_first_trial: bool = False,
    dummy_first_trial_fields: Tuple[str, ...] = ('choice', 'reward'),
) -> Tuple[Optional[float], Dict[str, Any]]:
  """Evaluates an agent on a dataset."""

  total_nll, (_, predictions) = compute_nll(
      params=params,
      agent_fun=agent_fun,
      inputs_targets=dataset,
      penalize_output_shape=penalize_output_shape,
      compute_predictions=True,
      dummy_first_trial=dummy_first_trial,
      dummy_first_trial_fields=dummy_first_trial_fields,
  )

  n_trials_total = np.sum(
      np.logical_not(np.array(dataset['targets']['ignore']))
  )
  norm_ll = float(negloglik_to_normlik(float(total_nll), n_trials_total))
  return norm_ll, predictions
