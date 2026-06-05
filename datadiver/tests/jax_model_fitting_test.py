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

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from datadiver.library import dataset_utils
from datadiver.library import jax_model_fitting
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


def initialize_if_none(agent_state):
  if agent_state is None:
    agent_state = jnp.array(0)
  return agent_state


def _agent_fun_for_test(params, c, r, agent_state):  # pylint: disable=unused-argument
  return (jnp.array([0, 0]), jnp.array(0))


def _noinfo_agent_fun_for_test(params, x, y, state):  # pylint: disable=unused-argument
  return (jnp.array([0, 0]), jnp.array(0))


class ProcessOneTrial(parameterized.TestCase):

  @parameterized.parameters([
      dict(returned_logits=[0.1, 0.2], expected_lls=[-0.744, -0.644]),
      dict(returned_logits=[-0.1, 0.2], expected_lls=[-0.854, -0.554]),
      dict(returned_logits=[-0.1, -0.2], expected_lls=[-0.644, -0.744]),
      dict(returned_logits=[1.0, -0.2], expected_lls=[-0.263, -1.463]),
  ])
  def test_with_dummy_agent(self, returned_logits, expected_lls):
    # Only next choice affects the total_ll in this test.
    agent_fun = lambda p, c, r, agent_state: (
        jnp.array(returned_logits),
        agent_state,
    )
    agent_state_val = jnp.array(27)  # Our agent_fun ignores agent_state.
    ll_state = (0.0, agent_state_val)
    for next_choice in range(2):
      inputs_targets = {
          'inputs': {'c': 1, 'r': 0},
          'targets': {'next_choice': next_choice, 'ignore': 0},
      }
      # choice_reward_ignore_next_choice = (1, 0, 0, next_choice)
      new_ll_state, _ = jax_model_fitting.process_one_trial(
          ll_state,
          inputs_targets,
          agent_fun=agent_fun,
          params=None,
          penalize_output_shape=None,
          compute_predictions=False,
      )
      self.assertLen(new_ll_state, 2)
      np.testing.assert_almost_equal(
          new_ll_state[0], expected_lls[next_choice], decimal=3
      )
      self.assertEqual(agent_state_val, new_ll_state[1])

  @parameterized.parameters([
      dict(returned_state=12),
      dict(returned_state=None),
      dict(returned_state=[]),
      dict(returned_state='some string'),
  ])
  def test_invalid_state_raises_value_error(self, returned_state):
    # Only next choice affects the total_ll in this test.
    agent_fun = lambda p, c, r, agent_state: (jnp.zeros((2,)), returned_state)
    ll_state = (0.0, None)
    for next_choice in range(2):
      inputs_targets = {
          'inputs': {'c': 1, 'r': 0},
          'targets': {'next_choice': next_choice, 'ignore': 0},
      }
      with self.assertRaises(ValueError):
        jax_model_fitting.process_one_trial(
            ll_state,
            inputs_targets,
            agent_fun=agent_fun,
            params=None,
            penalize_output_shape=None,
            compute_predictions=False,
        )

  @parameterized.parameters([
      dict(returned_logits=[0.1, 0.2], expected_lls=[-0.744, -0.644]),
      dict(returned_logits=[0.1, 0.2, 0.3], expected_lls=None),
  ])
  def test_with_shape_check(self, returned_logits, expected_lls):
    # Only next choice affects the total_ll in this test.
    agent_fun = lambda p, c, r, agent_state: (
        jnp.array(returned_logits),
        agent_state,
    )
    agent_state_val = jnp.array(27)  # Our agent_fun ignores agent_state.
    ll_state = (0.0, agent_state_val)
    for next_choice in range(2):
      inputs_targets = {
          'inputs': {'c': 1, 'r': 0},
          'targets': {'next_choice': next_choice, 'ignore': 0},
      }
      new_ll_state, _ = jax_model_fitting.process_one_trial(
          ll_state,
          inputs_targets,
          agent_fun=agent_fun,
          params=None,
          penalize_output_shape=(2,),
          compute_predictions=False,
      )
      self.assertLen(new_ll_state, 2)
      if expected_lls is None:  # Bad shape
        self.assertLess(new_ll_state[0], jax_model_fitting.LARGE_NUMBER)
      else:
        np.testing.assert_almost_equal(
            new_ll_state[0], expected_lls[next_choice], decimal=3
        )
      self.assertEqual(agent_state_val, new_ll_state[1])

  @parameterized.parameters(
      [dict(compute_predictions=True), dict(compute_predictions=False)]
  )
  def test_with_session_over(self, compute_predictions):
    # Only next choice affects the total_ll in this test.
    returned_logits = [0.1, 0.2, 0.3]
    agent_fun = lambda p, choice, reward, agent_state: (
        jnp.array(returned_logits),
        agent_state + 1,
    )
    agent_state_val = jnp.array(np.pi)  # Our agent_fun ignores agent_state.
    ll_state = (0.0, agent_state_val)
    inputs_targets = {
        'inputs': {'choice': -1, 'reward': -1},
        'targets': {'next_choice': -1, 'ignore': 1},
    }
    new_ll_state, _ = jax_model_fitting.process_one_trial(
        ll_state,
        inputs_targets,
        agent_fun=agent_fun,
        params=None,
        logits_shape=(3,),
        penalize_output_shape=None,
        compute_predictions=compute_predictions,
    )
    np.testing.assert_almost_equal(new_ll_state[0], ll_state[0])
    np.testing.assert_almost_equal(new_ll_state[1], ll_state[1])

  @parameterized.parameters(
      [dict(compute_predictions=True), dict(compute_predictions=False)]
  )
  def test_with_predictions_output(self, compute_predictions):
    # Only next choice affects the total_ll in this test.
    returned_logits = [0.1, 0.2]
    expected_lls = [-0.744, -0.644]
    agent_fun = lambda p, c, r, agent_state: (
        jnp.array(returned_logits),
        agent_state,
    )
    agent_state_val = jnp.array(np.pi)  # Our agent_fun ignores agent_state.
    ll_state = (0.0, agent_state_val)
    for next_choice in range(2):
      inputs_targets = {
          'inputs': {'c': 1, 'r': 0},
          'targets': {'next_choice': next_choice, 'ignore': 0},
      }
      new_ll_state, predictions = jax_model_fitting.process_one_trial(
          ll_state,
          inputs_targets,
          agent_fun=agent_fun,
          params=None,
          penalize_output_shape=None,
          compute_predictions=compute_predictions,
      )
      self.assertLen(new_ll_state, 2)
      np.testing.assert_almost_equal(
          new_ll_state[0], expected_lls[next_choice], decimal=3
      )
      self.assertEqual(agent_state_val, new_ll_state[1])
      if compute_predictions:
        expected_predictions = {
            'choice_logits': returned_logits,
            'chosen_logprob': expected_lls[next_choice],
            'agent_state': agent_state_val,
        }
        for k, v in expected_predictions.items():
          self.assertIn(k, predictions)
          np.testing.assert_almost_equal(predictions[k], v, decimal=3)
      else:
        self.assertLen(predictions, 1)

  @parameterized.parameters([
      dict(
          ignore_next=0,
          expected_lls=(-0.744, -0.644),
          returned_logits=(0.1, 0.2),
      ),
      dict(ignore_next=1, expected_lls=(0.0, 0.0), returned_logits=(0.1, 0.2)),
      dict(
          ignore_next=1,
          expected_lls=(0.0, 0.0),
          returned_logits=(np.nan, np.nan),
      ),
  ])
  def test_with_ignore_next(self, ignore_next, expected_lls, returned_logits):
    # Only next choice affects the total_ll in this test.
    agent_fun = lambda p, c, r, agent_state: (
        jnp.array(returned_logits),
        agent_state,
    )
    agent_state_val = jnp.array(27)  # Our agent_fun ignores agent_state.
    ll_state = (0.0, agent_state_val)
    for next_choice in range(2):
      inputs_targets = {
          'inputs': {'c': 1, 'r': 0},
          'targets': {'next_choice': next_choice, 'ignore': ignore_next},
      }
      new_ll_state, _ = jax_model_fitting.process_one_trial(
          ll_state,
          inputs_targets,
          agent_fun=agent_fun,
          params=None,
          penalize_output_shape=None,
          compute_predictions=False,
      )
      self.assertLen(new_ll_state, 2)
      np.testing.assert_almost_equal(
          new_ll_state[0], expected_lls[next_choice], decimal=3
      )
      self.assertEqual(agent_state_val, new_ll_state[1])

  @parameterized.parameters([
      dict(choice=0, reward=0),
      dict(choice=0, reward=1),
      dict(choice=1, reward=0),
      dict(choice=1, reward=1),
  ])
  def test_with_parameterized_agent(self, choice, reward):
    # Create a linear layer with a 2-dimensional output, initialized so output
    # is just the sum of the inputs.
    class Linear(nn.Module):

      @nn.compact
      def __call__(self, x):
        return nn.Dense(features=2, kernel_init=nn.initializers.ones)(x)

    model = Linear()
    params = model.init(jax.random.PRNGKey(0), jnp.array([0, 0]))

    # Stack choice and reward as input to model, and add one to agent state.
    def agent_fun(p, c, r, agent_state):
      x = jnp.stack([c, r])
      return (model.apply(p, x), agent_state + 1)

    agent_state_val = jnp.array(10)
    ll_state = (0.0, agent_state_val)
    model_output = choice + reward
    expected_logprob = -0.693  # Will be the same for both choices.
    inputs_targets = {
        'inputs': {'agent_state': agent_state_val, 'c': choice, 'r': reward},
        'targets': {'next_choice': 0, 'ignore': 0},
    }
    new_ll_state, predictions = jax_model_fitting.process_one_trial(
        ll_state,
        inputs_targets,
        agent_fun=agent_fun,
        params=params,
        penalize_output_shape=(2,),
        compute_predictions=True,
    )
    self.assertLen(new_ll_state, 2)
    np.testing.assert_almost_equal(new_ll_state[0], expected_logprob, decimal=3)
    self.assertEqual(new_ll_state[1], agent_state_val + 1)
    expected_predictions = {
        'choice_logits': [model_output, model_output],
        'chosen_logprob': expected_logprob,
        'agent_state': agent_state_val + 1,
    }
    for k, v in expected_predictions.items():
      self.assertIn(k, predictions)
      np.testing.assert_almost_equal(predictions[k], v, decimal=3)


class ProcessOneSession(parameterized.TestCase):

  @parameterized.parameters([
      dict(returned_logits=[0.1, 0.2], expected_lls=[-0.744, -0.644]),
      dict(returned_logits=[-0.1, 0.2], expected_lls=[-0.854, -0.554]),
      dict(returned_logits=[-0.1, -0.2], expected_lls=[-0.644, -0.744]),
      dict(returned_logits=[1.0, -0.2], expected_lls=[-0.263, -1.463]),
  ])
  def test_with_dummy_agent(self, returned_logits, expected_lls):
    agent_fun = lambda p, c, r, agent_state: (
        jnp.array(returned_logits),
        initialize_if_none(agent_state),
    )
    for next_choice in range(2):
      choices = np.array([0, 1, next_choice])
      rewards = np.array([0, 0, 0])
      ignores = np.array([0, 0, 0])
      # choice_reward_ignore = (choices, rewards, ignores)

      inputs_targets = {
          'inputs': {'c': choices[:-1], 'r': rewards[:-1]},
          'targets': {'next_choice': choices[1:], 'ignore': ignores[1:]},
      }
      session_ll, _ = jax_model_fitting.process_one_session(
          0.0,
          inputs_targets,
          agent_fun=agent_fun,
          params=None,
          penalize_output_shape=None,
          compute_predictions=False,
      )
      # This session is of length 3, so it will call process_trial twice, with
      # next_choice being set to choices[1] and choices[2].
      expected_ll = expected_lls[choices[1]] + expected_lls[choices[2]]
      np.testing.assert_almost_equal(session_ll, expected_ll, decimal=3)

  @parameterized.parameters(
      [dict(compute_predictions=True), dict(compute_predictions=False)]
  )
  def test_with_predictions_output(self, compute_predictions):
    # Only next choice affects the total_ll in this test.
    returned_logits = [0.1, 0.2]
    expected_lls = [-0.744, -0.644]
    # This agent function will always return π as the new agent state.
    agent_fun = lambda p, c, r, agent_state: (
        jnp.array(returned_logits),
        np.array(
            np.pi,
        ),
    )
    for next_choice in range(2):
      choices = np.array([0, 1, next_choice])
      rewards = np.array([0, 0, 0])
      ignores = np.array([0, 0, 0])
      inputs_targets = {
          'inputs': {'c': choices[:-1], 'r': rewards[:-1]},
          'targets': {'next_choice': choices[1:], 'ignore': ignores[1:]},
      }
      session_ll, (_, predictions) = jax_model_fitting.process_one_session(
          0.0,
          inputs_targets,
          agent_fun=agent_fun,
          params=None,
          penalize_output_shape=None,
          compute_predictions=compute_predictions,
      )
      # This session is of length 3, so it will call process_trial twice, with
      # next_choice being set to choices[1] and choices[2].
      chosen_logprob = np.array(
          [expected_lls[choices[1]], expected_lls[choices[2]]]
      )
      expected_ll = chosen_logprob[0] + chosen_logprob[1]
      np.testing.assert_almost_equal(session_ll, expected_ll, decimal=3)
      if compute_predictions:
        expected_predictions = {
            'choice_logits': np.array([returned_logits, returned_logits]),
            'chosen_logprob': chosen_logprob,
            'agent_state': np.array(
                np.pi,
            ),
        }
        for k, v in expected_predictions.items():
          self.assertIn(k, predictions)
          np.testing.assert_almost_equal(predictions[k], v, decimal=3)
      else:
        self.assertLen(predictions, 1)

  @parameterized.parameters([dict(ignore_next=0), dict(ignore_next=1)])
  def test_with_ignore_next(self, ignore_next):
    returned_logits = [0.1, 0.2]
    expected_lls = [0.0, 0.0] if ignore_next else [-0.744, -0.644]
    # Only next choice affects the total_ll in this test.
    agent_fun = lambda p, c, r, agent_state: (
        jnp.array(returned_logits),
        initialize_if_none(agent_state),
    )
    for next_choice in range(2):
      choices = np.array([0, 1, next_choice])
      rewards = np.array([0, 0, 0])
      ignores = np.array([0, ignore_next, ignore_next])
      inputs_targets = {
          'inputs': {'c': choices[:-1], 'r': rewards[:-1]},
          'targets': {'next_choice': choices[1:], 'ignore': ignores[1:]},
      }
      session_ll, _ = jax_model_fitting.process_one_session(
          0.0,
          inputs_targets,
          agent_fun=agent_fun,
          params=None,
          penalize_output_shape=None,
          compute_predictions=False,
      )
      # This session is of length 3, so it will call process_trial twice, with
      # next_choice being set to choices[1] and choices[2].
      expected_ll = expected_lls[choices[1]] + expected_lls[choices[2]]
      np.testing.assert_almost_equal(session_ll, expected_ll, decimal=3)

  @parameterized.parameters([
      dict(choice=0, reward=0),
      dict(choice=0, reward=1),
      dict(choice=1, reward=0),
      dict(choice=1, reward=1),
  ])
  def test_with_parameterized_agent(self, choice, reward):
    # Create a linear layer with a 2-dimensional output, initialized so output
    # is just the sum of the inputs.
    class Linear(nn.Module):

      @nn.compact
      def __call__(self, x):
        return nn.Dense(features=2, kernel_init=nn.initializers.ones)(x)

    model = Linear()
    params = model.init(jax.random.PRNGKey(0), jnp.array([0, 0]))

    # Stack choice and reward as input to model, and add one to agent state.
    def agent_fun(p, c, r, agent_state):
      x = jnp.stack([c, r])
      if agent_state is None:
        current_agent_state = jnp.array(10)
      else:
        current_agent_state = agent_state + 1
      return (model.apply(p, x), current_agent_state)

    model_output = choice + reward
    expected_logprob = -0.693  # Will be the same for both choices.
    choices = np.ones(3) * choice
    rewards = np.ones(3) * reward
    ignores = np.zeros_like(choices)
    inputs_targets = {
        'inputs': {'c': choices[:-1], 'r': rewards[:-1]},
        'targets': {'next_choice': choices[1:], 'ignore': ignores[1:]},
    }
    sess_ll, (sess_ll_0, predictions) = jax_model_fitting.process_one_session(
        0.0,
        inputs_targets,
        agent_fun=agent_fun,
        params=params,
        penalize_output_shape=(2,),
        compute_predictions=True,
    )
    expected_ll = expected_logprob * 2
    np.testing.assert_almost_equal(sess_ll, expected_ll, decimal=3)
    expected_predictions = {
        'choice_logits': [
            [model_output, model_output],
            [model_output, model_output],
        ],
        'chosen_logprob': [expected_logprob, expected_logprob],
        'agent_state': np.array([10, 11]),
    }
    for k, v in expected_predictions.items():
      self.assertIn(k, predictions)
      np.testing.assert_almost_equal(predictions[k], v, decimal=3)

    sess_ll, (sess_ll_1, _) = jax_model_fitting.process_one_session(
        sess_ll,
        inputs_targets,
        agent_fun=agent_fun,
        params=params,
        penalize_output_shape=(2,),
        compute_predictions=True,
    )
    np.testing.assert_almost_equal(sess_ll, sess_ll_0 + sess_ll_1)


class TrainToConvergenceTest(parameterized.TestCase):

  def test_session_over_does_not_affect_gradients(self):
    inputs_targets = {
        # The names of the keys are important, because these keys signal
        # when the session is over.
        'inputs': {
            'choice': np.array([[0, 0, -1]]),
            'reward': np.array([[0, 0, -1]]),
        },
        'targets': {
            'next_choice': np.array([[0, -1, -1]]),
            'ignore': np.array([[0, 1, 1]]),
        },
    }

    # This weird function will set the gradients and the state to NaN if the
    # reward is negative. The test is to ensure that this doesn't affect the
    # gradients.
    def agent_fun(params, choice, reward, agent_state):
      del params, choice
      if agent_state is None:
        agent_state = jnp.zeros((2,))
      logits_and_state = reward ** agent_state[0] + jnp.ones((2,)) * np.pi
      return logits_and_state, logits_and_state

    params, score = jax_model_fitting.train_to_convergence(
        agent_fun=agent_fun,
        inputs_targets=inputs_targets,
        rng=jax.random.PRNGKey(0),
    )
    self.assertIsNotNone(params)
    self.assertFalse(np.isnan(params).any())
    self.assertFalse(np.isnan(score).any())

  def test_exit_early_on_nan(self):
    inputs_targets = {
        'inputs': {'c': np.array([[0]]), 'r': np.array([[0]])},
        'targets': {'next_choice': np.array([[0]]), 'ignore': np.array([[0]])},
    }

    with mock.patch.object(jax_model_fitting, 'opt_step') as mock_opt_step:
      mock_opt_step.return_value = (jnp.array([np.nan, 0.0]), None)
      jax_model_fitting.train_to_convergence(
          agent_fun=_agent_fun_for_test,
          inputs_targets=inputs_targets,
          rng=jax.random.PRNGKey(0),
      )
      mock_opt_step.assert_called_once()


class TwoFoldCrossValidationTest(parameterized.TestCase):

  def test_get_likelihoods(self):

    random_state = np.random.RandomState(123)
    choices = random_state.randint(0, 1, (20, 11))
    r = random_state.randint(0, 1, (20, 10))
    c = choices[:, :-1]
    next_c = choices[:, 1:]
    ignore = np.zeros_like(c)

    even_inputs_targets = {
        'inputs': {'c': c[::2], 'r': r[::2]},
        'targets': {'next_choice': next_c[::2], 'ignore': ignore[::2]},
    }
    odd_inputs_targets = {
        'inputs': {'c': c[1::2], 'r': r[1::2]},
        'targets': {'next_choice': next_c[1::2], 'ignore': ignore[1::2]},
    }

    normalized_likelihood, summary = jax_model_fitting.twofold_crossvalidation(
        dataset0=even_inputs_targets,
        dataset1=odd_inputs_targets,
        agent_fun=_agent_fun_for_test,
        penalize_output_shape=(2,),
    )

    np.testing.assert_almost_equal(normalized_likelihood, 0.5, decimal=2)
    np.testing.assert_almost_equal(
        normalized_likelihood, summary['normalized_likelihood'], decimal=10
    )

  def test_get_likelihoods_with_lbfgs(self):

    random_state = np.random.RandomState(123)
    choices = random_state.randint(0, 1, (20, 11))
    r = random_state.randint(0, 1, (20, 10))
    c = choices[:, :-1]
    next_c = choices[:, 1:]
    ignore = np.zeros_like(c)

    even_inputs_targets = {
        'inputs': {'c': c[::2], 'r': r[::2]},
        'targets': {'next_choice': next_c[::2], 'ignore': ignore[::2]},
    }
    odd_inputs_targets = {
        'inputs': {'c': c[1::2], 'r': r[1::2]},
        'targets': {'next_choice': next_c[1::2], 'ignore': ignore[1::2]},
    }

    normalized_likelihood, summary = jax_model_fitting.twofold_crossvalidation(
        dataset0=even_inputs_targets,
        dataset1=odd_inputs_targets,
        agent_fun=_agent_fun_for_test,
        penalize_output_shape=(2,),
        is_lbfgs=True,
        optimizer=optax.lbfgs(),
    )

    np.testing.assert_almost_equal(normalized_likelihood, 0.5, decimal=2)
    np.testing.assert_almost_equal(
        normalized_likelihood, summary['normalized_likelihood'], decimal=10
    )

  def test_get_likelihoods_with_noinfo_agent(self):
    random_state = np.random.RandomState(456)  # Different seed

    choices_data = random_state.randint(0, 2, (20, 10))
    rewards_data = random_state.randint(0, 2, (20, 10))

    # Inputs to the agent function.
    # These keys MUST be 'choice' and 'reward' for _remap_kwargs to correctly
    # map them to 'x' and 'y' for _noinfo_agent_fun_for_test.
    input_choices = choices_data[:, :-1]
    input_rewards = rewards_data[:, :-1]

    # Targets for prediction
    target_next_choices = choices_data[:, 1:]
    target_ignores = np.zeros_like(input_rewards, dtype=int)

    even_inputs_targets = {
        'inputs': {'choice': input_choices[::2], 'reward': input_rewards[::2]},
        'targets': {
            'next_choice': target_next_choices[::2],
            'ignore': target_ignores[::2],
        },
    }
    odd_inputs_targets = {
        'inputs': {
            'choice': input_choices[1::2],
            'reward': input_rewards[1::2],
        },
        'targets': {
            'next_choice': target_next_choices[1::2],
            'ignore': target_ignores[1::2],
        },
    }

    normalized_likelihood, summary = jax_model_fitting.twofold_crossvalidation(
        dataset0=even_inputs_targets,
        dataset1=odd_inputs_targets,
        agent_fun=_noinfo_agent_fun_for_test,  # Use the noinfo agent
        penalize_output_shape=(2,),
    )

    np.testing.assert_almost_equal(normalized_likelihood, 0.5, decimal=2)
    np.testing.assert_almost_equal(
        normalized_likelihood, summary['normalized_likelihood'], decimal=10
    )


class JaxModelFittingTestExtras(parameterized.TestCase):

  @parameterized.parameters([
      dict(nll=0.1, n_valid_trials=10, exp_result=0.99),
      dict(nll=1.0, n_valid_trials=10, exp_result=0.905),
      dict(nll=-0.1, n_valid_trials=10, exp_result=1.01),
      dict(nll=0.1, n_valid_trials=100, exp_result=0.999),
      dict(nll=1.0, n_valid_trials=100, exp_result=0.99),
      dict(nll=-0.1, n_valid_trials=100, exp_result=1.001),
  ])
  def test_negloglik_to_normlik(self, nll, n_valid_trials, exp_result):
    np.testing.assert_almost_equal(
        jax_model_fitting.negloglik_to_normlik(nll, n_valid_trials),
        exp_result,
        decimal=3,
    )


class TestEvaluation(parameterized.TestCase):

  @parameterized.parameters([
      dict(
          choices=np.array([[1, 1]]), rewards=np.array([[1, 1]]), exp_score=1.0
      ),
      dict(
          choices=np.array([[1, 0]]), rewards=np.array([[1, 1]]), exp_score=0.0
      ),
  ])
  def test_evaluate(self, choices, rewards, exp_score):
    """Test evaluate_agent_on_dataset returns the expected score.

    Args:
      choices: n_sess x n_trials array of choices
      rewards: n_sess x n_trials array of rewards
      exp_score: expected score
    """

    large_number = 1e5

    def _win_stay_lose_switch(params, choice, reward, agent_state):
      del params
      if agent_state is None:
        current_agent_state = jnp.zeros(2)
      else:
        current_agent_state = agent_state
      reward_for_update = 2 * reward - 1
      new_agent_state = current_agent_state.at[choice].set(
          large_number * reward_for_update
      )
      new_agent_state = new_agent_state.at[1 - choice].set(
          large_number * -reward_for_update
      )
      return new_agent_state, new_agent_state

    dummy_dataset = {
        'choice': choices,
        'reward': rewards,
        'ignore': np.zeros_like(choices),
    }

    dataset_formatted = {}
    dataset_formatted['inputs'], dataset_formatted['targets'] = (
        dataset_utils.format_data_for_next_choice_prediction(
            dummy_dataset, ['reward', 'choice']
        )
    )

    pred_score, _ = jax_model_fitting.evaluate_agent_on_dataset(
        dataset_formatted,
        agent_fun=_win_stay_lose_switch,
        params=None,
    )

    np.testing.assert_almost_equal(exp_score, pred_score)


if __name__ == '__main__':
  absltest.main()
