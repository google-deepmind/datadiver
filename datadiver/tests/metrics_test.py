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
from datadiver.library import metrics


class MetricsTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(metric='volume', transform='identity'),
      dict(metric='difficulty', transform='identity'),
      dict(metric='effort', transform='identity'),
      dict(metric='volume', transform='inverse'),
      dict(metric='difficulty', transform='inverse'),
      dict(metric='effort', transform='inverse'),
      dict(metric='volume', transform='negative'),
      dict(metric='difficulty', transform='negative'),
      dict(metric='effort', transform='negative'),
  ])
  def test_compute_halstead(self, metric, transform):
    mock_metric_values = {
        'volume': 1.0,
        'effort': 2.0,
        'difficulty': 3.0,
    }
    expected_key = f'{transform}_halstead_{metric}'
    expected_value = mock_metric_values[metric]
    if transform == 'inverse':
      expected_value = 1.0 / expected_value
    elif transform == 'negative':
      expected_value = -expected_value
    with mock.patch.object(metrics, 'get_halstead_metrics') as mock_halstead:
      mock_halstead.return_value = mock_metric_values
      output = metrics.compute_halstead('dummy_code', metric, transform)
      self.assertAlmostEqual(expected_value, output.score)
      self.assertIn(expected_key, output.parsed_output)
      self.assertAlmostEqual(expected_value, output.parsed_output[expected_key])

  @parameterized.parameters([
      dict(metric='num_lines', transform='identity'),
      dict(metric='total_imports', transform='identity'),
      dict(metric='num_variables', transform='identity'),
      dict(metric='num_lines', transform='inverse'),
      dict(metric='total_imports', transform='inverse'),
      dict(metric='num_variables', transform='inverse'),
      dict(metric='num_lines', transform='negative'),
      dict(metric='total_imports', transform='negative'),
      dict(metric='num_variables', transform='negative'),
  ])
  def test_compute_code_stats(self, metric, transform):
    mock_metric_values = {
        'num_lines': 1.0,
        'code_stats': {
            'num_variables': 2.0,
            'total_imports': 3.0,
        },
    }
    expected_key = f'{transform}_{metric}'
    if metric in mock_metric_values:
      expected_value = mock_metric_values[metric]
    else:
      expected_value = mock_metric_values['code_stats'][metric]

    assert isinstance(expected_value, float)
    if transform == 'inverse':
      expected_value = 1.0 / expected_value
    elif transform == 'negative':
      expected_value = -expected_value

    with mock.patch.object(metrics, 'get_code_stats') as mock_code_stats:
      mock_code_stats.return_value = mock_metric_values
      output = metrics.compute_code_stats('dummy_code', metric, transform)
      self.assertAlmostEqual(expected_value, output.score)
      self.assertIn(expected_key, output.parsed_output)
      self.assertAlmostEqual(expected_value, output.parsed_output[expected_key])

  def test_invalid_transform_fails(self):
    output = metrics.compute_halstead('dummy_code', 'volume', 'badtransform')
    self.assertEqual(output.status, metrics.RUNTIME_ERROR_STATUS)
    output = metrics.compute_code_stats(
        'dummy_code', 'num_lines', 'badtransform'
    )
    self.assertEqual(output.status, metrics.RUNTIME_ERROR_STATUS)

  def test_invalid_halstead_metric_fails(self):
    mock_metric_values = {'volume': 1.0}
    with mock.patch.object(metrics, 'get_halstead_metrics') as mock_halstead:
      mock_halstead.return_value = mock_metric_values
      output = metrics.compute_halstead('dummy_code', 'badmetric', 'identity')
      self.assertEqual(output.status, metrics.RUNTIME_ERROR_STATUS)

  def test_invalid_code_stats_fails(self):
    mock_metric_values = {
        'num_lines': 1.0,
        'code_stats': {'num_variables': 2.0},
    }
    with mock.patch.object(metrics, 'get_code_stats') as mock_code_stats:
      mock_code_stats.return_value = mock_metric_values
      output = metrics.compute_code_stats('dummy_code', 'badmetric', 'identity')
      self.assertEqual(output.status, metrics.RUNTIME_ERROR_STATUS)


if __name__ == '__main__':
  absltest.main()
