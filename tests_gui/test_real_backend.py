"""GUI integration checks using frozen models and synthetic uploaded messages."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gui_backend import analyze_upload
from gui_data import normalize_service_result
import service


def messages():
    rows = []
    for event, risk, distance in [('001', -8, 8000), ('SAT-B', -3, 500), ('LATE', -5, 4000)]:
        for days in (4.0, 2.5, 0.5):
            rows.append(dict(event_id=event, time_to_tca=days, risk=risk,
                             miss_distance=distance, relative_speed=11000,
                             relative_position_r=100 if event == '001' else -200,
                             relative_position_t=200, relative_position_n=300,
                             c_object_type='DEBRIS', c_sigma_r=3, t_sigma_r=4,
                             max_risk_estimate=-4, F10=80, AP=12))
    source = pd.DataFrame(rows)
    source.loc[source.event_id.eq('LATE'), 'time_to_tca'] = [1.5, 1.0, 0.5]
    # An eligible historical warning must remain distinct from SAFE current risk.
    source.loc[0, 'F10'] = float('nan')
    return source


class RealBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = messages()
        cls.payload = cls.source.to_csv(index=False).encode()
        cls.data = analyze_upload(cls.payload)

    def test_real_contract_identity_summary_and_raw_exports(self):
        data = self.data
        self.assertEqual(set(data['events'].event_id), {'001', 'SAT-B', 'LATE'})
        self.assertEqual(data['summary']['event_count'], 3)
        self.assertEqual(data['summary']['event_count'], data['summary']['total_events'])
        self.assertEqual(len(data['raw_df']), 9)
        self.assertEqual(len(data['records']), 3)
        self.assertEqual(data['raw_df'].event_id.iloc[0], '001')
        self.assertEqual(data['summary']['status'], 'partial')
        self.assertIsNone(data['summary']['window_24h_count'])
        self.assertEqual(data['summary']['window_24h']['coverage_status'], 'INSUFFICIENT')
        self.assertIn('severity', data['issues'])
        events = data['events'].set_index('event_id')
        self.assertEqual(events.loc['001', 'risk_level'], 'SAFE')
        self.assertEqual(events.loc['001', 'data_quality_status'], 'NEEDS_REVIEW')
        self.assertEqual(events.loc['LATE', 'prediction_status'], 'NOT_PREDICTED')
        self.assertTrue(pd.isna(events.loc['LATE', 'predicted_pc']))

    def test_history_is_only_service_eligible_observations(self):
        result = service.analyze_csv(self.source, model_dir=ROOT / 'models')
        history = self.data['history']
        self.assertEqual(len(history), 4)
        self.assertTrue(history.time_to_tca_hours.ge(48).all())
        self.assertEqual(history.loc[history.event_id.eq('001'), 'risk'].tolist(), [-8, -8])
        self.assertFalse(history.event_id.eq('LATE').any())
        # Poison raw input and prediction records: the returned history is authoritative.
        changed = {**result, 'records': result['records'].assign(risk=1000)}
        adapted = normalize_service_result(changed, self.source.assign(risk=0))
        pd.testing.assert_frame_equal(history, adapted['history'])
        pd.testing.assert_frame_equal(result['issues'], adapted['issues'])
        self.assertEqual(result['summary'], adapted['summary'])
        absent = normalize_service_result({**result, 'history': result['history'].iloc[:0]}, self.source)
        self.assertTrue(absent['history'].empty)
        unknown_scale = normalize_service_result({**result, 'metadata': {}}, self.source)
        self.assertTrue(unknown_scale['history'].empty)

    def test_invalid_missing_empty_and_no_eligible_inputs(self):
        bad = self.source.copy()
        bad['risk'] = 'bad'
        data = analyze_upload(bad.to_csv(index=False).encode())
        self.assertEqual(data['summary']['status'], 'invalid_input')
        self.assertTrue(data['events'].prediction_status.eq('NOT_PREDICTED').all())
        missing = analyze_upload(b'event_id,risk\n001,-8\n')
        self.assertEqual(missing['summary']['status'], 'invalid_input')
        self.assertIn('MISSING_COLUMN', missing['issues'].code.tolist())
        empty = analyze_upload(self.source.iloc[:0].to_csv(index=False).encode())
        self.assertEqual(empty['summary']['status'], 'empty')
        late = self.source.assign(time_to_tca=[1.5, 1.0, 0.5] * 3)
        data = analyze_upload(late.to_csv(index=False).encode())
        self.assertEqual(data['summary']['status'], 'no_eligible_input')
        self.assertTrue(data['history'].empty)

    def test_service_model_errors_are_not_hidden(self):
        for error in [FileNotFoundError('Missing model'), RuntimeError('Model loading failed')]:
            with patch('service.analyze_csv', side_effect=error):
                with self.assertRaises(type(error)):
                    analyze_upload(self.payload)

    def widget(self, collection, label):
        return next(w for w in collection if w.label == label)

    def test_empty_simulation_backend_simulation_has_no_stale_data(self):
        at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertIsNone(self.widget(at.radio, 'Data source').value)
        self.assertEqual([m.value for m in at.metric], ['—'] * 4)
        self.widget(at.radio, 'Data source').set_value('Simulation').run()
        self.assertEqual(at.metric[0].value, '20')
        self.widget(at.button, 'Open event #1024').click().run()
        self.assertEqual(at.session_state.selected_event, 1024)
        self.widget(at.radio, 'Data source').set_value('Backend CSV').run()
        self.assertIsNone(at.session_state.selected_event)
        self.assertEqual(len(at.dataframe), 0)
        self.assertEqual(len(at.get('plotly_chart')), 0)
        upload = SimpleNamespace(getvalue=lambda: self.payload)
        with patch('streamlit.file_uploader', return_value=upload), patch('service.analyze_csv', wraps=service.analyze_csv) as call:
            at.run()
            self.assertEqual(call.call_count, 0)
            self.assertEqual(len(at.dataframe), 0)
            self.widget(at.button, 'Analyze CSV').click().run()
            self.assertEqual(call.call_count, 1)
            self.assertEqual(set(at.dataframe[-1].value['Event ID']), {'001', 'SAT-B', 'LATE'})
            self.widget(at.selectbox, 'Open event details').set_value('001').run()
            self.widget(at.button, 'Open selected event').click().run()
            self.assertEqual(at.session_state.selected_event, '001')
            self.widget(at.radio, 'Data source').set_value('Simulation').run()
            self.assertIsNone(at.session_state.selected_event)
            self.assertEqual(len(at.dataframe[-1].value), 15)
            self.assertIn(1024, at.dataframe[-1].value['Event ID'].tolist())
            self.assertNotIn('001', at.dataframe[-1].value['Event ID'].tolist())
            self.assertEqual(len(at.get('plotly_chart')), 0)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(len(at.get('download_button')), 1)
        self.widget(at.radio, 'Navigation').set_value('Mission Overview').run()
        self.assertEqual([m.value for m in at.metric], ['20', '5', '7', '9'])
        self.assertFalse(at.exception)

    def test_app_real_navigation_details_no_stale_charts_or_repeat_inference(self):
        upload = SimpleNamespace(getvalue=lambda: self.payload)
        with patch('streamlit.file_uploader', return_value=upload), patch('service.analyze_csv', wraps=service.analyze_csv) as call:
            at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
            self.widget(at.radio, 'Data source').set_value('Backend CSV').run()
            self.widget(at.button, 'Analyze CSV').click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.metric[0].value, '3')
            self.assertTrue(any('REAL BACKEND ANALYSIS' in c.value for c in at.caption))
            self.assertTrue(any('Partial analysis' in w.value for w in at.warning))
            self.widget(at.radio, 'Navigation').set_value('Collision Events').run()
            self.widget(at.selectbox, 'Open event details').set_value('001').run()
            self.widget(at.button, 'Open selected event').click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state.selected_event, '001')
            self.assertEqual(at.metric[0].label, 'Latest Known Pc')
            self.assertTrue(any('Predicted Final Tier:' in c.value for c in at.caption))
            charts = [c.proto.spec for c in at.get('plotly_chart')]
            self.assertEqual(len(charts), 2)
            self.widget(at.selectbox, 'Open event details').set_value('SAT-B').run()
            self.widget(at.button, 'Open selected event').click().run()
            self.assertNotEqual(charts[1], at.get('plotly_chart')[1].proto.spec)
            self.widget(at.selectbox, 'Open event details').set_value('LATE').run()
            self.widget(at.button, 'Open selected event').click().run()
            self.assertEqual(len(at.get('plotly_chart')), 0)
            self.assertTrue(any('No successful prediction' in w.value for w in at.warning))
            self.assertTrue(any('Historical data unavailable' in i.value for i in at.info))
            self.assertEqual(at.metric[1].value, '—')
            self.assertEqual(call.call_count, 1)
            self.assertFalse(at.exception)

    def test_app_empty_analysis_keeps_issues_metadata_and_downloads(self):
        upload = SimpleNamespace(getvalue=lambda: b'event_id,risk\n')
        with patch('streamlit.file_uploader', return_value=upload):
            at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
            self.widget(at.radio, 'Data source').set_value('Backend CSV').run()
            self.widget(at.button, 'Analyze CSV').click().run()
            self.assertFalse(at.exception)
            self.assertTrue(at.error)
            self.assertEqual(len(at.get('download_button')), 2)


if __name__ == '__main__':
    unittest.main()
