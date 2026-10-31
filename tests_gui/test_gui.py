import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import Mock
import pandas as pd
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gui_data import normalize_result, priority_events, seven_day_count, time_counts, explanation
from gui_mock import mock_result
from gui_charts import donut
from gui_charts import history_chart, position_chart

def service_result():
    mock = mock_result()
    events = mock['events'].copy()
    events['event_id'] = events.event_id.astype(str)
    events['prediction_status'] = 'PREDICTED'
    events['predicted_final_tier'] = 'Safe'
    summary = {'status': 'ok', 'total_events': 20, 'needs_review_events': 7,
               'critical_events': 5, 'elevated_events': 6, 'safe_events': 6, 'unknown_events': 3,
               'window_24h': {'coverage_status': 'INSUFFICIENT', 'coverage_note': 'Test coverage'}}
    summary.update(event_count=20, critical_count=5, elevated_count=6, safe_count=6, unknown_count=3, needs_review_count=7, window_24h_count=None, window_24h_note='Test coverage')
    return {'records': events.copy(), 'events': events, 'history': pd.DataFrame(columns=['event_id', 'time_to_tca_hours', 'risk', 'miss_distance_m']),
            'issues': pd.DataFrame(columns=['input_record_id', 'row_position', 'event_id', 'column', 'code', 'severity', 'value', 'message']),
            'summary': summary, 'metadata': {**mock['metadata'], 'units': {'time_to_tca': 'days'}, 'history_risk_scale': 'log10(Pc)'}}


APP = str(Path(__file__).resolve().parents[1] / 'app.py')


class DataTests(unittest.TestCase):
    def setUp(self):
        self.result = mock_result()
        self.data = normalize_result(self.result)
        self.events = self.data['events']

    def test_summary_and_donut(self):
        summary = self.result['summary']
        self.assertEqual(summary['event_count'], self.events.event_id.nunique())
        self.assertEqual(sum(summary[k + '_count'] for k in ['critical', 'elevated', 'safe', 'unknown']), 20)
        self.assertEqual(summary['needs_review_count'], 7)
        values = list(donut(summary).data[0].values)
        self.assertEqual(values, [6, 6, 5, 3])
        self.assertEqual(sum(values), 20)

    def test_time_boundaries(self):
        self.assertEqual(seven_day_count(self.events), 9)
        boundary = self.events.iloc[:7].copy()
        boundary['time_to_tca_hours'] = [0, 24, 48, 72, 120, 168, 168.1]
        counts = time_counts(boundary)
        self.assertEqual(counts.sum(axis=1).tolist(), [2, 1, 1, 1, 1])

    def test_order_search_and_filters(self):
        ordered = priority_events(self.events)
        self.assertEqual(ordered.event_id.head(5).tolist(), [32, 178, 55, 1401, 1402])
        self.assertEqual(priority_events(self.events, search='1024').event_id.tolist(), [1024])
        self.assertEqual(len(priority_events(self.events, risk='UNKNOWN', quality='NEEDS_REVIEW')), 3)
        self.assertEqual(priority_events(self.events, risk='SAFE').iloc[-1].event_id, 1404)

    def test_contract_types_missing_fields_and_invalid_values(self):
        self.assertEqual(len(normalize_result(SimpleNamespace(**self.result))['events']), 20)
        self.result['events'] = pd.DataFrame([{'event_id': 1, 'tier_pc': 2, 'risk_level': 'SAFE', 'alert_score': -1}])
        result = normalize_result(self.result)
        self.assertTrue(pd.isna(result['events'].iloc[0].tier_pc))
        self.assertTrue(pd.isna(result['events'].iloc[0].alert_score))
        self.assertTrue(result['warnings'])
        self.assertEqual(result['events'].iloc[0].risk_level, 'SAFE')  # Never overwrite backend classification.

    def test_duplicates_rejected_and_explanation_preserved(self):
        self.result['events'] = pd.concat([self.events, self.events.iloc[:1]])
        with self.assertRaises(ValueError):
            normalize_result(self.result)
        text = '最近距离 820 m，小于 1000 m'
        self.assertEqual(explanation(text), text)

    def test_distinct_mock_charts_and_historical_cutoff(self):
        coordinates = self.events[['relative_position_r', 'relative_position_t', 'relative_position_n']].dropna()
        self.assertEqual(len(coordinates.drop_duplicates()), len(coordinates))
        history = self.data['history']
        self.assertTrue((history.time_to_tca_hours >= self.data['metadata']['horizon_days'] * 24).all())
        first = history[history.event_id == 32]
        second = history[history.event_id == 178]
        self.assertFalse(first.empty)
        self.assertFalse(second.empty)
        self.assertNotEqual(list(history_chart(first, True).data[0].y), list(history_chart(second, True).data[0].y))
        a = position_chart(self.events[self.events.event_id == 32].iloc[0])
        b = position_chart(self.events[self.events.event_id == 178].iloc[0])
        self.assertNotEqual(list(a.data[0].y), list(b.data[0].y))


class AppTests(unittest.TestCase):
    def test_first_launch_has_no_source_or_dataset(self):
        with patch('gui_mock.mock_result', side_effect=AssertionError('Mock data loaded before selection')):
            at = AppTest.from_file(APP, default_timeout=20).run()
            self.assertEqual(len(at.exception), 0)
            self.assertIsNone(self.by_label(at.radio, 'Data source').value)
            self.assertEqual(self.by_label(at.radio, 'Data source').options, ['Simulation', 'Backend CSV'])
            self.assertEqual([m.value for m in at.metric], ['—'] * 4)
            self.assertEqual(len(at.get('plotly_chart')), 0)
            self.assertEqual(len(at.dataframe), 0)
            self.assertEqual(len(at.get('download_button')), 0)
            self.assertTrue(any('No analysis loaded.' in i.value for i in at.info))
            at.run()
            self.assertIsNone(self.by_label(at.radio, 'Data source').value)
            self.collision(at)
            self.assertEqual(len(at.dataframe), 0)
            self.assertEqual(len(at.get('plotly_chart')), 0)

    def app(self):
        at = AppTest.from_file(APP, default_timeout=20).run()
        self.assertEqual(len(at.exception), 0)
        self.by_label(at.radio, 'Data source').set_value('Simulation').run()
        return at

    def by_label(self, collection, label):
        return next(widget for widget in collection if widget.label == label)

    def collision(self, at):
        self.by_label(at.radio, 'Navigation').set_value('Collision Events').run()
        self.assertEqual(len(at.exception), 0)

    def test_overview_and_simulation_never_imports_backend(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'service':
                raise AssertionError('Mock mode imported backend')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded):
            at = self.app()
            self.assertEqual([m.value for m in at.metric], ['20', '5', '7', '9'])
            self.assertTrue(any('No common observation timestamp' in c.value for c in at.caption))
            self.collision(at)
        self.assertEqual(len(at.exception), 0)

    def test_search_filter_pagination(self):
        at = self.app()
        self.collision(at)
        self.assertEqual(len(at.dataframe[-1].value), 15)
        self.assertEqual(len(at.number_input), 0)
        self.assertNotIn('Rows per page', [widget.label for widget in at.selectbox])
        self.assertTrue(at.button(key='event_previous').disabled)
        at.button(key='event_next').click().run()
        self.assertEqual(at.dataframe[-1].value['Event ID'].tolist(), [1350, 1404, 520, 815, 1102])
        self.assertTrue(any('Page 2 of 2' in m.value and 'Showing 16–20 of 20 events' in m.value for m in at.markdown))
        self.assertTrue(at.button(key='event_next').disabled)
        at.run()
        self.assertEqual(at.session_state.event_page, 2)
        self.by_label(at.selectbox, 'Sort by').set_value('Time to TCA').run()
        self.assertEqual(at.session_state.event_page, 1)
        at.button(key='event_next').click().run()
        self.by_label(at.text_input, 'Search event ID').set_value('1024').run()
        self.assertEqual(at.session_state.event_page, 1)
        self.assertEqual(at.dataframe[-1].value.iloc[0]['Event ID'], 1024)
        self.by_label(at.selectbox, 'Current risk level').set_value('SAFE').run()
        self.assertTrue(any('No events match' in i.value for i in at.info))
        self.assertTrue(at.button(key='event_previous').disabled)
        self.assertTrue(at.button(key='event_next').disabled)
        self.assertTrue(self.by_label(at.selectbox, 'Open event details').disabled)
        self.assertTrue(self.by_label(at.button, 'Open selected event').disabled)
        self.assertTrue(any('Showing 0–0 of 0 events' in m.value for m in at.markdown))
        self.assertEqual(len(at.exception), 0)

    def test_event_page_change_clears_pending_selection_not_open_details(self):
        at = self.app()
        self.collision(at)
        self.by_label(at.selectbox, 'Open event details').set_value(32).run()
        self.by_label(at.button, 'Open selected event').click().run()
        self.assertEqual(at.session_state.selected_event, 32)
        at.button(key='event_next').click().run()
        self.assertIsNone(self.by_label(at.selectbox, 'Open event details').value)
        self.assertTrue(self.by_label(at.button, 'Open selected event').disabled)
        self.assertEqual(at.session_state.selected_event, 32)
        self.assertTrue(any('Details remain on Event #32' in caption.value for caption in at.caption))
        self.by_label(at.selectbox, 'Open event details').set_value(1350).run()
        self.by_label(at.button, 'Open selected event').click().run()
        self.assertEqual(at.session_state.selected_event, 1350)
        at.button(key='event_previous').click().run()
        self.assertIsNone(self.by_label(at.selectbox, 'Open event details').value)
        self.assertEqual(at.session_state.selected_event, 1350)
        self.by_label(at.selectbox, 'Current risk level').set_value('CRITICAL').run()
        self.assertEqual(at.session_state.event_page, 1)
        self.assertIsNone(self.by_label(at.selectbox, 'Open event details').value)
        self.assertEqual(at.session_state.selected_event, 1350)
        self.assertEqual(len(at.exception), 0)

    def test_detail_1024_close_and_missing_data(self):
        at = self.app()
        self.collision(at)
        self.assertTrue(any('Select an event' in m.value for m in at.markdown))
        self.by_label(at.selectbox, 'Open event details').set_value(1024).run()
        self.by_label(at.button, 'Open selected event').click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual([m.value for m in at.metric[:3]], ['2.1e-05', '3.0e-09', '4.2e-04'])
        self.assertTrue(any('4.2e-04' in w.value for w in at.warning))
        self.by_label(at.button, '✕').click().run()
        self.assertIsNone(at.session_state.selected_event)
        self.by_label(at.selectbox, 'Current risk level').set_value('UNKNOWN').run()
        self.by_label(at.selectbox, 'Open event details').set_value(520).run()
        self.by_label(at.button, 'Open selected event').click().run()
        self.assertEqual(at.metric[0].value, 'N/A')
        self.assertTrue(any('No historical observations' in i.value for i in at.info))
        self.assertTrue(any('coordinates are unavailable' in i.value for i in at.info))
        self.assertEqual(len(at.exception), 0)

    def test_close_details_preserves_operator_context(self):
        at = self.app()
        self.collision(at)
        self.by_label(at.selectbox, 'Data quality').set_value('NEEDS_REVIEW').run()
        self.by_label(at.selectbox, 'Sort by').set_value('Ranking score').run()
        self.by_label(at.text_input, 'Search event ID').set_value('1').run()
        self.by_label(at.selectbox, 'Open event details').set_value(1024).run()
        self.by_label(at.button, 'Open selected event').click().run()
        before = at.dataframe[-1].value.copy()
        page = at.session_state.event_page
        self.by_label(at.button, '✕').click().run()
        self.assertIsNone(at.session_state.selected_event)
        self.assertEqual(at.session_state.navigation, 'Collision Events')
        self.assertEqual(at.session_state.data_source, 'Simulation')
        self.assertEqual(at.session_state.event_page, page)
        self.assertEqual(self.by_label(at.text_input, 'Search event ID').value, '1')
        self.assertEqual(self.by_label(at.selectbox, 'Data quality').value, 'NEEDS_REVIEW')
        self.assertEqual(self.by_label(at.selectbox, 'Sort by').value, 'Ranking score')
        pd.testing.assert_frame_equal(before, at.dataframe[-1].value)
        self.assertTrue(any('Select an event' in m.value for m in at.markdown))
        self.assertEqual(len(at.exception), 0)

    def test_overview_drilldown_and_two_page_navigation(self):
        at = self.app()
        self.assertEqual(self.by_label(at.radio, 'Navigation').options, ['Mission Overview', 'Collision Events'])
        at.button(key='alert_32').click().run()
        self.assertEqual(at.session_state.navigation, 'Collision Events')
        self.assertEqual(at.session_state.selected_event, 32)
        self.by_label(at.radio, 'Navigation').set_value('Mission Overview').run()
        self.assertEqual(len(at.exception), 0)

    def test_overview_pagination_and_search_reset(self):
        at = self.app()
        self.assertEqual(len(at.dataframe[-1].value), 10)
        self.assertTrue(at.button(key='overview_previous').disabled)
        self.assertFalse(at.button(key='overview_next').disabled)
        self.assertTrue(any('Page 1 of 2' in m.value and 'Showing 1–10 of 20 events' in m.value for m in at.markdown))
        at.button(key='overview_next').click().run()
        self.assertEqual(at.dataframe[-1].value['Event ID'].tolist(), [1078, 441, 731, 933, 1216, 1350, 1404, 520, 815, 1102])
        self.assertTrue(at.button(key='overview_next').disabled)
        self.assertFalse(at.button(key='overview_previous').disabled)
        at.run()
        self.assertEqual(at.session_state.overview_page, 2)
        at.button(key='overview_previous').click().run()
        self.assertEqual(at.session_state.overview_page, 1)
        at.button(key='overview_next').click().run()
        self.by_label(at.text_input, 'Search event ID').set_value('1024').run()
        self.assertEqual(at.session_state.overview_page, 1)
        self.assertEqual(at.dataframe[-1].value['Event ID'].tolist(), [1024])
        self.assertTrue(at.button(key='overview_next').disabled)
        self.assertTrue(any('Showing 1–1 of 1 events' in m.value for m in at.markdown))
        self.assertEqual(len(at.number_input), 0)
        self.assertEqual(len(at.exception), 0)

    def test_overview_partial_final_page_and_empty_search(self):
        at = self.app()
        self.by_label(at.text_input, 'Search event ID').set_value('1').run()
        self.assertEqual(len(at.dataframe[-1].value), 10)
        at.button(key='overview_next').click().run()
        self.assertEqual(len(at.dataframe[-1].value), 5)
        self.assertTrue(any('Page 2 of 2' in m.value and 'Showing 11–15 of 15 events' in m.value for m in at.markdown))
        self.by_label(at.text_input, 'Search event ID').set_value('no-such-event').run()
        self.assertEqual(at.session_state.overview_page, 1)
        self.assertTrue(any('No events match your search' in info.value for info in at.info))
        self.assertTrue(at.button(key='overview_previous').disabled)
        self.assertTrue(at.button(key='overview_next').disabled)
        self.assertTrue(any('Page 1 of 1' in m.value and 'Showing 0–0 of 0 events' in m.value for m in at.markdown))
        self.assertEqual(len(at.exception), 0)

    def test_removed_settings_session_preserves_metadata_context(self):
        at = AppTest.from_file(APP, default_timeout=20)
        at.session_state.navigation = 'Settings'
        at.run()
        self.by_label(at.radio, 'Data source').set_value('Simulation').run()
        self.assertEqual(at.session_state.navigation, 'Mission Overview')
        self.assertEqual(self.by_label(at.radio, 'Navigation').options, ['Mission Overview', 'Collision Events'])
        self.assertTrue(any('DATA SOURCE' in m.value for m in at.markdown))
        self.assertTrue(any('WORKSPACE' in m.value for m in at.markdown))
        self.assertTrue(any('DATA STATUS' in m.value for m in at.markdown))
        self.assertTrue(any('model_version' in item.value and 'horizon_days' in item.value for item in at.json))
        self.assertEqual(len(at.exception), 0)

    def test_chart_refresh_and_no_stale_missing_event_charts(self):
        at = self.app()
        self.collision(at)
        self.by_label(at.selectbox, 'Open event details').set_value(1024).run()
        self.by_label(at.button, 'Open selected event').click().run()
        specs = [chart.proto.spec for chart in at.get('plotly_chart')]
        self.assertEqual(len(specs), 2)
        self.by_label(at.selectbox, 'Open event details').set_value(178).run()
        self.by_label(at.button, 'Open selected event').click().run()
        updated = [chart.proto.spec for chart in at.get('plotly_chart')]
        self.assertEqual(len(updated), 2)
        self.assertNotEqual(specs[0], updated[0])
        self.assertNotEqual(specs[1], updated[1])
        self.by_label(at.selectbox, 'Current risk level').set_value('UNKNOWN').run()
        self.by_label(at.selectbox, 'Open event details').set_value(520).run()
        self.by_label(at.button, 'Open selected event').click().run()
        self.assertEqual(len(at.get('plotly_chart')), 0)
        self.assertTrue(any('No historical observations' in info.value for info in at.info))
        self.assertTrue(any('coordinates are unavailable' in info.value for info in at.info))
        self.assertEqual(len(at.exception), 0)

    def test_backend_upload_missing_and_no_analysis_on_navigation(self):
        at = self.app()
        self.by_label(at.radio, 'Data source').set_value('Backend CSV').run()
        self.assertTrue(any('No analysis loaded.' in i.value for i in at.info))
        self.assertTrue(self.by_label(at.button, 'Analyze CSV').disabled)
        self.assertTrue(any('Choose a CSV to begin.' in m.value for m in at.sidebar.markdown))
        self.assertEqual(len(at.sidebar.info), 0)
        self.assertEqual(len(at.exception), 0)

    def test_backend_csv_ui_errors_and_cached_navigation(self):
        at = self.app()
        upload = SimpleNamespace(getvalue=lambda: b'event_id,risk\n1024,-5\n')
        backend = Mock(return_value=service_result())
        with patch('streamlit.file_uploader', return_value=upload), patch.dict(sys.modules, {'service': SimpleNamespace(analyze_csv=backend)}):
            self.by_label(at.radio, 'Data source').set_value('Backend CSV').run()
            self.assertTrue(any('Ready to analyze.' in m.value for m in at.sidebar.markdown))
            self.assertFalse(any('Analysis complete.' in m.value for m in at.sidebar.markdown))
            self.by_label(at.button, 'Analyze CSV').click().run()
            self.assertEqual(backend.call_count, 1)
            self.assertTrue(any('Analysis complete.' in m.value for m in at.sidebar.markdown))
            self.collision(at)
            self.by_label(at.text_input, 'Search event ID').set_value('1024').run()
            self.by_label(at.selectbox, 'Open event details').set_value('1024').run()
            self.by_label(at.button, 'Open selected event').click().run()
            self.by_label(at.button, '✕').click().run()
            self.assertEqual(backend.call_count, 1)
            self.assertEqual(at.session_state.data_source, 'Backend CSV')
            self.assertIsNone(at.session_state.selected_event)
            self.assertEqual(at.dataframe[-1].value['Event ID'].tolist(), ['1024'])
            backend.side_effect = RuntimeError('Model bundle unavailable')
            self.by_label(at.button, 'Analyze CSV').click().run()
            self.assertTrue(any('Model bundle unavailable' in e.value for e in at.error))
            self.assertNotIn('backend_result', at.session_state)
            self.assertFalse(any('Analysis complete.' in m.value for m in at.sidebar.markdown))
            at.run()
            self.assertEqual(backend.call_count, 2)
            self.assertTrue(any('Model bundle unavailable' in e.value for e in at.error))
            self.assertEqual(len(at.exception), 0)
        with patch('streamlit.file_uploader', return_value=SimpleNamespace(getvalue=lambda: b'')):
            at.run()
            self.by_label(at.button, 'Analyze CSV').click().run()
            self.assertTrue(any('empty' in e.value for e in at.error))

    def test_new_csv_does_not_reuse_another_files_status(self):
        at = self.app()
        upload_a = SimpleNamespace(getvalue=lambda: b'event_id,risk\n1024,-5\n')
        upload_b = SimpleNamespace(getvalue=lambda: b'event_id,risk\n32,-4\n')
        backend = Mock(side_effect=RuntimeError('Service unavailable'))
        with patch('streamlit.file_uploader', return_value=upload_a), patch.dict(sys.modules, {'service': SimpleNamespace(analyze_csv=backend)}):
            self.by_label(at.radio, 'Data source').set_value('Backend CSV').run()
            self.by_label(at.button, 'Analyze CSV').click().run()
            self.assertTrue(at.error)
        with patch('streamlit.file_uploader', return_value=upload_b):
            at.run()
            self.assertEqual(len(at.error), 0)
            self.assertTrue(any('Ready to analyze.' in m.value for m in at.sidebar.markdown))
            self.assertFalse(any('Analysis complete.' in m.value for m in at.sidebar.markdown))
            self.assertEqual(backend.call_count, 1)
            self.assertEqual(len(at.exception), 0)


class BackendAdapterTests(unittest.TestCase):
    def test_csv_validation_and_backend_errors(self):
        from gui_backend import analyze_upload
        backend = Mock(return_value=service_result())
        with patch.dict(sys.modules, {'service': SimpleNamespace(analyze_csv=backend)}):
            for payload in [b'', b' ', b'event_id,risk\n1,\"unterminated']:
                with self.assertRaises(ValueError):
                    analyze_upload(payload)
            backend.assert_not_called()
            data = analyze_upload(b'event_id,risk\n1024,-5\n')
            self.assertEqual(len(data['events']), 20)
            args = backend.call_args.args
            self.assertIsInstance(args[0], pd.DataFrame)
            self.assertEqual(backend.call_args.kwargs['model_dir'], Path(APP).parent / 'models')
            self.assertEqual(args[0].event_id.tolist(), ['1024'])
            backend.side_effect = RuntimeError('Model bundle unavailable')
            with self.assertRaisesRegex(RuntimeError, 'Model bundle unavailable'):
                analyze_upload(b'event_id,risk\n1024,-5\n')

    def test_empty_result_is_safe(self):
        from gui_backend import analyze_upload
        backend = Mock(return_value={**service_result(), 'events': service_result()['events'].iloc[:0], 'records': pd.DataFrame()})
        with patch.dict(sys.modules, {'service': SimpleNamespace(analyze_csv=backend)}):
            self.assertTrue(analyze_upload(b'event_id,risk\n1,-5\n')['events'].empty)


if __name__ == '__main__':
    unittest.main()
