"""ORBITGUARD operator GUI. Run: python -m streamlit run app.py."""
import hashlib
import html
import math
import pandas as pd
import streamlit as st
from gui_data import (COLORS, LEVELS, explanation, field, normalize_result, number as format_number,
                      priority_events, seven_day_count)
from gui_mock import mock_result
from gui_charts import donut, time_chart, history_chart, position_chart
from gui_style import CSS

st.set_page_config(page_title='ORBITGUARD · Satellite Risk Intelligence', page_icon='🛰️', layout='wide')
st.markdown(CSS, unsafe_allow_html=True)


EMPTY_MESSAGE = 'No analysis loaded. Select Simulation or upload a Backend CSV to begin.'


def clear_dataset_selection():
    st.session_state.selected_event = None
    st.session_state.pop('last_table_selection', None)
    st.session_state.overview_page = 1
    st.session_state.event_page = 1
    st.session_state.event_selection_generation = st.session_state.get('event_selection_generation', 0) + 1
    st.session_state.dataset_generation = st.session_state.get('dataset_generation', 0) + 1


def risk_criteria():
    st.subheader('Risk Level Criteria')
    st.caption('Airbus Challenge #1 · Reference only')
    criteria = [('CRITICAL', 'Pc ≥ 1e-4 OR miss distance < 1 km'),
                        ('ELEVATED', 'Pc ≥ 1e-6 OR miss distance ≤ 5 km, provided not Critical'),
                        ('SAFE', 'Pc < 1e-6 AND miss distance > 5 km'),
                        ('UNKNOWN', 'Insufficient or invalid evidence; unclassified, not an official risk tier')]
    with st.container(height='stretch', vertical_alignment='center', key='criteria_balance'):
        st.markdown('<div class="criteria-group">' + ''.join(
            '<div class="criteria" style="border-color:{}"><b style="color:{}">{}</b><br>{}</div>'.format(
                COLORS[level], COLORS[level], level, html.escape(text)) for level, text in criteria) + '</div>',
                    unsafe_allow_html=True)


def empty_dashboard():
    page = st.session_state.navigation
    st.markdown('<div class="eyebrow">ORBITGUARD / OPERATOR WORKSPACE</div>', unsafe_allow_html=True)
    st.title(page)
    st.caption('Explore and analyze satellite conjunction events' if page == 'Collision Events' else 'Satellite conjunction risk assessment · Human-led decisions')
    st.info(EMPTY_MESSAGE)
    if page == 'Mission Overview':
        cols = st.columns(4)
        for col, accent, label in zip(cols, ['total', 'critical', 'review', 'window'],
                                      ['Total Events', 'Critical Events', 'Needs Review', 'High-Risk within 7 Days']):
            with col.container(key='overview_kpi_' + accent):
                st.metric(label, '—')
        left, middle, right = st.columns([1.05, 1.15, .85])
        for col, key, title in [(left, 'overview_distribution', 'Risk Distribution'),
                                (middle, 'overview_time', 'Events by Time to TCA')]:
            with col, st.container(border=True, height=510, key=key):
                st.subheader(title)
                st.info(EMPTY_MESSAGE)
        with right, st.container(border=True, height=510, key='overview_criteria'):
            risk_criteria()
        left, right = st.columns([1.9, 1])
        for col, key, title in [(left, 'overview_priority', 'Events by Risk Priority'),
                                (right, 'overview_alerts', 'Priority Alerts')]:
            with col, st.container(border=True, height=640, key=key):
                st.subheader(title)
                st.info(EMPTY_MESSAGE)
    else:
        left, right = st.columns([1, 1.25], gap='medium')
        for col, key, title in [(left, 'event_list_panel', 'Event List'),
                                (right, 'event_detail_panel', 'Event Details')]:
            with col, st.container(border=True, key=key):
                st.subheader(title)
                st.info(EMPTY_MESSAGE)


def number(value, suffix='', scientific=False):
    rendered = format_number(value, suffix, scientific)
    return '—' if not simulation and rendered == 'N/A' else rendered


def badge(text):
    label = str(text) if pd.notna(text) else 'N/A'
    cls = label if label in LEVELS + ['OK', 'NEEDS_REVIEW'] else 'unavailable'
    return '<span class="badge {}">{}</span>'.format(cls, html.escape(label))


def open_event(eid):
    st.session_state.selected_event = eid
    st.session_state.navigation = 'Collision Events'


def change_overview_page(step):
    st.session_state.overview_page = max(1, st.session_state.get('overview_page', 1) + step)


def change_event_page(step):
    st.session_state.event_page = max(1, st.session_state.get('event_page', 1) + step)
    # A new page starts with no pending row/picker selection; open details stay put.
    st.session_state.event_selection_generation = st.session_state.get('event_selection_generation', 0) + 1


def csv_status(state, message):
    icon = {'idle': '○', 'ready': '○', 'success': '✓'}[state]
    st.markdown('<div class="csv-status csv-status-{}" role="status">{} {}</div>'.format(
        state, icon, html.escape(message)), unsafe_allow_html=True)


def normalized_status(data):
    return str(field(data.get('summary', {}), 'status', 'unknown'))


def backend_downloads(data):
    st.download_button('Download complete processing results (CSV)',
                       data['records'].to_csv(index=False).encode('utf-8'),
                       file_name='orbitguard_processing_results.csv', mime='text/csv', key='records_download')
    st.download_button('Download input issues (CSV)',
                       data['issues'].to_csv(index=False).encode('utf-8'),
                       file_name='orbitguard_input_issues.csv', mime='text/csv', key='issues_download')


def event_table(events, key, height=410):
    display = pd.DataFrame({
        'Event ID': events.event_id,
        'Current Risk Level': events.risk_level.fillna('N/A'),
        ('Latest Pc' if simulation else ('Latest Known Pc' if field(metadata, 'task') == 'final' else 'Current Estimated Pc')): events.tier_pc.map(lambda x: number(x, scientific=True)),
        'Conservative Estimate': events.conservative_pc.map(lambda x: number(x, scientific=True)),
        'Miss Distance': events.miss_distance_m.map(lambda x: number(x / 1000, ' km')),
        'Time to TCA': events.time_to_tca_hours.map(lambda x: number(x, ' h')),
        'Data Quality': events.data_quality_status.fillna('N/A'),
        'Ranking Score': events.alert_score.map(number),
    }).reset_index(drop=True)
    def cell_style(value):
        if value in COLORS:
            return 'color: {}; font-weight: 600'.format(COLORS[value])
        if value == 'NEEDS_REVIEW':
            return 'color: #ffc444; font-weight: 600'
        if value == 'OK':
            return 'color: #00d9a1'
        return ''
    styled = display.style.map(cell_style, subset=['Current Risk Level', 'Data Quality'])
    selected_id = st.session_state.get('selected_event')
    if selected_id is not None:
        styled = styled.apply(lambda row: ['background-color: #0b2d46' if row['Event ID'] == selected_id else ''] * len(row), axis=1)
    event = st.dataframe(styled, hide_index=True, width='stretch', height=height,
                         selection_mode='single-row', on_select='rerun', key=key, row_height=32)
    return display.iloc[event.selection.rows[0]]['Event ID'] if event.selection.rows else None


def details(events, history, simulation):
    selected = st.session_state.get('selected_event')
    rows = events[events.event_id == selected]
    with st.container(border=True, key='event_detail_panel', height='stretch'):
        with st.container(horizontal=True, horizontal_alignment='distribute', vertical_alignment='center',
                          key='event_detail_header'):
            st.subheader('Event Details')
            if selected is not None and st.button('✕', key='close_details', help='Close event details', width='content'):
                st.session_state.selected_event = None
                st.rerun()
        if rows.empty:
            st.markdown('<div class="empty">Select an event to view its detailed risk assessment.</div>', unsafe_allow_html=True)
            return
        row = rows.iloc[0]
        st.subheader('Event #{}'.format(row.event_id))
        st.markdown(badge(row.risk_level) + badge(row.data_quality_status), unsafe_allow_html=True)
        if not simulation:
            status = field(row, 'prediction_status')
            final_tier = field(row, 'predicted_final_tier')
            st.caption('Prediction Status: {}'.format(explanation(status)))
            if status != 'PREDICTED':
                st.warning('No successful prediction is available for this event.')
            elif pd.notna(final_tier):
                st.caption('Predicted Final Tier: {}'.format(final_tier))
        st.caption('Conjunction event · satellite identifier not supplied')
        cols = st.columns(3)
        for col, label, value in zip(cols, ['Latest Pc' if simulation else ('Latest Known Pc' if field(metadata, 'task') == 'final' else 'Current Estimated Pc'), 'Predicted Final Pc', 'Conservative Estimate'],
                                      [row.tier_pc, row.predicted_pc, row.conservative_pc]):
            col.metric(label, number(value, scientific=True))
        cols = st.columns(3)
        for col, label, value in zip(cols, ['Miss Distance', 'Time to TCA', 'Sigma Ratio'],
                                    [number(row.miss_distance_m / 1000, ' km'), number(row.time_to_tca_hours, ' h'), number(row.sigma_ratio)]):
            col.metric(label, value)
        if row.data_quality_status == 'NEEDS_REVIEW':
            st.warning('Needs review — ' + explanation(row.data_quality_reason))
        elif row.risk_level == 'UNKNOWN' or pd.isna(row.risk_level):
            st.warning('Current risk is unclassified. Review the available evidence.')
        st.markdown('<div class="detail-evidence"><b>Trigger Reason</b><p>{}</p>'
                    '<b>Data Quality Reason</b><p>{}</p></div>'.format(
                        html.escape(explanation(row.trigger_reason)), html.escape(explanation(row.data_quality_reason))),
                    unsafe_allow_html=True)
        if any('\u4e00' <= c <= '\u9fff' for c in explanation(row.trigger_reason) + explanation(row.data_quality_reason)):
            st.caption('Original backend explanation preserved; a verified English translation is not available.')
        st.caption('Ranking Score: {} · Used for prioritization, not collision probability.'.format(number(row.alert_score)))
        hist = history[history.event_id == row.event_id].dropna(subset=['time_to_tca_hours'])
        # Keys depend on the selected event so Plotly camera/state cannot leak between events.
        with st.container(key='detail_charts'):
            history_col, position_col = st.columns(2, gap='small')
            with history_col:
                st.subheader('Risk History · Before Cutoff')
                if hist.empty or hist[['risk', 'miss_distance_m']].isna().all().all():
                    st.info('No historical observations are available for this event.' if simulation else 'Historical data unavailable')
                else:
                    st.plotly_chart(history_chart(hist, risk_is_log10=True), use_container_width=True,
                                    key='history_{}'.format(row.event_id))
                    if not simulation:
                        st.caption('Observed input messages returned by the service · risk = log10(Pc). Smaller time to TCA means a newer observation. Final mode includes only eligible input at or before the model cutoff (time to TCA ≥ horizon).')
            with position_col:
                st.subheader('Relative Position · R / T / N')
                if row[['relative_position_r', 'relative_position_t', 'relative_position_n']].isna().any():
                    st.info('R/T/N coordinates are unavailable for this event.')
                else:
                    st.plotly_chart(position_chart(row), use_container_width=True,
                                    key='position_{}'.format(row.event_id))
                    st.caption('R: {} · T: {} · N: {}'.format(*[number(row[c], ' m') for c in ['relative_position_r', 'relative_position_t', 'relative_position_n']]))
                st.caption('Relative-position schematic only; this is not an orbital trajectory.')


with st.sidebar:
    st.markdown('<div class="sidebar-brand-header"><div class="brand">◉ ORBITGUARD</div>'
                '<div class="sidebar-subtitle">Satellite Risk Intelligence</div></div>', unsafe_allow_html=True)
    # Handle an old session still pointing at the removed read-only Settings page.
    if st.session_state.get('navigation') not in (None, 'Mission Overview', 'Collision Events'):
        st.session_state.navigation = 'Mission Overview'
    st.markdown('<div class="sidebar-section-label">WORKSPACE</div>', unsafe_allow_html=True)
    st.radio('Navigation', ['Mission Overview', 'Collision Events'], key='navigation', label_visibility='collapsed')
    st.markdown('<div class="sidebar-section-label">DATA SOURCE</div>', unsafe_allow_html=True)
    mode = st.radio('Data source', ['Simulation', 'Backend CSV'], index=None, key='data_source',
                    label_visibility='collapsed', on_change=clear_dataset_selection)
    simulation = mode == 'Simulation'
    st.markdown('<div class="sidebar-section-label">DATA STATUS</div>', unsafe_allow_html=True)
    if simulation:
        st.markdown('<div class="sim">SIMULATION DATA</div>', unsafe_allow_html=True)
        st.caption('Demonstration values. Not real model predictions.')
        if st.button('Reload simulation', width='stretch'):
            st.session_state.pop('selected_event', None)
            st.rerun()
        raw = mock_result()
    elif mode == 'Backend CSV':
        with st.container(key='csv_upload_panel'):
            upload = st.file_uploader('Input CSV', type=['csv'], label_visibility='collapsed')
            if upload is None:
                st.button('Analyze CSV', type='primary', disabled=True, width='stretch')
                csv_status('idle', 'Choose a CSV to begin.')
                raw = None
            else:
                payload = upload.getvalue()
                token = hashlib.sha256(payload).hexdigest()
                if st.button('Analyze CSV', type='primary', width='stretch'):
                    st.session_state.pop('backend_result', None)
                    st.session_state.pop('backend_error', None)
                    try:
                        with st.spinner('Analyzing CSV…'):
                            from gui_backend import analyze_upload
                            normalized = analyze_upload(payload)
                        st.session_state.backend_result = normalized
                        st.session_state.backend_token = token
                        clear_dataset_selection()
                    except Exception as exc:
                        st.session_state.backend_error = str(exc) or type(exc).__name__
                        st.session_state.backend_error_token = token
                raw = st.session_state.get('backend_result') if st.session_state.get('backend_token') == token else None
                if raw is not None:
                    csv_status('success' if normalized_status(raw) == 'ok' else 'ready',
                               'Analysis complete.' if normalized_status(raw) == 'ok' else 'Analysis returned ' + normalized_status(raw) + '; review warnings.')
                elif st.session_state.get('backend_error_token') == token and st.session_state.get('backend_error'):
                    st.error('Analysis failed: {}'.format(st.session_state.backend_error))
                else:
                    csv_status('ready', 'Ready to analyze.')
    else:
        raw = None
    st.markdown('<div class="sidebar-footer">Operator decision support · Human review required</div>', unsafe_allow_html=True)

try:
    data = normalize_result(raw) if simulation else raw
except (ValueError, TypeError) as exc:
    st.error('Analysis result cannot be displayed: {}'.format(exc))
    st.stop()

if data is None:
    empty_dashboard()
    st.stop()

events, history, summary, metadata = (data[k] for k in ['events', 'history', 'summary', 'metadata'])
for warning in data['warnings']:
    st.warning(warning)
if not data['issues'].empty:
    with st.expander('Input data warnings · {} issues'.format(len(data['issues']))):
        if not simulation:
            errors = int(data['issues'].severity.eq('ERROR').sum())
            warnings = int(data['issues'].severity.eq('WARNING').sum())
            if errors:
                st.error('{} input errors; affected input may not be predicted.'.format(errors))
            if warnings:
                st.warning('{} input warnings require review.'.format(warnings))
        # Convert only the displayed copy: text columns align left in the native grid.
        issues_display = data['issues'].copy()
        for column in ['row_number', 'input_record_id', 'row_position', 'event_id', 'value']:
            if column in issues_display:
                issues_display[column] = issues_display[column].map(lambda value: 'N/A' if pd.isna(value) else str(value))
        st.dataframe(issues_display, hide_index=True, width='stretch',
                     column_config={'row_number': st.column_config.TextColumn('row_number')})
if not simulation:
    st.caption('REAL BACKEND ANALYSIS · Uploaded CSV · {}'.format(normalized_status(data)))
    status = normalized_status(data)
    messages = {'partial': 'Partial analysis: some events or records were not predicted. Review input issues.',
                'invalid_input': 'Invalid input: no events could be predicted. Review the validation errors.',
                'no_eligible_input': 'No eligible input messages: no predictions were generated for this model horizon.',
                'empty': 'The uploaded CSV contains no input messages.'}
    if status in messages:
        (st.error if status == 'invalid_input' else st.warning)(messages[status])
    elif status != 'ok':
        st.warning('Unrecognized service status: {}'.format(status))
if events.empty:
    if not simulation:
        backend_downloads(data)
        with st.expander('Analysis metadata and data conventions'):
            st.json(metadata)
    st.info('The analysis returned no conjunction events.')
    st.stop()

page = st.session_state.navigation
st.markdown('<div class="eyebrow">ORBITGUARD / OPERATOR WORKSPACE</div>', unsafe_allow_html=True)
st.title(page)
st.caption('Explore and analyze satellite conjunction events' if page == 'Collision Events' else 'Satellite conjunction risk assessment · Human-led decisions')
if simulation:
    st.caption('SIMULATION DATA · {} unique demonstration events · No live telemetry'.format(len(events)))

if page == 'Mission Overview':
    cols = st.columns(4)
    for col, accent, label, value in zip(cols, ['total', 'critical', 'review', 'window'], ['Total Events', 'Critical Events', 'Needs Review', 'High-Risk within 7 Days'],
                                  [field(summary, 'event_count'), field(summary, 'critical_count'), field(summary, 'needs_review_count'), seven_day_count(events)]):
        with col.container(key='overview_kpi_' + accent):
            st.metric(label, 'N/A' if value is None else '{:,}'.format(value))
    with st.container(key='overview_coverage'):
        st.caption('7-day display statistic: unique CRITICAL or ELEVATED events with 0 ≤ time to TCA ≤ 168 hours. Current levels are supplied by the backend.')
    window = field(summary, 'window_24h_count')
    note = field(summary, 'window_24h_note')
    if not simulation:
        coverage = field(field(summary, 'window_24h', {}), 'coverage_status')
        note = ('Insufficient coverage. No reliable 24-hour risk conclusion can be drawn.' if coverage == 'INSUFFICIENT' else
                'Per-event latest-message coverage; not synchronized coverage. ' + str(note or ''))
    with st.container(key='overview_window_note'):
        st.caption('Backend 24-hour window: {}'.format(note or 'Coverage information unavailable' if window is None else '{} events{}'.format(window, ' · ' + str(note) if note else '')))
    left, middle, right = st.columns([1.05, 1.15, .85])
    with left, st.container(border=True, height=510, key='overview_distribution'):
        st.subheader('Risk Distribution')
        st.caption('Current risk levels · backend summary')
        fig = donut(summary)
        if fig is None:
            st.info('Risk distribution summary is unavailable.')
        else:
            st.plotly_chart(fig, use_container_width=True)
    with middle, st.container(border=True, height=510, key='overview_time'):
        st.subheader('Events by Time to TCA')
        chart_risk = st.selectbox('Chart risk level', ['All risk levels'] + LEVELS)
        chart_events = events if chart_risk == 'All risk levels' else events[events.risk_level == chart_risk]
        st.plotly_chart(time_chart(chart_events), use_container_width=True)
    with right, st.container(border=True, height=510, key='overview_criteria'):
        risk_criteria()
    left, right = st.columns([1.9, 1])
    with left, st.container(border=True, height=640, key='overview_priority'):
        st.subheader('Events by Risk Priority')
        search = st.text_input('Search event ID', key='overview_search', placeholder='Search event ID…', label_visibility='collapsed')
        ranked = priority_events(events, search=search)
        if st.session_state.get('overview_query') != search:
            st.session_state.overview_page = 1
            st.session_state.overview_query = search
        pages = max(1, math.ceil(len(ranked) / 10))
        st.session_state.setdefault('overview_page', 1)
        if st.session_state.get('overview_page', 1) > pages:
            st.session_state.overview_page = pages
        overview_page = st.session_state.overview_page
        start = (overview_page - 1) * 10
        end = min(start + 10, len(ranked))
        overview_rows = ranked.iloc[start:end]
        selected = None
        if overview_rows.empty:
            st.info('No events match your search.')
        else:
            selected = event_table(overview_rows, 'overview_table_{}_{}_{}'.format(search, overview_page, st.session_state.get('dataset_generation', 0)),
                                   height=32 * (len(overview_rows) + 1) + 22)
        if selected is not None:
            st.button('Open event #{}'.format(selected), on_click=open_event, args=(selected,))
        with st.container(key='overview_pagination'):
            previous, indicator, following = st.columns([1.2, 2.6, 1], vertical_alignment='center')
            previous.button('Previous', key='overview_previous', disabled=overview_page == 1,
                            on_click=change_overview_page, args=(-1,), width='stretch')
            indicator.markdown('<div class="pagination-summary"><strong>Page {} of {}</strong>'
                               '<span>Showing {}–{} of {} events</span></div>'.format(
                                   overview_page, pages, start + 1 if len(ranked) else 0, end, len(ranked)),
                               unsafe_allow_html=True)
            following.button('Next', key='overview_next', disabled=overview_page == pages,
                             on_click=change_overview_page, args=(1,), width='stretch')
    with right, st.container(border=True, height=640, key='overview_alerts'):
        st.subheader('Priority Alerts')
        st.caption('Critical or review-required events')
        alerts = priority_events(events[events.risk_level.eq('CRITICAL') | events.data_quality_status.eq('NEEDS_REVIEW')])
        with st.container(height=520, border=False, key='priority_alert_scroll'):
            for _, row in alerts.iterrows():
                review = row.data_quality_status == 'NEEDS_REVIEW'
                accent = 'review' if review else 'critical'
                with st.container(key='priority_alert_{}_{}'.format(accent, row.event_id)):
                    st.markdown('<div class="alert-card-content"><div class="alert-card-title">'
                                '<strong>Event #{}</strong><span>TCA {}</span></div>'
                                '<div class="alert-card-badges">{}{}</div></div>'.format(
                                    html.escape(str(row.event_id)), html.escape(number(row.time_to_tca_hours, ' h')),
                                    badge(row.risk_level), badge('NEEDS_REVIEW') if review else ''),
                                unsafe_allow_html=True)
                    # A native keyboard-accessible button overlays the visual card.
                    st.button('Open event #{}'.format(row.event_id), key='alert_{}'.format(row.event_id),
                              on_click=open_event, args=(row.event_id,), width='stretch',
                              help=explanation(row.data_quality_reason if review else row.trigger_reason))
else:
    a, b, c, d = st.columns([1.5, 1, 1.2, 1.1])
    search = a.text_input('Search event ID', placeholder='Search event ID…')
    risk = b.selectbox('Current risk level', ['All risk levels'] + LEVELS)
    quality = c.selectbox('Data quality', ['All data quality', 'OK', 'NEEDS_REVIEW'])
    sort = d.selectbox('Sort by', ['Risk priority', 'Time to TCA', 'Ranking score'])
    filtered = priority_events(events, search, risk, quality, sort)
    signature = (search, risk, quality, sort)
    if st.session_state.get('filter_signature') != signature:
        st.session_state.event_page = 1
        st.session_state.event_selection_generation = st.session_state.get('event_selection_generation', 0) + 1
        st.session_state.filter_signature = signature
    left, right = st.columns([1, 1.25], gap='medium')
    with left, st.container(border=True, key='event_list_panel'):
        st.subheader('Event List · {} events'.format(len(filtered)))
        total_pages = max(1, math.ceil(len(filtered) / 15))
        st.session_state.setdefault('event_page', 1)
        if st.session_state.get('event_page', 1) > total_pages:
            st.session_state.event_page = total_pages
        page_number = st.session_state.event_page
        generation = st.session_state.get('event_selection_generation', 0)
        start = (page_number - 1) * 15
        end = min(start + 15, len(filtered))
        visible = filtered.iloc[start:end]
        if visible.empty:
            st.info('No events match the selected filters.')
        else:
            selected = event_table(visible, 'events_{}_{}_{}'.format(signature, page_number, generation),
                                   height=min(550, 32 * (len(visible) + 1) + 22))
            if selected is not None:
                marker = (signature, page_number, generation, selected)
                if st.session_state.get('last_table_selection') != marker:
                    st.session_state.selected_event = selected
                    st.session_state.last_table_selection = marker
                    st.rerun()
        with st.container(key='event_pagination'):
            previous, indicator, following = st.columns([1.2, 2.6, 1], vertical_alignment='center')
            previous.button('Previous', key='event_previous', disabled=page_number == 1,
                            on_click=change_event_page, args=(-1,), width='stretch')
            indicator.markdown('<div class="pagination-summary"><strong>Page {} of {}</strong>'
                               '<span>Showing {}–{} of {} events</span></div>'.format(
                                   page_number, total_pages, start + 1 if len(filtered) else 0, end, len(filtered)),
                               unsafe_allow_html=True)
            following.button('Next', key='event_next', disabled=page_number == total_pages,
                             on_click=change_event_page, args=(1,), width='stretch')
        with st.container(key='event_open_controls'):
            options = [None] + visible.event_id.tolist()
            # Accessible alternative to row selection, also supported by keyboard and AppTest.
            choice = st.selectbox('Open event details', options, format_func=lambda x: 'Choose an event…' if x is None else 'Event #{}'.format(x),
                                  key='event_picker_{}_{}_{}'.format(signature, page_number, generation), disabled=visible.empty)
            if st.button('Open selected event', disabled=choice is None):
                st.session_state.selected_event = choice
                st.rerun()
            selected_id = st.session_state.get('selected_event')
            if selected_id is not None and selected_id not in visible.event_id.values:
                st.caption('Details remain on Event #{}. Choose an event on this page to change them.'.format(selected_id))
    with right:
        details(events, history, simulation)

st.divider()
if not simulation:
    backend_downloads(data)
st.download_button('↓ Download displayed events (CSV)',
                   (filtered if page == 'Collision Events' else events).to_csv(index=False).encode('utf-8'),
                   file_name='orbitguard_{}_events.csv'.format('simulation' if simulation else 'analysis'), mime='text/csv')
with st.expander('Analysis metadata and data conventions'):
    # The removed Settings page was read-only; retain its context here.
    st.caption('Source: {}'.format('SIMULATION DATA' if simulation else 'Uploaded CSV · backend analysis'))
    st.json({k: field(metadata, k) for k in ['task', 'horizon_days', 'alpha', 'model_version', 'service_version', 'rules_version', 'variant', 'history_risk_scale', 'units']})
    if simulation:
        st.caption('Task, horizon and alpha are fixture context. No trained model is loaded; no model version is claimed.')
    st.caption('Risk levels and trigger reasons are authoritative backend outputs. No frontend risk inference is performed. Missing values are shown as N/A.')
    st.write('Latest Known Pc (final mode) = tier_pc · Predicted Final Pc = predicted_pc · Conservative Estimate = conservative_pc. Ranking Score is not a collision probability.')
    st.caption('The seven-day KPI is a display window, not a universal model prediction horizon. Unknown means unclassified. Review flags require operator attention.')
    st.caption('Risk History uses the service’s eligible input observations, never historical predictions or Simulation data. The service enforces the model horizon.')
