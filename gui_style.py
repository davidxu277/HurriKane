CSS = """
<style>
:root {color-scheme: dark;}
.stApp {background:radial-gradient(ellipse at 80% 0%,#0c2232 0%,#030c14 65%);color:#e8f2fc;font-family:Inter,Arial,sans-serif;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#030d16;border-right:1px solid #173347;}
@media(min-width:900px) {[data-testid="stSidebar"][aria-expanded="true"] {width:264px!important;min-width:264px!important;max-width:264px!important;}}
[data-testid="stSidebarContent"] {padding-left:18px!important;padding-right:18px!important;}
[data-testid="stSidebar"] h1 {color:#00c9ff;font-size:1.5rem;letter-spacing:.03em;}
.block-container {padding-top:2rem;padding-bottom:2rem;max-width:1800px;}
h1,h2,h3 {letter-spacing:-.03em;color:#f4f8fe;}
h1 {font-size:2rem!important;} h3 {font-size:1.08rem!important;}
.stCaption,[data-testid="stCaptionContainer"] {color:#9bb5cb!important;}
[data-testid="stCaptionContainer"] p {color:#9bb5cb!important;}
.st-key-event_detail_panel [data-testid="stMetricValue"] {font-size:1.05rem!important;}
.st-key-event_detail_panel [data-testid="stMetricLabel"] p {font-size:.7rem;white-space:normal;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label {color:#bcd3e7;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:9px 12px;border-radius:7px;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#083658;color:#e8f7ff;}
[data-testid="stSidebarHeader"] {height:40px!important;min-height:40px!important;position:relative;}
/* Keep Streamlit 1.50's own controls and handlers; style only stable test IDs. */
[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarCollapseButton"] {position:absolute;top:16px;right:0;opacity:1!important;visibility:visible!important;}
[data-testid="stSidebarCollapseButton"] button,[data-testid="stExpandSidebarButton"] {width:32px;height:32px;min-width:32px;min-height:32px;padding:0;border:1px solid #29485d;border-radius:7px;background:#0a1d2b;color:#a8cde4;}
[data-testid="stExpandSidebarButton"] {position:fixed;top:16px;left:18px;z-index:1001;}
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {font-size:20px;line-height:1;}
[data-testid="stSidebarCollapseButton"] button:hover,[data-testid="stExpandSidebarButton"]:hover {border-color:#00bfff;background:#0b2f46;color:#dcf5ff;}
[data-testid="stSidebarCollapseButton"] button:focus-visible,[data-testid="stExpandSidebarButton"]:focus-visible {outline:2px solid #00c9ff;outline-offset:2px;}
[data-testid="stSidebarUserContent"] {padding:8px 0 20px!important;}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {gap:12px!important;}
.sidebar-brand-header {text-align:center;}
.sidebar-brand-header .brand {font-size:1.2rem;margin-bottom:3px;}
.sidebar-subtitle {font-size:.72rem;line-height:1.5;color:#9bb5cb;}
.sidebar-section-label {border-top:1px solid #183044;padding-top:10px;font-size:.63rem;font-weight:600;letter-spacing:.13em;color:#8ba9bf;text-align:center;line-height:1.5;}
.sidebar-footer {border-top:1px solid #183044;padding-top:12px;font-size:.7rem;line-height:1.5;color:#8ba9bf;text-align:center;}
/* Preserve Markdown's paragraph-tail space for plain HTML blocks in Streamlit. */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(> .sidebar-brand-header),
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(> .sidebar-section-label),
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(> .sidebar-footer),
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(> .sim) {padding-bottom:16px;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {font-size:.72rem;line-height:1.5;margin:0 0 16px;}
[data-testid="stSidebar"] .sim {padding:8px 10px;font-size:.65rem;}
[data-testid="stSidebar"] .stButton button {min-height:34px;font-size:.76rem;}
[data-testid="stSidebar"] .brand,[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {text-align:center;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {justify-content:center;text-align:center;width:100%;}
[data-testid="stSidebar"] [data-testid="stRadio"],
[data-testid="stSidebar"] [role="radiogroup"] {gap:5px;width:100%!important;}
[data-testid="stSidebar"] .st-key-navigation,[data-testid="stSidebar"] .st-key-data_source,
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(> .stButton) {width:100%!important;}
[data-testid="stSidebar"] [role="radiogroup"] label {justify-content:center;width:100%;margin:0;min-height:36px;padding:6px 10px;}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-size:.8rem;line-height:1.5;}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] {width:100%;text-align:center;}
[data-testid="stSidebar"] .stButton>button {width:100%;}
[data-testid="stSidebar"] .sim {text-align:center;}
.st-key-csv_upload_panel {gap:8px!important;}
.st-key-csv_upload_panel > [data-testid="stElementContainer"] {flex-shrink:0!important;}
.st-key-csv_upload_panel [data-testid="stFileUploader"] {min-height:102px;}
.st-key-csv_upload_panel [data-testid="stFileUploaderDropzone"] {min-height:0;padding:8px;border:1px dashed #2c526b;border-radius:8px;background:#081b29;}
.st-key-csv_upload_panel [data-testid="stFileUploaderDropzoneInstructions"] {display:none;}
.st-key-csv_upload_panel [data-testid="stFileUploaderDropzone"] > span {width:100%;margin:0;}
.st-key-csv_upload_panel [data-testid="stFileUploaderDropzone"] button {width:100%;min-height:34px;padding:5px 10px;font-size:.76rem;}
.st-key-csv_upload_panel [data-testid="stFileUploader"] small {font-size:.68rem;}
.st-key-csv_upload_panel [data-testid="stFileUploaderFile"] {gap:8px;padding:4px 0;}
.st-key-csv_upload_panel [data-testid="stFileUploader"] ul {padding:0;margin:6px 0 0;list-style:none;}
.st-key-csv_upload_panel [data-testid="stFileUploader"] li {padding:0;margin:0;}
.st-key-csv_upload_panel [data-testid="stFileUploaderFile"] > div:first-child {padding:0;}
.st-key-csv_upload_panel [data-testid="stFileUploaderFile"] svg {width:18px;height:18px;}
.st-key-csv_upload_panel .stFileUploaderFileData {min-width:0;flex:1;padding:0;}
.st-key-csv_upload_panel [data-testid="stFileUploaderFileName"] {font-size:.74rem;line-height:1.5;white-space:normal;overflow-wrap:anywhere;overflow:visible;text-overflow:clip;}
.st-key-csv_upload_panel [data-testid="stFileUploaderDeleteBtn"] button {min-height:28px;min-width:28px;padding:3px;}
.st-key-csv_upload_panel [data-testid="stSpinner"] {font-size:.75rem;}
.st-key-csv_upload_panel [data-testid="stAlertContainer"] {padding:9px 10px;border:1px solid #8d3747;background:#321720;}
.st-key-csv_upload_panel [data-testid="stAlertContainer"] p {font-size:.74rem;line-height:1.5;overflow-wrap:anywhere;}
.csv-status {font-size:.74rem;line-height:1.5;color:#a9c3d7;text-align:center;overflow-wrap:anywhere;}
.csv-status-ready {color:#91cfff;}
.csv-status-success {color:#62d8b3;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(> .csv-status) {padding-bottom:16px;}
.st-key-overview_kpi_total [data-testid="stMetric"] {border-top:3px solid #00c9ff;background:linear-gradient(135deg,#0b2b3c,#06131e);}
.st-key-overview_kpi_critical [data-testid="stMetric"] {border-top:3px solid #ff5569;background:linear-gradient(135deg,#291721,#06131e);}
.st-key-overview_kpi_review [data-testid="stMetric"] {border-top:3px solid #ffc444;background:linear-gradient(135deg,#282316,#06131e);}
.st-key-overview_kpi_window [data-testid="stMetric"] {border-top:3px solid #7ebfff;background:linear-gradient(135deg,#152b40,#06131e);}
[class*="st-key-overview_kpi_"] [data-testid="stMetric"] {height:124px;padding:17px 18px;}
[class*="st-key-overview_kpi_"] [data-testid="stMetricValue"] {font-size:2.35rem;line-height:1.3;}
.st-key-overview_kpi_critical [data-testid="stMetricValue"] {color:#ff6578;}
.st-key-overview_kpi_review [data-testid="stMetricValue"] {color:#ffd05a;}
.st-key-overview_coverage [data-testid="stCaptionContainer"] p,
.st-key-overview_window_note [data-testid="stCaptionContainer"] p {font-size:.73rem;line-height:1.4;margin:0;color:#85a3ba!important;}
.st-key-overview_distribution,.st-key-overview_time,.st-key-overview_criteria,
.st-key-overview_priority,.st-key-overview_alerts {padding:18px!important;border-color:#163247!important;border-radius:12px!important;background:linear-gradient(130deg,#081925aa,#06121caa);}
.st-key-overview_distribution h3,.st-key-overview_time h3,.st-key-overview_criteria h3,
.st-key-overview_priority h3,.st-key-overview_alerts h3 {padding-top:0!important;}
.st-key-priority_alert_scroll [data-testid="stVerticalBlock"] {gap:8px;}
.st-key-overview_criteria .criteria {padding:8px 10px;margin:5px 0;font-size:.76rem;line-height:1.5;}
.criteria-group {display:flex;flex-direction:column;gap:9px;}
.st-key-overview_criteria .criteria-group .criteria {margin:0;}
[class*="st-key-priority_alert_review_"],[class*="st-key-priority_alert_critical_"] {position:relative;min-height:84px;height:84px;padding:12px;border:1px solid #1b3b50;border-radius:9px;background:linear-gradient(110deg,#0b2130,#071722);}
[class*="st-key-priority_alert_review_"]:hover,[class*="st-key-priority_alert_critical_"]:hover,
[class*="st-key-priority_alert_review_"]:focus-within,[class*="st-key-priority_alert_critical_"]:focus-within {border-color:#00bfff;box-shadow:0 0 0 1px #00bfff44;background:#0b2a3c;}
.alert-card-content {pointer-events:none;}
.alert-card-title {display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:.84rem;color:#e8f2fc;}
.alert-card-title span {font-size:.72rem;color:#aec9de;white-space:nowrap;}
.alert-card-badges {margin-top:8px;display:flex;gap:5px;flex-wrap:wrap;}
.alert-card-badges .badge {font-size:.61rem;padding:2px 6px;margin:0;line-height:1.5;}
.st-key-priority_alert_scroll [data-testid="stElementContainer"]:has(> .stButton) {position:absolute!important;inset:0;width:100%!important;height:100%!important;z-index:2;}
.st-key-priority_alert_scroll .stButton,.st-key-priority_alert_scroll .stButton [data-testid="stTooltipHoverTarget"] {height:100%;}
.st-key-priority_alert_scroll .stButton button {width:100%;height:100%;min-height:0;opacity:0;cursor:pointer;}
.st-key-event_list_panel,.st-key-event_detail_panel {padding:16px;border:1px solid #163247!important;border-radius:12px;background:linear-gradient(130deg,#081925aa,#06121caa);}
.st-key-event_detail_panel {gap:10px!important;}
.st-key-event_detail_panel h3 {padding-top:0!important;padding-bottom:5px!important;}
.st-key-event_detail_header {min-height:32px;gap:10px!important;}
.st-key-event_detail_header h3 {padding:0!important;line-height:32px;}
.st-key-event_detail_header [data-testid="stElementContainer"]:has(> [data-testid="stHeading"]) {min-height:32px!important;}
.st-key-close_details button {width:32px;min-width:32px;height:32px;min-height:32px;padding:0;border:1px solid #29485d;border-radius:7px;background:#0a1d2b;color:#bcd8eb;}
.st-key-close_details button:hover {border-color:#00bfff;background:#0b2f46;color:#e5f8ff;}
.st-key-close_details button:focus-visible {outline:2px solid #00c9ff;outline-offset:2px;}
.st-key-event_detail_panel [data-testid="stMetric"] {min-height:74px;padding:10px;}
.st-key-event_detail_panel [data-testid="stAlert"] {font-size:.8rem;}
.st-key-event_detail_panel [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {background:#34270f;border:1px solid #81601f;color:#ffdc80;}
.detail-evidence {border:1px solid #19384d;border-radius:8px;padding:10px 12px;font-size:.75rem;line-height:1.5;}
.detail-evidence b {color:#a6c5dc;font-size:.72rem;}
.detail-evidence p {margin:3px 0 8px;}
.detail-evidence p:last-child {margin-bottom:0;}
.st-key-detail_charts {max-height:405px;overflow-y:auto;scrollbar-gutter:stable;border-top:1px solid #173449;padding-top:12px;}
.st-key-detail_charts [data-testid="stHorizontalBlock"] {flex-wrap:wrap;align-items:flex-start;}
.st-key-detail_charts [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 275px!important;min-width:min(275px,100%)!important;width:auto!important;}
.st-key-detail_charts h3 {font-size:.88rem!important;}
.pagination-summary {text-align:center;line-height:1.5;}
.pagination-summary strong {display:block;font-size:.8rem;color:#d2e6f6;font-weight:600;}
.pagination-summary span {display:block;font-size:.7rem;color:#9bb5cb;}
.st-key-overview_pagination .stButton button {min-height:36px;padding:5px 8px;font-size:.77rem;}
.st-key-event_pagination .stButton button {min-height:36px;padding:5px 8px;font-size:.77rem;}
.st-key-event_pagination .stButton button p {font-size:.75rem;white-space:nowrap;}
/* Native containers are flex columns: auto margins consume remaining space. */
.st-key-overview_priority {padding-bottom:22px!important;}
.st-key-overview_priority > [data-testid="stLayoutWrapper"]:has(> .st-key-overview_pagination) {margin-top:auto!important;flex-shrink:0!important;}
.st-key-event_list_panel > [data-testid="stLayoutWrapper"]:has(> .st-key-event_open_controls) {margin-top:12px!important;flex-shrink:0!important;}
.st-key-event_list_panel {padding-bottom:22px!important;}
.st-key-event_open_controls {gap:8px!important;}
.st-key-event_open_controls .stButton {width:100%;}
.st-key-event_open_controls .stButton button {width:100%;}
.st-key-event_pagination .pagination-summary span {font-size:.66rem;}
@media(max-width:1100px) {
  [data-testid="stHorizontalBlock"]:has(.st-key-event_list_panel) {flex-wrap:wrap;}
  [data-testid="stColumn"]:has(.st-key-event_list_panel),
  [data-testid="stColumn"]:has(.st-key-event_detail_panel) {min-width:min(340px,100%)!important;}
}
.brand {color:#00c9ff;font-size:1.25rem;font-weight:800;letter-spacing:.02em;white-space:nowrap;margin:0 0 8px;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#163247!important;border-radius:12px!important;background:linear-gradient(130deg,#081925aa,#06121caa);}
[data-testid="stMetric"] {min-height:105px;padding:15px;border:1px solid #19394f;border-radius:12px;background:linear-gradient(130deg,#0a1e2d,#06131e);}
[data-testid="stMetricValue"] {font-size:1.8rem;color:#eef7ff;font-weight:700;}
[data-testid="stMetricLabel"] {color:#bcd3e7;}
.stButton>button,.stDownloadButton>button {border:1px solid #20516d;background:#082239;color:#d5edff;border-radius:8px;}
.stButton>button:hover {border-color:#00bfff;color:#00cfff;}
.stTextInput input {background:#071724;color:#eef7ff;}
[data-baseweb="select"]>div {background:#071724;color:#d4e8fa;border-color:#1a3b52;}
.eyebrow {color:#00c9ff;font-size:.73rem;font-weight:700;letter-spacing:.14em;margin-bottom:.35rem;}
.badge {display:inline-block;padding:4px 10px;border:1px solid;border-radius:6px;font-size:.72rem;font-weight:700;margin:0 5px 7px 0;}
.CRITICAL {color:#ff6d7d;background:#4a1724;border-color:#913346;}
.ELEVATED,.NEEDS_REVIEW {color:#ffd254;background:#3c2e0d;border-color:#856a1e;}
.SAFE,.OK {color:#21e6ae;background:#063b32;border-color:#096651;}
.UNKNOWN {color:#c0d0e5;background:#23374c;border-color:#55728b;}
.unavailable {color:#c0d0e5;background:#23374c;border-color:#55728b;}
.sim {color:#ffc444;background:#352c16;border:1px solid #6e5929;padding:9px 12px;border-radius:7px;font-size:.75rem;font-weight:700;letter-spacing:.08em;}
.criteria {border-left:3px solid;padding:9px 12px;background:#091b27;margin:8px 0;border-radius:5px;font-size:.8rem;line-height:1.65;}
.empty {padding:75px 20px;text-align:center;color:#91abc2;border:1px dashed #244359;border-radius:10px;}
.stDataFrame {border:1px solid #163247;border-radius:8px;}
@media(max-width:900px) {.block-container {padding:1rem;} [data-testid="stMetricValue"] {font-size:1.4rem;}}
</style>
"""
