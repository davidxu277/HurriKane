"""Presentation-only Plotly figures."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gui_data import COLORS, LEVELS, field, time_counts


def theme(fig, height=300):
    fig.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#bcd2e5', family='Arial', size=12),
                      margin=dict(l=15, r=15, t=25, b=15),
                      legend=dict(orientation='h', y=1.15), hoverlabel=dict(bgcolor='#102438'))
    fig.update_xaxes(gridcolor='#153047', zerolinecolor='#28475d')
    fig.update_yaxes(gridcolor='#153047', zerolinecolor='#28475d')
    return fig


def donut(summary):
    levels = ['SAFE', 'ELEVATED', 'CRITICAL', 'UNKNOWN']
    values = [field(summary, v.lower() + '_count') for v in levels]
    if any(v is None or not isinstance(v, (int, float)) or v < 0 for v in values):
        return None
    fig = go.Figure(go.Pie(labels=levels, values=values, hole=.73, sort=False,
                           marker_colors=[COLORS[v] for v in levels], textinfo='none',
                           hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'))
    fig.update_layout(annotations=[dict(text='<b>{:,}</b><br>Events'.format(sum(values)),
                                        x=.5, y=.5, showarrow=False, font_size=23)], showlegend=True)
    theme(fig, 380)
    fig.update_traces(domain=dict(x=[0, 1], y=[.23, 1]))
    fig.update_layout(margin=dict(l=8, r=8, t=12, b=12),
                      legend=dict(orientation='h', x=.5, xanchor='center', y=0, yanchor='bottom',
                                  font_size=11, tracegroupgap=5),
                      annotations=[dict(text='<b>{:,}</b><br>Events'.format(sum(values)),
                                        x=.5, y=.615, showarrow=False, font_size=23)])
    return fig


def time_chart(events):
    counts = time_counts(events)
    fig = go.Figure()
    for level in ['SAFE', 'ELEVATED', 'CRITICAL', 'UNKNOWN']:
        fig.add_bar(name=level.title(), x=counts.index.tolist(), y=counts[level].tolist(), marker_color=COLORS[level])
    fig.update_layout(barmode='stack', xaxis_title='Time to TCA', yaxis_title='Events')
    theme(fig, 345)
    fig.update_layout(margin=dict(l=40, r=12, t=65, b=55),
                      legend=dict(orientation='h', x=0, xanchor='left', y=1.28, yanchor='bottom', font_size=10))
    return fig


def history_chart(history, risk_is_log10=False):
    data = history.sort_values('time_to_tca_hours', ascending=False)
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    risk_label = 'Risk (log10 Pc)' if risk_is_log10 else 'Risk (backend value; scale unverified)'
    fig.add_trace(go.Scatter(x=data.time_to_tca_hours, y=data.risk, name=risk_label,
                            mode='lines+markers', line_color='#00b7ff'), secondary_y=False)
    fig.add_trace(go.Scatter(x=data.time_to_tca_hours, y=data.miss_distance_m / 1000,
                            name='Miss distance (km)', mode='lines+markers', line_color='#ff5569'), secondary_y=True)
    fig.update_xaxes(title_text='Time to TCA (hours)', autorange='reversed')
    fig.update_yaxes(title_text='log10(Pc)' if risk_is_log10 else 'Risk (unverified scale)', secondary_y=False)
    fig.update_yaxes(title_text='Distance (km)', secondary_y=True)
    theme(fig, 290)
    fig.update_layout(margin=dict(l=42, r=42, t=65, b=45),
                      legend=dict(orientation='h', x=0, y=1.35, yanchor='bottom', font_size=10))
    fig.update_xaxes(title_font_size=10, tickfont_size=10)
    fig.update_yaxes(title_font_size=10, tickfont_size=10)
    return fig


def position_chart(row):
    r, t, n = (row[c] for c in ['relative_position_r', 'relative_position_t', 'relative_position_n'])
    fig = go.Figure(go.Scatter3d(x=[0, r], y=[0, t], z=[0, n], mode='lines+markers',
                               marker=dict(size=[5, 7], color=['#00b7ff', '#ff5569']),
                               line=dict(color='#ff5569', width=4), showlegend=False))
    theme(fig, 290)
    axis = dict(backgroundcolor='#081925', gridcolor='#28495f', zerolinecolor='#456079',
                tickfont=dict(size=10, color='#bcd2e5'), showbackground=True,
                title=dict(font=dict(size=11, color='#d5edff')), nticks=4)
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=28),
                      scene=dict(xaxis={**axis, 'title': {**axis['title'], 'text': 'R (m)'}},
                                 yaxis={**axis, 'title': {**axis['title'], 'text': 'T (m)'}},
                                 zaxis={**axis, 'title': {**axis['title'], 'text': 'N (m)'}},
                                 bgcolor='rgba(0,0,0,0)', aspectmode='cube',
                                 camera=dict(eye=dict(x=1.65, y=1.65, z=1.25)),
                                 domain=dict(x=[0, 1], y=[0, 1])))
    return fig
