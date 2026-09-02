# ============================================================
# SECOM YIELD RCA — STREAMLIT DASHBOARD v2
# Industrial Modern Theme | LLM Summary | Clean Navigation
# ============================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from groq import Groq

st.set_page_config(
    page_title="SECOM RCA System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0f1923; color: #cdd6e0; }
.stApp { background-color: #0f1923; }
[data-testid="stSidebar"] { background-color: #0a1018 !important; border-right: 1px solid #1e3a4a; }
[data-testid="stSidebar"] * { color: #cdd6e0 !important; }

.page-title {
    font-family: 'Share Tech Mono', monospace; font-size: 1.5rem; color: #00d4ff;
    border-bottom: 2px solid #1e3a4a; padding-bottom: 10px; margin-bottom: 18px; letter-spacing: 0.04em;
}
.metric-card {
    background: #131f2e; border: 1px solid #1e3a4a; border-top: 3px solid #00d4ff;
    border-radius: 6px; padding: 16px 18px; margin: 3px 0;
}
.metric-value { font-family: 'Share Tech Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #00d4ff; line-height: 1.2; }
.metric-label { font-size: 0.70rem; color: #7a9bb5; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 5px; }
.metric-status { font-size: 0.73rem; margin-top: 4px; color: #7a9bb5; }
.sec-header {
    font-family: 'Share Tech Mono', monospace; font-size: 0.70rem; color: #00d4ff;
    text-transform: uppercase; letter-spacing: 0.14em; border-bottom: 1px solid #1e3a4a;
    padding-bottom: 5px; margin: 14px 0 10px 0;
}
.badge-high { display:inline-block; background:rgba(255,70,70,0.15); border:1px solid #ff4646; color:#ff8080; border-radius:4px; padding:2px 10px; font-size:0.73rem; font-family:'Share Tech Mono',monospace; }
.badge-sus  { display:inline-block; background:rgba(255,186,0,0.12); border:1px solid #ffba00; color:#ffd060; border-radius:4px; padding:2px 10px; font-size:0.73rem; font-family:'Share Tech Mono',monospace; }
.badge-pass { display:inline-block; background:rgba(0,220,130,0.12); border:1px solid #00dc82; color:#00dc82; border-radius:4px; padding:2px 10px; font-size:0.73rem; font-family:'Share Tech Mono',monospace; }
.lot-card { background:#131f2e; border:1px solid #1e3a4a; border-radius:6px; padding:12px 16px; margin-bottom:8px; font-size:0.86rem; color:#e8f0f7; }
.lot-id { font-family:'Share Tech Mono',monospace; font-size:1.05rem; color:#00d4ff; font-weight:700; }
.llm-box { background:#0d1c2b; border:1px solid #1e3a4a; border-left:4px solid #00d4ff; border-radius:6px; padding:18px 22px; font-size:0.87rem; line-height:1.78; color:#cdd6e0; white-space:pre-wrap; }
.chat-user { background:#1a2d40; border-left:3px solid #00d4ff; padding:10px 14px; border-radius:0 6px 6px 0; margin:6px 0; font-size:0.87rem; }
.chat-bot  { background:#131f2e; border-left:3px solid #00dc82; padding:10px 14px; border-radius:0 6px 6px 0; margin:6px 0; font-size:0.87rem; line-height:1.7; }
.chat-lbl  { font-family:'Share Tech Mono',monospace; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:3px; }
.info-box  { background:#131f2e; border:1px solid #1e3a4a; border-radius:6px; padding:12px 16px; font-size:0.84rem; color:#9ab0c4; line-height:1.65; }
.nav-lbl   { font-family:'Share Tech Mono',monospace; font-size:0.62rem; color:#3a6070; text-transform:uppercase; letter-spacing:0.14em; padding:8px 0 3px 0; }
.stButton > button { background:#131f2e !important; color:#00d4ff !important; border:1px solid #00d4ff !important; border-radius:4px !important; font-family:'Share Tech Mono',monospace !important; font-size:0.78rem !important; padding:5px 16px !important; }
.stButton > button:hover { background:#00d4ff !important; color:#0f1923 !important; }
div[data-testid="stSelectbox"] label { color:#7a9bb5 !important; font-size:0.73rem !important; text-transform:uppercase !important; letter-spacing:0.09em !important; }
hr { border-color:#1e3a4a !important; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not set. Add it as an environment variable or in Streamlit secrets to enable LLM report generation.")
GROQ_MODEL   = "llama-3.1-8b-instant"

PT = dict(
    plot_bgcolor='#0d1c2b', paper_bgcolor='#131f2e',
    font=dict(color='#cdd6e0', family='Inter', size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)
GR = dict(gridcolor='#1e3a4a', linecolor='#1e3a4a', zerolinecolor='#1e3a4a')

@st.cache_data
def load_data():
    d = {}
    d['flagged']   = pd.read_csv(f'{OUTPUT_DIR}/data/p3_flagged_lots.csv')
    d['anomaly']   = pd.read_csv(f'{OUTPUT_DIR}/data/p3_anomaly_results.csv')
    d['shap']      = pd.read_csv(f'{OUTPUT_DIR}/data/p4_per_lot_shap.csv')
    d['top25']     = pd.read_csv(f'{OUTPUT_DIR}/data/p4_top25_stable_sensors.csv')
    d['stability'] = pd.read_csv(f'{OUTPUT_DIR}/data/p4_bootstrap_stability.csv')
    d['hyp']       = pd.read_csv(f'{OUTPUT_DIR}/data/p6_hypotheses.csv')
    d['csr']       = pd.read_csv(f'{OUTPUT_DIR}/data/p7_csr_results.csv')
    d['changes']   = pd.read_csv(f'{OUTPUT_DIR}/data/p7_sensor_changes.csv')
    d['X_test']    = pd.read_csv(f'{OUTPUT_DIR}/data/X_test_b.csv')
    d['X_train']   = pd.read_csv(f'{OUTPUT_DIR}/data/X_train_b.csv')
    d['y_train']   = pd.read_csv(f'{OUTPUT_DIR}/data/y_train_b.csv').squeeze()
    with open(f'{OUTPUT_DIR}/data/p9_all_reports.json') as f:
        d['reports'] = json.load(f)
    d['csr_score'] = float(open(f'{OUTPUT_DIR}/models/overall_csr.txt').read())
    d['pcr_score'] = float(open(f'{OUTPUT_DIR}/models/overall_pcr.txt').read())
    d['threshold'] = float(open(f'{OUTPUT_DIR}/models/best_threshold.txt').read())
    return d

@st.cache_resource
def get_client():
    return Groq(api_key=GROQ_API_KEY)

def groq_call(messages, max_tokens=700, temp=0.3):
    try:
        r = get_client().chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            max_tokens=max_tokens, temperature=temp
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq error: {e}"

try:
    d = load_data()
except Exception as e:
    st.error(f" Data load failed: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ SECOM RCA")
    st.markdown("<div class='nav-lbl'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio(
        "nav",
       ["Analyse a Lot", "Recommendations"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("<div class='nav-lbl'>Dataset Info</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    <b>SECOM</b> Semiconductor<br>
    1567 runs · 195 features<br>
    104 failures · 6.6% fail rate<br>
    8 lots flagged · 5 true fails
    </div>""", unsafe_allow_html=True)
    st.markdown("<div class='nav-lbl'>Model</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    RF + Isolation Forest<br>
    Threshold = 0.15<br>
    Hybrid weights 0.7 / 0.3<br>
    LLM: Llama 3.1 via Groq
    </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "Overview":
    st.markdown("<div class='page-title'>⚙️ Pipeline Overview</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    End-to-end summary of the SECOM yield RCA pipeline. Shows key performance metrics
    across all phases — from RF classification and hybrid anomaly detection to SHAP
    sensor attribution and DiCE counterfactual correction. Use this page to understand
    overall model performance and which sensors matter most.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    cols = st.columns(5)
    metrics = [
        ("RF AUC",         "0.7961", "✅ Target hit",     "#00d4ff"),
        ("Hybrid Prec.",   "0.6250", "✅ Target hit",     "#00dc82"),
        ("Bootstrap Stab.","56%",    "⚠️ Honest result",  "#ffd060"),
        ("PCR",            "1.0000", "✅ All lots viable", "#00dc82"),
        ("CSR",            "0.4800", "⚠️ Below 0.60",    "#ffd060"),
    ]
    for col, (lbl, val, status, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-value' style='color:{color}'>{val}</div>
                <div class='metric-status'>{status}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='sec-header'>Metrics vs Target (0.60)</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box' style='margin-bottom:8px'>Performance of each pipeline phase against the 0.60 target. Green = target met, Yellow = below target but acceptable.</div>", unsafe_allow_html=True)
        names  = ['RF AUC', 'Hybrid Prec', 'Bootstrap', 'PCR', 'CSR']
        values = [0.7961, 0.6250, 0.56, 1.00, 0.48]
        colors = ['#00dc82' if v >= 0.6 else '#ffd060' for v in values]
        fig = go.Figure(go.Bar(
            x=names, y=values, marker_color=colors,
            text=[f'{v:.2f}' for v in values], textposition='outside',
            textfont=dict(size=12, color='#cdd6e0')
        ))
        fig.add_hline(y=0.6, line_dash='dash', line_color='#ff4646', line_width=2,
                      annotation_text='Target 0.60',
                      annotation_font=dict(color='#ff8080', size=11))
        fig.update_layout(**PT, height=300, yaxis_range=[0,1.2], showlegend=False,
                          xaxis=dict(**GR, tickfont=dict(size=12)),
                          yaxis=dict(**GR))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='sec-header'>Risk Tier Distribution</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box' style='margin-bottom:8px'>Breakdown of 8 flagged lots by risk tier. High Risk lots have both high anomaly score and high SHAP deviation.</div>", unsafe_allow_html=True)
        rc = d['flagged']['risk_tier'].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=rc.index, values=rc.values, hole=0.55,
            marker=dict(colors=['#ff4646','#ffd060'],
                        line=dict(color='#0d1c2b', width=2)),
            textfont=dict(size=13, color='#cdd6e0')
        ))
        fig2.update_layout(**PT, height=300,
                           legend=dict(font=dict(size=12, color='#cdd6e0')))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("<div class='sec-header'>Top 15 Sensors — SHAP Importance</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box' style='margin-bottom:8px'>Sensors ranked by mean |SHAP| value across all flagged lots. Sensor 59 dominates — it is 100% bootstrap stable and appears as primary candidate in all 8 lots.</div>", unsafe_allow_html=True)
        t15 = d['top25'].head(15)
        fig3 = go.Figure(go.Bar(
            x=t15['mean_shap'], y=t15['sensor'].astype(str),
            orientation='h', marker_color='#00d4ff',
            text=[f'{v:.5f}' for v in t15['mean_shap']],
            textposition='outside', textfont=dict(size=10, color='#9ab0c4')
        ))
        fig3.update_layout(**PT, height=370, showlegend=False,
                           xaxis=dict(**GR, title='Mean |SHAP|'),
                           yaxis=dict(**GR, autorange='reversed'))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown("<div class='sec-header'>Bootstrap Stability — Top 20 Sensors</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box' style='margin-bottom:8px'>How consistently each sensor appears in the top 25 SHAP rankings across 30 bootstrap runs. Green = above 70% target. Yellow = above 50% threshold used. Red = unstable.</div>", unsafe_allow_html=True)
        stab = d['stability'].head(20)
        sc = ['#00dc82' if v>=70 else '#ffd060' if v>=50 else '#ff4646'
              for v in stab['stability_pct']]
        fig4 = go.Figure(go.Bar(
            x=stab['stability_pct'], y=stab['sensor'].astype(str),
            orientation='h', marker_color=sc,
            text=[f'{v:.0f}%' for v in stab['stability_pct']],
            textposition='outside', textfont=dict(size=10, color='#9ab0c4')
        ))
        fig4.add_vline(x=70, line_dash='dash', line_color='#ff4646', line_width=1.5,
                       annotation_text='70%', annotation_font=dict(color='#ff8080', size=10))
        fig4.add_vline(x=50, line_dash='dot', line_color='#ffd060', line_width=1.5,
                       annotation_text='50%', annotation_font=dict(color='#ffd060', size=10))
        fig4.update_layout(**PT, height=370, showlegend=False,
                           xaxis=dict(**GR, title='Stability (%)', range=[0,120]),
                           yaxis=dict(**GR, autorange='reversed'))
        st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# PAGE 2 — FLAGGED LOTS
# ============================================================
elif page == "Flagged Lots":
    st.markdown("<div class='page-title'>🚨 Flagged Lots</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    Lists all 8 production lots flagged by the hybrid anomaly detector. Each lot shows
    its true label, risk tier, anomaly score, primary candidate sensor, and deviation
    from normal. Red = confirmed failure, Green = false alarm. Use this page to
    prioritize which lots require immediate engineering investigation.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    merged = d['flagged'].merge(
        d['hyp'][['lot_index','primary_candidate','confidence','deviation_sigma','top3_sensors']],
        on='lot_index', how='left'
    )

    st.markdown("<div class='sec-header'>All 8 Flagged Lots</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box' style='margin-bottom:8px'>8 lots flagged by the hybrid anomaly detector. 5 are true failures, 3 are false alarms. All show Sensor 59 as primary candidate. Dev = how many standard deviations from normal process range.</div>", unsafe_allow_html=True)
    for _, row in merged.iterrows():
        tb = "<span class='badge-high'>HIGH RISK</span>" if row['risk_tier']=='High Risk' else "<span class='badge-sus'>SUSPICIOUS</span>"
        lb = "<span class='badge-high'>FAIL</span>" if row['true_label']==1 else "<span class='badge-pass'>PASS</span>"
        st.markdown(f"""
        <div class='lot-card'>
            <span class='lot-id'>LOT {row['lot_index']}</span> &nbsp; {lb} &nbsp; {tb}
            &nbsp;&nbsp; Score: <b style='color:#ffffff'>{row['hybrid_score']:.4f}</b>
            &nbsp;&nbsp; Primary: <b style='color:#ffd060'>Sensor {row['primary_candidate']}</b>
            &nbsp;&nbsp; Dev: <b style='color:#ffffff'>{row['deviation_sigma']:.2f}σ</b>
            &nbsp;&nbsp; Conf: <b style='color:#00dc82'>{row['confidence']}</b>
            <br><span style='color:#7a9bb5;font-size:0.78rem'>Top 3: {row['top3_sensors']}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='sec-header'>Anomaly Score Distribution</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box' style='margin-bottom:8px'>Distribution of hybrid anomaly scores across all 1567 lots. The blue threshold line at 0.519 separates flagged from normal lots. Ideal separation = FAIL lots on the right, PASS lots on the left.</div>", unsafe_allow_html=True)
    an = d['anomaly']
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=an[an['true_label']==-1]['hybrid_score'],
                               name='PASS', marker_color='#00dc82', opacity=0.55, nbinsx=35))
    fig.add_trace(go.Histogram(x=an[an['true_label']==1]['hybrid_score'],
                               name='FAIL', marker_color='#ff4646', opacity=0.65, nbinsx=35))
    fig.add_vline(x=0.5188, line_dash='dash', line_color='#00d4ff', line_width=2,
                  annotation_text='Threshold 0.519',
                  annotation_font=dict(color='#00d4ff', size=11))
    fig.update_layout(**PT, height=320, barmode='overlay',
                      xaxis=dict(**GR, title='Hybrid Anomaly Score'),
                      yaxis=dict(**GR, title='Count'),
                      legend=dict(font=dict(size=12, color='#cdd6e0')))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 3 — LOT INSPECTOR
# ============================================================
elif page == "🔍  Lot Inspector":
    st.markdown("<div class='page-title'>🔍 Lot Inspector</div>", unsafe_allow_html=True)

    flagged = d['flagged']
    sel = st.selectbox("Select Lot", flagged['lot_index'].tolist(),
        format_func=lambda x: (
            f"Lot {x}  —  "
            f"{'FAIL' if flagged[flagged['lot_index']==x]['true_label'].values[0]==1 else 'PASS'}  —  "
            f"{flagged[flagged['lot_index']==x]['risk_tier'].values[0]}"
        ))

    lot_row  = flagged[flagged['lot_index']==sel].iloc[0]
    hyp_row  = d['hyp'][d['hyp']['lot_index']==sel].iloc[0]
    csr_row  = d['csr'][d['csr']['lot_index']==sel]
    lot_shap = d['shap'][d['shap']['lot_index']==sel]

    c1,c2,c3,c4,c5 = st.columns(5)
    lc = "#ff4646" if lot_row['true_label']==1 else "#00dc82"
    lt = "FAIL" if lot_row['true_label']==1 else "PASS"
    tc = "#ff4646" if lot_row['risk_tier']=='High Risk' else "#ffd060"

    for col, lbl, val, color in zip([c1,c2,c3,c4,c5],
        ["True Label","Risk Tier","Hybrid Score","Primary Sensor","Deviation"],
        [lt, lot_row['risk_tier'], f"{lot_row['hybrid_score']:.4f}",
         f"#{hyp_row['primary_candidate']}", f"{hyp_row['deviation_sigma']:.2f}σ"],
        [lc, tc, "#00d4ff", "#ffd060", "#cdd6e0"]):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-value' style='color:{color};font-size:1.25rem'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='sec-header'>SHAP Sensor Contributions (Top 10)</div>", unsafe_allow_html=True)
        top10 = lot_shap.head(10)
        bc = ['#ffd060' if str(s)==str(hyp_row['primary_candidate']) else '#00d4ff'
              for s in top10['sensor']]
        fig = go.Figure(go.Bar(
            x=top10['shap_value'], y=top10['sensor'].astype(str),
            orientation='h', marker_color=bc,
            text=[f'{v:.5f}' for v in top10['shap_value']],
            textposition='outside', textfont=dict(size=10, color='#9ab0c4')
        ))
        fig.update_layout(**PT, height=320,
                          xaxis=dict(**GR, title='|SHAP|'),
                          yaxis=dict(**GR, autorange='reversed', tickfont=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='sec-header'>Sensor Deviation from Normal (σ)</div>", unsafe_allow_html=True)
        stable   = d['top25']['sensor'].astype(str).tolist()
        X_norm   = d['X_train'][d['y_train']==-1]
        lot_data = d['X_test'].iloc[sel]
        devs = []
        for s in stable:
            if s in lot_data.index:
                dev = (lot_data[s] - X_norm[s].mean()) / max(X_norm[s].std(), 0.001)
                devs.append({'sensor': s, 'dev': float(dev)})
        dev_df = pd.DataFrame(devs).sort_values('dev', key=abs, ascending=False).head(12)
        dc = ['#ff4646' if abs(v)>2 else '#ffd060' if abs(v)>1 else '#00dc82'
              for v in dev_df['dev']]
        fig2 = go.Figure(go.Bar(
            x=dev_df['sensor'].astype(str), y=dev_df['dev'],
            marker_color=dc,
            text=[f'{v:.2f}' for v in dev_df['dev']],
            textposition='outside', textfont=dict(size=10, color='#9ab0c4')
        ))
        fig2.add_hline(y=2,  line_dash='dash', line_color='#ff4646', line_width=1.5)
        fig2.add_hline(y=-2, line_dash='dash', line_color='#ff4646', line_width=1.5)
        fig2.add_hline(y=0,  line_color='#3a6070', line_width=1)
        fig2.update_layout(**PT, height=320,
                           xaxis=dict(**GR, tickfont=dict(size=10), tickangle=-35),
                           yaxis=dict(**GR, title='Deviation (σ)'))
        st.plotly_chart(fig2, use_container_width=True)

    lot_changes = d['changes'][d['changes']['lot_index']==sel]
    if len(lot_changes) > 0:
        st.markdown("---")
        st.markdown("<div class='sec-header'>DiCE — Suggested Process Adjustments</div>", unsafe_allow_html=True)
        c1, c2 = st.columns([3,1])
        with c1:
            chg = lot_changes.groupby('sensor')['change'].mean().sort_values()
            dc2 = ['#00dc82' if v>0 else '#ff4646' for v in chg.values]
            fig3 = go.Figure(go.Bar(
                x=chg.values, y=chg.index.astype(str),
                orientation='h', marker_color=dc2,
                text=[f'{v:+.3f}' for v in chg.values],
                textposition='outside', textfont=dict(size=10, color='#9ab0c4')
            ))
            fig3.add_vline(x=0, line_color='#3a6070', line_width=1)
            fig3.update_layout(**PT, height=290,
                               xaxis=dict(**GR, title='Suggested Change'),
                               yaxis=dict(**GR, tickfont=dict(size=11)))
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            if len(csr_row) > 0:
                for lbl, val, color in [("CSR", f"{csr_row['csr'].values[0]:.2f}", "#ffd060"),
                                         ("PCR", f"{csr_row['pcr'].values[0]:.2f}", "#00dc82")]:
                    st.markdown(f"""<div class='metric-card' style='margin-bottom:8px'>
                        <div class='metric-label'>{lbl}</div>
                        <div class='metric-value' style='color:{color}'>{val}</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='sec-header'>Pre-Generated Investigation Report</div>", unsafe_allow_html=True)
    rkey = str(sel)
    if rkey in d['reports']:
        st.markdown(f"<div class='llm-box'>{d['reports'][rkey]}</div>", unsafe_allow_html=True)
    else:
        st.info("No pre-generated report for this lot.")

    # ── Export ALL lot reports to PDF ─────────────────────
    st.markdown("---")
    st.markdown("<div class='sec-header'>Export All Reports</div>", unsafe_allow_html=True)
    if st.button("📄 Export All Lot Reports to PDF"):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
            from reportlab.lib.enums import TA_LEFT
            import io

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    leftMargin=2*cm, rightMargin=2*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('title', fontSize=16, fontName='Helvetica-Bold',
                                         textColor=colors.HexColor('#1a1a2e'), spaceAfter=6)
            lot_style   = ParagraphStyle('lot',   fontSize=13, fontName='Helvetica-Bold',
                                         textColor=colors.HexColor('#0066cc'), spaceAfter=4,
                                         spaceBefore=12)
            body_style  = ParagraphStyle('body',  fontSize=9,  fontName='Helvetica',
                                         leading=14, textColor=colors.HexColor('#2c2c2c'),
                                         spaceAfter=4)
            meta_style  = ParagraphStyle('meta',  fontSize=8,  fontName='Helvetica-Oblique',
                                         textColor=colors.HexColor('#666666'), spaceAfter=8)

            story = []
            story.append(Paragraph("SECOM Yield RCA — All Lot Investigation Reports", title_style))
            story.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", meta_style))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
            story.append(Spacer(1, 0.3*cm))

            flagged_df = d['flagged'].drop_duplicates(subset=['lot_index'])
            for _, row in flagged_df.iterrows():
                lot_idx = row['lot_index']
                label   = 'FAIL' if row['true_label'] == 1 else 'PASS'
                tier    = row['risk_tier']
                score   = row['hybrid_score']

                hyp = d['hyp'][d['hyp']['lot_index'] == lot_idx]
                primary  = hyp['primary_candidate'].values[0] if len(hyp) > 0 else 'N/A'
                dev      = hyp['deviation_sigma'].values[0]   if len(hyp) > 0 else 0
                conf     = hyp['confidence'].values[0]        if len(hyp) > 0 else 'N/A'

                story.append(Paragraph(f"LOT {lot_idx}", lot_style))
                story.append(Paragraph(
                    f"Label: {label}  |  Risk: {tier}  |  Score: {score:.4f}  |  "
                    f"Primary Sensor: {primary}  |  Deviation: {dev:.2f}σ  |  Confidence: {conf}",
                    meta_style))

                rkey = str(lot_idx)
                report_text = d['reports'].get(rkey, 'No report available for this lot.')
                for line in report_text.split('\n'):
                    line = line.strip()
                    if line:
                        story.append(Paragraph(line.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'), body_style))

                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#dddddd')))
                story.append(Spacer(1, 0.2*cm))

            doc.build(story)
            buf.seek(0)
            st.download_button(
                label="⬇️ Download All Reports PDF",
                data=buf,
                file_name="secom_all_lot_reports.pdf",
                mime="application/pdf"
            )
            st.success("✅ PDF ready — click above to download")
        except ImportError:
            st.error("Install reportlab first: pip install reportlab")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

# ============================================================
# PAGE 3 — Analyse a lot
# ============================================================



elif page == "Analyse a Lot":
    st.markdown("<div class='page-title'>🔍 Analyse a Lot</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    Select any flagged lot to run a full interactive analysis.
    Generates live SHAP attribution, sensor deviation, DiCE correction
    pathway and a fresh LLM engineering report via Groq.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    flagged = d['flagged']
    sel = st.selectbox("Select Lot to Analyse", flagged['lot_index'].drop_duplicates().tolist(),
        format_func=lambda x: (
            f"Lot {x}   "
        ))

    if st.button("▶  Run Full Analysis", key="run_analysis"):
        lot_row  = flagged[flagged['lot_index']==sel].iloc[0]
        hyp_row  = d['hyp'][d['hyp']['lot_index']==sel].iloc[0]
        csr_row  = d['csr'][d['csr']['lot_index']==sel]
        lot_shap = d['shap'][d['shap']['lot_index']==sel]

        # ── Header cards ──────────────────────────────
        lc = "#ff4646" if lot_row['true_label']==1 else "#00dc82"
        lt = "FAIL"    if lot_row['true_label']==1 else "PASS"
        tc = "#ff4646" if lot_row['risk_tier']=='High Risk' else "#ffd060"

        c1,c2,c3,c4,c5 = st.columns(5)
        for col, lbl, val, color in zip(
            [c1,c2,c3,c4,c5],
            ["True Label","Risk Tier","Hybrid Score","Primary Sensor","Deviation"],
            [lt, lot_row['risk_tier'],
             f"{lot_row['hybrid_score']:.4f}",
             f"#{hyp_row['primary_candidate']}",
             f"{hyp_row['deviation_sigma']:.2f}σ"],
            [lc, tc, "#00d4ff", "#ffd060", "#cdd6e0"]
        ):
            with col:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>{lbl}</div>
                    <div class='metric-value' style='color:{color};font-size:1.2rem'>{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Charts ────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='sec-header'>SHAP Sensor Contributions</div>",
                        unsafe_allow_html=True)
            top10 = lot_shap.head(10)
            bc = ['#ffd060' if str(s)==str(hyp_row['primary_candidate'])
                  else '#00d4ff' for s in top10['sensor']]
            fig = go.Figure(go.Bar(
                x=top10['shap_value'], y=top10['sensor'].astype(str),
                orientation='h', marker_color=bc,
                text=[f'{v:.5f}' for v in top10['shap_value']],
                textposition='outside',
                textfont=dict(size=10, color='#9ab0c4')
            ))
            fig.update_layout(**PT, height=300,
                              xaxis=dict(**GR, title='|SHAP|'),
                              yaxis=dict(**GR, autorange='reversed',
                                         tickfont=dict(size=11)))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<div class='sec-header'>Sensor Deviation from Normal (σ)</div>",
                        unsafe_allow_html=True)
            stable   = d['top25']['sensor'].astype(str).tolist()
            X_norm   = d['X_train'][d['y_train']==-1]
            lot_data = d['X_test'].iloc[sel]
            devs = []
            for s in stable:
                if s in lot_data.index:
                    dev = (lot_data[s] - X_norm[s].mean()) / max(X_norm[s].std(), 0.001)
                    devs.append({'sensor': s, 'dev': float(dev)})
            dev_df = pd.DataFrame(devs).sort_values('dev', key=abs, ascending=False).head(12)
            dc = ['#ff4646' if abs(v)>2 else '#ffd060' if abs(v)>1 else '#00dc82'
                  for v in dev_df['dev']]
            fig2 = go.Figure(go.Bar(
                x=dev_df['sensor'].astype(str), y=dev_df['dev'],
                marker_color=dc,
                text=[f'{v:.2f}' for v in dev_df['dev']],
                textposition='outside',
                textfont=dict(size=10, color='#9ab0c4')
            ))
            fig2.add_hline(y=2,  line_dash='dash', line_color='#ff4646', line_width=1.5)
            fig2.add_hline(y=-2, line_dash='dash', line_color='#ff4646', line_width=1.5)
            fig2.add_hline(y=0,  line_color='#3a6070', line_width=1)
            fig2.update_layout(**PT, height=300,
                               xaxis=dict(**GR, tickangle=-35,
                                          tickfont=dict(size=10)),
                               yaxis=dict(**GR, title='Deviation (σ)'))
            st.plotly_chart(fig2, use_container_width=True)

        # ── DiCE + CSR/PCR ────────────────────────────
        lot_changes = d['changes'][d['changes']['lot_index']==sel]
        if len(lot_changes) > 0:
            st.markdown("---")
            col3, col4 = st.columns([3,1])
            with col3:
                st.markdown("<div class='sec-header'>DiCE — Suggested Process Corrections</div>",
                            unsafe_allow_html=True)
                chg  = lot_changes.groupby('sensor')['change'].mean().sort_values()
                dc2  = ['#00dc82' if v>0 else '#ff4646' for v in chg.values]
                fig3 = go.Figure(go.Bar(
                    x=chg.values, y=chg.index.astype(str),
                    orientation='h', marker_color=dc2,
                    text=[f'{v:+.3f}' for v in chg.values],
                    textposition='outside',
                    textfont=dict(size=10, color='#9ab0c4')
                ))
                fig3.add_vline(x=0, line_color='#3a6070', line_width=1)
                fig3.update_layout(**PT, height=280,
                                   xaxis=dict(**GR, title='Suggested Change'),
                                   yaxis=dict(**GR, tickfont=dict(size=11)))
                st.plotly_chart(fig3, use_container_width=True)

            with col4:
                st.markdown("<div class='sec-header'>Correction</div>",
                            unsafe_allow_html=True)
                if len(csr_row) > 0:
                    for lbl, val, color in [
                        ("CSR", f"{csr_row['csr'].values[0]:.2f}", "#ffd060"),
                        ("PCR", f"{csr_row['pcr'].values[0]:.2f}", "#00dc82")
                    ]:
                        st.markdown(f"""<div class='metric-card' style='margin-bottom:10px'>
                            <div class='metric-label'>{lbl}</div>
                            <div class='metric-value' style='color:{color}'>{val}</div>
                        </div>""", unsafe_allow_html=True)

        # ── Live LLM Report ───────────────────────────
        st.markdown("---")
        st.markdown("<div class='sec-header'>Investigation Report</div>",
                    unsafe_allow_html=True)

        with st.spinner("Generating investigation report via Groq..."):
            import re
            SYS = """You are an expert semiconductor process engineer.
Generate a concise investigation report for a flagged production lot.
Rules: Say candidate sensor not root cause. Plain text only, no markdown, no asterisks."""

            prompt = f"""
Lot {sel} Investigation Report:
- True Label: {'FAIL' if lot_row['true_label']==1 else 'PASS'}
- Risk Tier: {lot_row['risk_tier']}
- Hybrid Score: {lot_row['hybrid_score']:.4f}
- Primary Candidate: Sensor {hyp_row['primary_candidate']}
- Deviation: {hyp_row['deviation_sigma']:.2f} sigma from normal
- Confidence: {hyp_row['confidence']}
- Top 3 Sensors: {hyp_row['top3_sensors']}
- CSR: {f"{csr_row['csr'].values[0]:.2f}" if len(csr_row)>0 else 'N/A'}
- PCR: {f"{csr_row['pcr'].values[0]:.2f}" if len(csr_row)>0 else 'N/A'}

Generate:
1. Executive Summary (2 sentences)
2. Primary Investigation Candidate (what to inspect)
3. Suggested Process Adjustments (3 specific actions)
4. Confidence Statement (1 sentence)
Plain text only. No bold, no asterisks."""

            report = groq_call(
                [{"role":"system","content":SYS},
                 {"role":"user","content":prompt}],
                600
            )
            report = re.sub(r'\*\*(.*?)\*\*', r'\1', report)
            report = re.sub(r'\*(.*?)\*',     r'\1', report)
            st.markdown(
                f"<div class='llm-box'>{report}</div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown("""<div class='info-box' style='text-align:center;padding:40px'>
        Select a lot above and click <b style='color:#00d4ff'>▶ Run Full Analysis</b>
        to generate live SHAP, deviation, DiCE correction and LLM report.
        </div>""", unsafe_allow_html=True)




# ============================================================
# PAGE 3 — Recomnedations
# ============================================================





elif page == "Recommendations":
    st.markdown("<div class='page-title'>🤖 Recommendations</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    Select a sensor to generate a detailed AI-powered investigation recommendation.
    Based on SHAP attribution, bootstrap stability, deviation analysis, and DiCE corrections.
    Export all sensor recommendations to PDF for engineering records.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    SYS_REC = """You are a senior semiconductor process engineer analyzing SECOM yield RCA results.
STRICT RULES:
- Say "candidate sensor" never "root cause"
- Say "statistical hypothesis" never "proven cause"  
- Be specific and actionable — give real process investigation steps
- Reference actual values provided (stability %, SHAP value, deviation, DiCE change)
- Plain text only, no markdown, no asterisks, no bold"""

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='sec-header'>Single Sensor Investigation</div>", unsafe_allow_html=True)
        sel_s = st.selectbox("Select Sensor", d['top25']['sensor'].astype(str).tolist(), key='srec')

        if st.button("Generate Detailed Recommendation"):
            with st.spinner("Analysing sensor..."):
                import math, re
                stab_r = d['stability'][d['stability']['sensor'].astype(str) == sel_s]
                shap_r = d['top25'][d['top25']['sensor'].astype(str) == sel_s]
                sv     = stab_r['stability_pct'].values[0] if len(stab_r) > 0 else 'N/A'
                sh     = float(shap_r['mean_shap'].values[0]) if len(shap_r) > 0 else 0
                lc     = d['shap'][d['shap']['sensor'].astype(str) == sel_s]['lot_index'].nunique()
                dc_raw = d['changes'][d['changes']['sensor'].astype(str) == sel_s]['change'].mean()
                dc_str = f"{round(float(dc_raw), 4)}" if (isinstance(dc_raw, float) and not math.isnan(dc_raw)) else 'N/A'

                # Get deviation across lots for this sensor
                stable   = d['top25']['sensor'].astype(str).tolist()
                X_norm   = d['X_train'][d['y_train'] == -1]
                all_devs = []
                for lid in d['flagged']['lot_index'].drop_duplicates():
                    try:
                        lot_data = d['X_test'].iloc[lid]
                        if sel_s in lot_data.index and sel_s in X_norm.columns:
                            dev = (lot_data[sel_s] - X_norm[sel_s].mean()) / max(X_norm[sel_s].std(), 0.001)
                            all_devs.append(dev)
                    except:
                        pass
                avg_dev = f"{np.mean(all_devs):.2f}" if all_devs else "N/A"

                p2 = f"""Sensor {sel_s} — SECOM Semiconductor Yield RCA Analysis:

Statistical Profile:
- Bootstrap stability: {sv}% (appears in top 25 SHAP sensors across bootstrap runs)
- Mean SHAP attribution: {sh:.5f} (contribution to failure prediction)
- Present in {lc} flagged lots out of 8 total flagged
- DiCE suggested correction direction: {dc_str} units (negative = decrease needed)
- Average deviation from normal: {avg_dev} sigma across flagged lots

Generate a detailed plain text investigation report with exactly these 4 sections:

SECTION 1 - SENSOR PROFILE (3 sentences):
What this sensor's statistical pattern tells us. Reference the stability percentage and SHAP value specifically.

SECTION 2 - INVESTIGATION STEPS (4 numbered steps):
Specific physical process checks an engineer should perform. Be concrete — mention calibration checks, process log review, equipment inspection, comparison to normal lots.

SECTION 3 - PROCESS LOG ANALYSIS (3 bullet points starting with -):
Specific things to look for in the fab's process monitoring logs. Include time windows, comparison methods, threshold values.

SECTION 4 - CONFIDENCE AND LIMITATIONS (2 sentences):
State confidence level based on stability percentage. State that this is a statistical hypothesis requiring physical validation.

Plain text only. No markdown. No asterisks."""

                r2 = groq_call(
                    [{"role": "system", "content": SYS_REC},
                     {"role": "user",   "content": p2}],
                    700
                )
                r2 = re.sub(r'\*\*(.*?)\*\*', r'\1', r2)
                r2 = re.sub(r'\*(.*?)\*',     r'\1', r2)
                st.session_state['srec_out'] = r2
                st.session_state['srec_sensor'] = sel_s

        if 'srec_out' in st.session_state:
            st.markdown(
                f"<div class='llm-box'>{st.session_state['srec_out']}</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Export ALL sensor recommendations to PDF ──────────
    st.markdown("<div class='sec-header'>Export All Sensor Recommendations</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box' style='margin-bottom:8px'>Generates recommendations for all top 25 stable sensors and exports to one PDF. Takes 1-2 minutes — Groq API is called for each sensor.</div>", unsafe_allow_html=True)

    if st.button("📄 Generate + Export All Sensor Recommendations to PDF"):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
            import io, math, re

            all_sensors = d['top25']['sensor'].astype(str).tolist()
            progress = st.progress(0)
            status   = st.empty()

            sensor_reports = {}
            for i, s in enumerate(all_sensors):
                status.text(f"Generating recommendation for Sensor {s} ({i+1}/{len(all_sensors)})...")
                progress.progress((i+1) / len(all_sensors))

                stab_r = d['stability'][d['stability']['sensor'].astype(str) == s]
                shap_r = d['top25'][d['top25']['sensor'].astype(str) == s]
                sv     = stab_r['stability_pct'].values[0] if len(stab_r) > 0 else 'N/A'
                sh     = float(shap_r['mean_shap'].values[0]) if len(shap_r) > 0 else 0
                lc     = d['shap'][d['shap']['sensor'].astype(str) == s]['lot_index'].nunique()
                dc_raw = d['changes'][d['changes']['sensor'].astype(str) == s]['change'].mean()
                dc_str = f"{round(float(dc_raw), 4)}" if (isinstance(dc_raw, float) and not math.isnan(dc_raw)) else 'N/A'

                prompt = f"""Sensor {s} — SECOM RCA:
- Stability: {sv}%, SHAP: {sh:.5f}, Lots: {lc}/8, DiCE change: {dc_str}

Write a plain text 4-section report:
1. SENSOR PROFILE (2 sentences referencing stability and SHAP)
2. INVESTIGATION STEPS (3 numbered concrete steps)  
3. PROCESS LOG CHECKS (2 bullet points starting with -)
4. CONFIDENCE STATEMENT (1 sentence)
Plain text only, no markdown, no asterisks."""

                resp = groq_call(
                    [{"role": "system", "content": SYS_REC},
                     {"role": "user",   "content": prompt}],
                    400
                )
                resp = re.sub(r'\*\*(.*?)\*\*', r'\1', resp)
                resp = re.sub(r'\*(.*?)\*',     r'\1', resp)
                sensor_reports[s] = resp

            status.text("Building PDF...")

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    leftMargin=2*cm, rightMargin=2*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)

            title_style  = ParagraphStyle('title', fontSize=15, fontName='Helvetica-Bold',
                                          textColor=colors.HexColor('#1a1a2e'), spaceAfter=6)
            sensor_style = ParagraphStyle('sensor', fontSize=12, fontName='Helvetica-Bold',
                                          textColor=colors.HexColor('#0066cc'),
                                          spaceAfter=3, spaceBefore=10)
            meta_style   = ParagraphStyle('meta',  fontSize=8,  fontName='Helvetica-Oblique',
                                          textColor=colors.HexColor('#666666'), spaceAfter=6)
            body_style   = ParagraphStyle('body',  fontSize=9,  fontName='Helvetica',
                                          leading=13, textColor=colors.HexColor('#2c2c2c'),
                                          spaceAfter=3)

            story = []
            story.append(Paragraph("SECOM Yield RCA — All Sensor Investigation Recommendations", title_style))
            story.append(Paragraph(f"Top {len(all_sensors)} Bootstrap-Stable Sensors  |  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", meta_style))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
            story.append(Spacer(1, 0.3*cm))

            for s, report_text in sensor_reports.items():
                stab_r = d['stability'][d['stability']['sensor'].astype(str) == s]
                shap_r = d['top25'][d['top25']['sensor'].astype(str) == s]
                sv_val = stab_r['stability_pct'].values[0] if len(stab_r) > 0 else 'N/A'
                sh_val = float(shap_r['mean_shap'].values[0]) if len(shap_r) > 0 else 0

                story.append(Paragraph(f"SENSOR {s}", sensor_style))
                story.append(Paragraph(
                    f"Bootstrap Stability: {sv_val}%  |  Mean SHAP: {sh_val:.5f}",
                    meta_style))
                for line in report_text.split('\n'):
                    line = line.strip()
                    if line:
                        story.append(Paragraph(
                            line.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'),
                            body_style))
                story.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor('#dddddd')))
                story.append(Spacer(1, 0.2*cm))

            doc.build(story)
            buf.seek(0)
            progress.empty()
            status.empty()

            st.download_button(
                label="⬇️ Download All Sensor Recommendations PDF",
                data=buf,
                file_name="secom_all_sensor_recommendations.pdf",
                mime="application/pdf"
            )
            st.success(f"✅ PDF ready with {len(all_sensors)} sensor reports — click above to download")

        except ImportError:
            st.error("Install reportlab first:  pip install reportlab")
        except Exception as e:
            st.error(f"Export failed: {e}")

