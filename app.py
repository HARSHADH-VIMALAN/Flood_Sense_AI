import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloodSense AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌊"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #07111f; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0c1e33 0%,#07111f 100%);
    border-right: 1px solid rgba(56,189,248,0.10);
}
[data-testid="stSidebar"] * { color: #90b8d0 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #dff0fc !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(12,28,48,0.85); border-radius:14px;
    padding:5px; gap:3px; border:1px solid rgba(56,189,248,0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius:11px !important; color:#5a90ad !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:600 !important; font-size:0.83rem !important;
    padding:9px 16px !important; transition:all .25s !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#0f4c75,#1a6fa8) !important;
    color:#e0f2fe !important;
    box-shadow:0 2px 14px rgba(56,189,248,0.22) !important;
}

.stButton > button {
    background:linear-gradient(135deg,#0369a1,#38bdf8) !important;
    color:#fff !important; font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:700 !important; font-size:1rem !important;
    border:none !important; border-radius:12px !important;
    padding:12px 32px !important;
    box-shadow:0 4px 20px rgba(56,189,248,0.28) !important;
    transition:all .25s !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 32px rgba(56,189,248,0.42) !important;
}

[data-testid="stFileUploader"] {
    border:2px dashed rgba(56,189,248,0.28) !important;
    border-radius:16px !important;
    background:rgba(14,116,144,0.04) !important;
}

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#07111f; }
::-webkit-scrollbar-thumb { background:#1e4d70; border-radius:10px; }

/* cards */
.hero-wrap {
    background:linear-gradient(135deg,rgba(3,105,161,0.20) 0%,rgba(7,17,31,0.0) 60%);
    border:1px solid rgba(56,189,248,0.15); border-radius:24px;
    padding:44px 52px; margin-bottom:26px; position:relative; overflow:hidden;
}
.hero-wrap::after {
    content:''; position:absolute; top:-80px; right:-80px;
    width:300px; height:300px;
    background:radial-gradient(circle,rgba(56,189,248,0.07) 0%,transparent 70%);
    border-radius:50%;
}
.hero-title {
    font-family:'Syne',sans-serif; font-size:2.9rem; font-weight:900;
    background:linear-gradient(90deg,#7dd3fc,#38bdf8,#0ea5e9);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    line-height:1.05; margin:0;
}
.hero-tagline { font-size:1.0rem; color:#5f9ab8; margin-top:10px; max-width:580px; }

.stat-card {
    background:linear-gradient(135deg,rgba(15,46,80,0.6),rgba(7,17,31,0.7));
    border:1px solid rgba(56,189,248,0.13); border-radius:16px;
    padding:20px 14px; text-align:center; transition:all .3s;
}
.stat-card:hover {
    border-color:rgba(56,189,248,0.35);
    box-shadow:0 0 22px rgba(56,189,248,0.09);
    transform:translateY(-2px);
}
.stat-num {
    font-family:'Syne',sans-serif; font-size:2.1rem;
    font-weight:900; color:#38bdf8; line-height:1;
}
.stat-lbl {
    font-size:0.70rem; color:#4a7a96;
    text-transform:uppercase; letter-spacing:1.4px; margin-top:5px;
}
.stat-tip { font-size:0.70rem; color:#2a5068; margin-top:5px; line-height:1.3; }

.sec-head {
    font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:800;
    color:#7dd3fc; margin:24px 0 12px; padding-left:11px;
    border-left:3px solid #0ea5e9;
}
.tip-box {
    background:rgba(14,116,144,0.09); border:1px solid rgba(56,189,248,0.17);
    border-radius:11px; padding:13px 17px; font-size:0.86rem;
    color:#90bdd6; margin:8px 0 16px; line-height:1.55;
}
.tip-box b { color:#38bdf8; }
.warn-box {
    background:rgba(234,88,12,0.07); border:1px solid rgba(234,88,12,0.28);
    border-radius:11px; padding:13px 17px; font-size:0.86rem;
    color:#fbbf7a; margin:8px 0 16px;
}
.glass-card {
    background:linear-gradient(135deg,rgba(15,46,80,0.5),rgba(7,17,31,0.75));
    border:1px solid rgba(56,189,248,0.11); border-radius:18px;
    padding:26px 28px; margin-bottom:16px;
}
.ins-card {
    background:linear-gradient(135deg,rgba(15,46,80,0.45),rgba(7,17,31,0.65));
    border:1px solid rgba(56,189,248,0.11); border-radius:13px;
    padding:17px 19px; margin-bottom:11px;
}
.ins-title { font-weight:700; color:#7dd3fc; font-size:0.88rem; }
.ins-text  { color:#6a9ab5; font-size:0.81rem; margin-top:4px; line-height:1.5; }

.risk-high {
    background:linear-gradient(135deg,rgba(239,68,68,0.18),rgba(239,68,68,0.04));
    border:1.5px solid rgba(239,68,68,0.45); border-radius:14px;
    padding:22px 30px; font-family:'Syne',sans-serif; font-size:1.5rem;
    font-weight:900; color:#fca5a5; text-align:center; width:100%;
}
.risk-low {
    background:linear-gradient(135deg,rgba(16,185,129,0.18),rgba(16,185,129,0.03));
    border:1.5px solid rgba(16,185,129,0.42); border-radius:14px;
    padding:22px 30px; font-family:'Syne',sans-serif; font-size:1.5rem;
    font-weight:900; color:#6ee7b7; text-align:center; width:100%;
}
.footer {
    text-align:center; margin-top:50px; padding:26px;
    border-top:1px solid rgba(56,189,248,0.07);
    color:#2e5870; font-size:0.80rem; line-height:1.8;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly base theme (NO margin key here) ────────────────────────────────────
PL = dict(
    paper_bgcolor='rgba(7,17,31,0)',
    plot_bgcolor='rgba(7,17,31,0)',
    font=dict(family='Plus Jakarta Sans', color='#5a8aa6', size=12),
    title_font=dict(family='Syne', color='#7dd3fc', size=14),
    xaxis=dict(gridcolor='rgba(56,189,248,0.05)',
               zerolinecolor='rgba(56,189,248,0.10)', color='#3a6070'),
    yaxis=dict(gridcolor='rgba(56,189,248,0.05)',
               zerolinecolor='rgba(56,189,248,0.10)', color='#3a6070'),
    legend=dict(bgcolor='rgba(7,17,31,0.6)',
                bordercolor='rgba(56,189,248,0.14)', borderwidth=1,
                font=dict(color='#6a9ab5'))
)

CF = "#ef4444"   # flood colour
CS = "#10b981"   # safe colour
CW = "#f97316"   # warn colour
CP = "#38bdf8"   # primary cyan
SRISK = [[0, CS],[0.5, CW],[1, CF]]

def risk_label(p):
    if p < 0.30: return "Very Low",  "🟢", CS
    if p < 0.50: return "Low",       "🟡", "#84cc16"
    if p < 0.70: return "Moderate",  "🟠", CW
    if p < 0.85: return "High",      "🔴", "#f87171"
    return             "Very High",  "🚨", CF

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 FloodSense AI")
    st.markdown("*Your flood risk companion*")
    st.markdown("---")
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Drop your flood dataset (.xlsx)",
        type=["xlsx","xls"],
        help="Excel file with columns: Rainfall (mm), Temperature (°C), Humidity (%), etc."
    )
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    with st.expander("Training Options"):
        model_choice = st.selectbox("AI Model", ["Random Forest 🌲","Gradient Boosting 🚀"])
        test_split   = st.slider("Test split (%)", 10, 35, 20)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem;color:#2e5870;line-height:1.7'>
    📌 <b style='color:#3a6a85'>How it works</b><br>
    Upload → AI trains → Explore → Predict<br><br>
    No coding needed. Fully guided.
    </div>""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <p class="hero-title">🌊 FloodSense AI</p>
  <p class="hero-tagline">
    Predict urban flood risk using environmental data &amp; machine learning —
    no technical knowledge required.
  </p>
  <div style="margin-top:18px;display:flex;gap:10px;flex-wrap:wrap">
    <span style="background:rgba(56,189,248,0.09);border:1px solid rgba(56,189,248,0.22);border-radius:20px;padding:4px 13px;font-size:0.78rem;color:#7dd3fc">🤖 AI-Powered</span>
    <span style="background:rgba(16,185,129,0.09);border:1px solid rgba(16,185,129,0.22);border-radius:20px;padding:4px 13px;font-size:0.78rem;color:#6ee7b7">📊 Visual Insights</span>
    <span style="background:rgba(249,115,22,0.09);border:1px solid rgba(249,115,22,0.22);border-radius:20px;padding:4px 13px;font-size:0.78rem;color:#fbbf7a">⚡ Real-Time Predictions</span>
    <span style="background:rgba(167,139,250,0.09);border:1px solid rgba(167,139,250,0.22);border-radius:20px;padding:4px 13px;font-size:0.78rem;color:#c4b5fd">🗺️ Geo Mapping</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── No file: welcome guide ────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown('<div class="sec-head">👋 Welcome! Here\'s how to get started</div>', unsafe_allow_html=True)
    for col_w, (num, icon, title, desc) in zip(
        st.columns(4),
        [("1","📂","Upload Data","Upload your Excel file using the sidebar. Your file should have flood & weather columns."),
         ("2","🤖","AI Trains","The AI automatically learns patterns from your data in seconds — no setup needed."),
         ("3","📊","Explore","Discover insights through charts: maps, trends, risk factors and more."),
         ("4","🔍","Predict","Enter real conditions and get an instant, plain-English flood risk prediction.")]
    ):
        with col_w:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;min-height:155px">
              <div style="font-size:2rem;margin-bottom:6px">{icon}</div>
              <div style="font-family:'Syne',sans-serif;font-weight:800;color:#7dd3fc;font-size:0.97rem;margin-bottom:7px">{title}</div>
              <div style="font-size:0.80rem;color:#3d6b87;line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">📋 What columns does my file need?</div>', unsafe_allow_html=True)
    st.markdown('<div class="tip-box"><b>Your Excel file must have these exact column names:</b></div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Column Name":   ["Latitude","Longitude","Rainfall (mm)","Temperature (°C)","Humidity (%)","River Discharge (m³/s)","Water Level (m)","Elevation (m)","Land Cover","Soil Type","Population Density","Infrastructure","Historical Floods","Flood Occurred"],
        "What it means": ["North-South GPS position","East-West GPS position","Rain amount in millimetres","Air temperature °C","Air moisture 0–100%","Volume of river water flow","Water surface height (metres)","Ground height above sea level","Land use type (urban/forest…)","Soil classification (clay/sand…)","People per sq km","Infrastructure quality level","Count of past flood events","⚠️ TARGET — 1=Flood, 0=No Flood"],
        "Example":       ["12.97","80.23","145","28","78","350","4.2","12","Urban","Clay","4500","Moderate","3","1"]
    }), use_container_width=True, hide_index=True)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════════
try:
    df_raw = pd.read_excel(uploaded_file, engine="openpyxl")
    df_raw.columns = df_raw.columns.str.strip()
except Exception as e:
    st.error(f"❌ Could not read file: {e}. Re-save as .xlsx and try again.")
    st.stop()

if "Flood Occurred" not in df_raw.columns:
    st.error("❌ Column **'Flood Occurred'** not found. Check your column names.")
    st.stop()

df = df_raw.dropna().copy()

# Encode categoricals
cat_cols = df.select_dtypes(include="object").columns.tolist()
if "Flood Occurred" in cat_cols:
    cat_cols.remove("Flood Occurred")
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
if df["Flood Occurred"].dtype == "object":
    df["Flood Occurred"] = LabelEncoder().fit_transform(df["Flood Occurred"])

X = df.drop("Flood Occurred", axis=1)
y = df["Flood Occurred"]

# ── Lock column order here — used for prediction input building ───────────────
FEATURE_COLS = list(X.columns)   # e.g. ['Latitude','Longitude','Rainfall (mm)',...]

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_split/100, random_state=42
)

# ── Scaler: fit & transform ONLY on .values (pure numpy, zero feature names) ──

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train.values)   # <-- .values strips column names
X_test_sc  = scaler.transform(X_test.values)        # <-- .values strips column names

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_model(choice, Xtr, ytr):
    m = (GradientBoostingClassifier(n_estimators=150, random_state=42)
         if "Gradient" in choice
         else RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
    m.fit(Xtr, ytr)
    return m

with st.spinner("🤖 Training AI model… (a few seconds)"):
    model = build_model(model_choice, X_train_sc, y_train)

y_pred  = model.predict(X_test_sc)
y_prob  = model.predict_proba(X_test_sc)[:,1]
acc     = accuracy_score(y_test, y_pred)
# cross_val also uses .values
cv_sc   = cross_val_score(model, scaler.transform(X.values), y, cv=5)
flood_rt = y.mean()

# ── Quick-stat bar ────────────────────────────────────────────────────────────
for col_w,(val,lbl,tip) in zip(
    st.columns(5),
    [(f"{acc*100:.1f}%",       "AI Accuracy",       "Correct predictions out of all"),
     (f"{len(df):,}",           "Records",           "Total observations in dataset"),
     (f"{flood_rt*100:.1f}%",  "Flood Rate",         "% of events that caused floods"),
     (f"{cv_sc.mean()*100:.1f}%","CV Reliability",   "Consistency across 5 test runs"),
     (f"{len(FEATURE_COLS)}",   "Risk Factors",      "Features the AI analyses")]
):
    with col_w:
        st.markdown(f"""
        <div class="stat-card" title="{tip}">
          <div class="stat-num">{val}</div>
          <div class="stat-lbl">{lbl}</div>
          <div class="stat-tip">{tip}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════════
T = st.tabs(["🏠 Dashboard","🔍 Predict Risk","🗺️ Flood Map",
             "📊 Risk Factors","🤖 AI Performance","📋 Data Explorer"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with T[0]:
    st.markdown('<div class="sec-head">📍 Flood Overview</div>', unsafe_allow_html=True)
    left, right = st.columns([1.1,1])

    with left:
        n_fl = int(y.sum()); n_sf = len(y)-n_fl
        fig_d = go.Figure(go.Pie(
            labels=["No Flood 🟢","Flood 🔴"], values=[n_sf,n_fl],
            hole=0.65,
            marker=dict(colors=[CS,CF], line=dict(color='rgba(7,17,31,1)',width=4)),
            textinfo='percent+label',
            textfont=dict(family='Plus Jakarta Sans',size=12,color='#c0dff0'),
            pull=[0,0.05]
        ))
        fig_d.add_annotation(
            text=f"<b>{len(y):,}</b><br><span style='font-size:11px'>Events</span>",
            x=0.5,y=0.5,showarrow=False,
            font=dict(size=20,family='Syne',color='#38bdf8')
        )
        fig_d.update_layout(title="Flood vs Safe Split",**PL,height=300,
                            margin=dict(l=20,r=20,t=45,b=20))
        st.plotly_chart(fig_d, use_container_width=True)

        st.markdown(f"""
        <div class="ins-card">
          <div class="ins-title">📌 Dataset Summary</div>
          <div class="ins-text">
            Out of <b style='color:{CP}'>{len(y):,} events</b>,
            <b style='color:{CF}'>{n_fl:,} ({flood_rt*100:.1f}%)</b> were floods and
            <b style='color:{CS}'>{n_sf:,} ({(1-flood_rt)*100:.1f}%)</b> were safe.
          </div>
        </div>""", unsafe_allow_html=True)

    with right:
        rain_col = next((c for c in df_raw.columns if "Rainfall" in c or "Rain" in c), None)
        if rain_col:
            bins = pd.cut(df_raw[rain_col], bins=10)
            bf = df_raw.groupby(bins, observed=False)["Flood Occurred"].mean().reset_index()
            bf.columns = ["Range","FloodRate"]
            bf["Range"] = bf["Range"].astype(str)
            bf["FloodRate%"] = (bf["FloodRate"]*100).round(1)
            fig_b = px.bar(bf, x="Range", y="FloodRate%",
                           color="FloodRate%", color_continuous_scale=SRISK,
                           text="FloodRate%", title="Flood Rate by Rainfall Band", height=300)
            fig_b.update_traces(texttemplate='%{text:.0f}%', textposition='outside',
                                marker_line_width=0)
            fig_b.update_layout(**PL, showlegend=False,
                                xaxis_tickangle=-35,
                                margin=dict(l=20,r=10,t=45,b=80))
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info("Add a Rainfall column to see this chart.")

    # Key insights
    st.markdown('<div class="sec-head">💡 Key Insights</div>', unsafe_allow_html=True)
    num_f  = df_raw.select_dtypes(include=np.number).drop(
                columns=["Flood Occurred","Latitude","Longitude"], errors="ignore")
    corrs  = num_f.corrwith(df_raw["Flood Occurred"]).abs().sort_values(ascending=False)
    top_f  = corrs.index[0] if len(corrs) else "N/A"
    top_c  = corrs.iloc[0]  if len(corrs) else 0

    avg_fl = df_raw[df_raw["Flood Occurred"]==1][rain_col].mean() if rain_col else None
    avg_sf = df_raw[df_raw["Flood Occurred"]==0][rain_col].mean() if rain_col else None

    i1,i2,i3 = st.columns(3)
    with i1:
        st.markdown(f"""
        <div class="ins-card">
          <div class="ins-title">🌧️ Strongest Risk Factor</div>
          <div class="ins-text">
            <b style='color:{CP}'>{top_f}</b> has the highest correlation with flood occurrence
            (r = {top_c:.2f}). Monitor this closely.
          </div>
        </div>""", unsafe_allow_html=True)
    with i2:
        if rain_col and avg_fl:
            st.markdown(f"""
            <div class="ins-card">
              <div class="ins-title">⚠️ Rainfall Difference</div>
              <div class="ins-text">
                Flood events averaged <b style='color:{CF}'>{avg_fl:.1f} mm</b> vs
                <b style='color:{CS}'>{avg_sf:.1f} mm</b> for safe events
                — a gap of <b>{avg_fl-avg_sf:.1f} mm</b>.
              </div>
            </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown(f"""
        <div class="ins-card">
          <div class="ins-title">🤖 Model Confidence</div>
          <div class="ins-text">
            The AI achieved <b style='color:{CP}'>{acc*100:.1f}% accuracy</b> and
            <b style='color:{CP}'>{cv_sc.mean()*100:.1f}% cross-validated reliability</b>.
          </div>
        </div>""", unsafe_allow_html=True)

    # Scatter of top 2 correlated features
    if len(corrs) >= 2:
        f1n, f2n = corrs.index[0], corrs.index[1]
        if f1n in df_raw.columns and f2n in df_raw.columns:
            st.markdown(f'<div class="sec-head">🔵 {f1n} vs {f2n}</div>', unsafe_allow_html=True)
            st.markdown("""<div class="tip-box">
            Each dot = one event. <span style='color:#ef4444'><b>Red = flood</b></span>,
            <span style='color:#10b981'><b>green = safe</b></span>.
            Flood events cluster at higher values.
            </div>""", unsafe_allow_html=True)
            fig_sc = px.scatter(
                df_raw, x=f1n, y=f2n,
                color=df_raw["Flood Occurred"].map({1:"Flood 🔴",0:"Safe 🟢"}),
                color_discrete_map={"Flood 🔴":CF,"Safe 🟢":CS},
                opacity=0.55, height=370,
                title=f"{f1n} vs {f2n} — coloured by outcome"
            )
            fig_sc.update_traces(marker=dict(size=7,
                                             line=dict(width=0.3,color='rgba(0,0,0,0.25)')))
            fig_sc.update_layout(**PL, margin=dict(l=40,r=20,t=50,b=40))
            st.plotly_chart(fig_sc, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PREDICT RISK
# ─────────────────────────────────────────────────────────────────────────────
with T[1]:
    st.markdown('<div class="sec-head">🔍 Predict Flood Risk for Any Location</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="tip-box">
    <b>How to use:</b> Adjust the sliders/dropdowns below to match the current environmental
    conditions, then press <b>Predict Now</b>. You'll get an instant risk level, probability
    score and an explanation — all in plain English.
    </div>""", unsafe_allow_html=True)

    # Group features for friendlier layout
    grp_weather = [c for c in FEATURE_COLS if any(k in c for k in ["Rainfall","Temperature","Humidity"])]
    grp_water   = [c for c in FEATURE_COLS if any(k in c for k in ["River","Water Level","Discharge"])]
    grp_geo     = [c for c in FEATURE_COLS if any(k in c for k in ["Elevation","Latitude","Longitude","Land","Soil"])]
    grp_social  = [c for c in FEATURE_COLS if any(k in c for k in ["Population","Infrastructure","Historical"])]
    grp_other   = [c for c in FEATURE_COLS
                   if c not in grp_weather+grp_water+grp_geo+grp_social]

    input_data = {}   # will be filled in FEATURE_COLS order

    def render_inputs(cols_list, n_cols=3):
        """Render slider/selectbox widgets and fill input_data."""
        rows = [cols_list[i:i+n_cols] for i in range(0, len(cols_list), n_cols)]
        for row in rows:
            wcols = st.columns(n_cols)
            for ci, col_name in enumerate(row):
                with wcols[ci]:
                    if col_name in le_dict:
                        opts = le_dict[col_name].classes_.tolist()
                        sel = st.selectbox(col_name, opts, key=f"inp_{col_name}")
                        input_data[col_name] = int(le_dict[col_name].transform([sel])[0])
                    else:
                        cmin  = float(X[col_name].min())
                        cmax  = float(X[col_name].max())
                        cmean = float(X[col_name].mean())
                        if col_name in ("Latitude","Longitude"):
                            input_data[col_name] = st.number_input(
                                col_name, cmin, cmax, cmean,
                                key=f"inp_{col_name}", format="%.4f")
                        else:
                            input_data[col_name] = st.slider(
                                col_name, cmin, cmax, cmean,
                                key=f"inp_{col_name}",
                                help=f"Data range: {cmin:.1f} – {cmax:.1f}")

    if grp_weather: st.markdown("**🌦️ Weather**");     render_inputs(grp_weather)
    if grp_water:   st.markdown("**💧 Water / River**"); render_inputs(grp_water)
    if grp_geo:     st.markdown("**🏔️ Geography**");    render_inputs(grp_geo)
    if grp_social:  st.markdown("**🏘️ Population & Infrastructure**"); render_inputs(grp_social)
    if grp_other:   st.markdown("**📌 Other**");         render_inputs(grp_other)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Predict Flood Risk Now")

    if predict_btn:
        # ── Build input as numpy array in exact FEATURE_COLS order ────────────
        inp_values = np.array([input_data[c] for c in FEATURE_COLS],
                              dtype=float).reshape(1, -1)
        inp_scaled = scaler.transform(inp_values)   # pure numpy → no name check
        pred       = model.predict(inp_scaled)[0]
        prob_flood = model.predict_proba(inp_scaled)[0][1]
        rlabel, ricon, rcolor = risk_label(prob_flood)

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1.1, 1, 1])

        with r1:
            pill = "risk-high" if pred==1 else "risk-low"
            txt  = "⚠️ Flood Likely" if pred==1 else "✅ Safe — No Flood"
            st.markdown(f"""
            <div class="{pill}">
              {txt}<br>
              <span style="font-size:1rem;opacity:0.85">Risk: {ricon} {rlabel}</span>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="ins-card" style="margin-top:13px">
              <div class="ins-title">🗣️ What does this mean?</div>
              <div class="ins-text">
                Based on the conditions you entered, the AI estimates a
                <b style='color:{rcolor}'>{prob_flood*100:.1f}% probability of flooding</b>.
                {"Flood-preparedness measures are <b>recommended</b>."
                 if pred==1 else
                 "Conditions appear <b>relatively safe</b> — continue monitoring water levels."}
              </div>
            </div>""", unsafe_allow_html=True)

        with r2:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=round(prob_flood*100,1),
                title={"text":"Flood Probability",
                       "font":{"family":"Syne","color":"#7dd3fc","size":13}},
                number={"suffix":"%","font":{"family":"Syne","color":"#dff0fc","size":30}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#2a5068",
                            "tickfont":{"size":9}},
                    "bar":{"color":rcolor,"thickness":0.28},
                    "bgcolor":"rgba(7,17,31,0.8)",
                    "borderwidth":1,"bordercolor":"rgba(56,189,248,0.14)",
                    "steps":[
                        {"range":[0,30],"color":"rgba(16,185,129,0.10)"},
                        {"range":[30,60],"color":"rgba(249,115,22,0.07)"},
                        {"range":[60,100],"color":"rgba(239,68,68,0.10)"}
                    ],
                    "threshold":{"line":{"color":CP,"width":2},"value":50}
                }
            ))
            fig_g.update_layout(
                paper_bgcolor='rgba(7,17,31,0)', height=255,
                font=dict(color="#5a8aa6"),
                margin=dict(l=30,r=30,t=45,b=10)
            )
            st.plotly_chart(fig_g, use_container_width=True)

        with r3:
            fig_prob = go.Figure(go.Bar(
                x=["No Flood","Flood"],
                y=[round((1-prob_flood)*100,1), round(prob_flood*100,1)],
                marker_color=[CS,CF],
                text=[f"{(1-prob_flood)*100:.1f}%", f"{prob_flood*100:.1f}%"],
                textposition="outside"
            ))
            fig_prob.update_layout(
                title="Confidence", yaxis_title="Probability (%)",
                yaxis_range=[0,115], height=255,
                **PL, margin=dict(l=10,r=10,t=45,b=10)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # Contributing factors
        if hasattr(model,"feature_importances_"):
            st.markdown('<div class="sec-head">🧪 What\'s Driving This Risk?</div>',
                        unsafe_allow_html=True)
            st.markdown("""<div class="tip-box">
            Larger bars = that factor has more influence on the flood risk prediction.
            </div>""", unsafe_allow_html=True)

            imp       = model.feature_importances_
            x_min     = X.min().values
            x_rng     = X.max().values - x_min
            x_rng[x_rng==0] = 1
            inp_norm  = (inp_values.flatten() - x_min) / x_rng
            contrib   = imp * inp_norm

            cdf = pd.DataFrame({
                "Factor":     FEATURE_COLS,
                "Influence":  contrib,
                "Your Value": [f"{v:.2f}" for v in inp_values.flatten()]
            }).sort_values("Influence", ascending=True).tail(10)

            fig_c = px.bar(cdf, x="Influence", y="Factor", orientation="h",
                           color="Influence", color_continuous_scale=SRISK,
                           text="Your Value", title="Top Risk Contributors", height=360)
            fig_c.update_traces(textposition='outside', marker_line_width=0)
            fig_c.update_layout(**PL, showlegend=False,
                                margin=dict(l=10,r=50,t=45,b=10))
            st.plotly_chart(fig_c, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — FLOOD MAP
# ─────────────────────────────────────────────────────────────────────────────
with T[2]:
    st.markdown('<div class="sec-head">🗺️ Where Did Floods Happen?</div>',
                unsafe_allow_html=True)

    if "Latitude" in df_raw.columns and "Longitude" in df_raw.columns:
        mdf = df_raw[["Latitude","Longitude","Flood Occurred"]].copy()
        mdf["Status"] = mdf["Flood Occurred"].map({1:"Flood 🔴",0:"Safe 🟢"})
        extras = [c for c in ["Rainfall (mm)","Water Level (m)",
                               "River Discharge (m³/s)","Temperature (°C)"]
                  if c in df_raw.columns]
        for e in extras: mdf[e] = df_raw[e]
        mdf["sz"] = mdf["Flood Occurred"].map({1:14,0:7})

        st.markdown("""<div class="tip-box">
        🔴 Red = flood location.  🟢 Green = safe location.
        Hover a dot for details. Zoom/pan the map freely.
        </div>""", unsafe_allow_html=True)

        fig_map = px.scatter_mapbox(
            mdf, lat="Latitude", lon="Longitude",
            color="Status",
            color_discrete_map={"Flood 🔴":CF,"Safe 🟢":CS},
            size="sz", zoom=5, height=500,
            hover_data=extras, title="Flood Event Map"
        )
        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            paper_bgcolor='rgba(7,17,31,0)',
            margin=dict(l=0,r=0,t=40,b=0),
            title_font=dict(family='Syne',color='#7dd3fc',size=14),
            legend=dict(bgcolor='rgba(7,17,31,0.7)',
                        bordercolor='rgba(56,189,248,0.18)')
        )
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown('<div class="sec-head">🌡️ Flood Hotspot Heatmap</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="tip-box">
        Darker red areas = higher concentration of flood events.
        Use this to identify the most at-risk zones.
        </div>""", unsafe_allow_html=True)

        fl_only = mdf[mdf["Flood Occurred"]==1]
        fig_heat = px.density_mapbox(
            fl_only, lat="Latitude", lon="Longitude",
            radius=22, zoom=4, height=430,
            color_continuous_scale=["#07111f","#0369a1","#f97316","#ef4444"],
            title="Flood Density Heatmap"
        )
        fig_heat.update_layout(
            mapbox_style="carto-darkmatter",
            paper_bgcolor='rgba(7,17,31,0)',
            margin=dict(l=0,r=0,t=40,b=0),
            title_font=dict(family='Syne',color='#7dd3fc',size=14)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Top zones
        if len(fl_only):
            st.markdown('<div class="sec-head">📊 Top Flood Zones</div>',
                        unsafe_allow_html=True)
            z = fl_only.copy()
            z["LatBin"] = z["Latitude"].round(1)
            z["LonBin"] = z["Longitude"].round(1)
            zt = (z.groupby(["LatBin","LonBin"])
                   .size().reset_index(name="Flood Events")
                   .sort_values("Flood Events",ascending=False).head(10))
            zt.columns = ["Latitude (approx)","Longitude (approx)","Flood Events"]
            st.dataframe(zt, use_container_width=True, hide_index=True)
    else:
        st.markdown("""<div class="warn-box">
        ⚠️ <b>Latitude / Longitude columns not found.</b>
        Add geographic coordinates to your dataset to enable the map.
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — RISK FACTORS
# ─────────────────────────────────────────────────────────────────────────────
with T[3]:
    st.markdown('<div class="sec-head">📊 Which Factors Drive Flood Risk?</div>',
                unsafe_allow_html=True)

    st.markdown("""<div class="tip-box">
    <b>Higher importance = the AI relies on it more.</b>
    Positive correlation = higher value → more flood risk.
    </div>""", unsafe_allow_html=True)

    la, ra = st.columns(2)

    # FEATURE IMPORTANCE
    with la:
        if hasattr(model,"feature_importances_"):
            idf = pd.DataFrame({
                "Factor": FEATURE_COLS,
                "Importance %": (model.feature_importances_ * 100).round(1)
            }).sort_values("Importance %", ascending=True)

            fig_i = px.bar(
                idf,
                x="Importance %",
                y="Factor",
                orientation="h",
                color="Importance %",
                color_continuous_scale=[[0,"#0b2a40"],[0.4,CP],[1,"#7dd3fc"]],
                text="Importance %",
                title="Feature Importance",
                height=420
            )

            fig_i.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                marker_line_width=0
            )

            fig_i.update_layout(
                **PL,
                showlegend=False,
                margin=dict(l=10,r=55,t=45,b=10)
            )

            st.plotly_chart(fig_i, use_container_width=True)

    # CORRELATION WITH FLOOD
    with ra:
        num_only = df_raw.select_dtypes(include=np.number).drop(
            columns=["Flood Occurred","Latitude","Longitude"], errors="ignore"
        )

        cs = num_only.corrwith(df_raw["Flood Occurred"]).sort_values()

        cdf2 = pd.DataFrame({
            "Factor": cs.index,
            "Correlation": cs.values
        })

        fig_cr = px.bar(
            cdf2,
            x="Correlation",
            y="Factor",
            orientation="h",
            color="Correlation",
            color_continuous_scale=[[0,CF],[0.5,"#07111f"],[1,CS]],
            text=[f"{v:.2f}" for v in cdf2["Correlation"]],
            title="Correlation with Flood Outcome",
            height=420
        )

        fig_cr.add_vline(
            x=0,
            line_color="rgba(56,189,248,0.25)",
            line_dash="dash"
        )

        fig_cr.update_traces(
            textposition='outside',
            marker_line_width=0
        )

        fig_cr.update_layout(
            **PL,
            showlegend=False,
            margin=dict(l=10,r=55,t=45,b=10)
        )

        st.markdown("""<div class="tip-box" style="font-size:0.79rem">
        ➡️ Right (positive) = higher value raises flood risk.<br>
        ⬅️ Left (negative) = higher value lowers flood risk.
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(fig_cr, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — AI PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with T[4]:
    st.markdown('<div class="sec-head">🤖 How Well Does the AI Perform?</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="tip-box">
    All metrics were measured on <b>data the AI had never seen before</b>.
    Higher numbers = better performance.
    </div>""", unsafe_allow_html=True)

    cr  = classification_report(y_test, y_pred, output_dict=True)
    key = "1" if "1" in cr else "weighted avg"
    prec = cr[key].get("precision",0)
    rec  = cr[key].get("recall",0)
    f1s  = cr[key].get("f1-score",0)

    for col_w,(val,lbl,tip) in zip(
        st.columns(5),
        [(f"{acc*100:.1f}%","Accuracy","Correct out of all predictions"),
         (f"{prec*100:.1f}%","Precision","When it says flood, how often right?"),
         (f"{rec*100:.1f}%","Recall","Of real floods, how many caught?"),
         (f"{f1s*100:.1f}%","F1 Score","Balance of Precision & Recall"),
         (f"{cv_sc.mean()*100:.1f}%","CV Score","5-fold cross-validated accuracy")]
    ):
        with col_w:
            st.markdown(f"""
            <div class="stat-card" title="{tip}">
              <div class="stat-num">{val}</div>
              <div class="stat-lbl">{lbl}</div>
              <div class="stat-tip">{tip}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr2 = st.columns(2)

    with cl:
        cm = confusion_matrix(y_test, y_pred)
        tn,fp,fn,tp = cm.ravel()
        fig_cm = go.Figure(go.Heatmap(
            z=[[tn,fp],[fn,tp]],
            x=["Predicted: Safe","Predicted: Flood"],
            y=["Actual: Safe","Actual: Flood"],
            colorscale=[[0,"#07111f"],[0.5,"#0369a1"],[1,CP]],
            text=[[f"✅ {tn}\nCorrectly Safe",f"❌ {fp}\nFalse Alarm"],
                  [f"❌ {fn}\nMissed Flood",f"✅ {tp}\nCorrectly Flood"]],
            texttemplate="%{text}", textfont=dict(size=11,family="Plus Jakarta Sans"),
            showscale=False
        ))
        fig_cm.update_layout(
            title="Confusion Matrix", **PL, height=350,
            margin=dict(l=10,r=10,t=45,b=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("""<div class="tip-box" style="font-size:0.78rem">
        ✅ Correctly Safe = AI said safe → was safe &nbsp;|&nbsp;
        ✅ Correctly Flood = AI said flood → was flood<br>
        ❌ False Alarm = AI said flood → was safe &nbsp;|&nbsp;
        ❌ Missed Flood = AI said safe → flood happened
        </div>""", unsafe_allow_html=True)

    with cr2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ra = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"AI Model (AUC={ra:.3f})",
            line=dict(color=CP,width=2.5),
            fill='tozeroy', fillcolor='rgba(56,189,248,0.05)'
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1],y=[0,1], name="Random Guess",
            line=dict(color="#1e4060",dash="dash",width=1.5)
        ))
        fig_roc.update_layout(
            title=f"ROC Curve — AUC = {ra:.3f}  (1.0 = perfect)",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=350, **PL, margin=dict(l=40,r=20,t=45,b=40)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown(f"""<div class="tip-box" style="font-size:0.78rem">
        AUC = <b>{ra:.3f}</b> → the AI is significantly better than random guessing (0.5).
        Closer to 1.0 = better at distinguishing floods from safe events.
        </div>""", unsafe_allow_html=True)

    # Cross-val bars
    st.markdown('<div class="sec-head">📉 5-Fold Cross-Validation</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="tip-box">
    The data was split 5 ways and the AI was tested on each slice.
    <b>Consistent high bars = stable, trustworthy model.</b>
    </div>""", unsafe_allow_html=True)

    bar_cols = [CS if v>0.80 else (CW if v>0.65 else CF) for v in cv_sc]
    fig_cv = go.Figure()
    fig_cv.add_trace(go.Bar(
        x=[f"Test {i+1}" for i in range(5)], y=cv_sc*100,
        marker=dict(color=bar_cols,line=dict(width=0)),
        text=[f"{v*100:.1f}%" for v in cv_sc], textposition="outside"
    ))
    fig_cv.add_hline(y=cv_sc.mean()*100, line_color=CP, line_dash="dot",
                     annotation_text=f"Average: {cv_sc.mean()*100:.1f}%",
                     annotation_font=dict(color=CP,size=12))
    fig_cv.update_layout(
        title="Accuracy per Fold", height=295,
        yaxis_range=[0,115], **PL,
        margin=dict(l=10,r=10,t=45,b=10)
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    with st.expander("📋 Full Classification Report"):
        st.dataframe(pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True)
        ).T.round(3), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with T[5]:
    st.markdown('<div class="sec-head">📋 Explore & Download Your Data</div>',
                unsafe_allow_html=True)

    fc1,fc2,fc3 = st.columns(3)
    with fc1:
        ff = st.selectbox("Show","All,Flood Events Only,Safe Events Only".split(","))
    with fc2:
        sc_col = st.selectbox("Sort by",
                              df_raw.select_dtypes(include=np.number).columns.tolist())
    with fc3:
        sd = st.radio("Order",["Descending","Ascending"],horizontal=True)

    fdf = df_raw.copy()
    if ff=="Flood Events Only": fdf=fdf[fdf["Flood Occurred"]==1]
    if ff=="Safe Events Only":  fdf=fdf[fdf["Flood Occurred"]==0]
    if sc_col in fdf.columns:
        fdf = fdf.sort_values(sc_col, ascending=(sd=="Ascending"))

    st.markdown(f"*Showing **{len(fdf):,}** of **{len(df_raw):,}** records*")
    st.dataframe(fdf, use_container_width=True, height=370, hide_index=True)

    d1,d2,_ = st.columns([1,1,3])
    with d1:
        st.download_button("⬇️ Download CSV",
                           fdf.to_csv(index=False).encode(),
                           "flood_filtered.csv","text/csv",
                           use_container_width=True)
    with d2:
        st.download_button("⬇️ Download TSV",
                           fdf.to_csv(index=False,sep="\t").encode(),
                           "flood_filtered.tsv","text/plain",
                           use_container_width=True)

    st.markdown('<div class="sec-head">🔍 Data Quality</div>', unsafe_allow_html=True)
    q1,q2,q3 = st.columns(3)
    miss_total = int(df_raw.isnull().sum().sum())
    dupes      = int(df_raw.duplicated().sum())
    complete   = (1 - df_raw.isnull().mean().mean())*100

    for col_w,(val,lbl,ok,note_ok,note_bad) in zip(
        [q1,q2,q3],
        [(miss_total,"Missing Values",  miss_total==0,"✅ Dataset complete!","⚠️ Rows with missing values were dropped."),
         (dupes,     "Duplicate Rows",  dupes==0,     "✅ No duplicates.",   "⚠️ Consider removing duplicates."),
         (f"{complete:.1f}%","Completeness",complete>95,"✅ Excellent quality.","⚠️ Some missing data detected.")]
    ):
        with col_w:
            c = CS if ok else CF
            st.markdown(f"""
            <div class="glass-card" style="text-align:center">
              <div class="stat-num" style="color:{c}">{val}</div>
              <div class="stat-lbl">{lbl}</div>
              <div class="stat-tip">{note_ok if ok else note_bad}</div>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  🌊 <b>FloodSense AI</b> — Urban Flood Risk Prediction System<br>
  🎓 Final Year AI/ML Project &nbsp;·&nbsp; 🌍 Climate-Tech &nbsp;·&nbsp; 🤖 Ensemble ML<br>
  <span style="font-size:0.72rem">Built with Streamlit · Scikit-learn · Plotly · Python</span>
</div>
""", unsafe_allow_html=True)
#---------------------------END OF CODE-----------------------------------------#



