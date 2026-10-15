"""Streamlit demo — Student Stress Predictor."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from feature_engineering import create_features

PIPELINE_PATH = Path(__file__).parent / "outputs/models/inference_pipeline.pkl"

_COMPOSITE_COLS = [
    "psychological_burden", "academic_stress", "wellbeing_index",
    "environmental_stress", "support_deficit",
]
_HIGHER_IS_BAD = {
    "psychological_burden": True,  "academic_stress": True,
    "wellbeing_index":      False, "environmental_stress": True,
    "support_deficit":      True,
}
_COMPOSITE_LABELS = {
    "psychological_burden": "Psychological Burden",
    "academic_stress":      "Academic Stress",
    "wellbeing_index":      "Wellbeing Index",
    "environmental_stress": "Environmental Stress",
    "support_deficit":      "Support Deficit",
}

# ── Preset profiles ────────────────────────────────────────────────────────────
_PRESETS = {
    "high": {
        # Psychological
        "anx": 17, "est": 8, "dep": 20, "mh_history": True,
        # Physical
        "head": 4, "bp": "High", "slp": 1, "brth": 3,
        # Academic & social
        "stl": 5, "perf": 1, "teach": 1, "car": 5,
        "peer": 4, "soc": 0, "ext": 1,
        # Environment
        "noi": 4, "liv": 1, "saf": 1, "bas": 1, "bul": 3,
    },
    "medium": {
        "anx": 11, "est": 14, "dep": 11, "mh_history": False,
        "head": 2, "bp": "Normal", "slp": 3, "brth": 1,
        "stl": 3, "perf": 3, "teach": 3, "car": 3,
        "peer": 2, "soc": 2, "ext": 2,
        "noi": 2, "liv": 3, "saf": 3, "bas": 3, "bul": 1,
    },
    "low": {
        "anx": 3, "est": 26, "dep": 2, "mh_history": False,
        "head": 0, "bp": "Normal", "slp": 5, "brth": 0,
        "stl": 2, "perf": 5, "teach": 5, "car": 1,
        "peer": 1, "soc": 3, "ext": 4,
        "noi": 1, "liv": 5, "saf": 5, "bas": 5, "bul": 0,
    },
}

def _apply_preset(name: str) -> None:
    for k, v in _PRESETS[name].items():
        st.session_state[k] = v


# ── Pipeline diagram (matplotlib) ─────────────────────────────────────────────
@st.cache_data
def _draw_pipeline() -> plt.Figure:
    CX = 6.2          # horizontal centre of the figure
    W_MAIN = 7.0      # width of main flow boxes
    W_SCALE = 5.8     # width of narrower scale box

    fig = plt.figure(figsize=(13, 9.2))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")

    # ── helpers ───────────────────────────────────────────────────────────────
    def rbox(x, y, w, h, fc, ec, lw=1.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            fc=fc, ec=ec, lw=lw, zorder=2,
        ))

    def txt(x, y, s, size=9, weight="normal", color="#111827", ha="center", va="center", **kw):
        ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                ha=ha, va=va, zorder=3, **kw)

    def arr(x, y1, y2):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="->", color="#94a3b8",
                                   lw=1.5, mutation_scale=13), zorder=1)

    def hline(x1, x2, y, color="#cbd5e1"):
        ax.plot([x1, x2], [y, y], color=color, lw=1.5, zorder=1)

    def vline(x, y1, y2, color="#cbd5e1"):
        ax.plot([x, x], [y1, y2], color=color, lw=1.5, zorder=1)

    # ── stage bracket (left bar) ──────────────────────────────────────────────
    def stage_bar(y_lo, y_hi, color, label):
        ax.plot([0.22, 0.22], [y_lo, y_hi], color=color, lw=4,
                solid_capstyle="round", zorder=1)
        ax.text(0.08, (y_lo + y_hi) / 2, label, fontsize=6.5, fontweight="600",
                color=color, ha="center", va="center", rotation=90,
                letter_spacing=1 if hasattr(ax.text, "__self__") else 0)

    ax.plot([0.22, 0.22], [7.2, 8.9], color="#e2e8f0", lw=4,
            solid_capstyle="round", zorder=1)
    ax.text(0.08, 8.05, "DATA\nPREP", fontsize=6, fontweight="bold",
            color="#94a3b8", ha="center", va="center", rotation=90)

    ax.plot([0.22, 0.22], [5.2, 7.0], color="#c4b5fd", lw=4,
            solid_capstyle="round", zorder=1)
    ax.text(0.08, 6.1, "PRE-\nPROC", fontsize=6, fontweight="bold",
            color="#8b5cf6", ha="center", va="center", rotation=90)

    ax.plot([0.22, 0.22], [3.1, 5.0], color="#86efac", lw=4,
            solid_capstyle="round", zorder=1)
    ax.text(0.08, 4.05, "ROUTE", fontsize=6, fontweight="bold",
            color="#059669", ha="center", va="center", rotation=90)

    ax.plot([0.22, 0.22], [1.4, 3.0], color="#e2e8f0", lw=4,
            solid_capstyle="round", zorder=1)
    ax.text(0.08, 2.2, "CLASSIF\nY", fontsize=6, fontweight="bold",
            color="#94a3b8", ha="center", va="center", rotation=90)

    # ══ SURVEY INPUT ══
    rbox(CX - W_MAIN/2, 8.15, W_MAIN, 0.75, "#eff6ff", "#93c5fd")
    txt(CX, 8.62, "Survey Input", size=12, weight="bold", color="#1d4ed8")
    txt(CX, 8.32, "20 psychometric features  ·  raw slider values", size=8.5, color="#3b82f6")

    arr(CX, 8.15, 7.9)

    # ══ FEATURE ENGINEERING ══
    rbox(CX - W_MAIN/2, 7.1, W_MAIN, 0.75, "#f0fdf4", "#86efac")
    txt(CX, 7.57, "Feature Engineering", size=12, weight="bold", color="#059669")
    txt(CX, 7.27, "+5 composite scores  →  25 features total", size=8.5, color="#16a34a")

    # Side note: composite formulas
    rbox(10.05, 6.85, 2.7, 1.2, "white", "#e2e8f0", lw=1)
    txt(11.4, 7.95, "COMPOSITE FEATURES", size=6.5, weight="bold", color="#9ca3af")
    ax.plot([10.1, 12.7], [7.88, 7.88], color="#e2e8f0", lw=0.8)
    for i, s in enumerate([
        "Psych Burden = anxiety+depression−esteem",
        "Academic Stress = load+pressure−perf.",
        "Wellbeing = sleep+support+safety−head.",
        "Environ. Stress = noise+bullying−living",
        "Support Deficit = pressure+career−support",
    ]):
        txt(10.12, 7.79 - i * 0.19, s, size=6.2, color="#6b7280", ha="left", va="center")
    # dashed connector
    ax.annotate("", xy=(10.05, 7.4), xytext=(9.7, 7.4),
                arrowprops=dict(arrowstyle="-", color="#e2e8f0",
                                lw=1, linestyle="dashed"), zorder=1)

    arr(CX, 7.1, 6.85)

    # ══ MINMAX SCALING ══
    rbox(CX - W_SCALE/2, 6.05, W_SCALE, 0.75, "#fefce8", "#fde68a")
    txt(CX, 6.52, "MinMax Scaling  [0 – 1]", size=11, weight="bold", color="#b45309")
    txt(CX, 6.22, "fitted on training data only  ·  prevents data leakage", size=8, color="#d97706")

    arr(CX, 6.05, 5.8)

    # ══ K-MEANS ══
    rbox(CX - W_MAIN/2 - 0.2, 5.0, W_MAIN + 0.4, 0.75, "#faf5ff", "#c4b5fd", lw=2)
    txt(CX, 5.47, "K-Means Clustering", size=12, weight="bold", color="#7c3aed")
    txt(CX, 5.17, "K = 4  ·  silhouette-selected  ·  each student routed to nearest cluster (unsupervised)", size=8.5, color="#8b5cf6")

    # ── branching ────────────────────────────────────────────────────────────
    CX_CLUSTERS = [1.55, 4.2, 6.85, 9.5]   # centre x of each cluster box
    BOX_W = 2.2
    BOX_H = 1.55

    vline(CX, 5.0, 4.72)
    hline(CX_CLUSTERS[0], CX_CLUSTERS[-1], 4.72)
    for cx in CX_CLUSTERS:
        arr(cx, 4.72, 4.65)

    # ══ CLUSTER BOXES ══
    clusters = [
        (CX_CLUSTERS[0], "#f0fdf4", "#6ee7b7", 1.5, "C0", "Low Stress",   "Pure cluster",   "Majority predictor", "#065f46", "#059669"),
        (CX_CLUSTERS[1], "#fef2f2", "#fca5a5", 1.5, "C1", "High Stress",  "Near-pure",      "Majority predictor", "#7f1d1d", "#dc2626"),
        (CX_CLUSTERS[2], "#fffbeb", "#fcd34d", 1.5, "C2", "Medium Stress","Pure cluster",   "Majority predictor", "#78350f", "#d97706"),
        (CX_CLUSTERS[3], "#f5f3ff", "#7c3aed", 2.2, "C3", "Mixed",        "Mixed cluster",  "Random Forest",      "#4c1d95", "#7c3aed"),
    ]

    for (cx, fc, ec, lw, cid, stress, ctype, clf, tc, lc) in clusters:
        x0 = cx - BOX_W / 2
        rbox(x0, 3.1, BOX_W, BOX_H, fc, ec, lw)
        txt(cx, 4.42, cid,    size=13, weight="bold", color=tc)
        txt(cx, 4.17, stress, size=9,  weight="bold", color=lc)
        txt(cx, 3.92, ctype,  size=7.5, color="#6b7280")
        ax.plot([x0 + 0.15, x0 + BOX_W - 0.15], [3.8, 3.8], color="#e2e8f0", lw=0.8)
        txt(cx, 3.58, clf, size=8.5,
            weight="bold" if cid == "C3" else "normal",
            color=tc if cid == "C3" else "#374151")
        if cid == "C3":
            txt(cx, 3.28, "+11 pp vs global RF", size=7, color="#8b5cf6")

    # ── merge ────────────────────────────────────────────────────────────────
    for cx in CX_CLUSTERS:
        vline(cx, 3.1, 2.82)
    hline(CX_CLUSTERS[0], CX_CLUSTERS[-1], 2.82)
    arr(CX, 2.82, 2.6)

    # ══ OUTPUT ══
    rbox(CX - W_MAIN/2 - 0.2, 1.55, W_MAIN + 0.4, 0.9, "#0f172a", "#334155")
    txt(CX, 2.12, "Stress Level Prediction", size=13, weight="bold", color="white")
    txt(CX - 1.0, 1.75, "Low  ·  Medium  ·  High", size=9, color="#94a3b8")
    # Macro-F1 badge
    rbox(9.05, 1.65, 1.6, 0.68, "#1e293b", "#475569", lw=1)
    txt(9.85, 2.08, "Macro-F1", size=7, color="#94a3b8")
    txt(9.85, 1.82, "0.873", size=13, weight="bold", color="#38bdf8")

    # ── title ────────────────────────────────────────────────────────────────
    txt(CX, 9.05, "PIPELINE ARCHITECTURE", size=8.5, weight="bold",
        color="#94a3b8", va="center",
        fontfamily="monospace")

    plt.tight_layout(pad=0)
    return fig


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, p, div, h1, h2, h3, h4, h5, h6, li, a, td, th, input, select, textarea {
  font-family: 'Inter', system-ui, sans-serif !important;
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { display: none !important; }
.block-container { padding: 1.75rem 2.5rem 3rem !important; max-width: 1280px; }

/* ── App header ── */
.app-header {
  background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
  border-radius: 14px; padding: 1.75rem 2rem; margin-bottom: 1.75rem; color: white;
}
.app-header h1 {
  margin: 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em;
}
.app-header p  { margin: 0.4rem 0 0; opacity: 0.75; font-size: 0.85rem; }
.badge {
  display: inline-block; background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.25);
  border-radius: 99px; padding: 0.2rem 0.65rem; font-size: 0.75rem;
  font-weight: 500; margin-right: 0.4rem; margin-top: 0.6rem;
}

/* ── Section labels ── */
.section-label {
  font-size: 0.68rem; font-weight: 600; color: #9ca3af;
  text-transform: uppercase; letter-spacing: .1em; margin-bottom: 0.6rem;
}

/* ── Tab pills ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 3px; background: #f1f5f9; padding: 4px; border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important; font-size: 0.8rem !important;
  font-weight: 500 !important; padding: 0.3rem 0.85rem !important; color: #64748b !important;
}
.stTabs [aria-selected="true"] {
  background: white !important;
  box-shadow: 0 1px 4px rgba(0,0,0,.1) !important;
  color: #1e3a5f !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem !important; }

/* ── Parameter rows ── */
.param-name { font-size: 0.875rem; font-weight: 600; color: #111827; margin-bottom: 0.1rem; }
.param-desc { font-size: 0.75rem; color: #6b7280; line-height: 1.45; margin-bottom: 0.25rem; }
.param-range {
  display: flex; justify-content: space-between;
  font-size: 0.67rem; color: #9ca3af; margin-bottom: -0.1rem;
}
div[data-testid="stSlider"] { padding-top: 0.1rem !important; }

/* ── Slider accent colour ── */
div[data-testid="stSlider"] [data-testid="stTickBar"] { background: #e2e8f0; }

/* ── Toggle label ── */
.param-toggle { display: flex; align-items: flex-start; gap: 0.65rem; margin-top: 0.5rem; }
.param-toggle-text .param-name { margin-bottom: 0.1rem; }

/* ── Result card ── */
.result-card {
  border-radius: 13px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
}
.result-icon  { font-size: 1.6rem; line-height: 1; margin-bottom: 0.3rem; }
.result-label { font-size: 1.45rem; font-weight: 700; margin: 0 0 0.2rem; }
.result-sub   { font-size: 0.8rem; color: #475569; }

/* ── Info mini-cards ── */
.mini-card {
  background: #f8fafc; border: 1px solid #e2e8f0;
  border-radius: 10px; padding: 0.8rem 1rem; height: 100%;
}
.mini-card-label { font-size: 0.67rem; font-weight: 600; color: #94a3b8;
                   text-transform: uppercase; letter-spacing: .08em; }
.mini-card-val   { font-size: 1rem; font-weight: 700; color: #0f172a; margin-top: 0.15rem; }
.mini-card-sub   { font-size: 0.72rem; color: #64748b; margin-top: 0.1rem; }

/* ── Progress bars ── */
.pbar-row    { margin-bottom: 0.85rem; }
.pbar-header { display: flex; justify-content: space-between; margin-bottom: 0.25rem; }
.pbar-label  { font-size: 0.8rem; font-weight: 500; color: #334155; }
.pbar-value  { font-size: 0.8rem; color: #94a3b8; }
.pbar-track  { background: #e2e8f0; border-radius: 99px; height: 7px; overflow: hidden; }
.pbar-fill   { height: 100%; border-radius: 99px; transition: width .4s ease; }

/* ── Divider ── */
.div { border: none; border-top: 1px solid #e2e8f0; margin: 1.25rem 0; }

/* ── Cluster table ── */
.stDataFrame { border-radius: 10px !important; }

/* ── Expander ── */
[data-testid="stExpander"] details > summary {
  font-size: 0.85rem !important; font-weight: 600 !important;
  display: flex !important; align-items: center !important;
  justify-content: space-between !important; list-style: none !important;
  padding: 0.65rem 0.8rem !important; cursor: pointer;
}
[data-testid="stExpander"] details > summary::-webkit-details-marker { display: none !important; }
[data-testid="stExpander"] details > summary::marker { display: none !important; }
[data-testid="stExpanderToggleIcon"] { display: none !important; }
[data-testid="stExpander"] svg { display: none !important; }

/* ── Preset buttons ── */
div[data-testid="stButton"] > button {
  border-radius: 8px; font-size: 0.78rem; font-weight: 500;
  border: 1.5px solid #e2e8f0; background: white;
  color: #374151; padding: 0.3rem 0.6rem;
  transition: border-color .15s, background .15s, color .15s;
  width: 100%;
}
div[data-testid="stButton"] > button:hover {
  border-color: #2563eb; color: #1d4ed8; background: #eff6ff;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    if not PIPELINE_PATH.exists():
        return None
    return joblib.load(PIPELINE_PATH)


def _scale(df_fe: pd.DataFrame, params: dict) -> pd.DataFrame:
    feat_cols = params["feature_cols"]
    if "scaler" in params:
        X = params["scaler"].transform(df_fe[feat_cols])
    else:
        X = (df_fe[feat_cols].values.astype(float) - params["min_"]) / np.where(
            params["range_"] == 0, 1, params["range_"])
    return pd.DataFrame(np.clip(X, 0, 1), columns=feat_cols, index=df_fe.index)


def run_inference(user_values: dict, bundle: dict):
    df_fe = create_features(pd.DataFrame([user_values]))
    df_sc = _scale(df_fe, bundle["scaler_params"])
    fc    = bundle["feature_cols"]
    cluster_id   = int(bundle["hybrid_model"].kmeans.predict(df_sc[fc])[0])
    pred_encoded = int(bundle["hybrid_model"].predict(df_sc)[0])
    stress_label = bundle["le_stress"].inverse_transform([pred_encoded])[0]
    return stress_label, cluster_id, {
        c: float(df_sc[c].iloc[0]) for c in _COMPOSITE_COLS if c in df_sc.columns
    }


def pslider(key: str, name: str, desc: str, lo: int, hi: int,
            default: int, lo_lbl: str = "", hi_lbl: str = "") -> int:
    """Slider widget with name, description and range hint labels."""
    st.markdown(
        f'<div class="param-name">{name}</div>'
        f'<div class="param-desc">{desc}</div>',
        unsafe_allow_html=True,
    )
    if lo_lbl or hi_lbl:
        st.markdown(
            f'<div class="param-range"><span>{lo_lbl}</span><span>{hi_lbl}</span></div>',
            unsafe_allow_html=True,
        )
    return st.slider(name, lo, hi, default, key=key, label_visibility="collapsed")


def pbar(label: str, value: float, higher_is_bad: bool) -> None:
    """Coloured progress bar: red when bad, green when good."""
    if higher_is_bad:
        r, g = int(215 * value + 40), int(170 * (1 - value) + 45)
    else:
        r, g = int(170 * (1 - value) + 45), int(215 * value + 40)
    color = f"rgb({r},{g},60)"
    st.markdown(f"""
    <div class="pbar-row">
      <div class="pbar-header">
        <span class="pbar-label">{label}</span>
        <span class="pbar-value">{value:.2f}</span>
      </div>
      <div class="pbar-track">
        <div class="pbar-fill" style="width:{int(value*100)}%;background:{color};"></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
bundle = load_bundle()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🧠 Student Stress Predictor</h1>
  <p>Hybrid K-Means + per-cluster classifier trained on 1,100 university students</p>
  <div style="margin-top:.5rem">
    <span class="badge">Macro-F1 = 0.873</span>
    <span class="badge">K = 4 clusters</span>
    <span class="badge">25 features</span>
  </div>
</div>
""", unsafe_allow_html=True)

if bundle is None:
    st.error(
        "**Model not found.** Run `py main.py` from the project directory "
        "to train the pipeline first, then relaunch this app."
    )
    st.stop()

# ── Two-column layout ──────────────────────────────────────────────────────────
col_form, col_result = st.columns([3, 2], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Assessment form
# ══════════════════════════════════════════════════════════════════════════════
with col_form:
    st.markdown('<div class="section-label">Assessment</div>', unsafe_allow_html=True)
    st.caption("Move each slider to reflect your current situation. Results update instantly.")

    # ── Preset profile loader ──────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:.78rem;font-weight:600;color:#64748b;margin-bottom:.4rem">'
        'Load a sample profile</div>',
        unsafe_allow_html=True,
    )
    pb1, pb2, pb3, pb4 = st.columns([1, 1, 1, 0.7])
    pb1.button("🔴  High Stress",   on_click=_apply_preset, args=("high",),   key="_ph")
    pb2.button("🟡  Medium Stress", on_click=_apply_preset, args=("medium",), key="_pm")
    pb3.button("🟢  Low Stress",    on_click=_apply_preset, args=("low",),    key="_pl")
    if pb4.button("↺  Reset", key="_pr"):
        for k in list(st.session_state.keys()):
            if not k.startswith("_"):
                del st.session_state[k]
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    tab_psych, tab_phys, tab_acad, tab_env = st.tabs([
        "🧠  Psychological",
        "🏥  Physical Health",
        "📚  Academic & Social",
        "🏠  Environment",
    ])

    # ── Psychological ──────────────────────────────────────────────────────────
    with tab_psych:
        anxiety = pslider(
            "anx", "Anxiety Level",
            "How often do you feel nervous, anxious, or unable to stop worrying? "
            "Based on the GAD-7 clinical scale — a standard tool used in mental health screening.",
            0, 21, 10, "No anxiety", "Severe / daily",
        )
        self_esteem = pslider(
            "est", "Self-Esteem",
            "Your overall sense of personal worth and confidence in your own abilities. "
            "Measured on the Rosenberg Self-Esteem Scale — higher scores reflect a more positive self-image.",
            0, 30, 15, "Very low", "Very high",
        )
        depression = pslider(
            "dep", "Depression Score",
            "How frequently have you experienced low mood, loss of interest, or feelings of "
            "hopelessness over the past two weeks? Based on the PHQ-9 depression inventory.",
            0, 27, 12, "No symptoms", "Severe",
        )

        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="param-name">Mental Health History</div>'
            '<div class="param-desc">Have you previously been diagnosed with or received '
            'treatment for any mental health condition (e.g. anxiety disorder, depression)?</div>',
            unsafe_allow_html=True,
        )
        mh_history = st.toggle("Mental health history", value=False, label_visibility="collapsed", key="mh_history")

    # ── Physical ───────────────────────────────────────────────────────────────
    with tab_phys:
        headache = pslider(
            "head", "Headache Frequency",
            "On average, how many days per week do you experience headaches or migraines? "
            "Frequent headaches are a common physiological marker of chronic stress.",
            0, 5, 2, "Never", "Every day",
        )

        st.markdown(
            '<div class="param-name">Blood Pressure Level</div>'
            '<div class="param-desc">Your most recent blood pressure reading or general category '
            'as advised by a healthcare provider. Elevated BP is linked to sustained stress responses.</div>',
            unsafe_allow_html=True,
        )
        bp_choice = st.radio(
            "Blood Pressure Level", ["Normal", "Elevated", "High"],
            index=1, horizontal=True, label_visibility="collapsed", key="bp",
        )
        blood_pressure = {"Normal": 1, "Elevated": 2, "High": 3}[bp_choice]

        sleep_quality = pslider(
            "slp", "Sleep Quality",
            "How rested and refreshed do you typically feel after a night's sleep? "
            "Poor sleep quality is both a symptom and a driver of elevated stress levels.",
            0, 5, 3, "Very poor", "Excellent",
        )
        breathing_prob = pslider(
            "brth", "Breathing Difficulties",
            "How often do you experience shortness of breath, chest tightness, or similar symptoms? "
            "Breathing problems can signal anxiety-related physical responses.",
            0, 5, 1, "Never", "Very often",
        )

    # ── Academic & Social ──────────────────────────────────────────────────────
    with tab_acad:
        study_load = pslider(
            "stl", "Study Load",
            "How heavy is your current academic workload relative to what you can comfortably handle? "
            "High study load is one of the strongest academic stressors for university students.",
            0, 5, 3, "Very light", "Overwhelming",
        )
        acad_perf = pslider(
            "perf", "Academic Performance",
            "Your current grade standing and overall results in your programme. "
            "Poor performance can compound anxiety, while strong performance acts as a buffer.",
            0, 5, 3, "Failing", "Excellent",
        )
        teacher_rel = pslider(
            "teach", "Teacher–Student Relationship",
            "Quality of communication, respect, and academic support from your teachers or professors. "
            "Positive relationships significantly reduce academic stress.",
            0, 5, 3, "Very poor", "Excellent",
        )
        career_concerns = pslider(
            "car", "Future Career Concerns",
            "How anxious are you about your employment prospects, career direction, or professional future? "
            "Career uncertainty is a major source of chronic stress among final-year students.",
            0, 5, 3, "Not worried", "Extremely worried",
        )
        peer_pressure = pslider(
            "peer", "Peer Pressure",
            "How much social or academic pressure do you experience from classmates to perform, "
            "conform, or compete? Peer pressure amplifies both academic and psychological burden.",
            0, 5, 2, "None", "Extreme",
        )
        social_support = pslider(
            "soc", "Social Support",
            "How much emotional encouragement and understanding do you receive from friends, "
            "family, or peers? Social support is one of the strongest protective factors against stress.",
            0, 3, 2, "No support", "Strong support",
        )
        extracurricular = pslider(
            "ext", "Extracurricular Activities",
            "How actively do you engage in clubs, sports, volunteering, or hobbies outside of class? "
            "Moderate engagement acts as a healthy stress outlet and strengthens social bonds.",
            0, 5, 2, "None", "Very active",
        )

    # ── Environment ────────────────────────────────────────────────────────────
    with tab_env:
        noise_level = pslider(
            "noi", "Noise Level",
            "How loud or disruptive is your primary study and living environment on a typical day? "
            "Chronic noise exposure raises cortisol levels and impairs concentration.",
            0, 5, 2, "Silent", "Very noisy",
        )
        living_cond = pslider(
            "liv", "Living Conditions",
            "Overall quality of your housing — cleanliness, personal space, comfort, and "
            "suitability for rest and study. Precarious living conditions are a persistent stressor.",
            0, 5, 3, "Very poor", "Excellent",
        )
        safety = pslider(
            "saf", "Sense of Safety",
            "How safe do you feel in your home, on campus, and in your surrounding neighbourhood? "
            "Feeling unsafe is a background stressor that compounds other pressures.",
            0, 5, 3, "Unsafe", "Completely safe",
        )
        basic_needs = pslider(
            "bas", "Basic Needs",
            "Do you have reliable access to food, clean water, stable shelter, and other daily essentials? "
            "Unmet basic needs represent the highest-priority stressor tier.",
            0, 5, 3, "Severely lacking", "Fully met",
        )
        bullying = pslider(
            "bul", "Bullying / Harassment",
            "How often do you experience intimidation, exclusion, or bullying from peers or others "
            "in your environment? Bullying has an outsized effect on psychological burden scores.",
            0, 5, 1, "Never", "Frequently",
        )

# ── Assemble user values ───────────────────────────────────────────────────────
user_values = {
    "anxiety_level":                anxiety,
    "self_esteem":                  self_esteem,
    "mental_health_history":        int(mh_history),
    "depression":                   depression,
    "headache":                     headache,
    "blood_pressure":               blood_pressure,
    "sleep_quality":                sleep_quality,
    "breathing_problem":            breathing_prob,
    "noise_level":                  noise_level,
    "living_conditions":            living_cond,
    "safety":                       safety,
    "basic_needs":                  basic_needs,
    "academic_performance":         acad_perf,
    "study_load":                   study_load,
    "teacher_student_relationship": teacher_rel,
    "future_career_concerns":       career_concerns,
    "social_support":               social_support,
    "peer_pressure":                peer_pressure,
    "extracurricular_activities":   extracurricular,
    "bullying":                     bullying,
}

stress, cluster_id, comp_scaled = run_inference(user_values, bundle)

cluster_info = bundle.get("cluster_info", {})
cinfo      = cluster_info.get(cluster_id, {})
dom_stress = cinfo.get("dominant_stress", "?")
c_type     = cinfo.get("type", "?")
c_n        = cinfo.get("n_train", "?")

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Results
# ══════════════════════════════════════════════════════════════════════════════
with col_result:
    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)

    _STRESS_CFG = {
        "High":   ("#dc2626", "#fef2f2", "#fca5a5", "🔴"),
        "Medium": ("#d97706", "#fffbeb", "#fcd34d", "🟡"),
        "Low":    ("#059669", "#f0fdf4", "#6ee7b7", "🟢"),
    }
    color, bg, border, icon = _STRESS_CFG[stress]

    st.markdown(f"""
    <div class="result-card" style="background:{bg};border:1.5px solid {border};">
      <div class="result-icon">{icon}</div>
      <div class="result-label" style="color:{color}">{stress.upper()} STRESS</div>
      <div class="result-sub">
        Your profile matches the <strong>{stress.lower()}-stress</strong> group.
        Routed through Cluster C{cluster_id}.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Cluster + confidence info cards
    _CONF_CFG = {
        "Pure":      ("Very High", "#059669"),
        "Near-Pure": ("High",      "#2563eb"),
        "Mixed":     ("Moderate",  "#d97706"),
    }
    conf_label, conf_color = _CONF_CFG.get(c_type, ("–", "#64748b"))

    mc1, mc2 = st.columns(2)
    mc1.markdown(f"""
    <div class="mini-card">
      <div class="mini-card-label">Cluster</div>
      <div class="mini-card-val">C{cluster_id} · {dom_stress}</div>
      <div class="mini-card-sub">{c_n} students in training</div>
    </div>""", unsafe_allow_html=True)
    mc2.markdown(f"""
    <div class="mini-card">
      <div class="mini-card-label">Confidence</div>
      <div class="mini-card-val" style="color:{conf_color}">{conf_label}</div>
      <div class="mini-card-sub">Profile type: {c_type}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # Factor score bars
    st.markdown(
        '<div class="section-label">Factor Scores  ·  0 – 1 scale</div>',
        unsafe_allow_html=True,
    )
    st.caption("Green = healthy direction · Red = stress-contributing direction · Midpoint = 0.5")

    for col_key in _COMPOSITE_COLS:
        if col_key in comp_scaled:
            pbar(_COMPOSITE_LABELS[col_key], comp_scaled[col_key], _HIGHER_IS_BAD[col_key])

# ── Cluster comparison table ───────────────────────────────────────────────────
st.markdown('<div class="div"></div>', unsafe_allow_html=True)

profiles_df = bundle.get("cluster_profiles", pd.DataFrame())
if not profiles_df.empty and all(c in profiles_df.columns for c in _COMPOSITE_COLS):
    st.markdown("### Cluster Comparison")
    st.caption(
        "Mean factor scores for each cluster's training population, "
        "alongside your scaled values."
    )

    cp   = [c for c in _COMPOSITE_COLS if c in profiles_df.columns]
    disp = profiles_df[["cluster", "dominant_stress"] + cp].copy()
    disp["cluster"] = disp["cluster"].astype(str)

    you_row = {"cluster": "You ★", "dominant_stress": stress}
    you_row.update({c: round(comp_scaled.get(c, 0), 3) for c in cp})
    disp = pd.concat([disp, pd.DataFrame([you_row])], ignore_index=True)
    disp.columns = (
        ["Cluster", "Dominant Stress"]
        + [_COMPOSITE_LABELS.get(c, c) for c in cp]
    )
    disp = disp.set_index("Cluster")

    st.dataframe(
        disp.style.format(
            {col: "{:.3f}" for col in disp.columns if col != "Dominant Stress"}
        ),
        width="stretch",
    )

# ── How it works ───────────────────────────────────────────────────────────────
st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
with st.expander("ℹ️  How the model works"):

    # ── Pipeline diagram ───────────────────────────────────────────────────────
    fig = _draw_pipeline()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

    # ── Text explanation ───────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
**Two-stage hybrid architecture**

**Routing (K-Means)** — All 25 features are fed into a K-Means model
(K = 4, selected by silhouette score). Each student is assigned to the nearest
cluster *before* any label is consulted — purely unsupervised geometry.

**Classification (per-cluster specialist)** — Each cluster has its own classifier
trained only on that cluster's data. The best candidate (by internal macro-F1) wins.
For C0, C1, C2 the dominant class is overwhelming, so a simple majority predictor
is unbeatable. C3 is mixed — a Real Forest learns real decision boundaries there
and outperforms the global baseline by **+11 pp accuracy** on that cluster alone.
        """)
    with c2:
        st.markdown("""
**Cluster profiles**

| Cluster | Dominant | Type | Key signature |
|---------|----------|------|---------------|
| C0 | Low | Pure | High wellbeing · low support deficit |
| C1 | High | Near-pure | High psych burden · high academic stress |
| C2 | Medium | Pure | Low blood pressure · moderate support |
| C3 | Mixed | Mixed | Low social support · high blood pressure |

**Evaluation**
Macro-F1 **0.873** · 5×2 nested CV · McNemar's test vs baselines
        """)

    # ── Models used ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""<div style="font-size:.72rem;font-weight:700;color:#2563eb;
        text-transform:uppercase;letter-spacing:.1em;margin-bottom:.8rem;">
        Models Used in This Pipeline</div>""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.markdown("""
**K-Means Clustering** *(Routing)*

Assigns each student to one of 4 clusters based on their 25 feature values.
Optimal K was selected by **silhouette score** (K=4 beat K=2,3,5,6).
This step is fully unsupervised — no stress labels are used. It splits the
student population into geometrically similar groups before any classification happens.
    """)
    m2.markdown("""
**Majority-Class Predictor** *(C0, C1, C2 specialists)*

For the three pure clusters where one stress class holds ≥ 90 % of members,
the best possible classifier is simply the **dominant-class predictor** — it achieves
near-perfect accuracy and cannot be beaten by any learner on near-uniform data.
Candidate models tested: Logistic Regression, XGBoost, Random Forest, Dummy.
    """)
    m3.markdown("""
**Random Forest Classifier** *(C3 specialist)*

The mixed cluster C3 has real class overlap, so a learner with genuine
decision boundaries is needed. A **Random Forest** (100 trees, Gini criterion)
trained only on C3 data wins the internal macro-F1 competition against
XGBoost and Logistic Regression for that cluster, contributing the **+11 pp**
accuracy gain that drives the hybrid's overall Macro-F1 of **0.873**.
    """)

