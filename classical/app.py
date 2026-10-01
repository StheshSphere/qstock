"""
QStock — Demo App
=========================
Streamlit presentation layer for the QStock demo:

  Pitch Summary    — judge-facing landing tab with live metrics
  Overview         — group health at a glance
  Members          — editable member roster (shared across tabs)
  Payout Scheduler — QAOA vs classical greedy rank-split
  Risk Classifier  — quantum kernel SVM vs classical SVM
  Pooled Funds     — illustrative fee/interest revenue projection

Run with: streamlit run classical/app.py   (from the repo root, so
.streamlit/config.toml theming is picked up)

NOTE: this file is the UI/UX layer only. All quantum logic lives in
quantum/qaoa_scheduler.py and quantum/qml_risk_classifier.py and is
treated as verified — the calls into those modules are unchanged.
"""

import html
import os
import sys
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow importing from the quantum/ package
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from quantum.qaoa_scheduler import run_qaoa_scheduler, decode_schedule, classical_baseline
from quantum.qml_risk_classifier import (
    compute_quantum_kernel,
    train_qsvm,
    train_classical_baseline,
    make_synthetic_data,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Windows compatibility shim (app layer only — quantum modules untouched)
# ---------------------------------------------------------------------------
# Aer's EstimatorV2/SamplerV2 execute each pub on a throwaway
# ThreadPoolExecutor thread (qiskit.primitives.primitive_job._submit). When
# the app script itself runs on a Streamlit worker thread, that pattern can
# trigger a native access violation inside qiskit's Rust circuit copy on
# Windows, killing the whole server process mid-demo. Every call site in this
# app waits for the job result immediately, so executing the job inline on
# the calling thread is behaviour-identical — which is what this patch does.
# It does not alter quantum/qaoa_scheduler.py or quantum/qml_risk_classifier.py
# in any way: same functions, same arguments, same results.
from concurrent.futures import Future

from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.providers import JobError


def _submit_inline(self):
    """Run the primitive job synchronously on the calling thread."""
    if self._future is not None:
        raise JobError("Primitive job has been submitted already.")
    future = Future()
    try:
        future.set_result(self._function(*self._args, **self._kwargs))
    except BaseException as exc:  # re-raised through the future by .result()
        future.set_exception(exc)
    self._future = future


PrimitiveJob._submit = _submit_inline

# ---------------------------------------------------------------------------
# Editable demo constants (tweak before presenting)
# ---------------------------------------------------------------------------
TEAM_NAME = "Team QStock"  # <- put your team name here
HACKATHON_NAME = "ADAPT IT Social Good Hackathon"
HACKATHON_TRACK = "Quantum Computing Track"
HACKATHON_ORGS = "IBM Research · Wits · SA QuTI"

# The MaxCut objective is symmetric in the early/late labels, so QAOA may return
# either mirror-image of the optimal cut. When True, the displayed schedule is
# oriented so the more-urgent half receives the early slots (display only —
# the objective/cut values are identical either way).
CANONICALIZE_SLOTS = True

# South-African-inspired palette: deep green + gold on a clean white base
GREEN_DARK = "#007749"
GREEN = "#009E60"
GREEN_SOFT = "#35A96B"
GOLD = "#FFB612"
GOLD_TEXT = "#8A6200"
RED = "#DE3831"
DONUT_PALETTE = ["#007749", "#009E60", "#39B54A", "#8CC63E", "#FFB612", "#E8A317", "#2F7D5B", "#B9D8A3"]

DEFAULT_MEMBERS = pd.DataFrame(
    {
        "name": ["Thabo", "Naledi", "Sipho", "Amahle", "Katlego", "Zanele"],
        "urgency": [0.9, 0.2, 0.7, 0.1, 0.8, 0.3],
        "contribution": [500, 750, 400, 600, 500, 850],
        "reliability": [0.92, 0.98, 0.74, 0.97, 0.68, 0.88],
    }
)

# Cosmetic labels for the synthetic members in the Risk Classifier dataset
SYNTH_NAMES = [
    "Thabo", "Naledi", "Sipho", "Amahle", "Katlego", "Zanele",
    "Lerato", "Themba", "Ayanda", "Bongani", "Refilwe", "Kagiso",
    "Palesa", "Sibusiso", "Nomvula", "Tshepo", "Unathi", "Vusi",
    "Yolanda", "Mandla",
]

# ---------------------------------------------------------------------------
# Inline SVG icon system (Feather-style shapes, MIT/ISC licensed, drawn inline
# so the app has zero CDN/network dependencies at judging time).
# One consistent grid for every icon: 24x24 viewBox, stroke-based,
# stroke-width 2, round caps/joins — and currentColor so icons always pick up
# the surrounding theme colour.
# ---------------------------------------------------------------------------
_ICON_BODY = {
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "alert-triangle": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "dollar-sign": '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "flag": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/>',
    "bar-chart-2": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "trending-up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "credit-card": '<rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/>',
    "landmark": '<line x1="3" y1="22" x2="21" y2="22"/><line x1="6" y1="18" x2="6" y2="11"/><line x1="10" y1="18" x2="10" y2="11"/><line x1="14" y1="18" x2="14" y2="11"/><line x1="18" y1="18" x2="18" y2="11"/><polygon points="12 2 20 7 4 7"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
    "play": '<polygon points="5 3 19 12 5 21 5 3"/>',
    "briefcase": '<rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
    "sun": '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>',
    "moon": '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "x": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "rotate-ccw": '<polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "arrow-up-right": '<line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/>',
    "arrow-down-right": '<line x1="7" y1="7" x2="17" y2="17"/><polyline points="17 7 17 17 7 17"/>',
    "minus": '<line x1="5" y1="12" x2="19" y2="12"/>',
}


def esc(text) -> str:
    """HTML-escape user-entered text before injecting it into custom markup."""
    return html.escape(str(text))


def icon(name: str, size: int = 16) -> str:
    """Inline monochrome SVG icon (consistent 24x24 grid, currentColor)."""
    return (
        f'<svg class="qs-i" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{_ICON_BODY[name]}</svg>'
    )


# Icons for the plain-text-only widget labels (tab bar + buttons), painted via
# CSS masks in _mask_css(). Panels are the last six div siblings inside stTabs:
# Members = 3rd panel, Payout Scheduler = 4th, Risk Classifier = 5th.
_BUTTON_MASKS = [
    ('div[role="tabpanel"]:nth-last-of-type(4) .stButton button', "rotate-ccw"),
    ('div[role="tabpanel"]:nth-last-of-type(3) .stButton button', "cpu"),
    ('div[role="tabpanel"]:nth-last-of-type(2) .stButton button', "shield"),
]
_TAB_MASKS = ["flag", "bar-chart-2", "users", "calendar", "shield", "dollar-sign"]


def _svg_mask_uri(name: str) -> str:
    """Encode an icon as a data-URI (for CSS mask-image)."""
    body = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{_ICON_BODY[name]}</svg>"
    )
    return "url('data:image/svg+xml;utf8," + quote(body, safe="") + "')"


def _mask_css() -> str:
    """CSS rules that paint monochrome icons via CSS masks for elements whose
    labels only accept plain text (tab labels and buttons). Mask + currentColor
    keeps each icon in sync with its label colour in every state."""
    rules = []
    for i, name in enumerate(_TAB_MASKS, start=1):
        rules.append(
            f'.stTabs [data-baseweb="tab"]:nth-of-type({i})::before {{ content: ""; '
            f"display: inline-block; width: 15px; height: 15px; margin-right: 7px; flex: none; "
            f"background-color: currentColor; "
            f"-webkit-mask: {_svg_mask_uri(name)} no-repeat center / contain; "
            f"mask: {_svg_mask_uri(name)} no-repeat center / contain; }}"
        )
    for selector, name in _BUTTON_MASKS:
        rules.append(
            f".stTabs {selector}::before {{ content: \"\"; "
            f"display: inline-block; width: 15px; height: 15px; margin-right: 8px; flex: none; "
            f"background-color: currentColor; "
            f"-webkit-mask: {_svg_mask_uri(name)} no-repeat center / contain; "
            f"mask: {_svg_mask_uri(name)} no-repeat center / contain; }}"
        )
    return "<style>" + "".join(rules) + "</style>"


def metric_card(icon_name: str, label: str, value: str, delta: str = "",
                delta_dir: int = 0, help_text: str = "") -> str:
    """st.metric-style card with an inline SVG icon (returns HTML for markdown)."""
    help_html = (
        f'<span class="qs-help" title="{esc(help_text)}">{icon("info", 13)}</span>'
        if help_text else ""
    )
    delta_html = ""
    if delta:
        cls = {1: "up", -1: "down"}.get(delta_dir, "flat")
        arrow = {1: "arrow-up-right", -1: "arrow-down-right"}.get(delta_dir, "minus")
        delta_html = f'<div class="qs-metric-delta {cls}">{icon(arrow, 14)}<span>{esc(delta)}</span></div>'
    return (
        f'<div class="qs-metric">'
        f'<div class="qs-metric-label">{icon(icon_name, 15)}<span>{esc(label)}</span>{help_html}</div>'
        f'<div class="qs-metric-value">{esc(value)}</div>{delta_html}</div>'
    )


def callout(icon_name: str, text: str, kind: str = "info"):
    """Icon + text callout box (replaces st.info where an icon is wanted)."""
    cls = "qs-callout warn" if kind == "warn" else "qs-callout"
    st.markdown(f'<div class="{cls}">{icon(icon_name, 17)}<div>{text}</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Brand styling (custom CSS on top of the .streamlit/config.toml theme)
# ---------------------------------------------------------------------------
QS_CSS = """
<style>
:root {
    --qs-green: #007749;
    --qs-green-mid: #009E60;
    --qs-green-soft: #35A96B;
    --qs-gold: #FFB612;
    --qs-gold-text: #8A6200;
    --qs-red: #DE3831;
    --qs-ink: #262626;
    --qs-muted: #5B6B62;
    --qs-card-border: #E2E9E4;
    --qs-font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

/* --- hide default Streamlit chrome --- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent;}
.stDeployButton, [data-testid="stDeployButton"], [data-testid="stDecoration"] {display: none;}

/* --- hero header --- */
.qs-hero {
    display: flex; align-items: center; gap: 20px;
    background: linear-gradient(120deg, var(--qs-green) 0%, var(--qs-green-mid) 75%, #23B573 100%);
    border-left: 8px solid var(--qs-gold);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 6px;
    box-shadow: 0 6px 18px rgba(0, 85, 55, 0.18);
}
.qs-hero-logo {
    flex: none;
    font-size: 42px; line-height: 1;
    width: 66px; height: 66px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(255, 255, 255, 0.14);
    border: 1.5px solid rgba(255, 255, 255, 0.35);
    border-radius: 16px;
}
.qs-hero-title {color: #ffffff; font-size: 2.15rem; font-weight: 800; letter-spacing: 0.3px; margin: 0; line-height: 1.15;}
.qs-hero-badge {
    display: inline-block; vertical-align: middle; margin-left: 12px;
    background: var(--qs-gold); color: #4A3300;
    font-size: 0.64rem; font-weight: 800; letter-spacing: 1.4px;
    padding: 4px 12px; border-radius: 999px; text-transform: uppercase;
}
.qs-hero-tag {color: rgba(255, 255, 255, 0.93); font-size: 1.02rem; margin: 6px 0 0 0;}

/* --- section headers --- */
.qs-section {display: flex; align-items: center; gap: 10px; margin: 2px 0 10px 0;}
.qs-section .qs-bar {flex: none; width: 6px; height: 24px; background: var(--qs-gold); border-radius: 3px;}
.qs-section .qs-title {font-size: 1.32rem; font-weight: 800; color: #0B3D2C; letter-spacing: 0.2px;}

/* --- chips / badges --- */
.chip {display: inline-block; padding: 4px 14px; margin: 3px 6px 3px 0; border-radius: 999px; font-weight: 700; font-size: 0.88rem; line-height: 1.5;}
.chip-early {background: #E3F4EA; color: var(--qs-green); border: 1.5px solid var(--qs-green-soft);}
.chip-late {background: #FFF4D6; color: var(--qs-gold-text); border: 1.5px solid var(--qs-gold);}
.chip-muted {background: #EEF1EE; color: #6B7280; border: 1.5px dashed #C7CDC7;}

/* --- cards --- */
.qs-card-title {font-size: 1.02rem; font-weight: 800; color: var(--qs-green); margin-bottom: 6px;}

/* --- global typography: offline-safe system stack (Inter first) --- */
.stApp, .stApp *:not(pre):not(code) {font-family: var(--qs-font) !important;}
/* ...but never override Streamlit's own bundled Material Symbols ligature font
   (expander chevrons, toolbar glyphs are ligatures of that font, not letters). */
.stApp [data-testid="stIconMaterial"] {font-family: "Material Symbols Rounded" !important;}

/* --- inline SVG icons --- */
.qs-i {vertical-align: -0.18em; flex: none;}

/* --- custom metric cards --- */
.qs-metric {
    background: #ffffff;
    border: 1px solid var(--qs-card-border);
    border-left: 4px solid var(--qs-green);
    border-radius: 12px;
    padding: 12px 16px 10px 16px;
    margin-bottom: 4px;
}
.qs-metric-label {
    display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
    font-size: 0.86rem; font-weight: 600; color: var(--qs-muted); line-height: 1.3;
}
.qs-metric-label .qs-i {color: var(--qs-green);}
.qs-metric-value {font-size: 2.1rem; font-weight: 600; color: var(--qs-ink); line-height: 1.2; margin-top: 2px;}
.qs-metric-delta {display: flex; align-items: center; gap: 4px; font-size: 0.84rem; margin-top: 2px;}
.qs-metric-delta.up {color: #0E7C46;}
.qs-metric-delta.down {color: var(--qs-red);}
.qs-metric-delta.flat {color: var(--qs-muted);}
.qs-help {display: inline-flex; cursor: help; color: #9AA6A0; margin-left: 2px;}

/* --- icon accents in titles / brand --- */
.qs-section .qs-title .qs-i {color: var(--qs-green); vertical-align: -3px;}
.qs-card-title .qs-i {color: var(--qs-green); vertical-align: -3px;}
.qs-hero-logo svg {color: #ffffff;}
.qs-side-icon svg {color: #ffffff;}
table.qs-grid th .qs-i {vertical-align: -2px; margin-right: 2px; color: var(--qs-muted);}

/* --- callout boxes --- */
.qs-callout {display: flex; align-items: flex-start; gap: 9px; border-radius: 12px;
    padding: 11px 15px; margin: 2px 0 10px 0; font-size: 0.92rem; line-height: 1.45;
    background: #EDF5F0; border: 1px solid #CFE5D8; border-left: 4px solid var(--qs-green); color: #1E4636;}
.qs-callout .qs-i {color: var(--qs-green); margin-top: 2px;}
.qs-callout.warn {background: #FFF8E6; border-color: #F2DFA8; border-left-color: var(--qs-gold); color: #5C4A12;}
.qs-callout.warn .qs-i {color: var(--qs-gold-text);}

/* --- icon lists (numbered + bulleted) --- */
.qs-ol, .qs-ul {list-style: none; padding-left: 0; margin: 4px 0 0 0;}
.qs-ol li, .qs-ul li {display: flex; align-items: flex-start; gap: 9px; margin: 6px 0; font-size: 0.92rem; color: var(--qs-ink);}
.qs-ol .qs-num {flex: none; width: 19px; height: 19px; margin-top: 1px; border-radius: 50%; background: #E3F4EA;
    color: var(--qs-green); font-size: 0.72rem; font-weight: 800; display: flex; align-items: center; justify-content: center;}
.qs-ul .qs-i {color: var(--qs-green); margin-top: 3px;}

/* --- early/late sub-headings --- */
.qs-sub {display: flex; align-items: center; gap: 7px; font-weight: 700; color: #0B3D2C; margin: 6px 0 2px 0; font-size: 0.95rem;}
.qs-sub.early .qs-i {color: var(--qs-green);}
.qs-sub.late .qs-i {color: var(--qs-gold-text);}

/* --- schedule comparison grid --- */
table.qs-grid {border-collapse: separate; border-spacing: 0 6px; width: 100%; font-size: 0.92rem;}
table.qs-grid th {text-align: left; color: var(--qs-muted); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.6px; padding: 2px 12px;}
table.qs-grid td {background: #F6F9F7; padding: 7px 12px; color: var(--qs-ink);}
table.qs-grid td:first-child {border-radius: 10px 0 0 10px; font-weight: 700;}
table.qs-grid td:last-child {border-radius: 0 10px 10px 0;}
table.qs-grid tr.qs-diff td {background: #FDF3F1;}
.slot-pill {display: inline-block; padding: 2px 12px; border-radius: 999px; font-weight: 700;}
.slot-pill.early {background: #E3F4EA; border: 1.5px solid var(--qs-green-soft); color: var(--qs-green);}
.slot-pill.late {background: #FFF4D6; border: 1.5px solid var(--qs-gold); color: var(--qs-gold-text);}
.grid-match-ok {color: var(--qs-green); font-weight: 800;}
.grid-match-no {color: var(--qs-red); font-weight: 800;}

/* --- sidebar brand --- */
.qs-side-brand {
    display: flex; align-items: center; gap: 12px;
    background: linear-gradient(120deg, var(--qs-green), var(--qs-green-mid));
    border-left: 5px solid var(--qs-gold);
    border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;
}
.qs-side-icon {flex: none; display: flex;}
.qs-side-name {color: #fff; font-weight: 800; font-size: 1.12rem; line-height: 1.15;}
.qs-side-sub {color: rgba(255, 255, 255, 0.85); font-size: 0.72rem;}

/* --- tabs --- */
.stTabs [data-baseweb="tab"] {font-weight: 600; display: inline-flex; align-items: center; justify-content: center;}
.stTabs [data-baseweb="tab-highlight"] {background-color: var(--qs-gold) !important;}
.stTabs [data-baseweb="tab"]:hover {color: var(--qs-green);}
</style>
"""

st.set_page_config(
    page_title="QStock — Quantum-Optimized Stokvel Management",
    page_icon="🪙",  # favicon only — browser chrome needs an emoji/char, not inline SVG
    layout="wide",
)
st.markdown(QS_CSS + _mask_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small render helpers
# ---------------------------------------------------------------------------
def section_header(icon_name: str, title: str):
    st.markdown(
        f'<div class="qs-section"><span class="qs-bar"></span>'
        f'<span class="qs-title">{icon(icon_name, 21)} {esc(title)}</span></div>',
        unsafe_allow_html=True,
    )


def card_title(icon_name: str, title: str):
    st.markdown(
        f'<div class="qs-card-title">{icon(icon_name, 17)} {esc(title)}</div>',
        unsafe_allow_html=True,
    )


def chips_html(names, kind: str) -> str:
    """Render a list of names as colored chips ('early' or 'late')."""
    if not names:
        return '<span class="chip chip-muted">none</span>'
    return " ".join(f'<span class="chip chip-{kind}">{esc(n)}</span>' for n in names)


def empty_members_notice():
    callout(
        "users",
        "No members yet — add or edit members in the <b>Members</b> tab. "
        "Everything else on this page updates instantly.",
    )


def synthetic_member_names(n: int) -> list:
    names = []
    for i in range(n):
        names.append(SYNTH_NAMES[i] if i < len(SYNTH_NAMES) else f"Member {i + 1:02d}")
    return names


# ---------------------------------------------------------------------------
# App-layer metrics (pure numpy on top of the returned quantum results —
# no changes to the quantum modules themselves)
# ---------------------------------------------------------------------------
def cut_value(urgencies, early_flags) -> float:
    """Total urgency-difference across members placed in different payout
    slots — the MaxCut objective. Higher = a sharper early/late split."""
    u = np.asarray(urgencies, dtype=float)
    e = np.asarray(early_flags)
    ii, jj = np.triu_indices(len(u), k=1)
    w = np.abs(u[ii] - u[jj])
    return float(w[e[ii] != e[jj]].sum())


def max_cut_value(urgencies, cap: int = 12):
    """Brute-force the best achievable cut for small groups (judge metric)."""
    n = len(urgencies)
    if n == 0 or n > cap:
        return None
    u = np.asarray(urgencies, dtype=float)
    ii, jj = np.triu_indices(n, k=1)
    w = np.abs(u[ii] - u[jj])
    best = 0.0
    for mask in range(1 << n):
        bits = np.array([(mask >> k) & 1 for k in range(n)])
        cut = w[bits[ii] != bits[jj]].sum()
        if cut > best:
            best = float(cut)
    return best


# ---------------------------------------------------------------------------
# Chart builders (plotly)
# ---------------------------------------------------------------------------
def urgency_bar_chart(df: pd.DataFrame):
    d = df.sort_values("urgency", ascending=False)
    fig = go.Figure(
        go.Bar(
            x=d["name"],
            y=d["urgency"],
            marker=dict(
                color=d["urgency"],
                colorscale=[[0.0, "#FFD666"], [1.0, GREEN_DARK]],
                cmin=0,
                cmax=1,
                showscale=False,
            ),
            text=[f"{u:.2f}" for u in d["urgency"]],
            textposition="outside",
            hovertemplate="%{x}: urgency %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=10, r=10, t=42, b=10),
        title=dict(text="Member urgency — who needs an early payout", font=dict(size=14)),
        yaxis=dict(range=[0, 1.08], title="Urgency (0–1)"),
        font=dict(family="Inter, Segoe UI, Roboto, Helvetica, sans-serif", size=13, color="#262626"),
    )
    return fig


def contribution_donut(df: pd.DataFrame):
    d = df.sort_values("contribution", ascending=False)
    colors = [DONUT_PALETTE[i % len(DONUT_PALETTE)] for i in range(len(d))]
    total = float(d["contribution"].sum())
    fig = go.Figure(
        go.Pie(
            labels=d["name"],
            values=d["contribution"],
            hole=0.58,
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            textinfo="label+percent",
            textposition="outside",
            hovertemplate="%{label}: R%{value:,.0f} (%{percent})<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"<b>R{total:,.0f}</b><br><span style='font-size:11px;color:#6B7280'>per month</span>",
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=17, color="#262626"),
    )
    fig.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=10, r=10, t=42, b=10),
        title=dict(text="Monthly contribution split", font=dict(size=14)),
        showlegend=False,
        font=dict(family="Inter, Segoe UI, Roboto, Helvetica, sans-serif", size=13, color="#262626"),
    )
    return fig


def risk_bar_chart(names, probs):
    order = np.argsort(probs)[::-1]
    names_s = [names[i] for i in order]
    probs_s = [probs[i] for i in order]
    flagged = [p > 0.5 for p in probs_s]
    x_risk = [p if f else None for p, f in zip(probs_s, flagged)]
    x_ok = [p if not f else None for p, f in zip(probs_s, flagged)]

    fig = go.Figure()
    fig.add_bar(
        y=names_s, x=x_risk, orientation="h", name="Flagged at risk",
        marker_color=RED,
        text=[f"{p:.0%}" if p is not None else "" for p in x_risk],
        textposition="outside",
        hovertemplate="%{y}: risk %{x:.0%}<extra></extra>",
    )
    fig.add_bar(
        y=names_s, x=x_ok, orientation="h", name="On track",
        marker_color=GREEN_SOFT,
        text=[f"{p:.0%}" if p is not None else "" for p in x_ok],
        textposition="outside",
        hovertemplate="%{y}: risk %{x:.0%}<extra></extra>",
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#9CA3AF")
    fig.update_layout(
        template="plotly_white",
        barmode="overlay",
        height=max(340, 26 * len(names_s) + 90),
        margin=dict(l=10, r=20, t=42, b=10),
        title=dict(text="Estimated risk of missing a contribution (highest first)", font=dict(size=14)),
        xaxis=dict(range=[0, 1.1], title="Risk score"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, Segoe UI, Roboto, Helvetica, sans-serif", size=13, color="#262626"),
    )
    return fig


def funds_projection_chart(months, cum_fee, cum_interest, balance):
    fig = go.Figure()
    fig.add_scatter(
        x=months, y=cum_fee, name="Platform fee revenue (cum.)",
        mode="lines+markers", line=dict(color=GREEN_DARK, width=3),
        hovertemplate="Month %{x}: R%{y:,.0f}<extra></extra>",
    )
    fig.add_scatter(
        x=months, y=cum_interest, name="Interest earned (cum.)",
        mode="lines+markers", line=dict(color=GOLD, width=3),
        hovertemplate="Month %{x}: R%{y:,.0f}<extra></extra>",
    )
    fig.add_scatter(
        x=months, y=balance, name="Pooled balance",
        mode="lines", line=dict(color="#6B7280", width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="Month %{x}: R%{y:,.0f}<extra></extra>",
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=42, b=10),
        title=dict(text="Illustrative revenue & balance projection", font=dict(size=14)),
        xaxis=dict(title="Month", dtick=1),
        yaxis=dict(title="Revenue (R)"),
        yaxis2=dict(title="Balance (R)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="Inter, Segoe UI, Roboto, Helvetica, sans-serif", size=13, color="#262626"),
    )
    return fig


def schedule_grid_html(members: pd.DataFrame, qaoa_early, classical_early) -> str:
    """Colored grid: which members landed in early vs late slots, QAOA vs greedy."""
    rows = []
    for _, r in members.iterrows():
        q = "early" if r["name"] in qaoa_early else "late"
        c = "early" if r["name"] in classical_early else "late"
        same = q == c
        rows.append(
            f'<tr class="{"" if same else "qs-diff"}">'
            f"<td>{esc(r['name'])}</td><td>{r['urgency']:.2f}</td>"
            f'<td><span class="slot-pill {q}">{"Early" if q == "early" else "Late"}</span></td>'
            f'<td><span class="slot-pill {c}">{"Early" if c == "early" else "Late"}</span></td>'
            f'<td><span class="{"grid-match-ok" if same else "grid-match-no"}">{icon("check", 14) if same else icon("x", 14)}</span></td>'
            "</tr>"
        )
    return (
        '<table class="qs-grid"><tr><th>Member</th><th>Urgency</th>'
        f'<th>{icon("cpu", 13)} QAOA slot</th><th>{icon("sliders", 13)} Classical greedy</th><th>Match</th></tr>'
        + "".join(rows)
        + "</table>"
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "members_df" not in st.session_state:
    st.session_state["members_df"] = DEFAULT_MEMBERS.copy()

# ---------------------------------------------------------------------------
# Sidebar — context for judges without scrolling the main page
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div class="qs-side-brand"><span class="qs-side-icon">{icon("users", 22)}</span><span>'
        '<span class="qs-side-name">QStock</span><br>'
        '<span class="qs-side-sub">Quantum-optimized stokvel management</span></span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**{esc(TEAM_NAME)}**")
    st.caption(f"{esc(HACKATHON_NAME)} — {esc(HACKATHON_TRACK)}")
    st.caption(f"Supported by {esc(HACKATHON_ORGS)}")

    st.divider()
    st.markdown("**About QStock**")
    st.write(
        "QStock brings bank-grade structure to South African stokvels: digital "
        "contribution tracking, pooled funds held with a partner bank, "
        "quantum-optimized payout schedules (QAOA) and quantum kernel risk "
        "flags (QSVM) — without losing the community heart that makes "
        "stokvels work."
    )

    st.divider()
    st.markdown("**Demo flow**")
    st.markdown(
        '<ul class="qs-ol">'
        '<li><span class="qs-num">1</span><span>Add or edit members</span></li>'
        '<li><span class="qs-num">2</span><span>Run the QAOA scheduler</span></li>'
        '<li><span class="qs-num">3</span><span>Run the risk classifier</span></li>'
        '<li><span class="qs-num">4</span><span>See the funds projection</span></li>'
        "</ul>",
        unsafe_allow_html=True,
    )
    st.caption("All computations run locally on Qiskit Aer simulators.")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="qs-hero">
        <div class="qs-hero-logo">{icon("users", 34)}</div>
        <div>
            <h1 class="qs-hero-title">QStock<span class="qs-hero-badge">Quantum</span></h1>
            <p class="qs-hero-tag">Quantum-optimized stokvel management — fair payouts, smarter risk tracking.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs (Members tab code runs first so its edited data is available everywhere)
# ---------------------------------------------------------------------------
TAB_PITCH, TAB_OVERVIEW, TAB_MEMBERS, TAB_SCHED, TAB_RISK, TAB_FUNDS = st.tabs(
    [
        "Pitch Summary",
        "Overview",
        "Members",
        "Payout Scheduler",
        "Risk Classifier",
        "Pooled Funds",
    ]
)

# ---------------- Members Tab ----------------
with TAB_MEMBERS:
    section_header("users", "Members")
    st.caption(
        "Edit the roster below — add or delete rows as needed. **Urgency** (0–1, higher = needs "
        "an earlier payout) drives the QAOA scheduler, **monthly contribution** drives the pooled "
        "funds, and **reliability** (payment history) is shown for committee context."
    )

    # Reset handling must happen before the editor is instantiated
    if st.session_state.pop("reset_members", False):
        st.session_state.pop("members_editor", None)
        st.session_state["members_df"] = DEFAULT_MEMBERS.copy()

    edited = st.data_editor(
        st.session_state["members_df"],
        num_rows="dynamic",
        key="members_editor",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Name", help="Member's display name"),
            "urgency": st.column_config.NumberColumn(
                "Urgency (0–1)", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"
            ),
            "contribution": st.column_config.NumberColumn(
                "Monthly contribution (R)", min_value=0, step=50, format="%.0f"
            ),
            "reliability": st.column_config.NumberColumn(
                "Reliability (0–1)", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"
            ),
        },
    )
    st.session_state["members_df"] = edited.copy()

    # Cleaned view of the roster used by every other tab
    members = edited.copy()
    members["name"] = members["name"].astype(str).str.strip()
    members = members[(members["name"] != "") & (members["name"].str.lower() != "nan")]
    members["urgency"] = pd.to_numeric(members["urgency"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    members["contribution"] = pd.to_numeric(members["contribution"], errors="coerce").fillna(0)
    members["reliability"] = pd.to_numeric(members["reliability"], errors="coerce")
    members = members.reset_index(drop=True)

    col_info, col_reset = st.columns([3, 1])
    with col_info:
        if len(members):
            top = members.loc[members["urgency"].idxmax()]
            st.caption(
                f"{len(members)} members · R{members['contribution'].sum():,.0f} pooled per month · "
                f"most urgent: **{esc(top['name'])}** ({top['urgency']:.2f})"
            )
        else:
            st.caption("The roster is empty — add at least two members to run the scheduler.")
    with col_reset:
        if st.button("Restore demo members", use_container_width=True, key="reset_members_btn"):
            st.session_state["reset_members"] = True
            st.rerun()

# Drop stale scheduler results if the underlying member data changed
if "qaoa_run" in st.session_state:
    current_input = members[["name", "urgency"]].reset_index(drop=True)
    if not st.session_state["qaoa_run"]["input"].equals(current_input):
        del st.session_state["qaoa_run"]

# ---------------- Pitch Summary Tab ----------------
with TAB_PITCH:
    section_header("flag", "Pitch Summary")

    qaoa_run = st.session_state.get("qaoa_run")
    qsvm_run = st.session_state.get("qsvm_run")

    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(metric_card("users", "Members", f"{len(members)}"), unsafe_allow_html=True)
    p2.markdown(
        metric_card("dollar-sign", "Pooled / month", f"R{members['contribution'].sum():,.0f}"),
        unsafe_allow_html=True,
    )
    if qaoa_run:
        p3.markdown(
            metric_card(
                "cpu",
                "QAOA cost",
                f"{qaoa_run['optimal_cost']:+.3f}",
                help_text="Last run's optimized expectation value of the cost Hamiltonian (⟨H⟩, lower = better split).",
            ),
            unsafe_allow_html=True,
        )
    else:
        p3.markdown(
            metric_card("cpu", "QAOA cost", "—", help_text="Run the Payout Scheduler tab to populate."),
            unsafe_allow_html=True,
        )
    if qsvm_run:
        delta_pts = (qsvm_run["qsvm_accuracy"] - qsvm_run["classical_accuracy"]) * 100
        p4.markdown(
            metric_card(
                "activity",
                "QSVM accuracy",
                f"{qsvm_run['qsvm_accuracy']:.0%}",
                delta=f"{delta_pts:+.0f} pts vs classical",
                delta_dir=1 if delta_pts > 0 else (-1 if delta_pts < 0 else 0),
            ),
            unsafe_allow_html=True,
        )
    else:
        p4.markdown(
            metric_card("activity", "QSVM accuracy", "—", help_text="Run the Risk Classifier tab to populate."),
            unsafe_allow_html=True,
        )

    if not qaoa_run or not qsvm_run:
        callout("zap", "Run the <b>Payout Scheduler</b> and <b>Risk Classifier</b> tabs to light up the quantum metrics above.")

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        card_title("globe", "The societal problem")
        st.write(
            "Stokvels pool the savings of an estimated 11 million South Africans — roughly "
            "R50 billion a year (industry estimates) — yet most still run on notebooks and "
            "WhatsApp threads. Payout order is decided by memory, contributions by trust, and "
            "nobody sees a shortfall coming until it becomes a dispute. QStock gives these groups "
            "the structure and auditability of a bank without losing the community spirit that "
            "makes stokvels work."
        )
    with c2.container(border=True):
        card_title("cpu", "The technical problem")
        st.write(
            "At QStock's core is a combinatorial optimization problem: assign N members to payout "
            "positions while balancing urgency, fairness and risk. The search space grows "
            "exponentially, so we map it to a QUBO (MaxCut-style cost Hamiltonian) and solve it "
            "with **QAOA**, benchmarked against a classical greedy rank-split. Separately, "
            "predicting who will miss a contribution is a small-data classification problem over "
            "behavioural features — solved with a **quantum kernel SVM** (ZZ feature map + "
            "state-fidelity kernel) and benchmarked against a classical RBF-SVM."
        )

    c3, c4 = st.columns(2)
    with c3.container(border=True):
        card_title("play", "How this demo works")
        st.markdown(
            '<ul class="qs-ol">'
            f'<li><span class="qs-num">1</span><span>{icon("users", 15)} <b>Members</b> — edit the roster (urgency drives the scheduler)</span></li>'
            f'<li><span class="qs-num">2</span><span>{icon("calendar", 15)} <b>Payout Scheduler</b> — run QAOA, compare with the greedy baseline</span></li>'
            f'<li><span class="qs-num">3</span><span>{icon("shield", 15)} <b>Risk Classifier</b> — run the QSVM on synthetic behaviour data</span></li>'
            f'<li><span class="qs-num">4</span><span>{icon("dollar-sign", 15)} <b>Pooled Funds</b> — explore the revenue projection</span></li>'
            "</ul>",
            unsafe_allow_html=True,
        )
        st.caption("The metrics at the top of this page update live as you run each tab.")
    with c4.container(border=True):
        card_title("briefcase", "Business model")
        st.markdown(
            '<ul class="qs-ul">'
            f'<li>{icon("credit-card", 15)}<span>Small flat <b>monthly platform fee</b> per group</span></li>'
            f'<li>{icon("landmark", 15)}<span><b>Interest-share partnership</b> with the banking partner on pooled funds</span></li>'
            f'<li>{icon("star", 15)}<span><b>Premium features</b>: credit-building reports, loan facilitation for members in good standing</span></li>'
            "</ul>",
            unsafe_allow_html=True,
        )

    st.caption("All results computed locally with Qiskit Aer simulators — no quantum hardware required for the demo.")

# ---------------- Overview Tab ----------------
with TAB_OVERVIEW:
    section_header("bar-chart-2", "Group Overview")

    if members.empty:
        empty_members_notice()
    else:
        avg_reliability = members["reliability"].mean()
        top = members.loc[members["urgency"].idxmax()]
        high_urgency = int((members["urgency"] >= 0.7).sum())
        o1, o2, o3, o4 = st.columns(4)
        o1.markdown(metric_card("users", "Members", f"{len(members)}"), unsafe_allow_html=True)
        o2.markdown(
            metric_card("dollar-sign", "Pooled / month", f"R{members['contribution'].sum():,.0f}"),
            unsafe_allow_html=True,
        )
        o3.markdown(
            metric_card(
                "zap",
                "High urgency",
                f"{high_urgency}",
                help_text=f"Members with urgency ≥ 0.7 (most urgent: {top['name']} at {top['urgency']:.2f}).",
            ),
            unsafe_allow_html=True,
        )
        o4.markdown(
            metric_card("shield", "Avg reliability", f"{avg_reliability:.0%}" if pd.notna(avg_reliability) else "—"),
            unsafe_allow_html=True,
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(urgency_bar_chart(members), use_container_width=True)
        with chart_col2:
            st.plotly_chart(contribution_donut(members), use_container_width=True)

        st.markdown("**Member details**")
        st.dataframe(
            members,
            use_container_width=True,
            hide_index=True,
            column_config={
                "name": st.column_config.TextColumn("Member"),
                "urgency": st.column_config.ProgressColumn(
                    "Urgency", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "contribution": st.column_config.NumberColumn(
                    "Monthly contribution (R)", format="%.0f"
                ),
                "reliability": st.column_config.ProgressColumn(
                    "Reliability", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )

# ---------------- Payout Scheduler Tab ----------------
with TAB_SCHED:
    section_header("calendar", "Fair Payout Rotation — QAOA")
    st.caption(
        "Member urgency is mapped onto a MaxCut-style QUBO cost Hamiltonian, optimized with QAOA "
        "(Qiskit + local Aer simulator), and the best bitstring is decoded into an early/late "
        "payout split — benchmarked against a classical greedy rank-split. Uses the members from "
        "the Members tab."
    )

    if members.empty:
        empty_members_notice()
    else:
        can_run = len(members) >= 2
        if st.button("Run QAOA Scheduler", type="primary", disabled=not can_run, key="run_qaoa_btn"):
            if members["urgency"].nunique() <= 1:
                st.warning(
                    "All urgency scores are equal — vary them in the Members tab so the "
                    "scheduler has a meaningful problem to optimize."
                )
            else:
                names = members["name"].tolist()
                urgencies = members["urgency"].tolist()
                try:
                    with st.spinner(
                        "Running QAOA optimization (8 COBYLA restarts on a local Aer "
                        "simulator — this can take a minute or two)…"
                    ):
                        t0 = time.time()
                        result = run_qaoa_scheduler(urgencies, reps=3)
                        schedule = decode_schedule(result, names)
                        baseline = classical_baseline(urgencies, names)
                        elapsed = time.time() - t0

                    # App-layer bookkeeping on the returned results
                    qaoa_flags = np.array([int(b) for b in reversed(schedule["bitstring"])])
                    ranked = sorted(range(len(urgencies)), key=lambda i: -urgencies[i])
                    greedy_flags = np.zeros(len(urgencies), dtype=int)
                    greedy_flags[ranked[: len(ranked) // 2]] = 1

                    # Display-only orientation of the symmetric cut (see
                    # CANONICALIZE_SLOTS above) — cut values are unaffected.
                    if (
                        CANONICALIZE_SLOTS
                        and schedule["early_payout"]
                        and schedule["late_payout"]
                    ):
                        urg_by_name = dict(zip(names, urgencies))
                        mean_early = float(np.mean([urg_by_name[n] for n in schedule["early_payout"]]))
                        mean_late = float(np.mean([urg_by_name[n] for n in schedule["late_payout"]]))
                        if mean_early < mean_late:
                            schedule = {
                                **schedule,
                                "early_payout": schedule["late_payout"],
                                "late_payout": schedule["early_payout"],
                            }

                    st.session_state["qaoa_run"] = {
                        "input": members[["name", "urgency"]].reset_index(drop=True),
                        "schedule": schedule,
                        "baseline": baseline,
                        "optimal_cost": result["optimal_cost"],
                        "restart_costs": result["restart_costs"],
                        "cut_qaoa": cut_value(urgencies, qaoa_flags),
                        "cut_greedy": cut_value(urgencies, greedy_flags),
                        "cut_max": max_cut_value(urgencies),
                        "seconds": elapsed,
                    }
                except Exception as exc:  # clean error instead of a raw traceback
                    st.error(f"The QAOA scheduler failed: {exc}")
                    st.info(
                        "Tip: check that the members table has valid names and varied urgency "
                        "scores (0–1), then try again."
                    )

        stored = st.session_state.get("qaoa_run")
        if stored:
            q_cut, g_cut, m_cut = stored["cut_qaoa"], stored["cut_greedy"], stored["cut_max"]
            s1, s2, s3, s4 = st.columns(4)
            s1.markdown(
                metric_card(
                    "cpu",
                    "QAOA cut",
                    f"{q_cut:.2f}",
                    delta=f"{q_cut - g_cut:+.2f} vs greedy" if q_cut != g_cut else "= greedy",
                    delta_dir=1 if q_cut > g_cut else (-1 if q_cut < g_cut else 0),
                    help_text="Total urgency-difference between members placed in different payout slots — the QAOA objective. Higher = a fairer split.",
                ),
                unsafe_allow_html=True,
            )
            s2.markdown(
                metric_card("sliders", "Greedy cut", f"{g_cut:.2f}", help_text="Classical rank-split baseline on the same objective."),
                unsafe_allow_html=True,
            )
            s3.markdown(
                metric_card("target", "Best cut", f"{m_cut:.2f}" if m_cut is not None else "n/a",
                            help_text="Brute-force optimum (computed for groups of up to 12 members)."),
                unsafe_allow_html=True,
            )
            s4.markdown(
                metric_card("clock", "Runtime", f"{stored['seconds']:.0f}s",
                            help_text=f"{len(stored['restart_costs'])} COBYLA restarts on the local Aer simulator."),
                unsafe_allow_html=True,
            )

            cA, cB = st.columns(2)
            with cA.container(border=True):
                card_title("cpu", "QAOA schedule")
                st.markdown(f'<div class="qs-sub early">{icon("sun", 15)}Early payout</div>', unsafe_allow_html=True)
                st.markdown(chips_html(stored["schedule"]["early_payout"], "early"), unsafe_allow_html=True)
                st.markdown(f'<div class="qs-sub late">{icon("moon", 15)}Late payout</div>', unsafe_allow_html=True)
                st.markdown(chips_html(stored["schedule"]["late_payout"], "late"), unsafe_allow_html=True)
            with cB.container(border=True):
                card_title("sliders", "Classical baseline (rank-split)")
                st.markdown(f'<div class="qs-sub early">{icon("sun", 15)}Early payout</div>', unsafe_allow_html=True)
                st.markdown(chips_html(stored["baseline"]["early_payout"], "early"), unsafe_allow_html=True)
                st.markdown(f'<div class="qs-sub late">{icon("moon", 15)}Late payout</div>', unsafe_allow_html=True)
                st.markdown(chips_html(stored["baseline"]["late_payout"], "late"), unsafe_allow_html=True)

            st.markdown("**Side-by-side slot assignment**")
            st.markdown(
                schedule_grid_html(members, stored["schedule"]["early_payout"], stored["baseline"]["early_payout"]),
                unsafe_allow_html=True,
            )
            st.caption(
                "Green pill = early payout slot · gold pill = late slot · highlighted rows are "
                "members the two methods place differently."
                + (
                    " The MaxCut objective treats the early/late labels symmetrically — QAOA may "
                    "return either mirror-image of the optimal cut, so the schedule is oriented "
                    "to give the more-urgent half the early slots (the cut value is identical "
                    "either way)."
                    if CANONICALIZE_SLOTS
                    else ""
                )
            )

            with st.expander("Under the hood — Qiskit Pattern & optimization run"):
                st.markdown(
                    "1. **MAP** — member urgencies → MaxCut-style cost Hamiltonian "
                    "H = Σ wᵢⱼ ZᵢZⱼ, with wᵢⱼ = |urgencyᵢ − urgencyⱼ|\n"
                    "2. **OPTIMIZE** — QAOA ansatz (reps=3), transpiled for the Aer simulator\n"
                    "3. **EXECUTE** — variational loop with COBYLA, restarted from 8 random "
                    "parameter sets to escape local minima\n"
                    "4. **POST-PROCESS** — sample the optimized circuit and decode the most "
                    "frequent bitstring into the schedule"
                )
                st.markdown("**QAOA restart costs** (each restart = a fresh random COBYLA start; lower is better)")
                st.line_chart(pd.DataFrame({"restart cost": stored["restart_costs"]}))
        elif can_run:
            callout("calendar", "Run the scheduler above to generate a QAOA payout schedule and compare it with the classical baseline.")

# ---------------- Risk Classifier Tab ----------------
with TAB_RISK:
    section_header("shield", "Member Risk Classification — Quantum Kernel SVM")
    st.caption(
        "Behavioural features (consistency, average days late, tenure) are encoded with a ZZ "
        "feature map; the pairwise state-fidelity kernel is fed to a classical SVM (QSVM) to flag "
        "members at risk of missing a contribution — benchmarked against a classical RBF-SVM. "
        "Synthetic member data for this demo; swap in real (anonymised) data in production."
    )

    n_samples = st.slider("Synthetic members in dataset", 10, 40, 20, key="n_samples_slider")

    if st.button("Run Risk Classifier", type="primary", key="run_risk_btn"):
        try:
            with st.spinner("Computing quantum kernels and training both classifiers…"):
                t0 = time.time()
                X, y = make_synthetic_data(n_samples=n_samples)
                # Small range keeps the ZZ kernel in the smooth, unwrapped regime
                # (see qml_risk_classifier.py); (0, pi) collapses all fidelities.
                X_scaled = MinMaxScaler(feature_range=(0, 0.25)).fit_transform(X)
                # (an index array is passed along purely to label the chart below;
                # the split itself is unchanged)
                X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                    X_scaled, y, np.arange(len(y)), test_size=0.3, random_state=42, stratify=y
                )

                qsvm_result = train_qsvm(X_train, y_train, X_test, y_test)
                baseline_result = train_classical_baseline(X_train, y_train, X_test, y_test)

                # App-layer extra: per-member risk scores from the trained QSVM
                combined = np.vstack([X_train, X_test])
                combined_names = [
                    synthetic_member_names(n_samples)[i] for i in list(idx_train) + list(idx_test)
                ]
                full_kernel = compute_quantum_kernel(combined)
                scores = qsvm_result["model"].decision_function(full_kernel[:, : len(X_train)])
                probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
                elapsed = time.time() - t0

            st.session_state["qsvm_run"] = {
                "n": n_samples,
                "names": combined_names,
                "probs": probs.tolist(),
                "qsvm_accuracy": qsvm_result["accuracy"],
                "classical_accuracy": baseline_result["accuracy"],
                "qsvm_report": qsvm_result["report"],
                "classical_report": baseline_result["report"],
                "seconds": elapsed,
            }
        except Exception as exc:  # clean error instead of a raw traceback
            st.error(f"The risk classifier failed: {exc}")
            st.info("Tip: try a different dataset size, then run again.")

    stored = st.session_state.get("qsvm_run")
    if stored:
        q_acc, c_acc = stored["qsvm_accuracy"], stored["classical_accuracy"]
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(
            metric_card(
                "activity",
                "QSVM acc.",
                f"{q_acc:.0%}",
                delta=f"{(q_acc - c_acc) * 100:+.0f} pts",
                delta_dir=1 if q_acc > c_acc else (-1 if q_acc < c_acc else 0),
                help_text="Accuracy on the held-out test set (delta = points vs the classical RBF-SVM baseline).",
            ),
            unsafe_allow_html=True,
        )
        r2.markdown(
            metric_card("sliders", "Classical SVM", f"{c_acc:.0%}", help_text="RBF-kernel SVM on the same features and split."),
            unsafe_allow_html=True,
        )
        r3.markdown(
            metric_card("users", "Scored", f"{stored['n']}", help_text="Synthetic members scored by the QSVM."),
            unsafe_allow_html=True,
        )
        r4.markdown(
            metric_card("clock", "Run time", f"{stored['seconds']:.1f}s",
                        help_text="Quantum kernel computation + training of both classifiers."),
            unsafe_allow_html=True,
        )

        st.plotly_chart(risk_bar_chart(stored["names"], stored["probs"]), use_container_width=True)
        st.caption(
            "Risk score = sigmoid of the QSVM decision value on the quantum kernel (a demo "
            "estimate over all synthetic members; the accuracy metrics above are held-out "
            "test-set). Bars above the dashed 50% line are flagged at risk."
        )

        with st.expander("Classification reports"):
            rep1, rep2 = st.columns(2)
            with rep1:
                st.markdown("**QSVM (quantum kernel)**")
                st.text(stored["qsvm_report"])
            with rep2:
                st.markdown("**Classical RBF-SVM baseline**")
                st.text(stored["classical_report"])
    else:
        callout("shield", "Run the classifier above to score members and compare the quantum kernel SVM with the classical baseline.")

# ---------------- Pooled Funds Tab ----------------
with TAB_FUNDS:
    section_header("dollar-sign", "Pooled Funds & Revenue Projection")
    st.caption(
        "Pooled monthly contributions are held with the banking partner. Explore how fee and "
        "interest revenue could grow if contributions stayed constant."
    )

    if members.empty:
        empty_members_notice()
    else:
        f1, f2, f3 = st.columns(3)
        monthly_fee = f1.number_input("Monthly platform fee per group (R)", 0, 500, 50, step=5, key="fee_input")
        interest_rate = f2.number_input("Interest on pooled funds (% p.a.)", 0.0, 15.0, 5.0, 0.5, key="rate_input")
        horizon = f3.slider("Projection horizon (months)", 3, 24, 12, key="horizon_slider")

        monthly_total = float(members["contribution"].sum())
        months = np.arange(1, horizon + 1)
        cum_fee, cum_interest, balances = [], [], []
        balance, fee_sum, interest_sum = 0.0, 0.0, 0.0
        for _ in months:
            balance += monthly_total
            interest = balance * interest_rate / 100.0 / 12.0
            balance += interest
            fee_sum += monthly_fee
            interest_sum += interest
            cum_fee.append(fee_sum)
            cum_interest.append(interest_sum)
            balances.append(balance)

        g1, g2, g3, g4 = st.columns(4)
        g1.markdown(
            metric_card("dollar-sign", "Pooled / month", f"R{monthly_total:,.0f}"),
            unsafe_allow_html=True,
        )
        g2.markdown(
            metric_card("landmark", f"Balance (mo {horizon})", f"R{balances[-1]:,.0f}",
                        help_text="Projected pooled balance at the end of the horizon."),
            unsafe_allow_html=True,
        )
        g3.markdown(
            metric_card("credit-card", "Fee revenue", f"R{cum_fee[-1]:,.0f}",
                        help_text=f"Cumulative platform-fee revenue by month {horizon}."),
            unsafe_allow_html=True,
        )
        g4.markdown(
            metric_card("trending-up", "Interest earned", f"R{cum_interest[-1]:,.0f}",
                        help_text=f"Cumulative interest earned on the pooled balance by month {horizon}."),
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            funds_projection_chart(months, cum_fee, cum_interest, balances),
            use_container_width=True,
        )
        callout(
            "alert-triangle",
            "<b>Illustrative projection only</b> — assumes constant contributions, no payouts or "
            "withdrawals, a flat monthly platform fee, and simple monthly interest on the pooled "
            "balance. Real stokvel cycles rotate payouts out of the pool; a partner-share of "
            "interest would be part of QStock's revenue.",
            kind="warn",
        )

# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "QStock — built for the ADAPT IT Social Good Hackathon, Quantum Computing Track · "
    "Powered by Qiskit (Aer) + Streamlit"
)
