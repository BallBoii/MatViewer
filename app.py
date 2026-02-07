"""
MatViewer — Scientific Image Gallery
=====================================
Interactive 3-D `.mat` file viewer built with Streamlit + Plotly.
Minimal / modern layout with full matplotlib-level adjustment tools.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.io as sio
import streamlit as st

# ─── Theme Tokens ─────────────────────────────────────────────────────────────
BG_PRIMARY = "#0E1117"
BG_CARD = "#161B22"
BORDER = "#30363d"
ACCENT = "#00F0FF"
ACCENT2 = "#FF2E97"
TEXT = "#E6EDF3"
TEXT_DIM = "#8B949E"

COLOR_SCALES: list[str] = [
    "Viridis", "Plasma", "Magma", "Inferno", "Cividis",
    "Gray", "Hot", "Jet", "Turbo", "Ice", "Electric",
]

INTERPOLATION_METHODS = ["none", "best", "fast"]


# ─── CSS ──────────────────────────────────────────────────────────────────────
def local_css() -> None:
    """Inject custom CSS to achieve a deep-dark / cyberpunk aesthetic."""
    st.markdown(
        f"""<style>
        /* remove red header */
        header[data-testid="stHeader"] {{ background: transparent !important; }}

        .stApp, [data-testid="stAppViewContainer"] {{
            background: {BG_PRIMARY} !important; color: {TEXT} !important;
        }}

        /* sidebar */
        section[data-testid="stSidebar"] {{ background: {BG_CARD} !important; }}
        section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

        /* expander headers */
        details summary span {{ font-size: 0.85rem !important; }}

        /* file uploader */
        [data-testid="stFileUploader"] {{
            border: 1.5px dashed {BORDER} !important;
            border-radius: 8px; padding: 0.5rem;
        }}
        [data-testid="stFileUploader"]:hover {{
            border-color: {ACCENT} !important;
        }}

        /* compact stat pill */
        .stat-row {{
            display: flex; gap: 0.6rem; flex-wrap: wrap;
            margin-bottom: 0.5rem;
        }}
        .stat-pill {{
            background: {BG_CARD}; border: 1px solid {BORDER};
            border-radius: 6px; padding: 0.35rem 0.75rem;
            font-size: 0.78rem; font-family: 'Consolas', monospace;
        }}
        .stat-pill b {{ color: {ACCENT}; margin-right: 0.4rem; }}
        .stat-pill span {{ color: {TEXT_DIM}; }}

        /* title */
        .app-title {{
            font-family: 'Segoe UI', Consolas, monospace;
            font-weight: 700; font-size: 1.1rem;
            background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}

        /* landing */
        .landing {{
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            height: 65vh; text-align: center;
        }}
        .landing h1 {{
            font-size: 2.8rem; margin-bottom: 0.3rem;
            background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .landing p {{ color: {TEXT_DIM}; max-width: 440px; font-size: 1rem; }}

        /* scrollbar */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: {BG_PRIMARY}; }}
        ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
        </style>""",
        unsafe_allow_html=True,
    )


# ─── Data Helpers ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading .mat file …")
def load_mat(raw_bytes: bytes) -> dict[str, Any]:
    """Load a `.mat` file from raw bytes (cached so it only runs once)."""
    buf = io.BytesIO(raw_bytes)
    return sio.loadmat(buf)


def find_3d_arrays(mat: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return only the variables whose value is a 3-D NumPy array."""
    return {
        k: v for k, v in mat.items()
        if isinstance(v, np.ndarray) and v.ndim == 3
    }


def transform_slice(
    arr: np.ndarray,
    flip_h: bool,
    flip_v: bool,
    rotate: int,
) -> np.ndarray:
    """Apply flip / rotate transforms to a 2-D slice."""
    if flip_v:
        arr = np.flipud(arr)
    if flip_h:
        arr = np.fliplr(arr)
    if rotate:
        arr = np.rot90(arr, k=rotate // 90)
    return arr


# ─── Figure Builder ──────────────────────────────────────────────────────────
def build_figure(
    slice_2d: np.ndarray,
    *,
    colorscale: str,
    zmin: float,
    zmax: float,
    reverse_cmap: bool,
    aspect: str,
    origin: str,
    smoothing: str,
    show_axis: bool,
    show_colorbar: bool,
    show_grid: bool,
    grid_opacity: float,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
) -> go.Figure:
    """Build a fully-configured Plotly heatmap figure."""
    cscale = colorscale + "_r" if reverse_cmap else colorscale

    img = slice_2d if origin == "upper" else slice_2d[::-1]

    fig = px.imshow(
        img,
        color_continuous_scale=cscale,
        zmin=zmin,
        zmax=zmax,
        aspect=aspect,
    )

    # smoothing (zsmooth in Plotly)
    zsmooth: str | bool = False
    if smoothing == "best":
        zsmooth = "best"
    elif smoothing == "fast":
        zsmooth = "fast"
    fig.update_traces(zsmooth=zsmooth)

    # layout
    fig.update_layout(
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_PRIMARY,
        font=dict(color=TEXT, family="Consolas, monospace", size=12),
        margin=dict(l=50, r=20, t=40 if title else 10, b=40),
        title=dict(text=title, font=dict(size=14, color=TEXT)) if title else None,
        xaxis=dict(
            title=xlabel or None,
            visible=show_axis,
            showgrid=show_grid,
            gridcolor=f"rgba(255,255,255,{grid_opacity})" if show_grid else None,
        ),
        yaxis=dict(
            title=ylabel or None,
            visible=show_axis,
            showgrid=show_grid,
            gridcolor=f"rgba(255,255,255,{grid_opacity})" if show_grid else None,
        ),
        coloraxis_colorbar=dict(
            title=dict(text=cbar_label, font=dict(color=TEXT_DIM, size=11)) if cbar_label else None,
            tickfont=dict(color=TEXT_DIM, size=10),
            len=0.85,
        ) if show_colorbar else None,
        coloraxis_showscale=show_colorbar,
    )

    # interactive config
    fig.update_layout(dragmode="pan")

    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def sidebar_controls():
    """Render all sidebar widgets, return a dict of settings."""
    with st.sidebar:
        st.markdown('<div class="app-title">🔬 MatViewer</div>', unsafe_allow_html=True)
        st.caption("Scientific Image Gallery")

        # ── File Upload ──────────────────────────────────────────────
        uploaded = st.file_uploader(
            "Drop 3D Matrix File Here",
            type=["mat"],
            help="MATLAB `.mat` file with at least one 3-D array.",
        )
        raw = uploaded.getvalue() if uploaded else None

        if raw is None:
            return {"raw": None}

        st.divider()

        # ── Data selection ───────────────────────────────────────────
        mat = load_mat(raw)
        arrays_3d = find_3d_arrays(mat)
        if not arrays_3d:
            return {"raw": raw, "arrays_3d": {}}

        var_names = list(arrays_3d.keys())
        chosen = st.selectbox("Variable", var_names, index=0) if len(var_names) > 1 else var_names[0]
        data = arrays_3d[chosen].astype(np.float64)
        nz = data.shape[2]
        gmin, gmax = float(np.nanmin(data)), float(np.nanmax(data))

        z_idx = st.slider("Z-Slice", 0, nz - 1, nz // 2,
                          help="Scrub through the Z-axis.")

        st.divider()

        # ══════════════════════════════════════════════════════════════
        #  Adjustment panels — matplotlib-equivalent controls
        # ══════════════════════════════════════════════════════════════

        # ── 🎨 Colormap ──────────────────────────────────────────────
        with st.expander("🎨 Colormap", expanded=True):
            cmap = st.selectbox("Scale", COLOR_SCALES, index=0,
                                label_visibility="collapsed")
            reverse = st.toggle("Reverse colormap", value=False)
            contrast = st.slider("Contrast", 0.0, 1.0, (0.0, 1.0), 0.01,
                                 help="Map to data range as vmin/vmax.")

        # ── 📐 Transform ─────────────────────────────────────────────
        with st.expander("📐 Transform"):
            col_a, col_b = st.columns(2)
            flip_h = col_a.toggle("Flip H", value=False)
            flip_v = col_b.toggle("Flip V", value=False)
            rotate = st.select_slider("Rotate", options=[0, 90, 180, 270],
                                      value=0, format_func=lambda x: f"{x}°")
            aspect = st.radio("Aspect", ["equal", "auto"], horizontal=True,
                              index=0)
            origin = st.radio("Origin", ["upper", "lower"], horizontal=True,
                              index=0, help="'upper' = row 0 at top (image-style).")

        # ── ✨ Smoothing ─────────────────────────────────────────────
        with st.expander("✨ Smoothing"):
            smoothing = st.radio(
                "Interpolation",
                INTERPOLATION_METHODS,
                index=0,
                horizontal=True,
                help="Plotly zsmooth: none, best, or fast.",
            )

        # ── 🏷 Labels ────────────────────────────────────────────────
        with st.expander("🏷 Labels"):
            title_text = st.text_input("Title", placeholder="(none)")
            lc1, lc2 = st.columns(2)
            xlabel = lc1.text_input("X label", placeholder="X")
            ylabel = lc2.text_input("Y label", placeholder="Y")
            cbar_label = st.text_input("Colorbar label", placeholder="Intensity")

        # ── 🔲 Axes & Grid ───────────────────────────────────────────
        with st.expander("🔲 Axes & Grid"):
            show_axis = st.toggle("Show axes", value=True)
            show_colorbar = st.toggle("Show colorbar", value=True)
            show_grid = st.toggle("Grid overlay", value=False)
            grid_opacity = st.slider("Grid opacity", 0.05, 0.5, 0.15, 0.05,
                                     disabled=not show_grid)

        # ── 🔍 Crop ROI ──────────────────────────────────────────────
        with st.expander("🔍 Crop (ROI)"):
            h, w = data.shape[0], data.shape[1]
            x_range = st.slider("X range", 0, w - 1, (0, w - 1))
            y_range = st.slider("Y range", 0, h - 1, (0, h - 1))

        return {
            "raw": raw,
            "arrays_3d": arrays_3d,
            "chosen": chosen,
            "data": data,
            "z_idx": z_idx,
            "gmin": gmin,
            "gmax": gmax,
            # colormap
            "cmap": cmap,
            "reverse": reverse,
            "contrast": contrast,
            # transform
            "flip_h": flip_h,
            "flip_v": flip_v,
            "rotate": rotate,
            "aspect": aspect,
            "origin": origin,
            # smoothing
            "smoothing": smoothing,
            # labels
            "title": title_text,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "cbar_label": cbar_label,
            # axes/grid
            "show_axis": show_axis,
            "show_colorbar": show_colorbar,
            "show_grid": show_grid,
            "grid_opacity": grid_opacity,
            # crop
            "x_range": x_range,
            "y_range": y_range,
        }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="MatViewer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    local_css()

    cfg = sidebar_controls()

    # ── Landing ──────────────────────────────────────────────────────
    if cfg["raw"] is None:
        st.markdown(
            """<div class="landing">
                <h1>🔬 MatViewer</h1>
                <p>Upload a <code>.mat</code> file from the sidebar to begin
                exploring 3-D datasets interactively.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    if not cfg.get("arrays_3d"):
        st.error("No 3-D arrays found in this file.")
        return

    # ── Unpack config ────────────────────────────────────────────────
    data: np.ndarray = cfg["data"]
    z_idx: int = cfg["z_idx"]
    gmin: float = cfg["gmin"]
    gmax: float = cfg["gmax"]
    cmin, cmax = cfg["contrast"]

    # ── Stat bar ─────────────────────────────────────────────────────
    st.markdown(
        f"""<div class="stat-row">
            <div class="stat-pill"><b>var</b><span>{cfg["chosen"]}</span></div>
            <div class="stat-pill"><b>shape</b><span>{data.shape[0]}×{data.shape[1]}×{data.shape[2]}</span></div>
            <div class="stat-pill"><b>dtype</b><span>{data.dtype}</span></div>
            <div class="stat-pill"><b>range</b><span>{gmin:.4g} — {gmax:.4g}</span></div>
            <div class="stat-pill"><b>slice</b><span>{z_idx} / {data.shape[2] - 1}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Prepare slice ────────────────────────────────────────────────
    slice_2d = data[:, :, z_idx]

    # crop
    y0, y1 = cfg["y_range"]
    x0, x1 = cfg["x_range"]
    slice_2d = slice_2d[y0 : y1 + 1, x0 : x1 + 1]

    # transform
    slice_2d = transform_slice(
        slice_2d,
        flip_h=cfg["flip_h"],
        flip_v=cfg["flip_v"],
        rotate=cfg["rotate"],
    )

    # contrast
    vmin = gmin + cmin * (gmax - gmin)
    vmax = gmin + cmax * (gmax - gmin)

    # ── Build & display ──────────────────────────────────────────────
    fig = build_figure(
        slice_2d,
        colorscale=cfg["cmap"],
        zmin=vmin,
        zmax=vmax,
        reverse_cmap=cfg["reverse"],
        aspect=cfg["aspect"],
        origin=cfg["origin"],
        smoothing=cfg["smoothing"],
        show_axis=cfg["show_axis"],
        show_colorbar=cfg["show_colorbar"],
        show_grid=cfg["show_grid"],
        grid_opacity=cfg["grid_opacity"],
        title=cfg["title"],
        xlabel=cfg["xlabel"],
        ylabel=cfg["ylabel"],
        cbar_label=cfg["cbar_label"],
    )

    plotly_cfg = {
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["drawrect", "eraseshape"],
        "scrollZoom": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "width": slice_2d.shape[1] * 3,
            "height": slice_2d.shape[0] * 3,
            "scale": 2,
        },
    }

    st.plotly_chart(fig, use_container_width=True, config=plotly_cfg)

    # ── Slice histogram (compact) ────────────────────────────────────
    with st.expander("📊 Slice Histogram"):
        hist_fig = px.histogram(
            slice_2d.ravel(), nbins=128,
            color_discrete_sequence=[ACCENT],
        )
        hist_fig.update_layout(
            paper_bgcolor=BG_PRIMARY,
            plot_bgcolor=BG_PRIMARY,
            font_color=TEXT_DIM,
            height=180,
            margin=dict(l=30, r=10, t=10, b=30),
            showlegend=False,
            xaxis_title="Pixel value",
            yaxis_title="Count",
        )
        # draw vmin/vmax lines
        for v, color in [(vmin, ACCENT), (vmax, ACCENT2)]:
            hist_fig.add_vline(x=v, line_dash="dash", line_color=color,
                               line_width=1.5)
        st.plotly_chart(hist_fig, use_container_width=True)


if __name__ == "__main__":
    main()
