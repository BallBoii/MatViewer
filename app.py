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
    """Inject custom CSS for a minimal dark aesthetic."""
    st.markdown(
        f"""<style>
        /* remove default header chrome */
        header[data-testid="stHeader"] {{ background: transparent !important; }}

        .stApp, [data-testid="stAppViewContainer"] {{
            background: {BG_PRIMARY} !important; color: {TEXT} !important;
        }}

        /* sidebar */
        section[data-testid="stSidebar"] {{ background: {BG_CARD} !important; }}
        section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

        /* expander: tighter, no emoji needed */
        details summary {{ padding: 0.4rem 0 !important; }}
        details summary span {{ font-size: 0.82rem !important; letter-spacing: 0.02em; }}

        /* file uploader */
        [data-testid="stFileUploader"] {{
            border: 1.5px dashed {BORDER} !important;
            border-radius: 8px; padding: 0.4rem;
        }}
        [data-testid="stFileUploader"]:hover {{ border-color: {ACCENT} !important; }}

        /* ── info table ── */
        .info-table {{
            width: 100%; border-collapse: collapse;
            font-family: 'Consolas', monospace;
            font-size: 0.78rem;
            margin-bottom: 0.5rem;
        }}
        .info-table th {{
            text-align: left;
            color: {TEXT_DIM};
            font-weight: 400;
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 0.35rem 0.7rem 0.2rem;
            border-bottom: 1px solid {BORDER};
        }}
        .info-table td {{
            color: {TEXT};
            padding: 0.4rem 0.7rem;
            border-bottom: 1px solid rgba(48,54,61,0.4);
        }}
        .info-table tr:last-child td {{ border-bottom: none; }}
        .info-section {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }}
        .info-section-title {{
            color: {TEXT_DIM};
            font-size: 0.6rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            padding: 0.5rem 0.7rem 0;
            font-family: 'Segoe UI', sans-serif;
        }}

        /* sidebar title */
        .app-title {{
            font-family: 'Segoe UI', Consolas, monospace;
            font-weight: 600; font-size: 1rem; letter-spacing: 0.03em;
            color: {TEXT} !important;
        }}

        /* landing */
        .landing {{
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            height: 65vh; text-align: center;
        }}
        .landing h1 {{
            font-size: 2.4rem; font-weight: 600; margin-bottom: 0.4rem;
            background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .landing p {{ color: {TEXT_DIM}; max-width: 400px; font-size: 0.95rem; line-height: 1.6; }}

        /* scrollbar */
        ::-webkit-scrollbar {{ width: 5px; }}
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
        st.markdown('<div class="app-title">MatViewer</div>', unsafe_allow_html=True)
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

        # ── Colormap ──────────────────────────────────────────────
        with st.expander("Colormap", expanded=True):
            cmap = st.selectbox("Scale", COLOR_SCALES, index=0,
                                label_visibility="collapsed")
            reverse = st.toggle("Reverse", value=False)
            
            # Contrast mode: Auto (slider) vs Manual (type values)
            use_manual = st.toggle("Manual vmin/vmax", value=False,
                                   help="Switch between slider (auto) and manual input.")
            
            if use_manual:
                col_min, col_max = st.columns(2)
                manual_vmin = col_min.number_input(
                    "vmin", value=float(gmin), format="%.6g",
                    help="Minimum display value"
                )
                manual_vmax = col_max.number_input(
                    "vmax", value=float(gmax), format="%.6g",
                    help="Maximum display value"
                )
                contrast = None
            else:
                contrast = st.slider("Contrast", 0.0, 1.0, (0.0, 1.0), 0.01,
                                     help="Map to data range as vmin / vmax.")
                manual_vmin = None
                manual_vmax = None

        # ── Transform ─────────────────────────────────────────────
        with st.expander("Transform"):
            col_a, col_b = st.columns(2)
            flip_h = col_a.toggle("Flip H", value=False)
            flip_v = col_b.toggle("Flip V", value=False)
            rotate = st.select_slider("Rotate", options=[0, 90, 180, 270],
                                      value=0, format_func=lambda x: f"{x}°")
            c1, c2 = st.columns(2)
            aspect = c1.radio("Aspect", ["equal", "auto"], index=0)
            origin = c2.radio("Origin", ["upper", "lower"], index=0,
                              help="Row 0 position.")

        # ── Interpolation ─────────────────────────────────────────
        with st.expander("Interpolation"):
            smoothing = st.radio(
                "Method",
                INTERPOLATION_METHODS,
                index=0,
                horizontal=True,
                label_visibility="collapsed",
            )

        # ── Labels ────────────────────────────────────────────────
        with st.expander("Labels"):
            title_text = st.text_input("Title", placeholder="(none)")
            lc1, lc2 = st.columns(2)
            xlabel = lc1.text_input("X", placeholder="X")
            ylabel = lc2.text_input("Y", placeholder="Y")
            cbar_label = st.text_input("Colorbar", placeholder="Intensity")

        # ── Axes & Grid ───────────────────────────────────────────
        with st.expander("Axes / Grid"):
            ga, gb = st.columns(2)
            show_axis = ga.toggle("Axes", value=True)
            show_colorbar = gb.toggle("Colorbar", value=True)
            show_grid = st.toggle("Grid", value=False)
            grid_opacity = st.slider("Opacity", 0.05, 0.5, 0.15, 0.05,
                                     disabled=not show_grid)

        # ── Crop ──────────────────────────────────────────────────
        with st.expander("Crop"):
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
            "manual_vmin": manual_vmin,
            "manual_vmax": manual_vmax,
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
    # Try to use logo as page icon
    from pathlib import Path
    logo_path = Path("public/logo.png")
    page_icon = str(logo_path) if logo_path.exists() else None
    
    st.set_page_config(
        page_title="MatViewer",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    local_css()

    cfg = sidebar_controls()

    # ── Landing ──────────────────────────────────────────────────────
    if cfg["raw"] is None:
        st.markdown(
            """<div class="landing">
                <h1>MatViewer</h1>
                <p>Upload a <code>.mat</code> file from the sidebar
                to explore 3-D datasets interactively.</p>
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
    
    # Handle contrast: manual or slider mode
    if cfg["manual_vmin"] is not None and cfg["manual_vmax"] is not None:
        vmin_input = cfg["manual_vmin"]
        vmax_input = cfg["manual_vmax"]
    else:
        cmin, cmax = cfg["contrast"]
        vmin_input = gmin + cmin * (gmax - gmin)
        vmax_input = gmin + cmax * (gmax - gmin)

    # ── Info bar ─────────────────────────────────────────────────
    # (moved below — layout: image → histogram → info)

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

    # apply vmin/vmax from earlier calculation
    vmin = vmin_input
    vmax = vmax_input

    # ━━ 1. IMAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

    # ━━ 2. HISTOGRAM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    flat = slice_2d.ravel()
    s_min = float(np.nanmin(flat))
    s_max = float(np.nanmax(flat))
    s_mean = float(np.nanmean(flat))
    s_std = float(np.nanstd(flat))
    s_med = float(np.nanmedian(flat))
    s_p1 = float(np.nanpercentile(flat, 1))
    s_p99 = float(np.nanpercentile(flat, 99))
    n_nan = int(np.count_nonzero(np.isnan(flat)))
    n_px = int(flat.size)

    hist_fig = px.histogram(
        flat, nbins=128,
        color_discrete_sequence=[ACCENT],
    )
    hist_fig.update_layout(
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_PRIMARY,
        font_color=TEXT_DIM,
        height=200,
        margin=dict(l=40, r=10, t=10, b=35),
        showlegend=False,
        xaxis_title="Pixel value",
        yaxis_title="Count",
    )
    for v, color, label in [
        (vmin, ACCENT, "vmin"),
        (vmax, ACCENT2, "vmax"),
        (s_mean, "#FFD866", "mean"),
        (s_med, "#A9DC76", "median"),
    ]:
        hist_fig.add_vline(
            x=v, line_dash="dash", line_color=color, line_width=1.2,
            annotation_text=label,
            annotation_font_color=color,
            annotation_font_size=10,
        )
    st.plotly_chart(hist_fig, use_container_width=True)

    # ━━ 3. INFO + STATS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            f"""<div class="info-section">
                <div class="info-section-title">Stack</div>
                <table class="info-table">
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>Variable</td><td>{cfg["chosen"]}</td></tr>
                    <tr><td>Shape</td><td>{data.shape[0]} × {data.shape[1]} × {data.shape[2]}</td></tr>
                    <tr><td>Dtype</td><td>{data.dtype}</td></tr>
                    <tr><td>Current slice</td><td>{z_idx} / {data.shape[2] - 1}</td></tr>
                    <tr><td>Pixels</td><td>{n_px:,}</td></tr>
                    <tr><td>NaN count</td><td>{n_nan}</td></tr>
                </table>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            f"""<div class="info-section">
                <div class="info-section-title">Slice Statistics</div>
                <table class="info-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Min</td><td>{s_min:.6g}</td></tr>
                    <tr><td>Max</td><td>{s_max:.6g}</td></tr>
                    <tr><td>Mean</td><td>{s_mean:.6g}</td></tr>
                    <tr><td>Std Dev</td><td>{s_std:.6g}</td></tr>
                    <tr><td>Median</td><td>{s_med:.6g}</td></tr>
                    <tr><td>P1</td><td>{s_p1:.6g}</td></tr>
                    <tr><td>P99</td><td>{s_p99:.6g}</td></tr>
                </table>
            </div>""",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
