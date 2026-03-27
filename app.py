import io
import zipfile
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDOF Pushover Dashboard", layout="wide")

G = 9.81
HINGE_ORDER = ["Elastic", "Beam IO", "Beam LS", "Column IO", "Column LS", "CP", "Failed"]
HINGE_COLOR = {
    "Elastic": "#dbeafe",
    "Beam IO": "#bbf7d0",
    "Beam LS": "#86efac",
    "Column IO": "#fde68a",
    "Column LS": "#fca5a5",
    "CP": "#ef4444",
    "Failed": "#7f1d1d",
}


@dataclass
class AnalysisResults:
    roof_disp: np.ndarray
    base_shear: np.ndarray
    first_mode_shapes: np.ndarray
    periods: np.ndarray
    target_disp: float
    yield_disp: float
    yield_shear: float
    peak_disp: float
    peak_shear: float
    final_disp_profile: np.ndarray
    final_drift_ratio: np.ndarray
    final_story_shear: np.ndarray
    story_k_initial: np.ndarray
    story_k_final: np.ndarray
    beam_state: np.ndarray
    col_state: np.ndarray
    story_label: list
    story_beam_m: np.ndarray
    story_col_m: np.ndarray
    floor_forces_target: np.ndarray
    notes: list


def default_table(n):
    return pd.DataFrame(
        {
            "Storey": np.arange(1, n + 1),
            "Height_m": [3.6] * n,
            "Columns": [3] * n,
            "Beams": [2] * n,
            "EI_column_kNm2": [42670.0] * n,
            "EI_beam_kNm2": [30000.0] * n,
            "Mpc_kNm": [310.0] * n,
            "Mpb_kNm": [550.0] * n,
            "Weight_kN": [270.0] * n,
            "Ultimate_Drift_Ratio": [0.04] * n,
        }
    )


def default_force_pattern(n, pattern):
    z = np.arange(1, n + 1, dtype=float)
    if pattern == "Uniform":
        f = np.ones(n)
    elif pattern == "Triangular":
        f = z
    elif pattern == "First-mode-like":
        f = np.sin((z / (n + 1.0)) * np.pi / 2.0)
    else:
        f = np.ones(n)
    f = np.maximum(f, 1e-12)
    return f / np.sum(f)


def calc_story_stiffness(n_cols, n_beams, ei_col, ei_beam, height, beam_participation=0.35):
    h = np.maximum(height, 1e-9)
    col_k = 12.0 * np.maximum(ei_col, 1e-9) * np.maximum(n_cols, 1e-9) / (h ** 3)
    beam_k = beam_participation * 12.0 * np.maximum(ei_beam, 1e-9) * np.maximum(n_beams, 1e-9) / (h ** 3)
    return col_k + beam_k


def assemble_shear_building_K(k_story):
    n = len(k_story)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        ki = k_story[i]
        if i == 0:
            K[i, i] += ki
        else:
            K[i, i] += ki
            K[i - 1, i - 1] += ki
            K[i, i - 1] -= ki
            K[i - 1, i] -= ki
    return K


def assemble_mass_matrix(weights_kN):
    masses = np.maximum(np.asarray(weights_kN, dtype=float), 1e-9) / G
    return np.diag(masses), masses


def modal_analysis(K, M):
    A = np.linalg.inv(M) @ K
    vals, vecs = np.linalg.eig(A)
    vals = np.real(vals)
    vecs = np.real(vecs)
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    vals = np.maximum(vals, 1e-12)
    omegas = np.sqrt(vals)
    periods = 2.0 * np.pi / omegas
    for i in range(vecs.shape[1]):
        if vecs[-1, i] < 0:
            vecs[:, i] *= -1.0
        if np.max(np.abs(vecs[:, i])) > 0:
            vecs[:, i] /= np.max(np.abs(vecs[:, i]))
    return periods, vecs


def lateral_forces_from_u(K, u):
    p = K @ u
    return np.real(p)


def story_shear_from_floor_forces(floor_forces):
    return np.flip(np.cumsum(np.flip(floor_forces)))


def cumulative_overturning_from_floor_forces(floor_forces, z):
    n = len(floor_forces)
    M = np.zeros(n)
    z0 = np.concatenate(([0.0], z[:-1]))
    for i in range(n):
        arm = z[i:] - z0[i]
        M[i] = np.sum(floor_forces[i:] * arm)
    return M


def classify_story(beam_state, col_state, drift_ratio, drift_cap):
    if drift_ratio >= 1.15 * drift_cap:
        return "Failed"
    if col_state >= 2 or drift_ratio >= 0.90 * drift_cap:
        return "CP"
    if col_state == 1:
        return "Column LS"
    if beam_state >= 2:
        return "Beam LS"
    if col_state == 0 and beam_state == 1:
        return "Beam IO"
    if col_state == 1 and beam_state == 0:
        return "Column IO"
    return "Elastic"


def degrade_story_stiffness(k0, beam_state, col_state, drift_ratio, drift_cap):
    factor = 1.0
    if beam_state == 1:
        factor *= 0.80
    elif beam_state >= 2:
        factor *= 0.60

    if col_state == 1:
        factor *= 0.55
    elif col_state >= 2:
        factor *= 0.25

    if drift_ratio >= drift_cap:
        factor *= 0.50
    elif drift_ratio >= 0.75 * drift_cap:
        factor *= 0.80

    return max(0.05 * k0, factor * k0)


def run_mdof_pushover(
    df,
    pattern_name,
    user_pattern,
    roof_disp_max,
    n_steps,
    beam_participation,
    pdelta_alpha,
    mode_update_every,
):
    n = len(df)
    h = df["Height_m"].to_numpy(dtype=float)
    n_cols = df["Columns"].to_numpy(dtype=float)
    n_beams = df["Beams"].to_numpy(dtype=float)
    ei_col = df["EI_column_kNm2"].to_numpy(dtype=float)
    ei_beam = df["EI_beam_kNm2"].to_numpy(dtype=float)
    mpc = df["Mpc_kNm"].to_numpy(dtype=float)
    mpb = df["Mpb_kNm"].to_numpy(dtype=float)
    w = df["Weight_kN"].to_numpy(dtype=float)
    drift_cap = df["Ultimate_Drift_Ratio"].to_numpy(dtype=float)

    k0 = calc_story_stiffness(n_cols, n_beams, ei_col, ei_beam, h, beam_participation=beam_participation)
    k_story = k0.copy()

    Mmat, masses = assemble_mass_matrix(w)
    z = np.cumsum(h)

    if pattern_name == "User-defined":
        patt = np.asarray(user_pattern, dtype=float)
        patt = np.maximum(patt, 0.0)
        if np.sum(patt) <= 0:
            patt = np.ones(n)
        patt = patt / np.sum(patt)
    else:
        patt = default_force_pattern(n, pattern_name)

    roof_hist = np.linspace(0.0, roof_disp_max, n_steps)
    base_hist = np.zeros(n_steps)
    mode_hist = np.zeros((n_steps, n))

    beam_state = np.zeros(n, dtype=int)  # 0 elastic, 1 IO, 2 LS+
    col_state = np.zeros(n, dtype=int)

    floor_force_hist = np.zeros((n_steps, n))
    disp_hist = np.zeros((n_steps, n))
    story_shear_hist = np.zeros((n_steps, n))
    beam_m_hist = np.zeros((n_steps, n))
    col_m_hist = np.zeros((n_steps, n))

    periods, modes = modal_analysis(assemble_shear_building_K(k_story), Mmat)
    phi = modes[:, 0].copy()
    phi /= max(phi[-1], 1e-9)

    yielded_once = False
    yield_disp = None
    yield_shear = None

    for j, roof_d in enumerate(roof_hist):
        if j == 0 or (mode_update_every > 0 and j % mode_update_every == 0):
            periods, modes = modal_analysis(assemble_shear_building_K(k_story), Mmat)
            phi = modes[:, 0].copy()
            phi /= max(phi[-1], 1e-9)

        u = roof_d * phi
        floor_forces = lateral_forces_from_u(assemble_shear_building_K(k_story), u)
        story_shear = story_shear_from_floor_forces(floor_forces)
        overturn = cumulative_overturning_from_floor_forces(floor_forces, z)
        drift = np.zeros(n)
        drift[0] = u[0] / max(h[0], 1e-9)
        for i in range(1, n):
            drift[i] = (u[i] - u[i - 1]) / max(h[i], 1e-9)

        pdelta_factor = 1.0 / (1.0 + pdelta_alpha * np.maximum(np.abs(drift) / np.maximum(drift_cap, 1e-9), 0.0))
        pdelta_factor = np.clip(pdelta_factor, 0.55, 1.0)
        floor_forces *= pdelta_factor
        story_shear = story_shear_from_floor_forces(floor_forces)
        overturn = cumulative_overturning_from_floor_forces(floor_forces, z)

        beam_m = np.abs(story_shear * h / np.maximum(2.0 * np.maximum(n_beams, 1.0), 1.0))
        col_m = np.abs(overturn / np.maximum(np.maximum(n_cols, 1.0), 1.0))

        for i in range(n):
            if beam_m[i] >= 0.85 * mpb[i]:
                beam_state[i] = max(beam_state[i], 1)
            if beam_m[i] >= 1.00 * mpb[i]:
                beam_state[i] = max(beam_state[i], 2)

            if col_m[i] >= 0.80 * mpc[i]:
                col_state[i] = max(col_state[i], 1)
            if col_m[i] >= 1.00 * mpc[i]:
                col_state[i] = max(col_state[i], 2)

            k_story[i] = degrade_story_stiffness(k0[i], beam_state[i], col_state[i], abs(drift[i]), drift_cap[i])

        if not yielded_once and (np.any(beam_state > 0) or np.any(col_state > 0)):
            yielded_once = True
            yield_disp = roof_d
            yield_shear = float(np.sum(floor_forces))

        periods, modes = modal_analysis(assemble_shear_building_K(k_story), Mmat)
        phi = modes[:, 0].copy()
        phi /= max(phi[-1], 1e-9)

        u = roof_d * phi
        floor_forces = lateral_forces_from_u(assemble_shear_building_K(k_story), u)
        drift = np.zeros(n)
        drift[0] = u[0] / max(h[0], 1e-9)
        for i in range(1, n):
            drift[i] = (u[i] - u[i - 1]) / max(h[i], 1e-9)
        pdelta_factor = 1.0 / (1.0 + pdelta_alpha * np.maximum(np.abs(drift) / np.maximum(drift_cap, 1e-9), 0.0))
        pdelta_factor = np.clip(pdelta_factor, 0.55, 1.0)
        floor_forces *= pdelta_factor
        story_shear = story_shear_from_floor_forces(floor_forces)
        overturn = cumulative_overturning_from_floor_forces(floor_forces, z)
        beam_m = np.abs(story_shear * h / np.maximum(2.0 * np.maximum(n_beams, 1.0), 1.0))
        col_m = np.abs(overturn / np.maximum(np.maximum(n_cols, 1.0), 1.0))

        roof_hist[j] = roof_d
        base_hist[j] = float(np.sum(floor_forces))
        mode_hist[j, :] = phi
        floor_force_hist[j, :] = floor_forces
        disp_hist[j, :] = u
        story_shear_hist[j, :] = story_shear
        beam_m_hist[j, :] = beam_m
        col_m_hist[j, :] = col_m

    peak_idx = int(np.argmax(base_hist))
    peak_disp = float(roof_hist[peak_idx])
    peak_shear = float(base_hist[peak_idx])

    if yield_disp is None:
        yield_disp = peak_disp * 0.6
        yield_shear = peak_shear * 0.75

    mu = max(roof_disp_max / max(yield_disp, 1e-9), 1.0)
    target_disp = min(roof_disp_max, yield_disp * mu ** 0.6)
    tgt_idx = int(np.argmin(np.abs(roof_hist - target_disp)))

    story_label = []
    for i in range(n):
        story_label.append(classify_story(beam_state[i], col_state[i], abs(disp_hist[tgt_idx, i] - (disp_hist[tgt_idx, i-1] if i > 0 else 0.0)) / h[i], drift_cap[i]))

    final_drift = np.zeros(n)
    final_drift[0] = disp_hist[tgt_idx, 0] / max(h[0], 1e-9)
    for i in range(1, n):
        final_drift[i] = (disp_hist[tgt_idx, i] - disp_hist[tgt_idx, i - 1]) / max(h[i], 1e-9)

    notes = [
        "This upgrade replaces the imposed displacement weights with a true shear-building MDOF stiffness and mass model.",
        "The first mode is recomputed as stiffness degrades, so hinge progression is influenced by evolving modal behavior.",
        "Beam and column yielding are still approximate story-level checks, not full member-end fiber or plastic-hinge elements.",
        "Base-first yielding is now more likely when overturning demand concentrates in the lower stories, similar to commercial nonlinear solvers.",
    ]

    return AnalysisResults(
        roof_disp=roof_hist,
        base_shear=base_hist,
        first_mode_shapes=mode_hist,
        periods=periods,
        target_disp=target_disp,
        yield_disp=float(yield_disp),
        yield_shear=float(yield_shear),
        peak_disp=peak_disp,
        peak_shear=peak_shear,
        final_disp_profile=disp_hist[tgt_idx, :],
        final_drift_ratio=final_drift,
        final_story_shear=story_shear_hist[tgt_idx, :],
        story_k_initial=k0,
        story_k_final=k_story.copy(),
        beam_state=beam_state.copy(),
        col_state=col_state.copy(),
        story_label=story_label,
        story_beam_m=beam_m_hist[tgt_idx, :],
        story_col_m=col_m_hist[tgt_idx, :],
        floor_forces_target=floor_force_hist[tgt_idx, :],
        notes=notes,
    )


def to_excel_bytes(inputs_df, results_df, summary_df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        inputs_df.to_excel(writer, index=False, sheet_name="Inputs")
        results_df.to_excel(writer, index=False, sheet_name="Story_Results")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
    return buffer.getvalue()


def to_zip_bytes(inputs_df, results_df, summary_df):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("inputs.csv", inputs_df.to_csv(index=False))
        zf.writestr("story_results.csv", results_df.to_csv(index=False))
        zf.writestr("summary.csv", summary_df.to_csv(index=False))
    return buffer.getvalue()


st.title("MDOF Nonlinear Pushover Dashboard")
st.caption("Shear-building MDOF upgrade with modal analysis, adaptive first-mode displacement shape, approximate beam/column yielding, and degrading story stiffness.")

with st.sidebar:
    st.header("Model Setup")
    n_storey = st.slider("Number of Storeys", 1, 15, 5)
    pattern_name = st.selectbox("Reference Lateral Pattern", ["Uniform", "Triangular", "First-mode-like", "User-defined"], index=2)
    roof_disp_max = st.number_input("Maximum Roof Displacement (m)", min_value=0.01, value=0.60, step=0.01)
    n_steps = st.slider("Pushover Steps", 20, 300, 120, 10)
    beam_participation = st.slider("Beam Stiffness Participation Factor", 0.00, 1.00, 0.35, 0.05)
    pdelta_alpha = st.slider("P-Delta Severity Factor", 0.00, 0.40, 0.06, 0.01)
    mode_update_every = st.slider("Recompute Mode Every N Steps", 1, 20, 2, 1)

st.subheader("Storey-by-Storey Frame Properties")
st.write("Enter equivalent story properties. Unlike the previous version, this upgrade explicitly forms a mass matrix, story stiffness matrix, and evolving first mode shape.")

if "table_mdof" not in st.session_state or len(st.session_state.table_mdof) != n_storey:
    st.session_state.table_mdof = default_table(n_storey)

edited_df = st.data_editor(
    st.session_state.table_mdof,
    num_rows="fixed",
    use_container_width=True,
    key="mdof_table",
    column_config={
        "Storey": st.column_config.NumberColumn(disabled=True),
        "Height_m": st.column_config.NumberColumn("Height (m)", min_value=2.0, step=0.1),
        "Columns": st.column_config.NumberColumn("No. of Columns", min_value=1, step=1),
        "Beams": st.column_config.NumberColumn("No. of Beams", min_value=1, step=1),
        "EI_column_kNm2": st.column_config.NumberColumn("Column EI (kN·m²)", min_value=1.0),
        "EI_beam_kNm2": st.column_config.NumberColumn("Beam EI (kN·m²)", min_value=1.0),
        "Mpc_kNm": st.column_config.NumberColumn("Column Plastic Moment Mpc (kN·m)", min_value=1.0),
        "Mpb_kNm": st.column_config.NumberColumn("Beam Plastic Moment Mpb (kN·m)", min_value=1.0),
        "Weight_kN": st.column_config.NumberColumn("Storey Seismic Weight (kN)", min_value=1.0),
        "Ultimate_Drift_Ratio": st.column_config.NumberColumn("Ultimate Drift Ratio", min_value=0.005, max_value=0.15, step=0.005),
    },
)
st.session_state.table_mdof = edited_df.copy()

user_pattern = None
if pattern_name == "User-defined":
    st.subheader("User-Defined Floor Force Pattern")
    pat_df = pd.DataFrame({"Storey": np.arange(1, n_storey + 1), "Relative Force": [1.0] * n_storey})
    pat_df = st.data_editor(pat_df, num_rows="fixed", use_container_width=True, key="user_pat_mdof")
    user_pattern = pat_df["Relative Force"].to_numpy(dtype=float)

run = st.button("Run MDOF Pushover Analysis", type="primary")

if run:
    results = run_mdof_pushover(
        edited_df.copy(),
        pattern_name=pattern_name,
        user_pattern=user_pattern,
        roof_disp_max=roof_disp_max,
        n_steps=n_steps,
        beam_participation=beam_participation,
        pdelta_alpha=pdelta_alpha,
        mode_update_every=mode_update_every,
    )

    results_df = pd.DataFrame(
        {
            "Storey": edited_df["Storey"],
            "Height_m": edited_df["Height_m"],
            "K_initial_kN_per_m": results.story_k_initial,
            "K_final_kN_per_m": results.story_k_final,
            "Disp_at_Target_m": results.final_disp_profile,
            "Drift_Ratio_at_Target": results.final_drift_ratio,
            "Story_Shear_at_Target_kN": results.final_story_shear,
            "Beam_Moment_Demand_kNm": results.story_beam_m,
            "Column_Moment_Demand_kNm": results.story_col_m,
            "Beam_State": results.beam_state,
            "Column_State": results.col_state,
            "Story_Label": results.story_label,
            "Floor_Force_at_Target_kN": results.floor_forces_target,
        }
    )

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "First-mode Period T1 (s)",
                "Yield Roof Displacement (m)",
                "Yield Base Shear (kN)",
                "Peak Roof Displacement (m)",
                "Peak Base Shear (kN)",
                "Target Roof Displacement (m)",
                "Maximum Drift Ratio at Target",
                "Stories at CP/Failed",
            ],
            "Value": [
                results.periods[0],
                results.yield_disp,
                results.yield_shear,
                results.peak_disp,
                results.peak_shear,
                results.target_disp,
                float(np.max(np.abs(results.final_drift_ratio))),
                int(np.sum(np.isin(results.story_label, ["CP", "Failed"]))),
            ],
        }
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("T1", f"{results.periods[0]:.3f} s")
    c2.metric("Yield Base Shear", f"{results.yield_shear:,.1f} kN")
    c3.metric("Peak Base Shear", f"{results.peak_shear:,.1f} kN")
    c4.metric("Critical Story State", max(results.story_label, key=lambda x: HINGE_ORDER.index(x)))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Capacity Curve", "Mode Shape", "Story Results", "Hinge Map", "Downloads"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(results.roof_disp, results.base_shear, linewidth=2.2, label="MDOF Pushover Curve")
        ax1.axvline(results.target_disp, linestyle=":", linewidth=2, label="Target Displacement")
        ax1.scatter([results.yield_disp], [results.yield_shear], label="First Yield")
        ax1.scatter([results.peak_disp], [results.peak_shear], label="Peak")
        ax1.set_xlabel("Roof Displacement (m)")
        ax1.set_ylabel("Base Shear (kN)")
        ax1.set_title("Adaptive-Mode MDOF Pushover Curve")
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.bar(results_df["Storey"].astype(str), np.abs(results_df["Drift_Ratio_at_Target"]))
        ax2.set_xlabel("Storey")
        ax2.set_ylabel("Drift Ratio")
        ax2.set_title("Storey Drift Ratio at Target Displacement")
        ax2.grid(True, axis="y")
        st.pyplot(fig2)

    with tab2:
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        current_mode = results.first_mode_shapes[min(len(results.first_mode_shapes) - 1, np.argmin(np.abs(results.roof_disp - results.target_disp))), :]
        z = np.cumsum(edited_df["Height_m"].to_numpy(dtype=float))
        ax3.plot(current_mode, z, marker="o")
        ax3.set_xlabel("Normalized First Mode Shape")
        ax3.set_ylabel("Elevation (m)")
        ax3.set_title("First Mode Shape at Target Displacement")
        ax3.grid(True)
        st.pyplot(fig3)

    with tab3:
        st.dataframe(results_df, use_container_width=True)
        st.dataframe(summary_df, use_container_width=True)
        for note in results.notes:
            st.caption(note)

    with tab4:
        cols = st.columns(len(results_df))
        for i, row in results_df.iterrows():
            label = row["Story_Label"]
            with cols[i]:
                st.markdown(
                    f"<div style='padding:18px;border-radius:10px;background:{HINGE_COLOR[label]};text-align:center;color:#111;'>"
                    f"<b>Storey {int(row['Storey'])}</b><br>{label}<br>Beam={int(row['Beam_State'])}, Col={int(row['Column_State'])}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with tab5:
        excel_bytes = to_excel_bytes(edited_df, results_df, summary_df)
        zip_bytes = to_zip_bytes(edited_df, results_df, summary_df)
        st.download_button(
            "Download Excel Results",
            data=excel_bytes,
            file_name="mdof_pushover_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download CSV ZIP Bundle",
            data=zip_bytes,
            file_name="mdof_pushover_results_bundle.zip",
            mime="application/zip",
        )

st.markdown("---")
st.write("Next logical upgrade: frame-element stiffness assembly with member-end rotational springs and tangent-stiffness iteration against SeismoBuild/OpenSees benchmarks.")
