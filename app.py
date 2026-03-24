
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Nonlinear Pushover Pro v3", layout="wide")

G = 9.81

# =========================================================
# HELPERS
# =========================================================
@dataclass
class HingeProperty:
    My: float          # kN-m
    theta_y: float     # rad
    theta_u: float     # rad
    alpha_post: float  # post-yield stiffness ratio
    Vy: float          # kN
    gamma_y: float     # shear distortion / proxy drift ratio
    gamma_u: float     # ultimate shear distortion
    weight: float      # relative share in storey behavior


def default_story_data(n_storey: int) -> pd.DataFrame:
    return pd.DataFrame({
        "Storey": np.arange(1, n_storey + 1),
        "Height_m": np.full(n_storey, 3.0),
        "Floor_Gravity_kN": np.full(n_storey, 1200.0),
        "Column_EI_kNm2_per_column": np.full(n_storey, 25000.0),
        "Columns_per_storey": np.full(n_storey, 4),
        "Beam_span_m": np.full(n_storey, 5.0),
    })


def default_hinge_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Element_Group": "Beam",
            "My_kNm": 220.0,
            "theta_y_rad": 0.010,
            "theta_u_rad": 0.060,
            "alpha_post": 0.03,
            "Vy_kN": 180.0,
            "gamma_y": 0.006,
            "gamma_u": 0.025,
            "weight": 1.0,
        },
        {
            "Element_Group": "Column",
            "My_kNm": 300.0,
            "theta_y_rad": 0.008,
            "theta_u_rad": 0.040,
            "alpha_post": 0.02,
            "Vy_kN": 260.0,
            "gamma_y": 0.005,
            "gamma_u": 0.020,
            "weight": 1.0,
        },
    ])


def hinge_map_from_df(df: pd.DataFrame) -> Dict[str, HingeProperty]:
    out = {}
    for _, r in df.iterrows():
        out[str(r["Element_Group"]).strip()] = HingeProperty(
            My=float(r["My_kNm"]),
            theta_y=float(r["theta_y_rad"]),
            theta_u=float(r["theta_u_rad"]),
            alpha_post=float(r["alpha_post"]),
            Vy=float(r["Vy_kN"]),
            gamma_y=float(r["gamma_y"]),
            gamma_u=float(r["gamma_u"]),
            weight=float(r["weight"]),
        )
    return out


def story_masses(story_df: pd.DataFrame) -> np.ndarray:
    return story_df["Floor_Gravity_kN"].to_numpy(dtype=float) / G  # kN / (m/s2) -> kN*s2/m


def story_elastic_stiffness(story_df: pd.DataFrame, pdelta_factor: np.ndarray | None = None) -> np.ndarray:
    h = story_df["Height_m"].to_numpy(dtype=float)
    ei = story_df["Column_EI_kNm2_per_column"].to_numpy(dtype=float)
    ncol = story_df["Columns_per_storey"].to_numpy(dtype=float)
    # Storey sway stiffness from columns only, fixed-fixed proxy
    k = ncol * 12.0 * ei / np.maximum(h, 1e-6) ** 3  # kN/m
    if pdelta_factor is not None:
        k = k * pdelta_factor
    return k


def pdelta_reduction(story_df: pd.DataFrame) -> np.ndarray:
    h = story_df["Height_m"].to_numpy(dtype=float)
    P = story_df["Floor_Gravity_kN"].to_numpy(dtype=float)
    k_el = story_elastic_stiffness(story_df, None)
    # simple reduction = max(0.55, 1 - P*Δ sensitivity proxy)
    # using P/(k*h) as dimensionless destabilizing index
    idx = P / np.maximum(k_el * h, 1e-6)
    red = 1.0 - 0.35 * np.clip(idx, 0, 1.0)
    return np.clip(red, 0.55, 1.0)


def build_K_from_story_stiffness(k_story: np.ndarray) -> np.ndarray:
    n = len(k_story)
    K = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            if n == 1:
                K[i, i] += k_story[i]
            else:
                K[i, i] += k_story[i] + k_story[i + 1]
                K[i, i + 1] -= k_story[i + 1]
        elif i == n - 1:
            K[i, i] += k_story[i]
            K[i, i - 1] -= k_story[i]
        else:
            K[i, i] += k_story[i] + k_story[i + 1]
            K[i, i - 1] -= k_story[i]
            K[i, i + 1] -= k_story[i + 1]
    return K


def modal_analysis(story_df: pd.DataFrame, n_modes: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = story_masses(story_df)
    k_story = story_elastic_stiffness(story_df, pdelta_reduction(story_df))
    K = build_K_from_story_stiffness(k_story)
    M = np.diag(m)
    # Solve generalized eigenvalue problem using M^-1 K
    A = np.linalg.inv(M) @ K
    vals, vecs = np.linalg.eig(A)
    idx = np.argsort(vals)
    vals = np.real(vals[idx])
    vecs = np.real(vecs[:, idx])

    vals = np.clip(vals, 1e-12, None)
    wn = np.sqrt(vals)
    T = 2 * np.pi / wn

    # mass-normalize and sign-fix
    phis = []
    gamma = []
    mass_part = []
    ones = np.ones(len(m))
    for i in range(min(n_modes, len(m))):
        phi = vecs[:, i]
        if phi[-1] < 0:
            phi = -phi
        mnorm = math.sqrt(phi.T @ M @ phi)
        phi = phi / mnorm
        phis.append(phi)
        g = (phi.T @ M @ ones) / (phi.T @ M @ phi)
        mp = (g ** 2) * (phi.T @ M @ phi) / np.sum(m)
        gamma.append(float(g))
        mass_part.append(float(mp))
    return T[:n_modes], np.column_stack(phis), np.array(gamma), np.array(mass_part)


def capacity_point_for_component(x, y, x1, y1, xu, alpha):
    """Bilinear elastic-plastic-softened proxy."""
    if x <= x1:
        return y1 / max(x1, 1e-9) * x
    k_post = alpha * y1 / max(x1, 1e-9)
    return y1 + k_post * (x - x1)


def component_state_flex(theta, hp: HingeProperty):
    demand_ratio = theta / max(hp.theta_u, 1e-9)
    if theta < hp.theta_y:
        return "Elastic", 1.0
    if theta < 0.5 * hp.theta_u:
        return "Yielded", max(hp.alpha_post, 0.15)
    if theta < hp.theta_u:
        return "LS/CP", max(hp.alpha_post * 0.7, 0.08)
    return "Failed", 0.02


def component_state_shear(gamma, hp: HingeProperty):
    if gamma < hp.gamma_y:
        return "Elastic", 1.0
    if gamma < 0.5 * hp.gamma_u:
        return "Yielded", max(hp.alpha_post, 0.12)
    if gamma < hp.gamma_u:
        return "LS/CP", max(hp.alpha_post * 0.5, 0.06)
    return "Failed", 0.01


def story_response(drift_i, h, beam_hp, col_hp, use_moment=True, use_shear=True):
    phi_beam = drift_i / max(h, 1e-9)      # rotation proxy
    phi_col = 1.35 * drift_i / max(h, 1e-9)

    states = []
    factors = []

    if use_moment:
        s_b, f_b = component_state_flex(phi_beam, beam_hp)
        s_c, f_c = component_state_flex(phi_col, col_hp)
        states.extend([f"Beam-M:{s_b}", f"Col-M:{s_c}"])
        factors.extend([f_b * beam_hp.weight, f_c * col_hp.weight])

    if use_shear:
        gam = drift_i / max(h, 1e-9)
        s_bs, f_bs = component_state_shear(gam, beam_hp)
        s_cs, f_cs = component_state_shear(1.1 * gam, col_hp)
        states.extend([f"Beam-V:{s_bs}", f"Col-V:{s_cs}"])
        factors.extend([f_bs * beam_hp.weight, f_cs * col_hp.weight])

    story_factor = np.clip(np.mean(factors) / max(np.mean([beam_hp.weight, col_hp.weight]), 1e-9), 0.01, 1.0)
    return story_factor, states


def run_pushover(
    story_df: pd.DataFrame,
    hinge_df: pd.DataFrame,
    roof_disp_max: float,
    n_steps: int,
    load_pattern: str,
    mode_number: int,
    include_moment: bool,
    include_shear: bool,
):
    n = len(story_df)
    heights = np.cumsum(story_df["Height_m"].to_numpy(dtype=float))
    m = story_masses(story_df)
    pdel = pdelta_reduction(story_df)
    k0_story = story_elastic_stiffness(story_df, pdelta_factor=pdel)
    K0 = build_K_from_story_stiffness(k0_story)

    T, PHI, Gamma, MP = modal_analysis(story_df, n_modes=min(5, n))
    mode_idx = min(max(mode_number - 1, 0), PHI.shape[1] - 1)

    if load_pattern == "Triangular":
        Fshape = heights / heights.sum()
    elif load_pattern == "Uniform":
        Fshape = np.ones(n) / n
    else:
        phi = PHI[:, mode_idx].copy()
        phi = np.abs(phi)
        Fshape = (m * phi) / np.sum(m * phi)

    beam_hp = hinge_map_from_df(hinge_df)["Beam"]
    col_hp = hinge_map_from_df(hinge_df)["Column"]

    roof_targets = np.linspace(0.0, roof_disp_max, n_steps)
    rows = []
    hinge_records = []

    for step, roof_disp in enumerate(roof_targets, start=1):
        # elastic / current profile by chosen load pattern
        profile = Fshape / max(Fshape[-1], 1e-9) * roof_disp
        story_drifts = np.empty(n)
        story_drifts[0] = profile[0]
        story_drifts[1:] = np.diff(profile)

        story_factors = []
        story_states = []
        for i in range(n):
            sf, states = story_response(
                story_drifts[i],
                float(story_df.iloc[i]["Height_m"]),
                beam_hp,
                col_hp,
                use_moment=include_moment,
                use_shear=include_shear,
            )
            # gravity effect further reduces stiffness for heavily loaded storeys
            grav = float(story_df.iloc[i]["Floor_Gravity_kN"])
            grav_red = np.clip(1.0 - 0.00004 * grav / max(float(story_df.iloc[i]["Columns_per_storey"]), 1.0), 0.60, 1.0)
            sf *= grav_red
            story_factors.append(sf)
            story_states.append(states)

        k_story = k0_story * np.array(story_factors)
        K = build_K_from_story_stiffness(k_story)

        # force needed to reach target roof displacement in current degraded system:
        # u = lam * K^-1 Fshape => lam = roof_disp / u_roof_for_unit
        u_unit = np.linalg.solve(K, Fshape)
        lam = roof_disp / max(u_unit[-1], 1e-9)
        u = lam * u_unit
        Vb = lam * np.sum(Fshape)
        story_forces = lam * Fshape
        story_shears = np.flip(np.cumsum(np.flip(story_forces)))

        # ADRS proxy using first mode participation
        phi1 = np.abs(PHI[:, 0])
        gamma1 = float(Gamma[0])
        mstar = float(phi1.T @ np.diag(m) @ phi1)
        roof_to_sd = max(gamma1 * phi1[-1], 1e-9)
        Sd = u[-1] / roof_to_sd
        Sa = Vb / max(mstar, 1e-9)

        failure_count = 0
        yielded_count = 0
        for i in range(n):
            joined = " | ".join(story_states[i])
            if "Failed" in joined:
                failure_count += 1
            if "Yielded" in joined or "LS/CP" in joined:
                yielded_count += 1
            hinge_records.append({
                "Step": step,
                "Storey": i + 1,
                "Drift_m": float(story_drifts[i]),
                "DriftRatio": float(story_drifts[i] / max(story_df.iloc[i]["Height_m"], 1e-9)),
                "StoryShear_kN": float(story_shears[i]),
                "States": joined,
                "StiffnessFactor": float(story_factors[i]),
            })

        rows.append({
            "Step": step,
            "RoofDisp_m": float(u[-1]),
            "BaseShear_kN": float(Vb),
            "SpectralDisp_m": float(Sd),
            "SpectralAccel_g_proxy": float(Sa / G),
            "YieldedOrWorse_Storeys": int(yielded_count),
            "Failed_Storeys": int(failure_count),
            "ModeUsed": int(mode_idx + 1),
            "Mode1Period_s": float(T[0]),
        })

        if failure_count >= max(1, math.ceil(0.2 * n)):
            break

    results = pd.DataFrame(rows)
    hinge_df_out = pd.DataFrame(hinge_records)

    # crude performance point: demand line intersection using equivalent period secant sweep
    if len(results) > 1:
        Sd = results["SpectralDisp_m"].to_numpy()
        Sa = results["SpectralAccel_g_proxy"].to_numpy()
        # simple elastic demand spectrum proxy
        Ts = max(float(T[0]), 0.2)
        demand = 0.35 / np.sqrt(np.maximum(Sd / max(Sd.max(), 1e-9), 0.05)) * (0.9 + 0.2 / Ts)
        idx_pp = int(np.argmin(np.abs(Sa - demand)))
        pp = {
            "SpectralDisp_m": float(Sd[idx_pp]),
            "SpectralAccel_g_proxy": float(Sa[idx_pp]),
            "RoofDisp_m": float(results.iloc[idx_pp]["RoofDisp_m"]),
            "BaseShear_kN": float(results.iloc[idx_pp]["BaseShear_kN"]),
            "Step": int(results.iloc[idx_pp]["Step"]),
        }
    else:
        pp = None

    modal = {
        "Periods_s": T,
        "Modes": PHI,
        "Gamma": Gamma,
        "MassParticipation": MP,
        "LoadShape": Fshape,
        "Heights_m": heights,
    }
    return results, hinge_df_out, pp, modal


def plot_capacity(results: pd.DataFrame, pp: dict | None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(results["RoofDisp_m"], results["BaseShear_kN"], linewidth=2)
    if pp:
        ax.scatter([pp["RoofDisp_m"]], [pp["BaseShear_kN"]], s=60)
        ax.annotate("Performance Point", (pp["RoofDisp_m"], pp["BaseShear_kN"]))
    ax.set_xlabel("Roof Displacement (m)")
    ax.set_ylabel("Base Shear (kN)")
    ax.set_title("Capacity Curve")
    ax.grid(True, alpha=0.3)
    return fig


def plot_adrs(results: pd.DataFrame, pp: dict | None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    Sd = results["SpectralDisp_m"]
    Sa = results["SpectralAccel_g_proxy"]
    ax.plot(Sd, Sa, linewidth=2, label="Capacity (ADRS)")
    if len(Sd) > 1:
        demand = 0.35 / np.sqrt(np.maximum(Sd / max(float(Sd.max()), 1e-9), 0.05))
        ax.plot(Sd, demand, linestyle="--", label="Demand Proxy")
    if pp:
        ax.scatter([pp["SpectralDisp_m"]], [pp["SpectralAccel_g_proxy"]], s=60)
    ax.set_xlabel("Spectral Displacement, Sd (m)")
    ax.set_ylabel("Spectral Acceleration, Sa (g-proxy)")
    ax.set_title("Capacity Spectrum Method View")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_modes(modal: dict, max_modes: int = 3):
    T = modal["Periods_s"]
    PHI = modal["Modes"]
    z = modal["Heights_m"]
    figs = []
    for i in range(min(max_modes, PHI.shape[1])):
        fig, ax = plt.subplots(figsize=(4.5, 5.0))
        x = PHI[:, i] / max(np.max(np.abs(PHI[:, i])), 1e-9)
        ax.plot(np.r_[0, x], np.r_[0, z], marker="o")
        ax.set_title(f"Mode {i+1}  |  T = {T[i]:.3f} s")
        ax.set_xlabel("Normalized mode shape")
        ax.set_ylabel("Height (m)")
        ax.grid(True, alpha=0.3)
        figs.append(fig)
    return figs


def story_color(states: str):
    if "Failed" in states:
        return "red"
    if "LS/CP" in states:
        return "orange"
    if "Yielded" in states:
        return "gold"
    return "green"


def plot_frame_hinges(story_df: pd.DataFrame, last_hinge_states: pd.DataFrame, n_bays: int):
    h = story_df["Height_m"].to_numpy(dtype=float)
    z = np.r_[0, np.cumsum(h)]
    span = float(story_df["Beam_span_m"].iloc[0])
    xs = np.arange(n_bays + 1) * span

    fig, ax = plt.subplots(figsize=(7, 7))
    # columns
    for x in xs:
        for i in range(len(h)):
            state = last_hinge_states[last_hinge_states["Storey"] == i + 1]["States"].iloc[0]
            ax.plot([x, x], [z[i], z[i + 1]], color=story_color(state), linewidth=3)
    # beams
    for i in range(1, len(z)):
        state = last_hinge_states[last_hinge_states["Storey"] == i]["States"].iloc[0]
        for j in range(n_bays):
            ax.plot([xs[j], xs[j + 1]], [z[i], z[i]], color=story_color(state), linewidth=3)

    for x in xs:
        for zz in z:
            ax.scatter(x, zz, s=20, color="black")

    ax.set_aspect("equal")
    ax.set_title("Frame Hinge-State Visualization")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Height (m)")
    ax.grid(True, alpha=0.2)
    return fig


# =========================================================
# UI
# =========================================================
st.title("Professional Nonlinear Pushover Dashboard v3")
st.caption("Includes user-defined plastic moment/shear hinges, gravity loads, modal analysis, and capacity spectrum view.")

with st.sidebar:
    st.header("Model")
    n_storey = st.slider("Number of storeys", 1, 10, 5)
    n_bays = st.slider("Number of bays", 1, 10, 3)
    load_pattern = st.selectbox("Pushover load pattern", ["Mode Shape", "Triangular", "Uniform"])
    mode_number = st.number_input("Mode number for modal / mode-shape pushover", min_value=1, max_value=max(1, n_storey), value=1, step=1)
    roof_disp_max = st.number_input("Maximum roof displacement (m)", min_value=0.01, max_value=2.0, value=0.25, step=0.01)
    n_steps = st.slider("Pushover steps", 10, 150, 60)
    include_moment = st.checkbox("Include flexural hinges", value=True)
    include_shear = st.checkbox("Include shear hinges", value=True)

if "story_df" not in st.session_state or len(st.session_state.story_df) != n_storey:
    st.session_state.story_df = default_story_data(n_storey)
if "hinge_df" not in st.session_state:
    st.session_state.hinge_df = default_hinge_table()

tab1, tab2, tab3, tab4 = st.tabs(["Geometry + Gravity", "Hinge Properties", "Modal Analysis", "Pushover Results"])

with tab1:
    st.subheader("Storey and gravity definition")
    st.write("Define height, gravity load, frame stiffness proxies, and span. Gravity load influences mass and P-Delta stiffness reduction.")
    story_df = st.data_editor(
        st.session_state.story_df,
        use_container_width=True,
        num_rows="fixed",
        key="story_editor",
    )
    st.session_state.story_df = story_df

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total gravity load (kN)", f"{story_df['Floor_Gravity_kN'].sum():,.1f}")
    with c2:
        st.metric("Approx. total seismic mass (kN·s²/m)", f"{(story_df['Floor_Gravity_kN'].sum()/G):,.1f}")

with tab2:
    st.subheader("User-defined hinge properties")
    st.write("Edit beam and column hinge capacities for plastic moment and shear hinges.")
    hinge_df = st.data_editor(
        st.session_state.hinge_df,
        use_container_width=True,
        num_rows="fixed",
        key="hinge_editor",
    )
    st.session_state.hinge_df = hinge_df
    st.info("My and Vy are capacity proxies used for hinge-state logic. theta and gamma define yielding and ultimate limits.")

with tab3:
    st.subheader("Modal analysis")
    try:
        T, PHI, Gamma, MP = modal_analysis(st.session_state.story_df, n_modes=min(5, n_storey))
        modal = {
            "Periods_s": T,
            "Modes": PHI,
            "Gamma": Gamma,
            "MassParticipation": MP,
            "Heights_m": np.cumsum(st.session_state.story_df["Height_m"].to_numpy(dtype=float)),
        }
        modal_df = pd.DataFrame({
            "Mode": np.arange(1, len(T) + 1),
            "Period_s": T,
            "Frequency_Hz": 1 / np.maximum(T, 1e-9),
            "ParticipationFactor": Gamma,
            "MassParticipationRatio": MP,
        })
        st.dataframe(modal_df, use_container_width=True)

        mode_figs = plot_modes(modal, max_modes=min(3, len(T)))
        cols = st.columns(len(mode_figs))
        for col, fig in zip(cols, mode_figs):
            with col:
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Modal analysis failed: {e}")

with tab4:
    st.subheader("Run nonlinear pushover")
    if st.button("Run Analysis", type="primary"):
        try:
            results, hinge_hist, pp, modal = run_pushover(
                st.session_state.story_df,
                st.session_state.hinge_df,
                roof_disp_max=roof_disp_max,
                n_steps=n_steps,
                load_pattern=load_pattern,
                mode_number=int(mode_number),
                include_moment=include_moment,
                include_shear=include_shear,
            )
            st.session_state.results = results
            st.session_state.hinge_hist = hinge_hist
            st.session_state.pp = pp
            st.session_state.modal_run = modal
            st.success("Analysis completed.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")

    if "results" in st.session_state:
        results = st.session_state.results
        hinge_hist = st.session_state.hinge_hist
        pp = st.session_state.pp
        modal = st.session_state.modal_run

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_capacity(results, pp))
        with c2:
            st.pyplot(plot_adrs(results, pp))

        if pp:
            st.write("**Performance Point (proxy):**")
            st.json(pp)

        last_step = int(results["Step"].iloc[-1])
        last_hinges = hinge_hist[hinge_hist["Step"] == last_step].copy()

        c3, c4 = st.columns([1.1, 1.2])
        with c3:
            st.pyplot(plot_frame_hinges(st.session_state.story_df, last_hinges, n_bays=n_bays))
        with c4:
            st.write("**Final step hinge summary by storey**")
            st.dataframe(last_hinges[["Storey", "DriftRatio", "StoryShear_kN", "States", "StiffnessFactor"]], use_container_width=True)

        st.write("**Pushover result table**")
        st.dataframe(results, use_container_width=True)

        csv1 = results.to_csv(index=False).encode("utf-8")
        csv2 = hinge_hist.to_csv(index=False).encode("utf-8")
        st.download_button("Download pushover results CSV", csv1, file_name="pushover_results.csv", mime="text/csv")
        st.download_button("Download hinge history CSV", csv2, file_name="hinge_history.csv", mime="text/csv")

st.markdown("---")
st.caption("Engineering note: this is a professional screening / teaching tool with user-defined hinges and modal analysis, but it remains an approximate nonlinear model rather than a full commercial finite-element constitutive solver.")
