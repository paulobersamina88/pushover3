import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Nonlinear Pushover Analysis (Conceptual Tool)")

# -----------------------------
# USER INPUT
# -----------------------------
st.sidebar.header("Model Parameters")

n_storey = st.sidebar.slider("Number of Storeys", 1, 10, 5)
n_bay = st.sidebar.slider("Number of Bays", 1, 10, 3)
bay_width = st.sidebar.number_input("Bay Width (m)", value=5.0)
storey_height = st.sidebar.number_input("Storey Height (m)", value=3.0)

base_shear_coeff = st.sidebar.number_input("Base Shear Coefficient", value=0.1)
yield_disp = st.sidebar.number_input("Yield Displacement (m)", value=0.05)
ultimate_disp = st.sidebar.number_input("Ultimate Displacement (m)", value=0.3)

# -----------------------------
# FRAME GEOMETRY
# -----------------------------
def generate_frame():
    nodes = []
    for i in range(n_storey + 1):
        for j in range(n_bay + 1):
            nodes.append((j * bay_width, i * storey_height))
    return np.array(nodes)

nodes = generate_frame()

# -----------------------------
# PUSHOVER ANALYSIS (SIMPLIFIED)
# -----------------------------
disp = np.linspace(0, ultimate_disp, 50)
force = []

for d in disp:
    if d <= yield_disp:
        f = d / yield_disp
    else:
        f = 1.0 - 0.3 * ((d - yield_disp) / (ultimate_disp - yield_disp))
    force.append(f)

force = np.array(force) * base_shear_coeff * 1000

# -----------------------------
# PLOTS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Capacity Curve")
    fig1, ax1 = plt.subplots()
    ax1.plot(disp, force)
    ax1.set_xlabel("Displacement (m)")
    ax1.set_ylabel("Base Shear")
    ax1.grid()
    st.pyplot(fig1)

with col2:
    st.subheader("Capacity Spectrum (ADRS)")
    Sd = disp
    Sa = force / 1000
    fig2, ax2 = plt.subplots()
    ax2.plot(Sd, Sa)
    ax2.set_xlabel("Spectral Displacement")
    ax2.set_ylabel("Spectral Acceleration")
    ax2.grid()
    st.pyplot(fig2)

# -----------------------------
# HINGE VISUALIZATION
# -----------------------------
st.subheader("Hinge Visualization")

fig3, ax3 = plt.subplots()

for (x, y) in nodes:
    color = "green"
    if y > storey_height * (n_storey * 0.7):
        color = "red"
    elif y > storey_height * (n_storey * 0.4):
        color = "orange"
    ax3.scatter(x, y, c=color)

ax3.set_aspect('equal')
ax3.set_title("Frame with Hinge States")
st.pyplot(fig3)

st.success("Analysis Complete (Conceptual Tool)")
