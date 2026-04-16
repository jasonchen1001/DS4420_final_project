import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Page Config
st.set_page_config(
    page_title="LIPISURV",
    layout="wide",
)

# Sidebar Navigation
st.sidebar.title("LIPISURV")
page = st.sidebar.radio(
    "Navigate",
    ["Project Overview", "Interactive Results"],
)

# PAGE 1: LANDING PAGE
if page == "Project Overview":
    st.title("LIPISURV: Lipid-Informed Multi-Omics Survival Prediction in Breast Cancer")
    st.markdown("**DS 4420: Machine Learning and Data Mining 2 — Final Project  |  Yiyang Bai, Yanzhen Chen  |  Northeastern University**")
    st.markdown("---")

    # Motivation
    st.header("Motivation")
    st.write(
        "Breast cancer is the most common cancer among women worldwide, yet its progression "
        "is highly unpredictable. Two patients with nearly identical clinical profiles can have "
        "drastically different outcomes. This suggests that molecular-level biomarkers may capture "
        "prognostic signals that clinical factors alone cannot explain."
    )



    # Approach
    st.header("Approach")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Method 1: Bayesian Cox Regression")
        st.write(
            "A three-step pipeline: (1) univariate Cox screening of 600+ features, "
            "(2) Lasso / Elastic Net / Adaptive Lasso penalized selection, "
            "(3) full Bayesian Cox model with Laplace priors and MCMC sampling. "
            "This yields interpretable Hazard Ratios with 95% credible intervals."
        )

    with col_b:
        st.subheader("Method 2: MLP Binary Classifier")
        st.write(
            "A multi-layer perceptron (Input -> 32 -> 16 -> 1) with BatchNorm, ReLU, and Dropout, "
            "trained with weighted binary cross-entropy loss (5x weight for the minority class). "
            "All layers, the loss function, and the Adam optimizer were implemented from scratch in NumPy."
        )

    st.markdown("---")

    # Key Findings
    st.header("Key Findings")
    findings = pd.DataFrame({
        "Biomarker": ["CYP4Z1", "Age at Diagnosis", "SPDYC", "ALOX15B", "FABP7"],
        "Hazard Ratio": [1.428, 1.356, 1.331, 0.675, 0.586],
        "95 pct CI": ["[1.105, 1.820]", "[1.075, 1.708]", "[1.059, 1.661]", "[0.487, 0.920]", "[0.428, 0.796]"],
        "Direction": ["Risk Factor", "Risk Factor", "Risk Factor", "Protective", "Protective"],
    })
    st.dataframe(findings, use_container_width=True, hide_index=True)

    st.markdown("---")


# PAGE 2: INTERACTIVE RESULTS
elif page == "Interactive Results":
    st.title("Interactive Results Explorer")
    st.markdown("---")

    (tab1,) = st.tabs([
        "Forest Plot"
    ])

    # TAB 1: Interactive Forest Plot
    with tab1:
        st.subheader("Bayesian Cox: Hazard Ratios with 95 pct Credible Intervals")

        forest_data = pd.DataFrame({
            "Biomarker": [
                "FABP7", "ALOX15B", "MS4A1", "TPRG1", "FOXJ1", "FREM2", "WDR72",
                "NEK10", "C16orf89", "ELOVL2", "CCL19", "GLRA3", "GFRA1", "CLIC6",
                "PIGR", "KRT15", "PSCA", "PAX7", "ZIC2", "NCCRP1", "CLEC3A",
                "SPDYC", "Age", "CYP4Z1"
            ],
            "HR": [
                0.586, 0.675, 0.680, 0.700, 0.720, 0.730, 0.740,
                0.780, 0.790, 0.800, 0.810, 0.820, 0.830, 0.850,
                0.870, 0.880, 0.920, 0.940, 0.960, 0.980, 1.050,
                1.331, 1.356, 1.428
            ],
            "CI_low": [
                0.428, 0.487, 0.440, 0.430, 0.470, 0.460, 0.480,
                0.470, 0.560, 0.560, 0.490, 0.500, 0.540, 0.540,
                0.620, 0.540, 0.670, 0.670, 0.670, 0.680, 0.740,
                1.059, 1.075, 1.105
            ],
            "CI_high": [
                0.796, 0.920, 1.050, 1.130, 1.100, 1.150, 1.140,
                1.280, 1.110, 1.140, 1.340, 1.340, 1.270, 1.340,
                1.220, 1.420, 1.260, 1.310, 1.370, 1.410, 1.490,
                1.661, 1.708, 1.820
            ],
        })

        forest_data["Significant"] = (
            (forest_data["CI_low"] > 1) | (forest_data["CI_high"] < 1)
        )

        plot_df = forest_data.copy()

        fig_forest = go.Figure()

        for _, row in plot_df.iterrows():
            color = "#e74c3c" if row["Significant"] else "#888888"
            fig_forest.add_trace(go.Scatter(
                x=[row["CI_low"], row["CI_high"]],
                y=[row["Biomarker"], row["Biomarker"]],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig_forest.add_trace(go.Scatter(
                x=[row["HR"]],
                y=[row["Biomarker"]],
                mode="markers",
                marker=dict(color=color, size=10, symbol="circle"),
                showlegend=False,
                name=row["Biomarker"],
            ))

        fig_forest.add_vline(x=1, line_dash="dash", line_color="red", opacity=0.5)
        fig_forest.update_layout(
            xaxis_title="Hazard Ratio (log scale)",
            xaxis_type="log",
            yaxis=dict(categoryorder="array", categoryarray=plot_df["Biomarker"].tolist()),
            height=max(400, len(plot_df) * 32),
            margin=dict(l=20, r=20, t=40, b=40),
            template="plotly_white",
        )
        st.plotly_chart(fig_forest, use_container_width=True)