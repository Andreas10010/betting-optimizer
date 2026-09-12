from __future__ import annotations

import streamlit as st
import requests

from data_loader import load_raw_data
from feature_engineering import engineer_features
from feature_explorer import render_feature_explorer
from model_lab import render_model_lab
from modeling import (
    FEATURE_GROUPS,
    build_modeling_data,
    compute_feature_importances,
    summarize_edge,
    train_baseline,
    train_ml_model,
    walk_forward_compare,
)


st.set_page_config(page_title="Betting Optimizer", layout="wide")

st.title("Betting Optimizer Dashboard")
st.caption("Pre-match feature engineering, time-based benchmarking, bookmaker edge analysis, and feature-combo comparison.")

with st.sidebar:
    st.header("Settings")
    page = st.radio("Sida", ["Modellering", "Features"], horizontal=True)
    rows = st.number_input("Rows to load", min_value=1000, max_value=50000, value=5000, step=1000)
    if page == "Features":
        feature_group_name = st.selectbox(
            "Feature combo", options=list(FEATURE_GROUPS.keys()), index=2,
        )
        edge_threshold = st.slider("Min edge to inspect (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
    gini_threshold = st.number_input("Minsta Gini för featurefilter", min_value=0.0, max_value=1.0,
                                    value=0.02, step=0.01, format="%.3f", key="screen_threshold")

@st.cache_data(ttl=3600, max_entries=4)
def load_and_engineer(rows):
    raw = load_raw_data(nrows=rows)
    return raw, engineer_features(raw)


with st.spinner("Loading data and engineering pre-match features..."):
    try:
        df, engineered = load_and_engineer(int(rows))
    except requests.RequestException as error:
        st.error("Kunde inte hämta matchdata. Kontrollera nätanslutningen och försök igen.")
        with st.expander("Felmeddelande"):
            st.code(str(error), language="text")
        st.stop()

st.caption(f"Inlästa matcher: {len(df):,}")
st.caption(f"Inlästa matchdatum: {engineered.MatchDate.min():%Y-%m-%d}–{engineered.MatchDate.max():%Y-%m-%d}. "
           "Radgränsen väljer de första raderna i datakällan. Tabellplaceringarna gäller före varje historisk matchdag.")
with st.expander("Visa rådata"):
    st.dataframe(df.head(10), width="stretch")

if page == "Modellering":
    try:
        render_model_lab(engineered, float(gini_threshold))
    except ValueError as error:
        st.error(str(error))
    st.stop()

@st.cache_resource(max_entries=4)
def fit_models(model_df, feature_cols):
    return train_baseline(model_df, feature_cols), train_ml_model(model_df, feature_cols)

with st.spinner("Training models..."):
    feature_cols = FEATURE_GROUPS[feature_group_name]
    model_df, feature_cols = build_modeling_data(engineered, feature_cols)
    try:
        baseline, ml = fit_models(model_df, feature_cols)
        baseline_pipe, baseline_metrics, baseline_splits = baseline
        ml_pipe, ml_metrics, ml_splits = ml
    except ValueError as error:
        st.error(str(error))
        st.stop()

st.subheader("Model comparison")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline log loss", round(baseline_metrics["log_loss"], 4))
col2.metric("Baseline accuracy", round(baseline_metrics["accuracy"], 4))
col3.metric("ML log loss", round(ml_metrics["log_loss"], 4))
col4.metric("ML accuracy", round(ml_metrics["accuracy"], 4))
st.caption(f"Multiclass Brier score — baseline: {baseline_metrics['brier_score']:.4f}, ML: {ml_metrics['brier_score']:.4f}. Lower is better.")
st.caption(f"Selected model inputs: {len(feature_cols)}. Utforska varje feature nedan.")

render_feature_explorer(engineered, feature_cols, float(gini_threshold))

st.subheader("Walk-forward feature comparison")
st.caption("Compare feature families on identical dates using logistic regression. The final 15% of dates are held out. Lower log loss and Brier score are better.")
if st.button("Run walk-forward comparison"):
    with st.spinner("Evaluating expanding training windows..."):
        try:
            comparison, calibration = walk_forward_compare(engineered)
            st.dataframe(comparison, width="stretch")
            st.caption("Calibration: predicted probabilities should be close to observed frequencies; inspect bin counts too.")
            st.dataframe(calibration, width="stretch")
        except ValueError as error:
            st.error(str(error))

st.subheader("Split sizes")
st.write(
    {
        "train_rows": baseline_splits["train_rows"],
        "val_rows": baseline_splits["val_rows"],
        "test_rows": baseline_splits["test_rows"],
    }
)

st.subheader("Feature importance (ML model)")
importance = compute_feature_importances(ml_pipe)
if importance.empty:
    st.info("Feature importance is not available for the trained model.")
else:
    st.dataframe(importance.head(25), width="stretch")
    st.bar_chart(importance.head(25).set_index("feature")["importance"])

st.subheader("Bookmaker edge analysis")
edge_df = summarize_edge(ml_pipe, feature_cols, ml_splits["test_df"])
if edge_df is None:
    st.warning("Bookmaker odds columns were not found in the loaded dataset, so edge analysis was skipped.")
else:
    edge_df = edge_df.copy()
    edge_df["best_edge_pct"] = edge_df["best_edge"] * 100
    filtered = edge_df[edge_df["best_edge_pct"] >= edge_threshold].copy()
    st.write(f"Rows with edge >= {edge_threshold}%: {len(filtered)}")
    st.dataframe(filtered[["MatchDate", "HomeTeam", "AwayTeam", "best_edge_pct", "prob_home", "prob_draw", "prob_away", "market_prob_home", "market_prob_draw", "market_prob_away"]].head(50), width="stretch")
