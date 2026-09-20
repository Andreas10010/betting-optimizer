"""Lagvy för den bästa logistiska modellen: matcher, sannolikheter och förklaring."""
from __future__ import annotations

import altair as alt
import pandas as pd
import requests
import streamlit as st

try:
    from .data_loader import load_raw_data
    from .feature_engineering import engineer_features
    from .model_experiments import screening_for_data
    from .team_dataset import build_team_match_dataset
    from .team_model import (
        BEST_LOGISTIC,
        CLASS_ORDER,
        LEAGUE,
        TEST_SEASON,
        add_predictions,
        artifact_path,
        class_shares,
        evaluate_predictions,
        explain_row,
        feature_label,
        kept_numeric_features,
        load_trained_model,
        numeric_feature_columns,
        save_trained_model,
        split_train_test,
        train_all_history_model,
    )
except ImportError:
    from data_loader import load_raw_data
    from feature_engineering import engineer_features
    from model_experiments import screening_for_data
    from team_dataset import build_team_match_dataset
    from team_model import (
        BEST_LOGISTIC,
        CLASS_ORDER,
        LEAGUE,
        TEST_SEASON,
        add_predictions,
        artifact_path,
        class_shares,
        evaluate_predictions,
        explain_row,
        feature_label,
        kept_numeric_features,
        load_trained_model,
        numeric_feature_columns,
        save_trained_model,
        split_train_test,
        train_all_history_model,
    )


@st.cache_data(max_entries=4)
def cached_screening(data):
    return screening_for_data(data)


@st.cache_data(ttl=3600, max_entries=2)
def load_team_model_data(league=LEAGUE, through_season=TEST_SEASON):
    raw = load_raw_data()
    history = raw.loc[raw["League"].eq(league) & raw["Season"].astype(str).le(through_season)].copy()
    if history.empty:
        raise ValueError(f"Ingen data för {league} t.o.m. {through_season}.")
    engineered = engineer_features(history)
    return build_team_match_dataset(history, engineered, season_label=None)


@st.cache_data(max_entries=2)
def cached_bundle(path, mtime):
    _ = mtime
    bundle = load_trained_model(path)
    if "test" not in bundle or "Predicted" not in bundle["test"].columns:
        train, test = split_train_test(bundle["team_data"])
        bundle["test"] = add_predictions(bundle["model"], test)
        bundle["metrics"] = evaluate_predictions(bundle["test"], dummy_priors=class_shares(train["Target"]))
    return bundle


FAIR_ODDS_TICKS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0)
FAIR_ODDS_PROB_TICKS = [100.0 / odds for odds in FAIR_ODDS_TICKS]


def fair_decimal_odds(probability):
    """Fair decimal odds implied by a model probability: 1 / p."""
    try:
        value = float(probability)
    except (TypeError, ValueError):
        return float("nan")
    if value <= 0:
        return float("nan")
    return 1.0 / value


def probability_figure(probabilities):
    frame = pd.DataFrame({
        "Utfall": list(CLASS_ORDER),
        "Sannolikhet": [100 * probabilities[label] for label in CLASS_ORDER],
        "Odds": [fair_decimal_odds(probabilities[label]) for label in CLASS_ORDER],
    })
    shared_x = alt.X("Utfall:N", sort=list(CLASS_ORDER), title="Utfall")
    probability_y = alt.Y(
        "Sannolikhet:Q",
        scale=alt.Scale(domain=[0, 108]),
        axis=alt.Axis(
            title="Sannolikhet (%)",
            orient="left",
            values=[0, 20, 40, 60, 80, 100],
        ),
    )
    tooltip = [
        "Utfall",
        alt.Tooltip("Sannolikhet:Q", title="Sannolikhet (%)", format=".1f"),
        alt.Tooltip("Odds:Q", title="Fair odds", format=".2f"),
    ]
    bars = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=shared_x,
            y=probability_y,
            color=alt.Color(
                "Utfall:N",
                scale=alt.Scale(domain=list(CLASS_ORDER), range=["#b2182b", "#7f7f7f", "#2c7bb6"]),
                legend=None,
            ),
            tooltip=tooltip,
        )
    )
    odds_axis = (
        alt.Chart(frame)
        .mark_point(opacity=0)
        .encode(
            x=shared_x,
            y=alt.Y(
                "Sannolikhet:Q",
                scale=alt.Scale(domain=[0, 108]),
                axis=alt.Axis(
                    title="Fair odds",
                    orient="right",
                    values=FAIR_ODDS_PROB_TICKS,
                    labelExpr="format(100 / datum.value, '.2f')",
                    grid=False,
                ),
            ),
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(dy=-10, fontSize=12, fontWeight=600)
        .encode(
            x=shared_x,
            y=alt.Y("Sannolikhet:Q", scale=alt.Scale(domain=[0, 108]), axis=None),
            text=alt.Text("Odds:Q", format=".2f"),
            tooltip=tooltip,
        )
    )
    return (
        alt.layer(bars, odds_axis, labels)
        .resolve_scale(y="shared")
        .resolve_axis(y="independent")
        .properties(title="Modellens sannolikheter för det valda laget", height=260)
    )


def contribution_figure(explanation):
    intercept_row = pd.DataFrame({
        "label": [feature_label("intercept")],
        "contribution": [explanation["intercept"]],
    })
    chart = pd.concat(
        [explanation["contributions"][["label", "contribution"]], intercept_row],
        ignore_index=True,
    )
    chart["riktning"] = chart["contribution"].ge(0).map({True: "För prediktionen", False: "Mot prediktionen"})
    order = chart.sort_values("contribution")["label"].tolist()
    bars = (
        alt.Chart(chart)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Bidrag till log-odds"),
            y=alt.Y("label:N", sort=order, title="Feature"),
            color=alt.Color(
                "riktning:N",
                scale=alt.Scale(
                    domain=["För prediktionen", "Mot prediktionen"],
                    range=["#2c7bb6", "#b2182b"],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=["label", alt.Tooltip("contribution:Q", format=".3f")],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#888").encode(x="x:Q")
    return (bars + zero).properties(
        title=f"Varför modellen lutar mot {explanation['outcome']}",
        height=max(360, 26 * len(chart)),
    )


def render_match_explanation(model, match):
    explanation = explain_row(model, match)
    actual = match["Target"]
    predicted = explanation["predicted"]
    headline = f"{match['Team']} mot {match['Opponent']} · {pd.to_datetime(match['MatchDate']):%Y-%m-%d}"
    st.subheader(headline)
    a, b, c, d = st.columns(4)
    a.metric("Plats", match["Venue"])
    b.metric("Prediktion", predicted, f"{explanation['probabilities'][predicted]:.0%}")
    c.metric("Faktiskt utfall", actual)
    d.metric("Rätt?", "Ja" if predicted == actual else "Nej")
    st.altair_chart(probability_figure(explanation["probabilities"]), width="stretch")
    st.caption(
        "Höger axel och siffrorna ovanför staplarna är fair decimalodds (1 / sannolikhet): "
        "vad ett spelbolag borde sätta utan marginal. Lika chans (33 %) ger 3,00; 50 % ger 2,00."
    )
    st.altair_chart(contribution_figure(explanation), width="stretch")
    st.caption(
        "Staplarna visar hur mycket varje feature, efter skalning, flyttar log-odds för det predicerade "
        "utfallet. Positiva värden stöder prognosen, negativa motverkar den. Sannolikheten beror också på "
        "log-odds för de två andra utfallen. Basnivån är modellens intercept."
    )
    detail = explanation["contributions"].copy()
    raw = explanation["raw_values"]
    detail["raw_value"] = detail["feature"].map(lambda name: raw.get("Venue") if name.startswith("Venue") else raw.get(name))
    display = detail[["label", "raw_value", "contribution"]].rename(columns={
        "label": "Feature",
        "raw_value": "Värde före skalning",
        "contribution": "Bidrag till log-odds",
    })
    intercept = pd.DataFrame({
        "Feature": [feature_label("intercept")],
        "Värde före skalning": [None],
        "Bidrag till log-odds": [explanation["intercept"]],
    })
    st.dataframe(
        pd.concat([display, intercept], ignore_index=True).round(3),
        hide_index=True,
        width="stretch",
    )


def render_model_lab(engineered=None, gini_threshold=None):
    _ = (engineered, gini_threshold)
    st.subheader("Bästa logistiska modellen")
    st.caption(
        f"Samma LG-upplägg som i all-historik-notebooken: featuregrupp `{BEST_LOGISTIC['feature_set']}`, "
        f"C={BEST_LOGISTIC['C']}, ingen klassvikt. Korrelationsrensning (0,90) och skalning lärs på train. "
        f"Train är all Premier League-historik före {TEST_SEASON}; test är orörd {TEST_SEASON}."
    )
    path = artifact_path()
    retrain = st.button("Träna om och spara modellen", help="Hämtar data, tränar LG på all historik före 2023/24 och skriver över den sparade filen.")
    bundle = None
    if path.exists() and not retrain:
        try:
            bundle = cached_bundle(str(path), path.stat().st_mtime)
            st.caption(f"Laddad sparad modell: `{path}`")
        except Exception as error:
            st.warning(f"Kunde inte läsa den sparade modellen ({error}). Tränar om.")
    if bundle is None:
        with st.spinner("Laddar Premier League, bygger lagfeatures, tränar LG på all historik och sparar modellen..."):
            try:
                team_data = load_team_model_data()
            except requests.RequestException as error:
                st.error("Kunde inte hämta matchdata. Kontrollera nätanslutningen och försök igen.")
                with st.expander("Felmeddelande"):
                    st.code(str(error), language="text")
                return
            bundle = train_all_history_model(team_data)
            save_trained_model(path, bundle)
            st.success(f"Sparad modell: `{path}`")
    model = bundle["model"]
    predicted_test = bundle["test"]
    metrics = bundle["metrics"]
    kept = kept_numeric_features(model)
    dropped = [name for name in numeric_feature_columns() if name not in kept]
    st.write(
        f"**Behållna features efter korrelationsrensning:** {len(kept) + 1} "
        f"(Venue plus {len(kept)} numeriska). "
        + (f"Borttagna: {', '.join(dropped)}." if dropped else "")
    )
    a, b, c, d = st.columns(4)
    a.metric("Testaccuracy", f"{metrics['accuracy']:.1%}", f"{metrics['accuracy'] - metrics['dummy_accuracy']:+.1%} mot dummy")
    b.metric("Test log loss", f"{metrics['log_loss']:.3f}", f"{metrics['dummy_log_loss'] - metrics['log_loss']:+.3f} mot dummy")
    c.metric("Test Gini, medel", f"{metrics['gini_macro']:.3f}")
    d.metric("Testmatcher", f"{metrics['rows']:,}")
    st.caption(
        f"Train: {bundle.get('train_rows', '—')} lagrader, {bundle.get('train_start', '—')}–{bundle.get('train_end', '—')}. "
        f"Dummy använder utfallsandelarna i den historiken. "
        f"Gini per utfall på test: "
        + ", ".join(f"{label} {metrics['gini'][label]:.3f}" for label in CLASS_ORDER)
        + "."
    )

    teams = sorted(predicted_test["Team"].dropna().unique())
    default_index = teams.index("Arsenal") if "Arsenal" in teams else 0
    st.subheader("Matcher per lag")
    team = st.selectbox("Välj lag", teams, index=default_index)
    team_matches = (
        predicted_test.loc[predicted_test["Team"].eq(team)]
        .sort_values("MatchDate")
        .reset_index(drop=True)
    )
    correct = int(team_matches["Correct"].sum())
    st.write(
        f"**{team} i {TEST_SEASON}:** {correct} rätt av {len(team_matches)} matcher "
        f"({correct / len(team_matches):.0%}). Klicka på en rad för att se förklaringen."
    )
    table = team_matches[[
        "MatchDate", "Opponent", "Venue", "Target", "Predicted",
        "ProbabilityWin", "ProbabilityDraw", "ProbabilityLoss", "Correct",
    ]].copy()
    table["MatchDate"] = pd.to_datetime(table["MatchDate"]).dt.strftime("%Y-%m-%d")
    event = st.dataframe(
        table.rename(columns={
            "MatchDate": "Datum",
            "Opponent": "Motståndare",
            "Venue": "Plats",
            "Target": "Utfall",
            "Predicted": "Prediktion",
            "ProbabilityWin": "P(vinst)",
            "ProbabilityDraw": "P(oavgjort)",
            "ProbabilityLoss": "P(förlust)",
            "Correct": "Rätt",
        }),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"team_matches_{team}",
        column_config={
            "P(vinst)": st.column_config.NumberColumn(format="%.1%"),
            "P(oavgjort)": st.column_config.NumberColumn(format="%.1%"),
            "P(förlust)": st.column_config.NumberColumn(format="%.1%"),
        },
    )
    selected = []
    if event is not None and event.selection is not None:
        selected = list(event.selection.rows)
    if not selected:
        st.info("Klicka på en match i tabellen för att se vilka features som driver prediktionen.")
        return
    render_match_explanation(model, team_matches.iloc[selected[0]])
