"""Streamlit view for feature definitions and standalone predictive diagnostics."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from feature_catalog import build_feature_catalog
from feature_diagnostics import OUTCOMES, evaluate_individual_features
from feature_engineering import FEATURE_ALIASES
from modeling import FEATURE_GROUPS
from model_lab import cached_screening
from model_experiments import eligible_features


@st.cache_data(max_entries=4)
def cached_diagnostics(engineered):
    return evaluate_individual_features(engineered)


def _metric(value):
    return "Ej definierat" if pd.isna(value) else f"{value:.4f}"


def render_feature_explorer(engineered, selected_model_features, gini_threshold=.02):
    st.subheader("Features: förklaring och värde mot dummy")
    st.write("Varje feature får en egen logistisk regression med just den featuren som input. "
             "Alla modeller tränas på samma äldre matcher och jämförs på samma senare testmatcher.")
    with st.spinner("Beräknar Gini och dummyjämförelse för varje feature..."):
        scores, outcomes, dummy, split = cached_diagnostics(engineered)
    st.caption(
        f"Train: {split['train_start']:%Y-%m-%d}–{split['train_end']:%Y-%m-%d} ({split['train_rows']:,} matcher). "
        f"Test: {split['test_start']:%Y-%m-%d}–{split['test_end']:%Y-%m-%d} ({split['test_rows']:,} matcher). "
        f"{split['validation_rows']:,} valideringsmatcher används inte i denna jämförelse."
    )
    st.write("**Dummy: träningsdatans utfallsandelar**")
    st.caption("Samma tre sannolikheter ges till varje testmatch. Utfallsandelarna i test visas enbart som jämförelse.")
    for column, row in zip(st.columns(3), outcomes.itertuples()):
        column.metric(row.outcome, f"{row.dummy_probability:.1%}")
        column.caption(f"Observerat i test: {row.test_frequency:.1%} ({row.test_count:,} matcher)")
    a, b, c = st.columns(3)
    a.metric("Dummy log loss", _metric(dummy["log_loss"]))
    b.metric("Dummy Gini, medel H/D/A", _metric(dummy["gini_macro"]))
    c.metric("Dummy accuracy", f"{dummy['accuracy']:.1%}")
    st.caption("Dummy accuracy väljer alltid det vanligaste utfallet i train. "
               "Gini = 2 × AUC − 1 för varje utfall mot de övriga: 0 = ingen rangordningsförmåga, "
               "1 = perfekt rangordning. Gini kan vara negativt. Om ett utfall saknas i test är dess Gini inte definierat.")

    catalog = build_feature_catalog()
    view = scores.copy()
    view["group"] = view.feature.map(lambda name: catalog[name]["group"])
    view["in_model"] = view.feature.isin(selected_model_features)
    screening = cached_screening(engineered)
    eligible = eligible_features(screening, gini_threshold)
    view = view.merge(screening[["feature", "screen_gini"]], on="feature", how="left")
    view["weak"] = ~view.feature.isin(eligible)
    hide_weak = st.checkbox("Dölj features med svagt självständigt Gini", value=False)
    st.caption(f"Röda features klarar inte gränsen Gini > {gini_threshold:.3f} på den interna valideringsdelen av train. "
               "Det är samma filter som på modelleringssidan. Kolumnerna Gini H/D/A avser fortfarande test.")
    show_only_selected = st.checkbox("Visa endast features i vald modell", value=False)
    query = st.text_input("Sök feature eller kategori", placeholder="Exempel: position, Tabell, elo, skott")
    if show_only_selected:
        view = view[view.in_model]
    if hide_weak:
        view = view[~view.weak]
    if query:
        descriptions = view.feature.map(lambda name: catalog[name]["description"])
        mask = (view.feature.str.contains(query, case=False, regex=False)
                | view.group.str.contains(query, case=False, regex=False)
                | descriptions.str.contains(query, case=False, regex=False))
        view = view[mask]
    view = view.reset_index(drop=True)
    options = sorted(scores.feature)
    if st.session_state.get("feature_to_inspect") not in options:
        st.session_state["feature_to_inspect"] = "home_position"
    st.selectbox("Feature att granska", options=options, index=None, key="feature_to_inspect")
    st.caption("Klicka på en rad för att öppna dess förklaring, eller välj en feature ovan. "
               "Positiv Δ log loss betyder att featuren slår dummy; tabellen sorteras efter denna förbättring.")
    table_key = f"feature_ranking_{show_only_selected}_{hide_weak}_{gini_threshold}_{query}"

    def select_feature():
        rows = st.session_state[table_key]["selection"]["rows"]
        if rows and rows[0] < len(view):
            st.session_state["feature_to_inspect"] = view.iloc[rows[0]]["feature"]

    weak_names = set(view.loc[view.weak, "feature"])
    display = view[["feature", "group", "in_model", "screen_gini", "gini_macro", "gini_home", "gini_draw", "gini_away",
                    "log_loss", "log_loss_gain", "test_missing", "status"]]
    st.dataframe(
        display.style.map(lambda name: "color: #d62728" if name in weak_names else "", subset=["feature"]),
        hide_index=True, width="stretch", on_select=select_feature, selection_mode="single-row", key=table_key,
        column_config={
            "feature": "Feature", "group": "Kategori", "in_model": "I vald modell",
            "screen_gini": st.column_config.NumberColumn("Gini för filter (inom train)", format="%.4f"),
            "gini_macro": st.column_config.NumberColumn("Gini medel", format="%.4f"),
            "gini_home": st.column_config.NumberColumn("Gini H", format="%.4f"),
            "gini_draw": st.column_config.NumberColumn("Gini D", format="%.4f"),
            "gini_away": st.column_config.NumberColumn("Gini A", format="%.4f"),
            "log_loss": st.column_config.NumberColumn("Log loss", format="%.4f"),
            "log_loss_gain": st.column_config.NumberColumn("Δ log loss mot dummy", format="%.4f"),
            "test_missing": st.column_config.NumberColumn("Andel saknat i test", format="%.3f"),
            "status": "Status",
        },
    )
    st.download_button("Ladda ner alla featuremått (CSV)", scores.to_csv(index=False),
                       file_name="feature_diagnostics.csv", mime="text/csv")
    feature = st.session_state["feature_to_inspect"] or "home_position"
    definition = catalog[feature]
    row = scores.set_index("feature").loc[feature]
    st.markdown(f"**{feature}**")
    st.write(definition["description"])
    st.code(definition["formula"], language="text")
    st.caption(definition["notes"])
    included = [name for name, columns in FEATURE_GROUPS.items() if feature in columns]
    st.caption("Ingår i: " + ", ".join(included))
    aliases = [alias for alias, canonical in FEATURE_ALIASES.items() if canonical == feature]
    if aliases:
        st.caption("Samma definition finns även under namnen: " + ", ".join(aliases))
    if row["status"] != "OK":
        st.info(row["status"] + ". En konstant eller helt saknad träningsfeature använder dummyprognosen.")
    a, b, c = st.columns(3)
    a.metric("Feature log loss", _metric(row["log_loss"]), delta=f"{row['log_loss_gain']:+.4f} förbättring mot dummy")
    b.metric("Feature Gini, medel H/D/A", _metric(row["gini_macro"]))
    c.metric("Feature accuracy", f"{row['accuracy']:.1%}", delta=f"{100 * row['accuracy_gain']:+.2f} procentenheter mot dummy")
    detail = []
    for name, (_, title) in OUTCOMES.items():
        detail.append({"Utfall": title, "Dummy Gini": dummy[f"gini_{name}"], "Feature Gini": row[f"gini_{name}"],
                       "Dummy log loss": dummy[f"log_loss_{name}"], "Feature log loss": row[f"log_loss_{name}"],
                       "Δ log loss mot dummy": row[f"log_loss_gain_{name}"]})
    st.dataframe(pd.DataFrame(detail), hide_index=True, width="stretch")
    st.caption("Log loss per utfall avser det utfallet mot de två övriga (binärt); övergripande log loss avser alla tre utfall. "
               f"Saknade värden: {row['train_missing']:.1%} i train och {row['test_missing']:.1%} i test. "
               "Imputering, skalning och modellparametrar lärs enbart på train.")
    with st.expander("Se featurevärden för testmatcher"):
        test = engineered[engineered.MatchDate.between(split["test_start"], split["test_end"]) & engineered.target_label.notna()]
        columns = list(dict.fromkeys(["MatchDate", "League", "HomeTeam", "AwayTeam", feature, "target"]))
        st.dataframe(test[columns].tail(100), hide_index=True, width="stretch")
    st.caption("Detta mäter featurens prognosvärde ensam. Gini och förbättring mot dummy visar inte bettingavkastning. "
               "Om du väljer features utifrån denna testvy blir perioden en del av modellutvecklingen; "
               "slutlig prestanda behöver då mätas på en ny, orörd period.")
