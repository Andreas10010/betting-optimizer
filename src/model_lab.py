"""Interactive logistic model comparison and random feature-subset experiments."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from modeling import FEATURE_GROUPS
from model_experiments import (confidence_summary, eligible_features, logistic_report, rank_combinations,
                               run_combination_search, screening_for_data)


@st.cache_data(max_entries=6)
def cached_report(data, columns):
    return logistic_report(data, list(columns))


@st.cache_data(max_entries=4)
def cached_screening(data):
    return screening_for_data(data)


def probability_summary(report, title):
    st.subheader(title)
    if not report['converged']:
        st.warning('Optimeringen konvergerade inte. Tolka modellens mått med försiktighet.')
    outcomes = report['outcomes']
    train = outcomes[outcomes.partition.eq('Train')].set_index('outcome')
    test = outcomes[outcomes.partition.eq('Test')].set_index('outcome')
    table = pd.DataFrame({
        'Dummy H/D/A': train.dummy_probability,
        'LG medel, train': train.model_probability,
        'Observerat, train': train.observed_frequency,
        'LG medel, test': test.model_probability,
        'Observerat, test': test.observed_frequency,
    })
    st.dataframe(table.map(lambda value: f'{value:.1%}'), width='stretch')
    st.caption('Procenttalen ovan är genomsnittliga sannolikheter över matcher. '
               'Dummy använder samma utfallsandelar från train för alla matcher. '
               'LG ger en egen sannolikhet för varje match; medelvärdet är inte modellens träffsäkerhet. '
               'En förbättrad modell kan ha samma genomsnitt som dummy och ändå ge bättre prognoser för enskilda matcher.')
    metrics = report['metrics']
    visible = metrics[metrics.partition.isin(['Train', 'Test'])][
        ['partition', 'model', 'rows', 'accuracy', 'gini_macro', 'gini_home', 'gini_draw', 'gini_away', 'log_loss', 'brier_score']]
    st.dataframe(visible.rename(columns={'partition': 'Period', 'model': 'Modell', 'rows': 'Matcher',
        'accuracy': 'Accuracy', 'gini_macro': 'Gini medel', 'gini_home': 'Gini H', 'gini_draw': 'Gini D',
        'gini_away': 'Gini A', 'log_loss': 'Log loss', 'brier_score': 'Brier'}).round(4),
        hide_index=True, width='stretch')
    test_scores = metrics[metrics.partition.eq('Test')].set_index('model')
    a, b, c = st.columns(3)
    a.metric('Gini på test', f"{test_scores.loc['LG', 'gini_macro']:.4f}")
    gain = test_scores.loc['Dummy', 'log_loss'] - test_scores.loc['LG', 'log_loss']
    b.metric('Förbättring i log loss mot dummy', f'{gain:+.4f}')
    c.metric('Accuracy på test', f"{test_scores.loc['LG', 'accuracy']:.1%}")


def model_curves(report, key):
    st.write('**Gini / ROC per utfall**')
    roc = report['roc']
    roc = roc[roc.partition.isin(['Train', 'Test'])].copy()
    roc['serie'] = roc.partition + ' · ' + roc.outcome
    diagonal = pd.DataFrame({'fpr': [0., 1.], 'tpr': [0., 1.], 'serie': ['Dummy', 'Dummy']})
    st.line_chart(pd.concat([roc[['fpr', 'tpr', 'serie']], diagonal], ignore_index=True),
                  x='fpr', y='tpr', color='serie', x_label='Falskt positiv andel', y_label='Sant positiv andel')
    st.caption('Gini = 2 × arean under ROC-kurvan − 1. Gini beräknas på alla matcher; '
               'linjerna kan vara förenklade för visning. Utfallet med saknad positiv eller negativ klass får ingen kurva.')
    st.write('**Kalibrering: stämmer sannolikheterna?**')
    calibration = report['calibration']
    calibration = calibration[calibration.partition.isin(['Train', 'Test'])].copy()
    calibration['serie'] = calibration.partition + ' · ' + calibration.outcome
    reference = pd.DataFrame({'mean_probability': [0., 1.], 'observed_frequency': [0., 1.], 'serie': ['Perfekt kalibrering'] * 2})
    st.line_chart(pd.concat([calibration[['mean_probability', 'observed_frequency', 'serie']], reference], ignore_index=True),
                  x='mean_probability', y='observed_frequency', color='serie',
                  x_label='Modellens sannolikhet', y_label='Observerad utfallsandel')
    st.caption('Om matcher ges 70 % sannolikhet för hemmavinst bör ungefär 70 % av dem sluta med hemmavinst. '
               'En linje nära diagonalen visar bättre kalibrering; små grupper ger osäkrare skattningar.')
    with st.expander('Antal matcher bakom kalibreringskurvan'):
        st.dataframe(calibration, hide_index=True, width='stretch')
    threshold = (st.session_state.get('match_confidence_threshold', .60)
                 if st.session_state.get('match_confidence_enabled', True) else 0.)
    st.caption(f'Matchurvalet nedan använder säkerhetsfiltret längst ner på sidan: minst {threshold:.0%}.')
    confidence_rows = []
    for partition in ('Train', 'Test'):
        data = report['predictions'][partition]
        selected = data[data.confidence >= threshold]
        confidence_rows.append({'Period': partition, 'Matcher': len(selected), 'Andel matcher': len(selected) / len(data),
                                'Medelsäkerhet': selected.confidence.mean(),
                                'Andel rätt': selected.predicted_label.eq(selected.target_label).mean() if len(selected) else np.nan})
    st.dataframe(pd.DataFrame(confidence_rows).round(4), hide_index=True, width='stretch')
    test = report['predictions']['Test']
    with st.expander('Matchernas sannolikheter jämfört med dummy'):
        st.dataframe(test[test.confidence >= threshold].tail(100), hide_index=True, width='stretch')
    st.download_button('Ladda ner testprediktioner', test.to_csv(index=False), file_name='lg_test_predictions.csv',
                       mime='text/csv', key=f'download_predictions_{key}')


def render_confidence_footer(report, model_name):
    st.subheader('H/D/A för matcherna vi är säkra på')
    st.write(f'**Modell: {model_name}**')
    enabled = st.checkbox('Filtrera bort osäkra matcher', value=True, key='match_confidence_enabled')
    threshold = st.slider('Minsta sannolikhet för modellens mest sannolika utfall',
                          min_value=0.0, max_value=1.0, value=0.60, step=0.01, format='%.2f',
                          key='match_confidence_threshold', disabled=not enabled)
    st.caption(f'Vid {threshold:.0%} behålls en match om max(P(H), P(D), P(A)) är minst {threshold:.0%}. '
               'Filtret använder modellens förmatchprognos. Resultatet används först när urvalet utvärderas. '
               'Modellen tränas inte om när du ändrar gränsen.')
    table, coverage, selected = confidence_summary(report, threshold, enabled)
    display = coverage.copy()
    for name in ('Andel kvar', 'Andel rätt'):
        display[name] = display[name].map(lambda value: '—' if pd.isna(value) else f'{value:.1%}')
    st.dataframe(display, hide_index=True, width='stretch', key='confidence_coverage')
    empty = coverage.loc[coverage.Kvar.eq(0), 'Period'].tolist()
    if empty:
        st.info('Inga matcher kvar i ' + ' och '.join(empty) + '. Sänk gränsen för att få ett underlag.')
    st.dataframe(table.map(lambda value: '—' if pd.isna(value) else f'{value:.1%}'),
                 width='stretch', key='confidence_outcomes')
    st.bar_chart(table * 100, stack=False, y_label='Andel (%)', x_label='Utfall')
    st.caption('LG medel och observerade andelar räknas om på de kvarvarande matcherna inom respektive period. '
               'Dummy behåller utfallsandelarna från hela train. En högre gräns ger färre matcher och garanterar inte högre träffsäkerhet. '
               'Om gränsen väljs efter testresultat ingår också det valet i modellutvecklingen.')
    with st.expander('Visa testmatcherna som klarade filtret'):
        st.dataframe(selected['Test'].tail(100), hide_index=True, width='stretch')
    st.download_button('Ladda ner filtrerade testmatcher', selected['Test'].to_csv(index=False),
                       file_name='lg_confident_test_matches.csv', mime='text/csv', key='download_confident_matches')


def render_model_lab(engineered, gini_threshold):
    all_columns = FEATURE_GROUPS['all_features']
    st.caption('Alla LG-modeller använder logistisk regression med C=1. Imputering, skalning och koefficienter lärs på train. '
               'Trainresultat beskriver anpassningen; testresultat visar hur modellen fungerar på senare matcher. '
               'Gini mäter rangordning. För sannolikheternas kvalitet jämför du även log loss, Brier och kalibrering.')
    with st.spinner('Tränar LG med alla features och jämför med dummy...'):
        full = cached_report(engineered, tuple(all_columns))
    active_report, active_name = full, f'LG med alla {len(all_columns)} features'
    probability_summary(full, f'Dummy och LG med alla {len(all_columns)} features')
    st.dataframe(full['splits'].rename(columns={'partition': 'Period', 'start': 'Från', 'end': 'Till', 'rows': 'Matcher'}),
                 hide_index=True, width='stretch')
    with st.expander('Gini, kalibrering och matchprognoser för LG med alla features'):
        model_curves(full, 'all')

    st.subheader('Filtrera features före modellering')
    with st.spinner('Mäter varje feature på en senare del av train...'):
        screening = cached_screening(engineered)
    candidates = eligible_features(screening, gini_threshold)
    apply_filter = st.checkbox('Filtrera bort svaga features före LG och kombinationssökning', value=True, key='apply_gini_filter')
    pool = candidates if apply_filter else list(all_columns)
    st.caption(f"Filtret tränar på train fram till {screening.fit_end.iloc[0]:%Y-%m-%d} och mäter Gini "
               f"på {screening.screen_start.iloc[0]:%Y-%m-%d}–{screening.screen_end.iloc[0]:%Y-%m-%d}, också inom train. "
               f"Gini måste vara större än {gini_threshold:.3f}. Validering och test används inte för detta filter.")
    st.write(f'**{len(pool)} features tillgängliga**, varav {len(candidates)} klarar Gini-gränsen.')
    st.caption('Röd text betyder lågt eller ej definierat självständigt Gini, alternativt utebliven konvergens. '
               'Svaga features ensamma kan fortfarande vara användbara tillsammans med andra.')
    screening_view = screening[['feature', 'screen_gini', 'screen_log_loss_gain', 'screen_status']].copy()
    screening_view['inkluderas'] = screening_view.feature.isin(pool)
    bad = set(screening.feature) - set(candidates)
    styled = screening_view.style.map(lambda value: 'color: #d62728' if value in bad else '', subset=['feature'])
    st.dataframe(styled, hide_index=True, width='stretch')
    if not pool:
        st.info('Inga features klarar gränsen. Sänk Gini-gränsen eller stäng av filtret för att träna en modell.')
        render_confidence_footer(active_report, active_name)
        return
    if apply_filter:
        with st.spinner('Tränar LG efter featurefiltret...'):
            filtered = cached_report(engineered, tuple(pool))
        active_report, active_name = filtered, f'LG efter featurefiltret ({len(pool)} features)'
        probability_summary(filtered, f'LG efter filtret: {len(pool)} features')
        with st.expander('Gini och kalibrering för filtrerad LG'):
            model_curves(filtered, 'filtered')

    st.subheader('Sök bland featurekombinationer')
    a, b, c, d = st.columns(4)
    n_simulations = int(a.number_input('Antal simuleringar', min_value=1, max_value=1000, value=50, step=1, key='simulation_count'))
    minimum = int(b.number_input('Min features per modell', min_value=1, max_value=len(all_columns), value=5, key='min_features'))
    maximum = int(c.number_input('Max features per modell', min_value=1, max_value=len(all_columns), value=20, key='max_features'))
    seed = int(d.number_input('Slumpfrö', min_value=0, max_value=2147483647, value=42, key='simulation_seed'))
    valid = minimum <= maximum <= len(pool)
    if not valid:
        st.info(f'Välj min ≤ max och högst {len(pool)} features per modell, eller ändra filtret.')
    st.caption('Varje simulering tränar en unik, slumpad kombination på samma train och mäts på samma validering och test. '
               'Slumpfröet gör kombinationerna reproducerbara. Finns färre unika kombinationer än önskat körs alla tillgängliga.')
    signature = hashlib.sha256(pd.util.hash_pandas_object(engineered, index=True).values.tobytes()).hexdigest()
    settings = (tuple(pool), n_simulations, minimum, maximum, seed)
    if st.button('Kör modellsimuleringar', type='primary', disabled=not valid):
        bar = st.progress(0., text='Tränar kombinationer...')
        try:
            results = run_combination_search(engineered, pool, n_simulations, minimum, maximum, seed,
                progress=lambda current, total: bar.progress(current / total, text=f'Modell {current} av {total}'))
            st.session_state['model_search'] = {'signature': signature, 'settings': settings, 'results': results,
                'run': st.session_state.get('model_search', {}).get('run', 0) + 1}
        except ValueError as error:
            st.error(str(error))
        finally:
            bar.empty()
    saved = st.session_state.get('model_search')
    if saved is None or saved['signature'] != signature:
        render_confidence_footer(active_report, active_name)
        return
    if saved['settings'] != settings:
        st.info('Inställningarna har ändrats. Resultaten nedan gäller föregående körning; tryck Kör för nya resultat.')
    results = saved['results']
    _, requested, used_min, used_max, used_seed = saved['settings']
    st.write(f'**{len(results)} av {requested} önskade kombinationer körda**, {used_min}–{used_max} features, slumpfrö {used_seed}.')
    partition = st.radio('Rangordna modeller på', ['Test', 'Validering'], horizontal=True, key='ranking_partition')
    metric_name = st.selectbox('Välj bästa modell efter', ['Gini (högst)', 'Log loss (lägst)', 'Brier (lägst)'], key='ranking_metric')
    metric = {'Gini (högst)': 'gini_macro', 'Log loss (lägst)': 'log_loss', 'Brier (lägst)': 'brier_score'}[metric_name]
    prefix = 'test' if partition == 'Test' else 'validation'
    ranked = rank_combinations(results, prefix, metric)
    st.caption('Att välja bäst på test gör testperioden till modellvalsdata. Vinnaren är explorativ och behöver '
               'en ny orörd period för en slutlig bedömning. Du kan också rangordna på validering.')
    if ranked.empty:
        st.info('Ingen konvergerad modell har ett definierat värde för valt mått. Prova log loss om Gini saknas.')
        render_confidence_footer(active_report, active_name)
        return
    winner = ranked.iloc[0]
    st.success(f"Bäst enligt {metric_name.lower()} på {partition.lower()}: modell {winner.simulation}, "
               f"{winner.n_features} features, {winner[f'{prefix}_{metric}']:.4f}.")
    visible = ranked[['simulation', 'n_features', 'train_gini_macro', 'validation_gini_macro', 'test_gini_macro',
                      'train_log_loss', 'validation_log_loss', 'test_log_loss', 'test_log_loss_gain', 'features']].copy()
    visible['features'] = visible.features.map(', '.join)
    st.dataframe(visible, hide_index=True, width='stretch')
    failed = results[~results.status.eq('OK')]
    if not failed.empty:
        st.warning(f'{len(failed)} modeller konvergerade inte och kan inte utses till vinnare.')
        st.dataframe(failed[['simulation', 'status']], hide_index=True)
    chart = results.sort_values('simulation').set_index('simulation')
    curves = chart[['train_gini_macro', 'validation_gini_macro', 'test_gini_macro']].rename(
        columns={'train_gini_macro': 'Train', 'validation_gini_macro': 'Validering', 'test_gini_macro': 'Test'})
    curves['Bästa test hittills'] = chart['test_gini_macro'].where(chart.status.eq('OK')).cummax()
    st.write('**Gini för kombinationerna**')
    st.line_chart(curves, x_label='Simulering', y_label='Gini, medel H/D/A')
    st.caption('Varje punkt är en separat modell. Linjen Bästa test hittills visar högsta observerade test-Gini, '
               'inte en bevisad förbättring på nya matcher.')
    export = results.copy()
    export['features'] = export.features.map(', '.join)
    st.download_button('Ladda ner simuleringarna (CSV)', export.to_csv(index=False), file_name='model_simulations.csv', mime='text/csv')
    chosen = st.selectbox('Granska en simulerad modell', ranked.simulation.tolist(),
                          key=f"chosen_simulation_{saved['run']}_{prefix}_{metric}")
    selected = results.loc[results.simulation.eq(chosen)].iloc[0]
    st.write('**Features i modellen:** ' + ', '.join(selected.features))
    report = cached_report(engineered, tuple(selected.features))
    probability_summary(report, f'Dummy och modell {chosen}: train jämfört med test')
    model_curves(report, f"simulation_{saved['run']}_{chosen}")
    render_confidence_footer(report, f'Simulerad modell {chosen} ({len(selected.features)} features)')
