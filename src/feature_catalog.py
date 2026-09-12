"""Human-readable definitions for every exported feature; no silent fallback."""
from __future__ import annotations

try:
    from .feature_engineering import ALL_FEATURE_COLUMNS, FEATURE_ALIASES
except ImportError:
    from feature_engineering import ALL_FEATURE_COLUMNS, FEATURE_ALIASES


TEMPORAL_NOTE = "Endast avslutade matcher på tidigare datum används. Lagets historik hålls separat per liga."
STANDINGS_NOTE = (
    "Tabellen gäller lagets liga och säsong precis före matchdagen. Poängen nollställs varje säsong. "
    "Endast ditt datasets kända lag och resultat ingår; poängavdrag och ligaspecifika inbördes möten saknas. "
    "Detta är en historisk förmatchplacering, inte en tabell hämtad live."
)
COUNT_NOTE = (
    "Tidsviktad Poissonmodell per liga: log(λ hemma) = bas + hemmafördel + anfall hemma + försvar borta; "
    "log(λ borta) = bas + anfall borta + försvar hemma. 180 dagars halveringstid, högst 1 095 dagar historik, "
    "omskattning tidigast efter 28 dagar och minst 20 historiska matcher. L2-regularisering stabiliserar "
    "lagparametrarna. Nya lag börjar neutralt. Detaljer och startvärden finns i README."
)


def build_feature_catalog():
    catalog = {}

    def add(name, description, formula, group, notes=TEMPORAL_NOTE):
        catalog[name] = {"feature": name, "description": description, "formula": formula,
                         "group": group, "notes": notes, "canonical": name}

    add("League", "Ligan där matchen spelas.", "Kategorin League kodas med one-hot från träningsdata.", "Liga")
    sums = {"goals_for": "gjorda mål", "goals_against": "insläppta mål", "goal_diff": "målskillnad",
            "shots_for": "egna skott", "shots_against": "motståndarnas skott",
            "sot_for": "egna skott på mål", "sot_against": "motståndarnas skott på mål",
            "corners_for": "egna hörnor", "corners_against": "motståndarnas hörnor"}
    for side, team in (("home", "Hemmalaget"), ("away", "Bortalaget")):
        for window in (5, 10):
            note = f"R{window} = senaste min({window}, antal tillgängliga) matcher. Summan är saknad om någon observation i fönstret saknas. " + TEMPORAL_NOTE
            add(f"{side}_points_last_{window}", f"{team}s poäng under sina senaste {window} matcher.",
                f"Σ poäng i R{window}; vinst = 3, oavgjort = 1, förlust = 0.", "Form", note)
            add(f"{side}_ppg_l{window}", f"{team}s genomsnittliga poäng per match under de senaste {window} matcherna.",
                f"Σ poäng i R{window} / antal matcher i R{window}.", "Form")
            for metric, label in sums.items():
                add(f"{side}_{metric}_l{window}", f"{team}s sammanlagda {label} under sina senaste {window} matcher.",
                    f"Σ {label} i R{window}. Målskillnad = gjorda − insläppta mål.", "Matchstatistik", note)
        for window in (5, 8):
            for suffix, description, formula in (
                (f"recent_form_{window}", "summerade form", f"Σ resultatscore i R{window}"),
                (f"recent_form_{window}_avg", "genomsnittliga form", f"Σ resultatscore i R{window} / antal matcher i R{window}"),
                (f"recent_goal_diff_{window}", "summerade målskillnad", f"Σ (gjorda mål − insläppta mål) i R{window}"),
            ):
                add(f"{side}_{suffix}", f"{team}s {description} under sina senaste {window} matcher.",
                    formula + "; resultatscore: vinst = +1, oavgjort = 0, förlust = −1.", "Form")
        for result, label in (("win", "vinster"), ("draw", "oavgjorda"), ("loss", "förluster")):
            add(f"{side}_{result}_rate_l5", f"Andelen {label} i {team.lower()}s senaste fem matcher.",
                f"Antal {label} i R5 / antal matcher i R5.", "Form")
        for color, label in (("yellow", "gula"), ("red", "röda")):
            add(f"{side}_{color}_cards_last_5", f"{team}s sammanlagda {label} kort under sina senaste fem matcher.",
                f"Σ {label} kort i R5; saknat om någon kortobservation saknas.", "Disciplin")
        venue = "hemmamatcher" if side == "home" else "bortamatcher"
        for window in (5, 10):
            add(f"{side}_{side}_ppg_l{window}", f"{team}s poäng per match i sina senaste {window} {venue}.",
                f"Σ poäng i senaste {window} {venue} / antal tillgängliga sådana matcher.", "Hemma-/bortaform")
        for metric in ("goals_for", "goals_against", "shots_for", "sot_for", "corners_for"):
            add(f"{side}_{side}_{metric}_l5", f"{team}s {sums[metric]} i sina senaste fem {venue}.",
                f"Σ {sums[metric]} i senaste min(5, tillgängliga) {venue}; kräver kompletta observationer.", "Hemma-/bortaform")
        for name, label in (("ppg", "poäng"), ("shot", "skott"), ("sot", "skott på mål"), ("corner", "hörnor")):
            add(f"{side}_{name}_momentum_l3_l10", f"Förändring i {team.lower()}s {label} per match: kort form mot längre form.",
                f"Medel({label} i R3) − medel({label} i R10).", "Momentum",
                "Varje medel använder sina observerade värden. Positivt betyder högre korttidssnitt. " + TEMPORAL_NOTE)
        for suffix, description, formula in (
            ("shot_accuracy_l5", "andel skott som träffar mål", "Σ skott på mål i R5 / Σ skott i R5"),
            ("sot_conversion_l5", "gjorda mål per skott på mål", "Σ gjorda mål i R5 / Σ skott på mål i R5"),
        ):
            add(f"{side}_{suffix}", f"{team}s {description}.", formula + ". Saknat vid noll eller saknad nämnare.", "Effektivitet")
        for kind, goals in (("attack", "gjorda"), ("defensive", "insläppta")):
            add(f"{side}_{kind}_strength_l10", f"{team}s relativa nivå för {goals} mål per match.",
                f"Medel({goals} mål i R10) / (ligans totala mål denna säsong / (2 × antal spelade matcher)).",
                "Relativ målstyrka", "1 motsvarar ligans snitt per lag. Högre försvarsvärde betyder fler insläppta mål. "
                "Ligasnittet återställs per säsong; R10 kan sträcka sig över säsongsgränsen. Saknat om nämnaren är noll. " + TEMPORAL_NOTE)
        add(f"{side}_position", f"{team}s tabellplacering precis före matchdagen; 1 betyder förstaplats.",
            "Rangordna säsongens kända lag efter poäng ↓, målskillnad ↓, gjorda mål ↓, lagnamn A–Ö.", "Tabell", STANDINGS_NOTE)
        add(f"{side}_points", f"{team}s ackumulerade ligapoäng denna säsong före matchdagen.",
            "3 × vinster + 1 × oavgjorda denna säsong; 0 innan första matchen.", "Tabell", STANDINGS_NOTE)
        add(f"{side}_points_per_game", f"{team}s ligapoäng per spelad match denna säsong.",
            "Säsongspoäng / spelade matcher denna säsong; saknat om inga matcher har spelats.", "Tabell", STANDINGS_NOTE)
        add(f"{side}_elo", f"{team}s Elo-rating före matchen.",
            "Start = 1500. R_ny = R_gammal + 32 × (S − E); S = 1 vid vinst, 0,5 vid oavgjort, 0 vid förlust.",
            "Elo", "E använder motståndarens Elo och en hemmafördel på 60 Elo-poäng som standard. Elo följer laget mellan säsonger inom ligan. " + TEMPORAL_NOTE)
        expectation = "1 / (1 + 10^((away_elo − home_elo − H) / 400))" if side == "home" else "1 − home_elo_expected"
        add(f"{side}_elo_expected", f"{team}s förväntade resultatpoäng enligt Elo, med hemmafördel.",
            expectation + "; H = 60 som standard.", "Elo",
            "Resultatpoäng är 1 / 0,5 / 0. Detta är inte en ren vinstsannolikhet eftersom matcher kan sluta oavgjort. " + TEMPORAL_NOTE)
        context = {
            "history_matches": ("antal tidigare matcher i denna liga", "Antal avslutade matcher före dagens datum, över alla säsonger."),
            "history_matches_l10": ("antal tillgängliga matcher i tiomatchersfönstret", "min(10, history_matches)"),
            "shot_observations_l10": ("antal observerade egna skottvärden i tiomatchersfönstret", "Antal matcher i R10 med ett ändligt värde för lagets egna skott."),
            "rest_days": ("antal vilodagar sedan senaste registrerade match", "Dagens datum − senaste avslutade matchens datum, i kalenderdagar; saknat utan historik."),
            "matches_last_14_days": ("matchtäthet under föregående 14 dagar", "Antal avslutade matcher med datum i [dagens datum − 14 dagar, dagens datum)."),
            "shot_share_l10": ("andel av matchernas skott", "Σ egna skott / Σ (egna + motståndarnas skott) i R10."),
            "sot_share_l10": ("andel av matchernas skott på mål", "Σ egna SOT / Σ (egna + motståndarnas SOT) i R10."),
            "sot_diff_per_match_l10": ("övertag i skott på mål per match", "Medel(egna SOT − motståndarnas SOT) i R10."),
            "opponent_elo_mean_l5": ("genomsnittliga motståndsstyrka i sina senaste fem matcher", "Medel av motståndarnas Elo precis före respektive historisk match i R5."),
            "result_overperformance_l5": ("överprestation jämfört med Elo i sina senaste fem matcher", "Medel(S − E) i R5; S = 1 / 0,5 / 0, E = lagets förmatchförväntan med hemmafördel."),
            "goal_diff_std_l10": ("variation i målskillnad", "sqrt(Σ (målskillnad − medelmålskillnad)² / (n − 1)) i R10; kräver n ≥ 2."),
        }
        for suffix, (description, formula) in context.items():
            note = TEMPORAL_NOTE
            if suffix in ("rest_days", "matches_last_14_days"):
                note += " Cup- och landskamper utanför datasetet ingår inte."
            if suffix in ("shot_share_l10", "sot_share_l10", "sot_diff_per_match_l10"):
                note += " Endast matcher med både egna och motståndarnas observationer används; saknat utan giltigt underlag."
            add(f"{side}_{suffix}", f"{team}s {description}.", formula, "Kontext", note)
        for metric in ("goals_for", "goals_against", "shots_for", "shots_against", "sot_for", "sot_against"):
            add(f"{side}_{metric}_ewm", f"{team}s tidsviktade snitt för {sums[metric]}; färska matcher väger mer.",
                f"Σ(w × {sums[metric]}) / Σw; w = 2^(−ålder i dagar / 60).",
                "Tidsviktad form", "Observerade värden från högst 1 095 dagar används. Saknat utan observationer. " + TEMPORAL_NOTE)
        for stat, label in (("shot", "skott"), ("sot", "skott på mål")):
            for kind, explanation in (("attack", "skapade"), ("defence", "tillåtna")):
                add(f"{side}_{stat}_{kind}_rating", f"{team}s motståndsjusterade rating för {explanation} {label}.",
                    f"exp(lagets {'anfallsparameter' if kind == 'attack' else 'försvarsparameter'}) i modellen för {label}.",
                    "Skottrating", "1 är neutral nivå. Högre försvarsrating betyder fler tillåtna skott. " + COUNT_NOTE)
        for stat, label in (("goals", "mål"), ("shots", "skott"), ("sot", "skott på mål")):
            formula = "exp(bas + hemmafördel + anfall hemma + försvar borta)" if side == "home" else "exp(bas + anfall borta + försvar hemma)"
            add(f"expected_{side}_{stat}", f"Förväntat antal {label} för {team.lower()} i den kommande matchen.",
                formula + f" i modellen för {label}.", "Målmodell" if stat == "goals" else "Skottrating", COUNT_NOTE)
    for name, description, formula in (
        ("position_diff", "Skillnad i tabellplacering; negativt betyder att hemmalaget ligger högre.", "home_position − away_position"),
        ("points_per_game_diff", "Skillnad i säsongens poäng per match.", "home_points_per_game − away_points_per_game"),
        ("elo_diff", "Hemmalagets Elo minus bortalagets, före hemmafördel.", "home_elo − away_elo"),
        ("home_adjusted_elo_diff", "Elo-skillnad inklusive hemmafördel.", "home_elo + H − away_elo; H = 60 som standard"),
        ("rest_days_diff", "Hemmalagets vilodagar minus bortalagets.", "home_rest_days − away_rest_days"),
        ("team_strength_delta", "Skillnad i lagens poäng under de senaste fem matcherna.", "home_points_last_5 − away_points_last_5"),
        ("goal_delta_last_5", "Skillnad i lagens gjorda mål under de senaste fem matcherna.", "home_goals_for_l5 − away_goals_for_l5"),
        ("shot_delta_last_5", "Skillnad i lagens egna skott under de senaste fem matcherna.", "home_shots_for_l5 − away_shots_for_l5"),
        ("recent_form_delta_5", "Skillnad i lagens summerade formscore över fem matcher.", "home_recent_form_5 − away_recent_form_5"),
        ("recent_form_delta_8", "Skillnad i lagens summerade formscore över åtta matcher.", "home_recent_form_8 − away_recent_form_8"),
        ("expected_total_goals", "Matchens förväntade totala antal mål.", "expected_home_goals + expected_away_goals"),
        ("expected_goal_diff", "Hemmalagets förväntade målövertag.", "expected_home_goals − expected_away_goals"),
    ):
        add(name, description, formula, "Skillnad mellan lagen", STANDINGS_NOTE if name in ("position_diff", "points_per_game_diff") else TEMPORAL_NOTE)
    for side, label, operator in (("home", "hemmavinst", "> 0"), ("draw", "oavgjort", "= 0"), ("away", "bortavinst", "< 0")):
        add(f"poisson_prob_{side}", f"Sannolikhet för {label} enligt den oberoende Poissonmodellen.",
            f"P(X − Y {operator}), där X ~ Poisson(expected_home_goals), Y ~ Poisson(expected_away_goals).",
            "Målmodell", "Beräknas med Skellamfördelningen utan avklippt målmatris. Ingen Dixon–Coles-korrigering. " + COUNT_NOTE)
    for alias, canonical in FEATURE_ALIASES.items():
        catalog[alias] = {**catalog[canonical], "feature": alias, "canonical": canonical,
                          "notes": f"Alias för {canonical}; samma definition används bara en gång i modellen. " + catalog[canonical]["notes"]}
    exported = {"League", *ALL_FEATURE_COLUMNS}
    missing = exported - catalog.keys()
    if missing:
        raise ValueError(f"Features missing documentation: {sorted(missing)}")
    return {name: catalog[name] for name in sorted(exported)}
