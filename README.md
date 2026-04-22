# NCAA NET Predictor

This project is for building preseason projections of NCAA D-I men's basketball end-of-season NET outcomes, with schedule-building decisions as the main use case.

The first scaffold focuses on collecting KenPom historical/team context. Roster continuity, EvanMiya class rankings, On3 transfer rankings, CBB Analytics player data, schedules, and final NET targets can be layered in next.

## Local Setup

Keep your KenPom API token in a local `.env` file:

```bash
KENPOM_API_TOKEN=your_token_here
CBB_ANALYTICS_API_KEY=your_key_here
```

The real `.env` file is ignored by git. Use `.env.example` as the shareable template.

## Nightly Dashboard Refresh On GitHub Actions

The dashboard can refresh on GitHub without your MacBook being awake. The workflow lives at
`.github/workflows/refresh-dashboard.yml` and runs every morning, with a manual
`workflow_dispatch` button available in GitHub.

Before the workflow can run, add these repository secrets in GitHub under
`Settings > Secrets and variables > Actions`:

```text
KENPOM_API_TOKEN
CBB_ANALYTICS_API_KEY
```

The workflow runs `python3 scripts/update_dashboard.py`, uploads the generated dashboard as an
artifact, commits changed dashboard files back to the repo, and deploys the `dashboard/` folder to
GitHub Pages:

```text
dashboard/index.html
dashboard/dashboard_payload.json
```

In GitHub, set `Settings > Pages > Build and deployment > Source` to `GitHub Actions`.

One important data note: `data/raw/` and `data/processed/` are ignored by git so licensed and
large local data are not accidentally published. GitHub Actions restores those folders from an
Actions cache. A brand-new GitHub repo will need a first successful seeded run before the nightly
refresh is fully independent. Until that cache exists, the runner will not have the same historical
KenPom, CBB Analytics, NET target, and hoopR schedule files your laptop has locally.

## Fetch KenPom Data

Fetch the default season endpoints:

```bash
python3 scripts/fetch_kenpom_season.py --season 2025
```

Fetch only one endpoint:

```bash
python3 scripts/fetch_kenpom_season.py --season 2025 --endpoint ratings
```

Fetch preseason archive data only:

```bash
python3 scripts/fetch_kenpom_season.py --season 2025 --endpoint preseason-archive
```

Fetch a historical range of team, final rating, and preseason archive files:

```bash
python3 scripts/fetch_kenpom_history.py --start-season 2016 --end-season 2026
```

Output is written under `data/raw/kenpom/<season>/` and is intentionally ignored by git.

## Fetch NCAA NET Rankings

Fetch the current NCAA.com NET rankings page and save parsed JSON/CSV:

```bash
python3 scripts/fetch_ncaa_net_rankings.py --save-html
```

If you already have an NCAA.com HTML file saved locally, parse it without making a network request:

```bash
python3 scripts/fetch_ncaa_net_rankings.py --html-file /path/to/net-rankings.html
```

Output is written under `data/raw/ncaa_net/` and is intentionally ignored by git.

## Fetch Selection Sunday NET Targets

For model targets, use the official NCAA Statistics archive's `(Selections)` snapshots, because those are the NET rankings available when the committee selects the NCAA tournament field.

Discover the men's D-I NET season archive pages from a known NCAA Stats season page:

```bash
python3 scripts/discover_ncaa_net_season_pages.py
```

List the available Selection Sunday snapshots from those discovered season pages:

```bash
python3 scripts/discover_ncaa_net_season_pages.py --list-selections
```

List discovered selection snapshots from a saved archive index:

```bash
python3 scripts/fetch_ncaa_net_selections.py --index-html-file /path/to/nitty-gritties.html --list-selections
```

Parse a saved Selection Sunday snapshot:

```bash
python3 scripts/fetch_ncaa_net_selections.py --selection-html-file /path/to/selection-snapshot.html
```

If NCAA Statistics allows direct scripted access from your network, this can fetch a season by ending year:

```bash
python3 scripts/fetch_ncaa_net_selections.py --index-url https://stats.ncaa.org/selection_rankings/season_divisions/18403/nitty_gritties --season 2025 --save-html
```

Output is written under `data/raw/ncaa_net_selections/` and is intentionally ignored by git.

## Fetch On3 Recruiting And Portal Snapshots

On3 rankings move during the portal window, so each pull is saved as a dated snapshot.

Fetch high school recruiting class rankings:

```bash
python3 scripts/fetch_on3_rankings.py --source hs --year 2026
```

Fetch transfer portal class rankings:

```bash
python3 scripts/fetch_on3_rankings.py --source transfer --year 2026
```

Fetch a historical range:

```bash
python3 scripts/fetch_on3_history.py --source hs --start-year 2022 --end-year 2026
python3 scripts/fetch_on3_history.py --source transfer --start-year 2022 --end-year 2026
```

Build incoming-talent features from all fetched On3 snapshots:

```bash
python3 scripts/build_on3_features.py
```

Parse a saved On3 HTML page without making a network request:

```bash
python3 scripts/fetch_on3_rankings.py --source transfer --year 2026 --html-file /path/to/on3-transfer.html
```

Output is written under `data/raw/on3/<source>/<year>/` and is intentionally ignored by git.

## Fetch CBB Analytics Data

Keep your CBB Analytics API key in `.env`:

```bash
CBB_ANALYTICS_API_KEY=your_key_here
```

Find competition IDs:

```bash
python3 scripts/fetch_cbb_analytics.py --endpoint competitions --version v1
```

Fetch player season aggregate box score stats:

```bash
python3 scripts/fetch_cbb_analytics.py --endpoint player-agg-box --version v1 --competition-ids 38409 --splits season
```

Fetch historical player aggregates for roster continuity:

```bash
python3 scripts/fetch_cbb_roster_history.py --start-season 2016 --end-season 2026
```

Fetch team season aggregate box score stats:

```bash
python3 scripts/fetch_cbb_analytics.py --endpoint team-agg-box --version v1 --competition-ids 38409 --splits season
```

Output is written under `data/raw/cbb_analytics/` and is intentionally ignored by git.

## Build Roster Status Features

Build first-pass player statuses and team continuity summaries from CBB Analytics player aggregates:

```bash
python3 scripts/build_roster_status.py --season 2026
```

This classifies players as probable returners, committed transfers out, portal/pending, draft prospect review, or senior eligibility review using CBB Analytics fields like `nextTeamId`, `willTransfer`, `inPortalAfterSeason`, and `isDraftProspect`.

Build combined historical roster-status and team continuity files from all fetched player aggregate snapshots:

```bash
python3 scripts/build_roster_history.py --min-season 2016 --max-season 2026
```

Build incoming-transfer production features from CBB Analytics transfer destinations and KenPom source-team context:

```bash
python3 scripts/build_transfer_features.py
```

## Fetch HoopDirt Coaching Changes

HoopDirt's tracker is live and should be treated as a dated snapshot during the coaching carousel.

Fetch the D-I 2026 coaching changes table:

```bash
python3 scripts/fetch_hoopdirt_coaching_changes.py --season 2026
```

Parse a saved HoopDirt Ninja Tables AJAX JSON response without making a network request:

```bash
python3 scripts/fetch_hoopdirt_coaching_changes.py --season 2026 --ajax-json-file /path/to/hoopdirt-d1.json
```

Output is written under `data/raw/hoopdirt/coaching_changes/` and is intentionally ignored by git.

## Build Coach Features

Join HoopDirt coaching changes to KenPom team/coach rows:

```bash
python3 scripts/build_coach_features.py --season 2026
```

The builder uses the latest dated HoopDirt CSV snapshot plus all available default KenPom `teams.json` files for 2025 and 2026. It creates one row per current KenPom team with `coach_changed`, `coach_change_status`, old-coach match confidence, available KenPom tenure, and whether the new coach appears in the available KenPom history.

## Build Historical Coach Metrics

Build leak-safe coach history features from KenPom final ratings and preseason archive files:

```bash
python3 scripts/build_coach_history.py --min-season 2016 --max-season 2026
```

The key metric is expectation-adjusted performance: final KenPom `AdjEMFinal` minus preseason KenPom `AdjEM`. Positive values mean the team outperformed its preseason KenPom expectation. The output includes prior-career averages, last-three/last-five averages, same-school history, and current-season observed over/under fields for backtesting.

## Build Team Crosswalk

Build a reviewable crosswalk centered on KenPom team IDs:

```bash
python3 scripts/build_team_crosswalk.py
```

This writes all source matches plus a smaller manual-review file for fuzzy or low-confidence matches.

## Build NET Targets And Model Table

Build all NET target variants from parsed Selection Sunday snapshots. Use `--include-current` only for a current-season placeholder or smoke test:

```bash
python3 scripts/build_net_targets.py --include-current
```

Build the first season-level modeling table:

```bash
python3 scripts/build_model_table.py --min-season 2016 --max-season 2026
```

By default, the modeling table joins historical roster-status summaries with a one-season lag, CBB Analytics incoming-transfer production keyed to the destination season, and incoming-freshman/team-class features from On3 ranking year `season - 1`. For example, the 2026 preseason row uses 2025 roster/departure signals, CBB transfers committed for 2026, and 2025 On3 class rankings when available.

## Backtest Models

Run rolling-season backtests for the preseason NET model:

```bash
python3 scripts/backtest_models.py
```

This evaluates KenPom preseason rank as a baseline plus conservative residual ridge models. Each test season is predicted using only earlier seasons with NET targets.

The current backtest includes roster diagnostics:

- direct roster-talent ridge
- direct KenPom ridge
- direct KenPom plus roster-talent ridge
- KenPom preseason baseline
- residual KenPom ridge
- residual KenPom plus coach ridge
- residual KenPom plus roster-talent ridge
- residual full ridge
- direct and residual gradient-boosted tree models
- simple blends of KenPom/ridge predictions with residual tree predictions

Roster-talent candidates are selected inside each rolling train split before fitting, and the selected features are written to `data/processed/backtests/rolling_feature_selections.csv`. Slice metrics for high-churn, low-returning-production, stable-roster, major HS class, and major transfer class groups are written to `data/processed/backtests/rolling_slice_metrics.csv`.

The backtest also writes `data/processed/backtests/rolling_band_metrics.csv`, which compares raw rank-threshold decisions against leakage-safe calibrated thresholds for top 50, top 75, top 100, and top 135 schedule-building bands. For each model and test season, calibrated thresholds are learned only from that model's prior-season out-of-fold predictions.

For schedule building before KenPom preseason ratings exist, use the schedule-safe models. These exclude same-season KenPom preseason columns and SOS/NCSOS history, and instead use prior program quality, coach-history scores, roster continuity, and incoming recruiting/transfer features:

```bash
python3 scripts/build_program_history.py
python3 scripts/build_model_table.py
python3 scripts/backtest_models.py
python3 scripts/predict_current_season.py
```

Current-season schedule-safe predictions are written to `data/processed/predictions/current_2027_schedule_predictions.csv`. Use `opponent_quality_tier` as the schedule-facing label; it combines the calibrated model band with a prior program-consistency band, and spans `top_25`, `26_50`, `51_75`, `76_100`, `101_135`, `136_160`, `161_200`, `201_250`, `251_300`, and `301_plus`.

## Build Guarantee-Game Upset Risk

The upset-risk layer is a separate schedule-building model. It uses historical game results to find high-major home, early-season, non-conference games against non-high-major opponents. The label is whether the non-high-major road opponent won.

Download the SportsDataverse/hoopR schedule master to `data/raw/hoopr/mbb_schedule_master.csv`, then make sure KenPom style endpoints are populated for the seasons you want in the training window:

```bash
python3 scripts/fetch_kenpom_season.py --season 2026 --endpoint four-factors --endpoint misc-stats --endpoint height --endpoint pointdist
```

Build the training table, rolling backtest, coefficients, and current candidate board:

```bash
python3 scripts/build_upset_risk.py
```

Outputs are written under `data/processed/upset_risk/`. The most useful schedule-building file is `current_2027_guarantee_risk_board.csv`, which combines `opponent_quality_tier` with `upset_probability_vs_median_high_major`, `risk_bucket`, `danger_index`, `safe_value_score`, and a first-pass `recommendation`.

## First Modeling Targets

The early model should predict useful schedule-building bands instead of only exact rank:

- final NET percentile
- probability of top 50
- probability of top 75
- probability of top 100
- probability of top 135
- probability of falling below 200

Exact final NET rank can still be tracked, but the band probabilities should be the first product-facing output.
