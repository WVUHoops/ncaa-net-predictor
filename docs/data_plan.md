# Data Plan

## KenPom Ingestion

Pull these endpoints for each historical season:

- `teams`
- `conferences`
- `conf-ratings`
- `ratings`
- `archive` with `preseason=true`
- `four-factors`
- `height`
- `misc-stats`
- `pointdist`

KenPom should supply the program baseline: prior strength, coach, conference, style, height, experience, bench strength, continuity, and preseason expectation.

## Roster Layer

Use CBB Analytics player data to classify last season's players:

- returning to same team
- transferred out and committed
- transferred out and uncommitted
- exhausted eligibility or graduated
- pro/NBA
- unknown

CBB Analytics API endpoints to prioritize:

- `/competitions` to map seasons to `competitionIds`
- `/competition-team-players` to identify season/team/player roster rows
- `/stats/player/agg-box` with `splits=season` for player-level box score production
- `/stats/team/agg-box` with `splits=season` for team-level aggregate baselines

Roll those into team-season snapshot features:

- returning minutes percentage
- returning usage percentage
- returning points, assists, rebounds, blocks, and steals percentage
- returning rotation count
- returning top-player production
- known departure production
- draft prospect production
- unknown production
- roster completion score

## Incoming Talent

Add class and portal context:

- EvanMiya freshman class rank and score
- On3 high school recruiting class rank and score
- On3 transfer class rank and score
- incoming transfer count
- incoming transfer prior minutes, usage, and efficiency where player matching is possible
- source team and conference strength for transfers

On3 sources:

- `https://www.on3.com/rivals/rankings/industry-team/basketball/2026/`
- `https://www.on3.com/transfer-portal/team-rankings/basketball/2026/`

These should be treated as dated snapshots because the portal and recruiting rankings will change through the offseason.

## Coach Factor

Use KenPom `teams` data as the stable team/coach spine and HoopDirt's D-I coaching-change tracker as the live offseason change source:

- `https://hoopdirt.com/2026-coaching-changes-tracker/`

Treat HoopDirt as a dated snapshot because the tracker updates during the coaching carousel. The first coach feature set should include:

- head coach changed flag
- known new coach vs TBD/open job flag
- old coach matches KenPom coach field flag
- available KenPom coach tenure at that program
- new coach appears in available KenPom history flag
- new coach's last seen D-I team, conference, and season
- unmatched HoopDirt rows for manual review

These features are mostly risk/context flags at first. Once more historical KenPom seasons are loaded, expand the factor into prior head-coach performance, coach movement source strength, and first-year transition effects.

The preferred historical coach metric is not raw team strength. Use KenPom preseason archive expectations against final KenPom results:

- `AdjEMFinal - preseason AdjEM`
- `preseason RankAdjEM - RankAdjEMFinal`
- offensive and defensive over/under components where available

Aggregate those by coach using only seasons before the modeled season:

- career average over/under
- last-three and last-five average over/under
- positive over/under rate
- big overperform and big underperform rates
- same-school prior over/under
- prior top-25/top-50/top-100 rates as context, not as the main coach metric

## Targets

Use Selection Sunday NET as the schedule-building target. The current public NCAA.com page is:

- `https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings`

That page also links to the official NCAA Statistics archive/team sheets:

- `https://stats.ncaa.org/selection_rankings/season_divisions/17783/nitty_gritties`

The archive is the better historical source because it exposes prior seasons and daily snapshots. Use the snapshot labeled `(Selections)` for each season, not the post-tournament `(Final)` snapshot.

Current official NCAA Stats coverage in this project is 2021-2026 Selection Sunday snapshots. The 2019-20 NCAA Stats men’s D-I NET page does not expose a `(Selections)` snapshot, and the seed page used for discovery does not list 2018-19.

- final NET rank
- final NET percentile
- top 50 flag
- top 75 flag
- top 100 flag
- top 135 flag
- below 200 flag

Also keep final KenPom `RankAdjEMFinal` as an auxiliary target and benchmark.

## Crosswalk And Modeling Table

Use KenPom team IDs as the model spine. Every non-KenPom source should be joined through a reviewable crosswalk with:

- source system
- source team ID when available
- source team name and normalized key
- matched KenPom team ID/name/key
- match type and score
- manual review flag

The initial modeling table should be keyed by `season` and KenPom-normalized team key, starting from KenPom preseason archive rows. Join coach-history features and NET targets first, then layer roster continuity, incoming talent, and schedule features as historical coverage is backfilled.

CBB Analytics player aggregate coverage currently produces usable roster-status rows for 2019-2026. The API returned empty player aggregate responses for 2016-2018, so roster continuity features should be treated as unavailable for those seasons unless another historical source fills the gap. In the modeling table, roster summaries are joined with a one-season lag: season `Y` uses roster/departure signals from season `Y-1`.

On3 incoming-talent features follow the same season convention: On3 ranking year `N` is joined to model season `N+1`, because the class/portal cycle feeds the following college basketball season. Current fetched coverage is HS class rankings for 2020-2026 and transfer portal team rankings for 2023-2026, with 2020-2021 transfer pages unavailable and 2022 returning no team rows.

## Backtesting

Use rolling season validation. For each target season, train only on prior seasons and evaluate:

- mean absolute NET rank error
- bucket accuracy
- top-band calibration
- lift from roster features over KenPom-only features

The first implemented backtest evaluates seasons 2022-2026, using 2021 and earlier labeled seasons only where available for the first fold. It compares KenPom preseason rank against direct and residual ridge models. Roster-talent models now include returning production, lost/uncertain production, rotation-core continuity, player-rate quality metrics, On3 incoming talent, and train-split feature selection diagnostics.

Current results show that roster talent is predictive on its own, but it does not yet improve rank MAE over the KenPom preseason baseline once KenPom information is present. That likely means the roster signal is partly duplicated by KenPom preseason expectation and partly too noisy in the current player-status and On3 team-class features. The next modeling priority is to improve player-level transfer/recruit mapping and test roster effects on band probabilities and subgroup slices, especially high-churn teams.

The first subgroup slice pass writes `rolling_slice_metrics.csv` and `rolling_slice_metrics.json`. Early slice results still do not show roster features improving rank MAE over KenPom residual models, though roster-heavy models slightly improve RMSE in the small major-transfer-class slice. Treat those slices as diagnostics rather than model-selection evidence until the transfer/recruit/player mapping is stronger.

CBB Analytics is now used for player-level incoming transfer features. The transfer layer follows `nextTeamId`/`nextTeamMarket` from prior-season player aggregates, aggregates prior production to the destination team-season, and adjusts selected production by the source team's prior-season KenPom context. On3 is still used for incoming freshman/team recruiting class signal. This materially improves roster-only model performance, but the KenPom-plus-roster models still do not beat the KenPom residual model on overall rank MAE.

The first nonlinear model pass adds dependency-free gradient-boosted regression trees. Direct tree models are poorly calibrated, but residual tree models improve RMSE, and simple blends of KenPom/ridge predictions with residual tree predictions improve MAE. Current best overall MAE is the KenPom-baseline plus residual-tree blend; current best RMSE is the KenPom-plus-roster residual tree.

The first schedule-band tuning pass adds prior-season threshold calibration for top 50, top 75, top 100, and top 135 decisions. Calibrated thresholds are learned from each model's earlier out-of-fold predictions only, then scored separately in `rolling_band_metrics.csv` alongside raw rank-threshold decisions. Early results suggest calibration mainly improves recall and F1 for top 75 and top 135 decisions, while top 100 often remains strongest with the raw rank threshold for already well-calibrated models.

For the schedule-building use case, same-season KenPom preseason ratings are excluded because they will not exist when the schedule is built. Other teams' schedule quality is also excluded, so SOS/NCSOS history is not eligible for schedule-safe models. KenPom remains valid for prior program-quality metrics and coach-history scores. The first schedule-safe model combines prior program performance, coach history, roster continuity, and current On3 recruiting/transfer class signals; adding program history improved the schedule-safe ridge model from 47.04 MAE to about 45.56 MAE in the 2022-2026 rolling backtest.

The schedule-facing tier is `opponent_quality_tier`, not the compressed score rank. It combines the calibrated model band with a prior program-consistency band so consistently elite programs can land in `top_25` even when the no-preseason-KenPom model score compresses them into the `26_50` range. The tier ladder now runs through the guarantee-game range: `top_25`, `26_50`, `51_75`, `76_100`, `101_135`, `136_160`, `161_200`, `201_250`, `251_300`, and `301_plus`.
