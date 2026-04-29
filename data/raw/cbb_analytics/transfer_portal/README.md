Place downloaded CBB Analytics transfer-portal CSV files here.

Preferred location for the freshest file:
- `data/raw/cbb_analytics/transfer_portal/current/`

Fallback location:
- `data/raw/cbb_analytics/transfer_portal/`

The dashboard refresh pipeline will:
1. look for the newest CSV in those folders,
2. use that file as the transfer movement ledger,
3. join the transferred players onto the frozen season stat rows already stored locally,
4. rebuild `data/processed/transfer_features/current/cbb_incoming_transfer_features.csv`.

Expected transfer-ledger information:
- player name
- source / prior team
- destination / committed team
- player ID if available

If no transfer-ledger CSV is present, the pipeline falls back to the older roster-status transfer detection path.
