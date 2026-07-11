# RUNBOOK — ASRI live dashboard & API refresh

Keeps https://asri.dissensus.ai / https://api.dissensus.ai current. Written
2026-07-11 after backfilling the Jan–Jul 2026 gap.

## Architecture

| Piece | What | Where |
|---|---|---|
| Data store | Cloudflare D1 `asri-db` (`f8c08c15-a596-4d7f-8c00-0f7592da26f4`), table `asri_daily`, one row per day | Cloudflare account `917877c0…` |
| API | Python Worker `asri-api` (`workers/src/main.py`), routes `api.dissensus.ai/*` and `asri.dissensus.ai/api/*`; recomputes ASRI + 30d avg from stored sub-indices at read time | Cloudflare Workers |
| Dashboard | Vite/React app (`frontend/`), fetches `/asri/current` + `/asri/timeseries` at page load — it needs **no redeploy** when data updates | Cloudflare (asri.dissensus.ai) |
| Generator | `scripts/generate_asri_series.py` → `ASRIBacktester` (`src/asri/backtest/`), live DeFiLlama + FRED inputs, frozen universe snapshot | this repo, `.venv` (py3.12 — 3.14 segfaults pandas) |

## Why it went stale (Jan–Jul 2026)

The D1 table was a **one-shot load of the frozen canonical paper series**
(2021-01-01 → 2026-01-15, 1841 rows = `results/data/asri_history.parquet`).
No cron trigger existed on the Worker, and the old `asri-scheduler.service`
pointed at a directory that no longer exists and was never installed. Nothing
was ever wired to append new days.

## The refresh mechanism (since 2026-07-11)

One idempotent entry point:

```bash
cd ~/Resurrexi/projects/papers/papers-official/asri/code
.venv/bin/python scripts/refresh_asri.py          # to yesterday UTC
.venv/bin/python scripts/refresh_asri.py --end 2026-08-31   # explicit end
```

It: reads D1 `max(date)` → extends `data/peg_history.csv` additively
(`extend_peg_history.py`) → runs the committed generator for exactly the
missing window under the pinned universe snapshot (`ASRI_SNAPSHOT_AS_OF=2026-07-11`)
→ loads rows with `INSERT OR IGNORE` (`load_d1_backfill.py`; frozen rows can
never be overwritten) → appends an audit line per row to
`results/data/asri_live_appends.csv` → verifies the live API serves the new
date. Exits 0 doing nothing when already current; a multi-day gap (machine off)
is filled in one run.

**Scheduled:** systemd *user* timer on PurrPower, daily 07:30 (+ jitter),
`Persistent=true` (catches up after downtime):

```bash
systemctl --user list-timers asri-refresh.timer
journalctl --user -u asri-refresh.service -n 50
# unit files live in scripts/asri-refresh.{service,timer}; reinstall with:
cp scripts/asri-refresh.{service,timer} ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now asri-refresh.timer
```

Credentials: Cloudflare global key in `~/.env.cloudflare`
(`export CLOUDFLARE_API_KEY=…`, used with X-Auth-Email); `FRED_API_KEY` in
`./.env`. DeFiLlama endpoints used are keyless.

## Data provenance — the 2026-01-15/16 seam (do not "fix")

- **≤ 2026-01-15**: frozen canonical paper series (Zenodo 10.5281/zenodo.17918239).
  Its original generation inputs were never archived and it is NOT reproducible
  from this codebase (`DATA_PROVENANCE.md`).
- **≥ 2026-01-16**: computed by the committed Jun-2026 fixed pipeline
  (D2 coin-ids, real peg data, rolling-365 TVL, frozen universe snapshot).

These differ in level. Overlap week 2026-01-08..15, fixed pipeline minus
frozen: **ASRI +10.3** (SCR +24.8, DeFi +10.9, Contagion −12.8, Arb +16.4).
The step at 2026-01-16 on the dashboard chart is therefore a *methodology
regime change*, not a market event. Rescaling either side to hide it would
misrepresent published numbers — don't. If it needs explaining on the
dashboard, add an annotation in the frontend.

## Pinned universe & known fragilities

- `data/snapshots/protocols_2026-07-11.json` (8.5 MB, not in git;
  sha256 `143bda3530d7d0a6b987ccd7f19e7ce2243ffb9b949c66ed8ac6bd5632e3e27a`).
  Re-dump with `scripts/dump_universe_snapshot.py <date>` ONLY as a deliberate
  decision — it level-shifts the DeFi/Contagion/Arb channels; update
  `PINNED_UNIVERSE` in `refresh_asri.py` and note the date here.
- `data/snapshots/bridges_2026-07-11.json` (committed): **bridges.llama.fi
  went behind DeFiLlama's paid API ~mid-2026.** This file holds the endpoint's
  last public output (Wayback, 2024-12-22, 63 bridges) and is effectively
  permanent. Only feeds `bridge_exploit_frequency` (2.5% of ASRI).
- Frontend timeseries end date is computed at fetch time (tomorrow UTC,
  `frontend/src/components/ASRIDashboard.tsx`) — fixed 2026-07-11; it was a
  hardcoded `end=2026-12-31` that would have truncated the chart from
  2027-01-01. The fix is in source but the LIVE bundle predates it; it goes
  live on the next frontend deploy.
- Worker source verified against live (2026-07-11): the live `asri-api`
  script was pulled via the CF API and is byte-identical to the committed
  `workers/src/main.py` (EOL-insensitive diff = 0; the file is CRLF) — this
  repo IS the deployed v2.1.0 and is safe to deploy from with
  `wrangler deploy` in `workers/`. The worker's own omitted-`end` default
  (2026-12-31, the same year-end fuse) is fixed in source to tomorrow-UTC;
  it goes live on the next worker deploy. A stale v2.0.0 copy that
  previously prompted a clobber warning here lives in the OLD duplicate
  clone at `papers/github-repos/asri`, not in this repo.
- Frontend deploys are **direct-upload Cloudflare Pages** (project
  `asri-dashboard`), NOT git-connected — pushing to GitHub does not deploy.
  From `frontend/`: `npm run build`, then
  `wrangler pages deploy dist --project-name=asri-dashboard`
  (credentials: `~/.env.cloudflare`).
- FRED publishes T+1 business day; weekend/holiday runs reuse the last
  observation (same closest-at-or-before rule as the paper pipeline).
- If DeFiLlama TVL/stablecoin/coins endpoints break or paywall, the generator
  errors per date and loads nothing — the refresh fails loudly rather than
  writing degraded rows. Check `journalctl --user -u asri-refresh.service`.
