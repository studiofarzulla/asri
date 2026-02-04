"""
Publication-Lag Aware Backtesting for ASRI

Implements realistic data availability constraints to simulate pseudo-real-time
evaluation, addressing JFS reviewer concern about look-ahead bias in historical
backtests.

Key insight: Different data sources have different publication lags. A "real-time"
system would only have access to data published by the target date, not data
that was observed on the target date but published later.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog

from asri.backtest.historical import HistoricalDataFetcher, HistoricalSnapshot
from asri.backtest.backtest import ASRIBacktester, CRISIS_EVENTS, BacktestResult

logger = structlog.get_logger()


@dataclass
class PublicationLag:
    """Publication lag configuration for a data source."""
    name: str
    lag: timedelta
    description: str


# Conservative publication lag estimates for each data source
# These represent the delay between observation and public availability
DATA_LAGS = {
    'defillama_tvl': PublicationLag(
        name='DeFi Llama TVL',
        lag=timedelta(hours=6),
        description='DeFi Llama updates TVL data with ~6 hour latency'
    ),
    'defillama_stables': PublicationLag(
        name='DeFi Llama Stablecoins',
        lag=timedelta(hours=12),
        description='Stablecoin market cap data aggregated with ~12 hour delay'
    ),
    'fred_treasury': PublicationLag(
        name='FRED Treasury Rates',
        lag=timedelta(days=2),
        description='Treasury rates published with 1-2 business day lag'
    ),
    'fred_vix': PublicationLag(
        name='FRED VIX',
        lag=timedelta(days=1),
        description='VIX published next day after market close'
    ),
    'fred_yield_spread': PublicationLag(
        name='FRED Yield Curve Spread',
        lag=timedelta(days=2),
        description='Yield spread published with 1-2 business day lag'
    ),
    'fred_sp500': PublicationLag(
        name='FRED S&P 500',
        lag=timedelta(days=1),
        description='S&P 500 published next day after close'
    ),
    'coingecko_btc': PublicationLag(
        name='BTC Price',
        lag=timedelta(hours=1),
        description='CoinGecko BTC data near real-time'
    ),
    'news_sentiment': PublicationLag(
        name='News Sentiment',
        lag=timedelta(hours=2),
        description='News aggregation and NLP processing delay'
    ),
    'protocols': PublicationLag(
        name='Protocol Data',
        lag=timedelta(hours=6),
        description='Protocol list and audit status updates'
    ),
    'bridges': PublicationLag(
        name='Bridge Data',
        lag=timedelta(hours=6),
        description='Bridge TVL and exploit tracking'
    ),
}


def get_lag_summary() -> dict[str, dict[str, Any]]:
    """Return summary of all publication lags for documentation."""
    return {
        name: {
            'lag_hours': lag.lag.total_seconds() / 3600,
            'lag_days': lag.lag.total_seconds() / 86400,
            'description': lag.description,
        }
        for name, lag in DATA_LAGS.items()
    }


class LagAwareHistoricalFetcher(HistoricalDataFetcher):
    """
    Historical data fetcher that simulates publication lags.

    When simulate_lags=True, only returns data that would have been
    PUBLISHED by the target_date, not data observed on target_date.
    """

    def __init__(self, simulate_lags: bool = False, timeout: float = 60.0):
        super().__init__(timeout=timeout)
        self.simulate_lags = simulate_lags

    def _apply_lag(self, target_date: datetime, source: str) -> datetime:
        """
        Calculate the data cutoff date given publication lag.

        If we want data "as of" target_date in real-time, we can only
        access data that was published by target_date. This means the
        most recent observable data is from (target_date - publication_lag).
        """
        if not self.simulate_lags:
            return target_date

        lag = DATA_LAGS.get(source)
        if lag is None:
            logger.warning(f"No lag defined for source {source}, using target date")
            return target_date

        return target_date - lag.lag

    def _find_closest_tvl_lagged(
        self, target_date: datetime, data: list[dict]
    ) -> tuple[float, datetime]:
        """Find TVL closest to lag-adjusted target date."""
        adjusted_date = self._apply_lag(target_date, 'defillama_tvl')
        target_ts = adjusted_date.timestamp()

        # Only consider data points before the adjusted cutoff
        valid_points = [p for p in data if p['date'] <= target_ts]
        if not valid_points:
            return 0.0, target_date

        closest = max(valid_points, key=lambda x: x['date'])
        return closest['tvl'], datetime.fromtimestamp(closest['date'])

    def _get_tvl_range_lagged(
        self, end_date: datetime, days: int, data: list[dict]
    ) -> list[float]:
        """Get TVL values for a date range with lag adjustment."""
        adjusted_end = self._apply_lag(end_date, 'defillama_tvl')
        end_ts = adjusted_end.timestamp()
        start_ts = (adjusted_end - timedelta(days=days)).timestamp()

        values = [
            point['tvl'] for point in data
            if start_ts <= point['date'] <= end_ts
        ]
        return values if values else [0.0]

    def _get_max_tvl_before_lagged(self, date: datetime, data: list[dict]) -> float:
        """Get max TVL up to lag-adjusted date."""
        adjusted_date = self._apply_lag(date, 'defillama_tvl')
        target_ts = adjusted_date.timestamp()
        values = [point['tvl'] for point in data if point['date'] <= target_ts]
        return max(values) if values else 0.0

    def _get_fred_value_at_date_lagged(
        self, series_id: str, target_date: datetime
    ) -> float | None:
        """Get FRED value with publication lag applied."""
        # Map FRED series to lag source
        source_map = {
            'DGS10': 'fred_treasury',
            'VIXCLS': 'fred_vix',
            'T10Y2Y': 'fred_yield_spread',
            'SP500': 'fred_sp500',
        }
        source = source_map.get(series_id, 'fred_treasury')
        adjusted_date = self._apply_lag(target_date, source)

        observations = self._fred_cache.get(series_id, [])
        if not observations:
            return None

        target_str = adjusted_date.strftime("%Y-%m-%d")

        closest_value = None
        for obs in observations:
            obs_date = obs.get("date", "")
            if obs_date <= target_str:
                value = obs.get("value")
                if value not in [".", None, ""]:
                    closest_value = float(value)
            else:
                break

        return closest_value

    async def _fetch_btc_prices_lagged(
        self, end_date: datetime, days: int = 90
    ) -> list[float]:
        """Get BTC prices with lag adjustment."""
        await self._ensure_btc_cache()

        if not self._btc_cache:
            return []

        adjusted_end = self._apply_lag(end_date, 'coingecko_btc')
        end_ts = adjusted_end.timestamp()
        start_ts = (adjusted_end - timedelta(days=days)).timestamp()

        prices = [
            point['price'] for point in self._btc_cache
            if start_ts <= point['timestamp'] <= end_ts
        ]

        return prices

    async def _fetch_sp500_prices_lagged(
        self, end_date: datetime, days: int = 90
    ) -> list[float]:
        """Fetch S&P 500 prices with lag adjustment."""
        await self._ensure_sp500_cache()

        if not self._sp500_cache:
            return []

        adjusted_end = self._apply_lag(end_date, 'fred_sp500')
        end_str = adjusted_end.strftime("%Y-%m-%d")
        start_str = (adjusted_end - timedelta(days=days)).strftime("%Y-%m-%d")

        prices = []
        for obs in self._sp500_cache:
            obs_date = obs.get("date", "")
            if start_str <= obs_date <= end_str:
                value = obs.get("value")
                if value not in [".", None, ""]:
                    prices.append(float(value))

        return prices

    async def _fetch_stablecoin_at_date_lagged(
        self, stablecoin_id: int, target_date: datetime
    ) -> float:
        """Get stablecoin market cap with lag adjustment."""
        await self._ensure_stablecoin_cache(stablecoin_id)
        data = self._stablecoin_cache.get(stablecoin_id, [])

        if not data:
            return 0.0

        adjusted_date = self._apply_lag(target_date, 'defillama_stables')
        target_ts = adjusted_date.timestamp()

        # Find closest data point at or before adjusted date
        closest = None
        min_diff = float('inf')

        for point in data:
            ts = int(point.get('date', 0))
            if ts <= target_ts:
                diff = target_ts - ts
                if diff < min_diff:
                    min_diff = diff
                    closest = point

        if closest:
            circulating = closest.get('totalCirculating', {}).get('peggedUSD', 0)
            return float(circulating) if circulating else 0.0

        return 0.0

    async def fetch_snapshot(self, target_date: datetime) -> HistoricalSnapshot:
        """
        Fetch all data needed for ASRI calculation at a specific date.

        If simulate_lags is True, applies publication lag to each source,
        returning only data that would have been available in real-time.
        """
        logger.info(
            f"Fetching {'lag-adjusted' if self.simulate_lags else 'perfect'} "
            f"snapshot for {target_date.date()}"
        )

        data_quality: dict[str, str] = {}

        # 1. TVL Data (with optional lag)
        await self._ensure_tvl_cache()

        if self.simulate_lags:
            current_tvl, actual_date = self._find_closest_tvl_lagged(
                target_date, self._tvl_cache
            )
            max_historical_tvl = self._get_max_tvl_before_lagged(
                target_date, self._tvl_cache
            )
            tvl_30d_history = self._get_tvl_range_lagged(
                target_date, 30, self._tvl_cache
            )
            lag_info = f" (lagged by {DATA_LAGS['defillama_tvl'].lag})"
        else:
            current_tvl, actual_date = self._find_closest_tvl(
                target_date, self._tvl_cache
            )
            max_historical_tvl = self._get_max_tvl_before(
                target_date, self._tvl_cache
            )
            tvl_30d_history = self._get_tvl_range(target_date, 30, self._tvl_cache)
            lag_info = ""

        data_quality['tvl'] = f"ok (closest: {actual_date.date()}){lag_info}"

        # 2. Stablecoin Data (with optional lag)
        stablecoin_caps: dict[str, float] = {}
        for stable_id, symbol in self.MAJOR_STABLES.items():
            if self.simulate_lags:
                cap = await self._fetch_stablecoin_at_date_lagged(stable_id, target_date)
            else:
                cap = await self._fetch_stablecoin_at_date(stable_id, target_date)
            if cap > 0:
                stablecoin_caps[symbol] = cap

        total_stablecoin = sum(stablecoin_caps.values())
        lag_info = f" (lagged)" if self.simulate_lags else ""
        data_quality['stablecoins'] = f"ok ({len(stablecoin_caps)} tracked){lag_info}"

        # 3. FRED Data (with optional lag)
        if self.simulate_lags:
            await self._ensure_fred_cache("DGS10")
            await self._ensure_fred_cache("VIXCLS")
            await self._ensure_fred_cache("T10Y2Y")
            treasury_10y = self._get_fred_value_at_date_lagged("DGS10", target_date)
            vix = self._get_fred_value_at_date_lagged("VIXCLS", target_date)
            yield_spread = self._get_fred_value_at_date_lagged("T10Y2Y", target_date)
            sp500_prices = await self._fetch_sp500_prices_lagged(target_date, days=90)
        else:
            treasury_10y = await self._fetch_fred_series("DGS10", target_date)
            vix = await self._fetch_fred_series("VIXCLS", target_date)
            yield_spread = await self._fetch_fred_series("T10Y2Y", target_date)
            sp500_prices = await self._fetch_sp500_prices(target_date, days=90)

        # Use defaults for missing data
        lag_suffix = " (lagged)" if self.simulate_lags else ""
        if treasury_10y is None:
            treasury_10y = 3.5
            data_quality['treasury'] = f"missing (using default 3.5%){lag_suffix}"
        else:
            data_quality['treasury'] = f"ok{lag_suffix}"

        if vix is None:
            vix = 20.0
            data_quality['vix'] = f"missing (using default 20){lag_suffix}"
        else:
            data_quality['vix'] = f"ok{lag_suffix}"

        if yield_spread is None:
            yield_spread = 0.5
            data_quality['yield_curve'] = f"missing (using default 0.5){lag_suffix}"
        else:
            data_quality['yield_curve'] = f"ok{lag_suffix}"

        data_quality['sp500'] = (
            f"ok ({len(sp500_prices)} days){lag_suffix}"
            if sp500_prices else f"missing{lag_suffix}"
        )

        # 4. BTC Prices (with optional lag)
        if self.simulate_lags:
            btc_prices = await self._fetch_btc_prices_lagged(target_date, days=90)
        else:
            btc_prices = await self._fetch_btc_prices(target_date, days=90)

        data_quality['btc'] = (
            f"ok ({len(btc_prices)} days){lag_suffix}"
            if btc_prices else f"missing{lag_suffix}"
        )

        # 5. Protocol/Bridge data (can't easily backtest, use current)
        try:
            resp = await self.client.get(f"{self.DEFILLAMA_BASE}/protocols")
            protocols = resp.json() if resp.status_code == 200 else []
            data_quality['protocols'] = f"current snapshot ({len(protocols)} protocols)"
        except Exception:
            protocols = []
            data_quality['protocols'] = "failed"

        try:
            resp = await self.client.get("https://bridges.llama.fi/bridges")
            bridges = resp.json().get("bridges", []) if resp.status_code == 200 else []
            data_quality['bridges'] = f"current snapshot ({len(bridges)} bridges)"
        except Exception:
            bridges = []
            data_quality['bridges'] = "failed"

        return HistoricalSnapshot(
            date=target_date,
            current_tvl=current_tvl,
            max_historical_tvl=max_historical_tvl,
            tvl_30d_history=tvl_30d_history,
            stablecoin_market_caps=stablecoin_caps,
            total_stablecoin_supply=total_stablecoin,
            treasury_10y_rate=treasury_10y,
            vix=vix,
            yield_curve_spread=yield_spread,
            sp500_prices=sp500_prices,
            btc_prices=btc_prices,
            protocols=protocols,
            bridges=bridges,
            data_quality=data_quality,
        )


class LagAwareBacktester(ASRIBacktester):
    """
    ASRI backtester that can optionally simulate publication lags
    for pseudo-real-time evaluation.
    """

    def __init__(self, simulate_lags: bool = False):
        self.simulate_lags = simulate_lags
        self.fetcher = LagAwareHistoricalFetcher(simulate_lags=simulate_lags)

    def get_mode_description(self) -> str:
        return "lag-simulated" if self.simulate_lags else "perfect-foresight"


@dataclass
class LagComparisonResult:
    """Comparison of detection performance with and without lag simulation."""
    crisis_name: str
    crisis_date: datetime

    # Perfect foresight (no lag)
    baseline_peak_asri: float
    baseline_lead_time: int  # days before crisis
    baseline_alert_level: str
    baseline_detected: bool

    # Lag-simulated
    lagged_peak_asri: float
    lagged_lead_time: int
    lagged_alert_level: str
    lagged_detected: bool

    # Differences
    asri_degradation: float  # percentage reduction in peak ASRI
    lead_time_change: int  # days lost (negative = lost, positive = gained)
    detection_maintained: bool

    # Contributing factors
    limiting_source: str  # which data source has the longest lag affecting detection
    notes: list[str]


async def compare_lag_impact(
    threshold: float = 50.0
) -> list[LagComparisonResult]:
    """
    Compare ASRI detection performance with and without lag simulation.

    Args:
        threshold: ASRI threshold for detection (default 50 = "elevated")

    Returns:
        List of comparison results for each crisis event
    """
    # Run both backtesters
    baseline_tester = LagAwareBacktester(simulate_lags=False)
    lagged_tester = LagAwareBacktester(simulate_lags=True)

    results = []

    try:
        for crisis in CRISIS_EVENTS:
            logger.info(f"Comparing lag impact for {crisis.name}")

            # Get baseline result
            baseline_result = await baseline_tester.validate_crisis(crisis)

            # Get lagged result
            lagged_result = await lagged_tester.validate_crisis(crisis)

            # Calculate lead times
            def calculate_lead_time(
                daily_values: list[tuple[datetime, float, str]],
                crisis_peak: datetime,
                threshold: float
            ) -> int:
                """Calculate days between first threshold breach and crisis peak."""
                for date, asri, _ in daily_values:
                    if asri >= threshold:
                        return (crisis_peak - date).days
                return 0  # Never crossed threshold

            baseline_lead = calculate_lead_time(
                baseline_result.daily_values,
                crisis.peak_date,
                threshold
            )
            lagged_lead = calculate_lead_time(
                lagged_result.daily_values,
                crisis.peak_date,
                threshold
            )

            # Determine limiting source
            # The source with longest lag most likely affects detection timing
            max_lag_source = max(DATA_LAGS.items(), key=lambda x: x[1].lag.total_seconds())

            # Check if both detected
            baseline_detected = baseline_result.peak_asri >= threshold
            lagged_detected = lagged_result.peak_asri >= threshold

            # Calculate ASRI degradation
            if baseline_result.peak_asri > 0:
                degradation = (
                    (baseline_result.peak_asri - lagged_result.peak_asri)
                    / baseline_result.peak_asri * 100
                )
            else:
                degradation = 0.0

            notes = []
            if degradation > 5:
                notes.append(f"ASRI degraded by {degradation:.1f}% due to data lag")
            if lagged_lead < baseline_lead:
                notes.append(f"Lost {baseline_lead - lagged_lead} days of lead time")
            if not lagged_detected and baseline_detected:
                notes.append("Crisis detection FAILED under lag simulation")

            comparison = LagComparisonResult(
                crisis_name=crisis.name,
                crisis_date=crisis.peak_date,
                baseline_peak_asri=baseline_result.peak_asri,
                baseline_lead_time=baseline_lead,
                baseline_alert_level=baseline_result.peak_alert_level,
                baseline_detected=baseline_detected,
                lagged_peak_asri=lagged_result.peak_asri,
                lagged_lead_time=lagged_lead,
                lagged_alert_level=lagged_result.peak_alert_level,
                lagged_detected=lagged_detected,
                asri_degradation=degradation,
                lead_time_change=lagged_lead - baseline_lead,
                detection_maintained=lagged_detected if baseline_detected else True,
                limiting_source=max_lag_source[0],
                notes=notes,
            )
            results.append(comparison)

    finally:
        await baseline_tester.close()
        await lagged_tester.close()

    return results


def format_lag_comparison_table(results: list[LagComparisonResult]) -> str:
    """Format comparison results as a LaTeX table."""
    lines = [
        r"\begin{table}[H]",
        r"\begin{threeparttable}",
        r"\centering",
        r"\caption{Pseudo-Real-Time Evaluation: Detection Performance with Publication Lags}",
        r"\label{tab:lag_comparison}",
        r"\small",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Perfect Foresight} & \multicolumn{3}{c}{Lag-Simulated} & \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Crisis & Peak & Lead & Det. & Peak & Lead & Det. & Deg. \\",
        r"\midrule",
    ]

    for r in results:
        baseline_det = "Yes" if r.baseline_detected else "No"
        lagged_det = "Yes" if r.lagged_detected else "No"
        deg_str = f"{r.asri_degradation:.1f}\\%"

        lines.append(
            f"{r.crisis_name[:12]} & {r.baseline_peak_asri:.1f} & {r.baseline_lead_time}d & "
            f"{baseline_det} & {r.lagged_peak_asri:.1f} & {r.lagged_lead_time}d & "
            f"{lagged_det} & {deg_str} \\\\"
        )

    # Summary row
    total = len(results)
    baseline_detected = sum(1 for r in results if r.baseline_detected)
    lagged_detected = sum(1 for r in results if r.lagged_detected)
    avg_degradation = np.mean([r.asri_degradation for r in results])
    avg_lead_change = np.mean([r.lead_time_change for r in results])

    lines.extend([
        r"\midrule",
        f"\\textit{{Summary}} & -- & -- & {baseline_detected}/{total} & "
        f"-- & -- & {lagged_detected}/{total} & {avg_degradation:.1f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Peak = peak ASRI during crisis window. Lead = days between threshold crossing (ASRI $>$ 50) and crisis peak.",
        r"\item Det. = crisis detected (threshold breach before peak). Deg. = ASRI degradation from lag simulation.",
        f"\\item Average lead time change: {avg_lead_change:+.1f} days (negative = lead time lost).",
        r"\item Publication lags: DeFi Llama TVL (6h), Stablecoins (12h), FRED Treasury (2d), VIX (1d), BTC (1h).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])

    return "\n".join(lines)


async def main():
    """Run lag comparison analysis."""
    import json

    print("=" * 70)
    print("PUBLICATION LAG IMPACT ANALYSIS")
    print("=" * 70)
    print()

    # Show lag configuration
    print("Data Source Publication Lags:")
    print("-" * 40)
    for name, lag in DATA_LAGS.items():
        hours = lag.lag.total_seconds() / 3600
        print(f"  {name}: {hours:.1f} hours ({lag.description})")
    print()

    # Run comparison
    print("Running comparison analysis...")
    results = await compare_lag_impact(threshold=50.0)

    # Print results
    print()
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    for r in results:
        status = "MAINTAINED" if r.detection_maintained else "DEGRADED"
        print(f"\n{r.crisis_name} [{status}]")
        print("-" * 50)
        print(f"  Baseline: Peak ASRI = {r.baseline_peak_asri:.1f}, "
              f"Lead = {r.baseline_lead_time} days, "
              f"Level = {r.baseline_alert_level}")
        print(f"  Lagged:   Peak ASRI = {r.lagged_peak_asri:.1f}, "
              f"Lead = {r.lagged_lead_time} days, "
              f"Level = {r.lagged_alert_level}")
        print(f"  Degradation: {r.asri_degradation:.1f}%, "
              f"Lead time change: {r.lead_time_change:+d} days")
        if r.notes:
            print("  Notes:")
            for note in r.notes:
                print(f"    - {note}")

    # Summary statistics
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(results)
    maintained = sum(1 for r in results if r.detection_maintained)
    avg_degradation = np.mean([r.asri_degradation for r in results])
    avg_lead_change = np.mean([r.lead_time_change for r in results])

    print(f"  Detection maintained: {maintained}/{total} ({maintained/total*100:.0f}%)")
    print(f"  Average ASRI degradation: {avg_degradation:.2f}%")
    print(f"  Average lead time change: {avg_lead_change:+.1f} days")

    # Save results
    results_dict = [
        {
            'crisis': r.crisis_name,
            'crisis_date': r.crisis_date.isoformat(),
            'baseline_peak': r.baseline_peak_asri,
            'baseline_lead_days': r.baseline_lead_time,
            'baseline_alert': r.baseline_alert_level,
            'lagged_peak': r.lagged_peak_asri,
            'lagged_lead_days': r.lagged_lead_time,
            'lagged_alert': r.lagged_alert_level,
            'asri_degradation_pct': r.asri_degradation,
            'lead_time_change_days': r.lead_time_change,
            'detection_maintained': r.detection_maintained,
            'notes': r.notes,
        }
        for r in results
    ]

    print()
    print("LaTeX table output:")
    print("-" * 70)
    print(format_lag_comparison_table(results))

    return results, results_dict


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
