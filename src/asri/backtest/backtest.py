"""
ASRI Backtester

Validates ASRI methodology against historical crises and
provides database backfill functionality.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete

from asri.config import settings
from asri.models.asri import ASRIDaily
from asri.models.base import Base
from asri.signals.calculator import (
    calculate_aggregate_asri,
    determine_alert_level,
    SubIndexValues,
)
from asri.pipeline.transform import (
    StablecoinRiskInputs,
    DeFiLiquidityRiskInputs,
    ContagionRiskInputs,
    ArbitrageOpacityRiskInputs,
    calculate_hhi,
    normalize_hhi_to_risk,
    normalize_to_100,
)
from asri.ingestion.defillama import StablecoinData
from .historical import HistoricalDataFetcher, HistoricalSnapshot

logger = structlog.get_logger()


@dataclass
class CrisisEvent:
    """Definition of a historical crisis event for validation."""
    name: str
    start_date: datetime
    peak_date: datetime
    end_date: datetime
    expected_level: str  # Expected alert level at peak: elevated, high, critical
    description: str
    key_indicators: list[str] = field(default_factory=list)


# Major crypto crises to validate against
CRISIS_EVENTS = [
    CrisisEvent(
        name="Luna/UST Collapse",
        start_date=datetime(2022, 5, 7),
        peak_date=datetime(2022, 5, 12),
        end_date=datetime(2022, 5, 20),
        expected_level="critical",
        description="UST depeg and Luna death spiral, $40B+ wiped",
        key_indicators=["stablecoin_risk", "defi_liquidity_risk"],
    ),
    CrisisEvent(
        name="3AC Liquidation",
        start_date=datetime(2022, 6, 13),
        peak_date=datetime(2022, 6, 17),
        end_date=datetime(2022, 6, 25),
        expected_level="high",
        description="Three Arrows Capital insolvency, cascading liquidations",
        key_indicators=["contagion_risk", "defi_liquidity_risk"],
    ),
    CrisisEvent(
        name="FTX Collapse",
        start_date=datetime(2022, 11, 6),
        peak_date=datetime(2022, 11, 11),
        end_date=datetime(2022, 11, 20),
        expected_level="critical",
        description="FTX/Alameda fraud, massive contagion",
        key_indicators=["contagion_risk", "arbitrage_opacity"],
    ),
    CrisisEvent(
        name="SVB Collapse",
        start_date=datetime(2023, 3, 8),
        peak_date=datetime(2023, 3, 11),
        end_date=datetime(2023, 3, 20),
        expected_level="elevated",
        description="Silicon Valley Bank failure, USDC brief depeg",
        key_indicators=["stablecoin_risk", "contagion_risk"],
    ),
]


@dataclass
class BacktestResult:
    """Result of backtesting ASRI against a crisis event."""
    crisis: CrisisEvent
    pre_crisis_asri: float
    peak_asri: float
    post_crisis_asri: float
    peak_alert_level: str
    validation_passed: bool
    sub_indices_at_peak: dict[str, float]
    daily_values: list[tuple[datetime, float, str]]  # (date, asri, alert_level)
    notes: list[str] = field(default_factory=list)


class ASRIBacktester:
    """
    Backtests ASRI methodology against historical events
    and provides database backfill functionality.
    """

    # Neutral sentiment for backtesting (can't get historical news)
    NEUTRAL_SENTIMENT = 40.0

    # Sub-index calculation weights (from transform layer)
    STABLECOIN_WEIGHTS = {
        'tvl_ratio': 0.4,
        'treasury_stress': 0.3,
        'concentration_hhi': 0.2,
        'peg_volatility': 0.1,
    }
    DEFI_WEIGHTS = {
        'top10_concentration': 0.35,
        'tvl_volatility': 0.25,
        'smart_contract_risk': 0.20,
        'flash_loan_proxy': 0.10,
        'leverage_change': 0.10,
    }
    CONTAGION_WEIGHTS = {
        'rwa_growth_rate': 0.30,
        'bank_exposure': 0.25,
        'tradfi_linkage': 0.20,
        'crypto_equity_correlation': 0.15,
        'bridge_exploit_frequency': 0.10,
    }
    ARBITRAGE_WEIGHTS = {
        'unregulated_exposure': 0.25,
        'multi_issuer_risk': 0.25,
        'custody_concentration': 0.20,
        'regulatory_sentiment': 0.15,
        'transparency_score': 0.15,
    }

    def __init__(self):
        self.fetcher = HistoricalDataFetcher()

    def _calculate_sub_index(self, inputs: Any, weights: dict[str, float]) -> float:
        """Calculate a sub-index from inputs using specified weights."""
        total = 0.0
        for field, weight in weights.items():
            value = getattr(inputs, field, 50.0)
            total += value * weight
        return min(100, max(0, total))

    async def close(self):
        await self.fetcher.close()

    def _snapshot_to_inputs(
        self, snapshot: HistoricalSnapshot
    ) -> tuple[StablecoinRiskInputs, DeFiLiquidityRiskInputs, ContagionRiskInputs, ArbitrageOpacityRiskInputs]:
        """Convert historical snapshot to calculator inputs."""

        # Calculate BTC-S&P500 correlation
        if len(snapshot.btc_prices) >= 20 and len(snapshot.sp500_prices) >= 20:
            # Align lengths
            min_len = min(len(snapshot.btc_prices), len(snapshot.sp500_prices))
            btc = np.array(snapshot.btc_prices[-min_len:])
            sp500 = np.array(snapshot.sp500_prices[-min_len:])

            # Calculate returns
            btc_returns = np.diff(btc) / btc[:-1]
            sp_returns = np.diff(sp500) / sp500[:-1]

            # Correlation
            if len(btc_returns) > 5 and len(sp_returns) > 5:
                corr = np.corrcoef(btc_returns, sp_returns)[0, 1]
                crypto_equity_corr = corr if not np.isnan(corr) else 0.3
            else:
                crypto_equity_corr = 0.3
        else:
            crypto_equity_corr = 0.3

        # Convert stablecoin caps to StablecoinData objects
        stablecoins = [
            StablecoinData(
                name=symbol,
                symbol=symbol,
                circulating=cap,
                price=1.0,  # Assume at peg for historical
                peg_deviation=0.0,
            )
            for symbol, cap in snapshot.stablecoin_market_caps.items()
        ]

        # Stablecoin Risk Inputs
        tvl_ratio = snapshot.current_tvl / snapshot.max_historical_tvl if snapshot.max_historical_tvl > 0 else 0.5
        tvl_risk = normalize_to_100(1 - tvl_ratio, 0, 0.5)
        treasury_stress = normalize_to_100(snapshot.treasury_10y_rate, 2.0, 6.0)

        circulating_values = [s.circulating for s in stablecoins if s.circulating > 0]
        hhi = calculate_hhi(circulating_values)
        concentration_risk = normalize_hhi_to_risk(hhi)

        stablecoin_inputs = StablecoinRiskInputs(
            tvl_ratio=tvl_risk,
            treasury_stress=treasury_stress,
            concentration_hhi=concentration_risk,
            peg_volatility=10.0,  # Default - can't easily get historical peg data
        )

        # DeFi Liquidity Risk Inputs
        protocols_with_tvl = [p for p in snapshot.protocols if (p.get('tvl') or 0) > 0]
        tvl_values = [p.get('tvl') or 0 for p in protocols_with_tvl]
        top10_tvl = sorted(tvl_values, reverse=True)[:10]
        top10_hhi = calculate_hhi(top10_tvl)

        # TVL volatility from 30-day history
        if len(snapshot.tvl_30d_history) > 1:
            tvl_std = np.std(snapshot.tvl_30d_history)
            tvl_mean = np.mean(snapshot.tvl_30d_history)
            volatility = tvl_std / tvl_mean if tvl_mean > 0 else 0
            tvl_volatility = normalize_to_100(volatility * 100, 0, 20)
        else:
            tvl_volatility = 30.0

        def has_audit(p):
            audits = p.get('audits')
            if audits is None:
                return False
            try:
                return int(audits) > 0
            except (ValueError, TypeError):
                return bool(audits)

        audited = sum(1 for p in protocols_with_tvl if has_audit(p))
        audit_ratio = audited / len(protocols_with_tvl) if protocols_with_tvl else 0.3
        smart_contract_risk = (1 - audit_ratio) * 100

        defi_inputs = DeFiLiquidityRiskInputs(
            top10_concentration=normalize_hhi_to_risk(top10_hhi),
            tvl_volatility=tvl_volatility,
            smart_contract_risk=smart_contract_risk,
            flash_loan_proxy=30.0,  # Default
            leverage_change=40.0,  # Default
        )

        # Contagion Risk Inputs
        vix_stress = normalize_to_100(snapshot.vix, 12.0, 40.0)
        bank_exposure = (treasury_stress * 0.6 + vix_stress * 0.4)

        if snapshot.yield_curve_spread < 0:
            tradfi_linkage = normalize_to_100(abs(snapshot.yield_curve_spread), 0, 2) + 50
        else:
            tradfi_linkage = max(0, 50 - normalize_to_100(snapshot.yield_curve_spread, 0, 2))

        correlation_risk = normalize_to_100(abs(crypto_equity_corr), 0, 1)

        contagion_inputs = ContagionRiskInputs(
            rwa_growth_rate=25.0,  # Default
            bank_exposure=bank_exposure,
            tradfi_linkage=tradfi_linkage,
            crypto_equity_correlation=correlation_risk,
            bridge_exploit_frequency=normalize_to_100(len(snapshot.bridges), 0, 150),
        )

        # Arbitrage Opacity Risk Inputs
        num_issuers = len([s for s in stablecoins if s.circulating > 1e9])
        if num_issuers < 3:
            multi_issuer_risk = 70.0
        elif num_issuers < 10:
            multi_issuer_risk = 30.0
        else:
            multi_issuer_risk = 50.0 + (num_issuers - 10) * 2

        total_supply = sum(s.circulating for s in stablecoins)
        top2 = sorted(stablecoins, key=lambda x: x.circulating, reverse=True)[:2]
        top2_share = sum(s.circulating for s in top2) / total_supply * 100 if total_supply > 0 else 85

        arbitrage_inputs = ArbitrageOpacityRiskInputs(
            unregulated_exposure=35.0,
            multi_issuer_risk=multi_issuer_risk,
            custody_concentration=normalize_to_100(top2_share, 50, 100),
            regulatory_sentiment=self.NEUTRAL_SENTIMENT,
            transparency_score=audit_ratio * 100,
        )

        return stablecoin_inputs, defi_inputs, contagion_inputs, arbitrage_inputs

    async def calculate_for_date(self, target_date: datetime) -> dict[str, Any]:
        """
        Calculate ASRI for a specific historical date.

        Returns dict with asri, sub-indices, alert_level, and data_quality.
        """
        snapshot = await self.fetcher.fetch_snapshot(target_date)

        stablecoin_inputs, defi_inputs, contagion_inputs, arbitrage_inputs = \
            self._snapshot_to_inputs(snapshot)

        # Calculate sub-indices
        stablecoin_risk = self._calculate_sub_index(stablecoin_inputs, self.STABLECOIN_WEIGHTS)
        defi_risk = self._calculate_sub_index(defi_inputs, self.DEFI_WEIGHTS)
        contagion_risk = self._calculate_sub_index(contagion_inputs, self.CONTAGION_WEIGHTS)
        arbitrage_risk = self._calculate_sub_index(arbitrage_inputs, self.ARBITRAGE_WEIGHTS)

        # Calculate aggregate ASRI
        sub_indices = SubIndexValues(
            stablecoin_risk=stablecoin_risk,
            defi_liquidity_risk=defi_risk,
            contagion_risk=contagion_risk,
            arbitrage_opacity=arbitrage_risk,
        )
        asri = calculate_aggregate_asri(sub_indices)

        alert_level = determine_alert_level(asri)

        return {
            'date': target_date,
            'asri': asri,
            'alert_level': alert_level,
            'stablecoin_risk': stablecoin_risk,
            'defi_liquidity_risk': defi_risk,
            'contagion_risk': contagion_risk,
            'arbitrage_opacity': arbitrage_risk,
            'data_quality': snapshot.data_quality,
            'raw_snapshot': {
                'tvl': snapshot.current_tvl,
                'max_tvl': snapshot.max_historical_tvl,
                'stablecoin_supply': snapshot.total_stablecoin_supply,
                'treasury_10y': snapshot.treasury_10y_rate,
                'vix': snapshot.vix,
                'yield_spread': snapshot.yield_curve_spread,
            }
        }

    async def validate_crisis(self, crisis: CrisisEvent) -> BacktestResult:
        """
        Validate ASRI against a specific crisis event.

        Checks if ASRI properly elevated during the crisis period.
        """
        logger.info(f"Validating ASRI against {crisis.name}")

        daily_values: list[tuple[datetime, float, str]] = []
        notes: list[str] = []

        # Calculate ASRI for the crisis period
        current = crisis.start_date
        while current <= crisis.end_date:
            try:
                result = await self.calculate_for_date(current)
                daily_values.append((current, result['asri'], result['alert_level']))
            except Exception as e:
                logger.warning(f"Failed to calculate for {current.date()}", error=str(e))
                notes.append(f"Missing data for {current.date()}: {e}")

            current += timedelta(days=1)

        if not daily_values:
            return BacktestResult(
                crisis=crisis,
                pre_crisis_asri=0,
                peak_asri=0,
                post_crisis_asri=0,
                peak_alert_level="unknown",
                validation_passed=False,
                sub_indices_at_peak={},
                daily_values=[],
                notes=["No data available for crisis period"],
            )

        # Find peak ASRI during crisis
        peak_idx = max(range(len(daily_values)), key=lambda i: daily_values[i][1])
        peak_date, peak_asri, peak_alert = daily_values[peak_idx]

        # Get pre-crisis (first day) and post-crisis (last day) values
        pre_crisis_asri = daily_values[0][1]
        post_crisis_asri = daily_values[-1][1]

        # Get sub-indices at peak
        peak_result = await self.calculate_for_date(peak_date)
        sub_indices_at_peak = {
            'stablecoin_risk': peak_result['stablecoin_risk'],
            'defi_liquidity_risk': peak_result['defi_liquidity_risk'],
            'contagion_risk': peak_result['contagion_risk'],
            'arbitrage_opacity': peak_result['arbitrage_opacity'],
        }

        # Validation logic
        alert_levels = ['low', 'moderate', 'elevated', 'high', 'critical']
        expected_idx = alert_levels.index(crisis.expected_level)
        actual_idx = alert_levels.index(peak_alert) if peak_alert in alert_levels else 0

        # Pass if actual >= expected - 1 (allow one level tolerance)
        validation_passed = actual_idx >= expected_idx - 1

        if not validation_passed:
            notes.append(
                f"ASRI peaked at {peak_alert} but expected {crisis.expected_level}"
            )

        # Check if ASRI increased from pre to peak
        if peak_asri > pre_crisis_asri * 1.1:
            notes.append(f"ASRI increased {((peak_asri / pre_crisis_asri) - 1) * 100:.1f}% from pre-crisis")
        else:
            notes.append("ASRI did not significantly increase during crisis")
            validation_passed = False

        # Check key indicators
        for indicator in crisis.key_indicators:
            if indicator in sub_indices_at_peak:
                value = sub_indices_at_peak[indicator]
                if value > 60:
                    notes.append(f"{indicator} elevated at {value:.1f}")
                else:
                    notes.append(f"{indicator} surprisingly low at {value:.1f}")

        logger.info(
            f"Crisis validation complete",
            crisis=crisis.name,
            peak_asri=peak_asri,
            peak_alert=peak_alert,
            expected=crisis.expected_level,
            passed=validation_passed,
        )

        return BacktestResult(
            crisis=crisis,
            pre_crisis_asri=pre_crisis_asri,
            peak_asri=peak_asri,
            post_crisis_asri=post_crisis_asri,
            peak_alert_level=peak_alert,
            validation_passed=validation_passed,
            sub_indices_at_peak=sub_indices_at_peak,
            daily_values=daily_values,
            notes=notes,
        )

    async def run_full_backtest(self) -> list[BacktestResult]:
        """Run validation against all defined crisis events."""
        results = []

        for crisis in CRISIS_EVENTS:
            result = await self.validate_crisis(crisis)
            results.append(result)

        return results

    async def backfill_database(
        self,
        start_date: datetime,
        end_date: datetime,
        step_days: int = 1,
        clear_existing: bool = False,
    ) -> int:
        """
        Backfill database with historical ASRI calculations.

        Args:
            start_date: Start date for backfill
            end_date: End date for backfill
            step_days: Days between calculations
            clear_existing: Whether to clear existing records in range

        Returns:
            Number of records created
        """
        engine = create_async_engine(settings.database_url)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        records_created = 0
        current = start_date

        async with async_session() as session:
            # Optionally clear existing records in range
            if clear_existing:
                stmt = delete(ASRIDaily).where(
                    ASRIDaily.date >= start_date,
                    ASRIDaily.date <= end_date,
                )
                await session.execute(stmt)
                await session.commit()
                logger.info(f"Cleared existing records from {start_date.date()} to {end_date.date()}")

            while current <= end_date:
                try:
                    # Check if record already exists
                    existing = await session.execute(
                        select(ASRIDaily).where(
                            ASRIDaily.date >= current,
                            ASRIDaily.date < current + timedelta(days=1),
                        )
                    )
                    if existing.scalar_one_or_none():
                        logger.debug(f"Record exists for {current.date()}, skipping")
                        current += timedelta(days=step_days)
                        continue

                    result = await self.calculate_for_date(current)

                    # Calculate 30-day average (from prior records)
                    prior_records = await session.execute(
                        select(ASRIDaily.asri).where(
                            ASRIDaily.date >= current - timedelta(days=30),
                            ASRIDaily.date < current,
                        ).order_by(ASRIDaily.date.desc())
                    )
                    prior_values = [r[0] for r in prior_records.fetchall()]
                    prior_values.append(result['asri'])
                    asri_30d_avg = np.mean(prior_values)

                    # Calculate trend
                    if len(prior_values) >= 7:
                        recent = prior_values[:7]
                        if recent[0] > recent[-1] * 1.05:
                            trend = "increasing"
                        elif recent[0] < recent[-1] * 0.95:
                            trend = "decreasing"
                        else:
                            trend = "stable"
                    else:
                        trend = "stable"

                    record = ASRIDaily(
                        date=current,
                        asri=result['asri'],
                        asri_normalized=result['asri'],
                        asri_30d_avg=asri_30d_avg,
                        trend=trend,
                        alert_level=result['alert_level'],
                        stablecoin_risk=result['stablecoin_risk'],
                        defi_liquidity_risk=result['defi_liquidity_risk'],
                        contagion_risk=result['contagion_risk'],
                        arbitrage_opacity=result['arbitrage_opacity'],
                    )

                    session.add(record)
                    records_created += 1

                    # Commit every 30 records
                    if records_created % 30 == 0:
                        await session.commit()
                        logger.info(f"Backfilled {records_created} records (up to {current.date()})")

                except Exception as e:
                    logger.error(f"Failed to backfill {current.date()}", error=str(e))

                current += timedelta(days=step_days)

            await session.commit()

        await engine.dispose()

        logger.info(f"Backfill complete: {records_created} records created")
        return records_created


def print_backtest_report(results: list[BacktestResult]):
    """Print a formatted backtest report."""
    print("\n" + "=" * 70)
    print("ASRI BACKTEST VALIDATION REPORT")
    print("=" * 70)

    passed = sum(1 for r in results if r.validation_passed)
    print(f"\nOverall: {passed}/{len(results)} validations passed\n")

    for result in results:
        status = "PASS" if result.validation_passed else "FAIL"
        print(f"\n{result.crisis.name} [{status}]")
        print("-" * 50)
        print(f"  Period: {result.crisis.start_date.date()} to {result.crisis.end_date.date()}")
        print(f"  Description: {result.crisis.description}")
        print()
        print(f"  Pre-Crisis ASRI:  {result.pre_crisis_asri:.1f}")
        print(f"  Peak ASRI:        {result.peak_asri:.1f} ({result.peak_alert_level})")
        print(f"  Post-Crisis ASRI: {result.post_crisis_asri:.1f}")
        print(f"  Expected Level:   {result.crisis.expected_level}")
        print()
        print("  Sub-Indices at Peak:")
        for name, value in result.sub_indices_at_peak.items():
            indicator = "*" if name in result.crisis.key_indicators else " "
            print(f"    {indicator} {name}: {value:.1f}")
        print()
        print("  Notes:")
        for note in result.notes:
            print(f"    - {note}")

    print("\n" + "=" * 70)


async def main():
    """Run full backtest validation."""
    import asyncio

    backtester = ASRIBacktester()

    try:
        print("Running ASRI backtest validation...")
        results = await backtester.run_full_backtest()
        print_backtest_report(results)

    finally:
        await backtester.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
