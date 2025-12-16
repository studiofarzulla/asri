"""
ASRI Data Transform Layer

Converts raw API data from DeFiLlama, FRED, and CoinGecko
into normalized sub-index inputs (0-100 scale).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class StablecoinRiskInputs:
    """Inputs for Stablecoin Concentration & Treasury Exposure sub-index."""
    tvl_ratio: float  # current_tvl / max_historical_tvl
    treasury_stress: float  # normalized treasury rate stress
    concentration_hhi: float  # Herfindahl-Hirschman Index (0-10000 normalized)
    peg_volatility: float  # weighted average peg deviation


@dataclass
class DeFiLiquidityRiskInputs:
    """Inputs for DeFi Liquidity & Composability Risk sub-index."""
    top10_concentration: float  # HHI of top 10 protocols
    tvl_volatility: float  # std(tvl) / mean(tvl) over 30 days
    smart_contract_risk: float  # inverse of average audit score
    flash_loan_proxy: float  # yield volatility as proxy
    leverage_change: float  # change in lending TVL


@dataclass
class ContagionRiskInputs:
    """Inputs for Cross-Market Interconnection & Contagion Risk sub-index."""
    rwa_growth_rate: float  # RWA TVL growth
    bank_exposure: float  # composite of treasury + VIX
    tradfi_linkage: float  # yield curve inversion signal
    crypto_equity_correlation: float  # BTC-SPY correlation
    bridge_exploit_frequency: float  # number of active bridges as proxy


@dataclass
class ArbitrageOpacityRiskInputs:
    """Inputs for Regulatory Arbitrage & Opacity Risk sub-index."""
    unregulated_exposure: float  # % on unregulated chains
    multi_issuer_risk: float  # number of stablecoin issuers
    custody_concentration: float  # chain concentration for stables
    regulatory_sentiment: float  # manual input (0-100)
    transparency_score: float  # average audit coverage


@dataclass
class TransformedData:
    """All transformed inputs for ASRI calculation."""
    timestamp: datetime
    stablecoin_risk: StablecoinRiskInputs
    defi_liquidity_risk: DeFiLiquidityRiskInputs
    contagion_risk: ContagionRiskInputs
    arbitrage_opacity_risk: ArbitrageOpacityRiskInputs
    raw_metrics: dict[str, Any]  # Store raw values for debugging


def calculate_hhi(values: list[float]) -> float:
    """
    Calculate Herfindahl-Hirschman Index.

    HHI = Σ(market_share²) * 10000

    Interpretation:
    - < 1500: Competitive market
    - 1500-2500: Moderate concentration
    - > 2500: Highly concentrated
    """
    if not values or sum(values) == 0:
        return 0.0

    total = sum(values)
    shares = [v / total for v in values]
    hhi = sum(s ** 2 for s in shares) * 10000
    return hhi


def normalize_to_100(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to 0-100 scale."""
    if max_val == min_val:
        return 50.0
    normalized = (value - min_val) / (max_val - min_val) * 100
    return max(0, min(100, normalized))


def normalize_hhi_to_risk(hhi: float) -> float:
    """
    Convert HHI to risk score (0-100).

    HHI 0-1500 → Risk 0-30 (low risk)
    HHI 1500-2500 → Risk 30-60 (moderate)
    HHI 2500-5000 → Risk 60-90 (high)
    HHI 5000-10000 → Risk 90-100 (critical)
    """
    if hhi < 1500:
        return hhi / 1500 * 30
    elif hhi < 2500:
        return 30 + (hhi - 1500) / 1000 * 30
    elif hhi < 5000:
        return 60 + (hhi - 2500) / 2500 * 30
    else:
        return 90 + (hhi - 5000) / 5000 * 10


class DataTransformer:
    """Transforms raw API data into ASRI sub-index inputs."""

    # Thresholds for normalization
    VIX_LOW = 12.0
    VIX_HIGH = 40.0
    TREASURY_10Y_LOW = 2.0
    TREASURY_10Y_HIGH = 6.0

    def __init__(self):
        self.logger = structlog.get_logger()

    def transform_stablecoin_risk(
        self,
        stablecoins: list[Any],
        current_tvl: float,
        max_historical_tvl: float,
        treasury_10y_rate: float,
    ) -> StablecoinRiskInputs:
        """Transform raw data into stablecoin risk inputs."""

        # TVL ratio (higher ratio = lower risk, invert for risk score)
        tvl_ratio = current_tvl / max_historical_tvl if max_historical_tvl > 0 else 0.5
        # Invert: at max TVL = low risk, at 50% of max = high risk
        tvl_risk = normalize_to_100(1 - tvl_ratio, 0, 0.5)

        # Treasury stress (higher rates = higher stress)
        treasury_stress = normalize_to_100(
            treasury_10y_rate,
            self.TREASURY_10Y_LOW,
            self.TREASURY_10Y_HIGH
        )

        # Stablecoin concentration HHI
        circulating_values = [s.circulating for s in stablecoins if s.circulating > 0]
        hhi = calculate_hhi(circulating_values)
        concentration_risk = normalize_hhi_to_risk(hhi)

        # Peg volatility (weighted average deviation from $1)
        total_supply = sum(s.circulating for s in stablecoins if s.circulating > 0)
        if total_supply > 0:
            weighted_deviation = sum(
                s.peg_deviation * s.circulating
                for s in stablecoins if s.circulating > 0
            ) / total_supply
            # Normalize: 0% deviation = 0 risk, 5% deviation = 100 risk
            peg_volatility = normalize_to_100(weighted_deviation * 100, 0, 5)
        else:
            peg_volatility = 50.0

        self.logger.info(
            "Transformed stablecoin risk",
            tvl_risk=tvl_risk,
            treasury_stress=treasury_stress,
            concentration_risk=concentration_risk,
            peg_volatility=peg_volatility,
            hhi=hhi,
        )

        return StablecoinRiskInputs(
            tvl_ratio=tvl_risk,
            treasury_stress=treasury_stress,
            concentration_hhi=concentration_risk,
            peg_volatility=peg_volatility,
        )

    def transform_defi_liquidity_risk(
        self,
        protocols: list[dict],
        tvl_history: list[float] | None = None,
    ) -> DeFiLiquidityRiskInputs:
        """Transform raw data into DeFi liquidity risk inputs."""

        # Filter to protocols with TVL
        protocols_with_tvl = [p for p in protocols if (p.get('tvl') or 0) > 0]

        # Top 10 concentration
        tvl_values = [p.get('tvl') or 0 for p in protocols_with_tvl]
        top10_tvl = sorted(tvl_values, reverse=True)[:10]
        top10_hhi = calculate_hhi(top10_tvl)
        top10_concentration = normalize_hhi_to_risk(top10_hhi)

        # TVL volatility (if historical data available)
        if tvl_history and len(tvl_history) > 1:
            tvl_std = np.std(tvl_history)
            tvl_mean = np.mean(tvl_history)
            volatility = tvl_std / tvl_mean if tvl_mean > 0 else 0
            # Normalize: 0% volatility = 0 risk, 20% volatility = 100 risk
            tvl_volatility = normalize_to_100(volatility * 100, 0, 20)
        else:
            tvl_volatility = 30.0  # Default moderate if no history

        # Smart contract risk (inverse of audit coverage)
        def has_audit(p):
            audits = p.get('audits')
            if audits is None:
                return False
            try:
                return int(audits) > 0
            except (ValueError, TypeError):
                return bool(audits)  # Non-empty string = has audit

        audited = sum(1 for p in protocols_with_tvl if has_audit(p))
        audit_ratio = audited / len(protocols_with_tvl) if protocols_with_tvl else 0
        # Invert: high audit coverage = low risk
        smart_contract_risk = (1 - audit_ratio) * 100

        # Flash loan proxy (use change_1d as volatility proxy)
        changes = [abs(p.get('change_1d') or 0) for p in protocols_with_tvl]
        avg_change = np.mean(changes) if changes else 5.0
        flash_loan_proxy = normalize_to_100(avg_change, 0, 20)

        # Leverage change (lending protocol TVL delta)
        lending = [p for p in protocols_with_tvl if p.get('category') == 'Lending']
        lending_tvl = sum(p.get('tvl') or 0 for p in lending)
        total_tvl = sum(tvl_values)
        leverage_ratio = lending_tvl / total_tvl * 100 if total_tvl > 0 else 10
        # Higher lending ratio = more leverage = higher risk
        leverage_change = normalize_to_100(leverage_ratio, 0, 30)

        self.logger.info(
            "Transformed DeFi liquidity risk",
            top10_concentration=top10_concentration,
            tvl_volatility=tvl_volatility,
            smart_contract_risk=smart_contract_risk,
            flash_loan_proxy=flash_loan_proxy,
            leverage_change=leverage_change,
        )

        return DeFiLiquidityRiskInputs(
            top10_concentration=top10_concentration,
            tvl_volatility=tvl_volatility,
            smart_contract_risk=smart_contract_risk,
            flash_loan_proxy=flash_loan_proxy,
            leverage_change=leverage_change,
        )

    def transform_contagion_risk(
        self,
        protocols: list[dict],
        treasury_10y_rate: float,
        vix: float,
        yield_curve_spread: float,
        bridges: list[dict],
        crypto_equity_corr: float = 0.5,  # Default moderate correlation
    ) -> ContagionRiskInputs:
        """Transform raw data into contagion risk inputs."""

        # RWA growth rate
        rwa = [p for p in protocols if p.get('category') == 'RWA']
        total_tvl = sum(p.get('tvl') or 0 for p in protocols if (p.get('tvl') or 0) > 0)
        rwa_tvl = sum(p.get('tvl') or 0 for p in rwa)
        rwa_share = rwa_tvl / total_tvl * 100 if total_tvl > 0 else 0
        # Higher RWA share = more TradFi linkage = higher risk
        rwa_growth_rate = normalize_to_100(rwa_share, 0, 10)

        # Bank exposure (composite of treasury stress + VIX)
        treasury_stress = normalize_to_100(
            treasury_10y_rate,
            self.TREASURY_10Y_LOW,
            self.TREASURY_10Y_HIGH
        )
        vix_stress = normalize_to_100(vix, self.VIX_LOW, self.VIX_HIGH)
        bank_exposure = (treasury_stress * 0.6 + vix_stress * 0.4)

        # TradFi linkage (yield curve inversion)
        # Negative spread = inverted = high risk
        if yield_curve_spread < 0:
            tradfi_linkage = normalize_to_100(abs(yield_curve_spread), 0, 2) + 50
        else:
            # Positive spread = normal = lower risk
            tradfi_linkage = max(0, 50 - normalize_to_100(yield_curve_spread, 0, 2))

        # Crypto-equity correlation
        # Higher correlation = more contagion risk
        correlation_risk = normalize_to_100(abs(crypto_equity_corr), 0, 1)

        # Bridge exploit proxy (more bridges = more attack surface)
        num_bridges = len(bridges)
        # 0-20 bridges = low risk, 50+ = moderate, 100+ = high
        bridge_risk = normalize_to_100(num_bridges, 0, 150)

        self.logger.info(
            "Transformed contagion risk",
            rwa_growth_rate=rwa_growth_rate,
            bank_exposure=bank_exposure,
            tradfi_linkage=tradfi_linkage,
            correlation_risk=correlation_risk,
            bridge_risk=bridge_risk,
        )

        return ContagionRiskInputs(
            rwa_growth_rate=rwa_growth_rate,
            bank_exposure=bank_exposure,
            tradfi_linkage=tradfi_linkage,
            crypto_equity_correlation=correlation_risk,
            bridge_exploit_frequency=bridge_risk,
        )

    def transform_arbitrage_opacity_risk(
        self,
        stablecoins: list[Any],
        protocols: list[dict],
        regulatory_sentiment: float = 50.0,  # Manual input default
    ) -> ArbitrageOpacityRiskInputs:
        """Transform raw data into arbitrage/opacity risk inputs."""

        # Define "regulated" vs "unregulated" chains
        REGULATED_CHAINS = {'ethereum', 'polygon', 'arbitrum', 'optimism', 'base'}

        # Calculate stablecoin distribution by chain
        # This is a simplification - would need more detailed chain data
        total_supply = sum(s.circulating for s in stablecoins if s.circulating > 0)

        # Multi-issuer risk (more issuers = more complexity but also diversification)
        num_issuers = len([s for s in stablecoins if s.circulating > 1e9])  # Only count significant ones
        # Sweet spot is 3-5, too few = concentration, too many = chaos
        if num_issuers < 3:
            multi_issuer_risk = 70.0  # Too concentrated
        elif num_issuers < 10:
            multi_issuer_risk = 30.0  # Good diversification
        else:
            multi_issuer_risk = 50.0 + (num_issuers - 10) * 2  # Getting fragmented

        # Custody concentration (top 2 stablecoins share)
        top2 = sorted(stablecoins, key=lambda x: x.circulating, reverse=True)[:2]
        top2_share = sum(s.circulating for s in top2) / total_supply * 100 if total_supply > 0 else 0
        custody_concentration = normalize_to_100(top2_share, 50, 100)

        # Unregulated exposure (simplified: assume 30% on unregulated chains)
        # Would need actual chain breakdown for accurate calculation
        unregulated_exposure = 35.0  # Placeholder - needs real chain data

        # Transparency score (based on audit coverage)
        protocols_with_tvl = [p for p in protocols if (p.get('tvl') or 0) > 0]

        def has_audit(p):
            audits = p.get('audits')
            if audits is None:
                return False
            try:
                return int(audits) > 0
            except (ValueError, TypeError):
                return bool(audits)

        audited = sum(1 for p in protocols_with_tvl if has_audit(p))
        audit_ratio = audited / len(protocols_with_tvl) if protocols_with_tvl else 0
        transparency_score = audit_ratio * 100

        self.logger.info(
            "Transformed arbitrage/opacity risk",
            unregulated_exposure=unregulated_exposure,
            multi_issuer_risk=multi_issuer_risk,
            custody_concentration=custody_concentration,
            regulatory_sentiment=regulatory_sentiment,
            transparency_score=transparency_score,
        )

        return ArbitrageOpacityRiskInputs(
            unregulated_exposure=unregulated_exposure,
            multi_issuer_risk=multi_issuer_risk,
            custody_concentration=custody_concentration,
            regulatory_sentiment=regulatory_sentiment,
            transparency_score=transparency_score,
        )

    def calculate_sub_index(self, inputs: Any, weights: dict[str, float]) -> float:
        """Calculate a sub-index from inputs using specified weights."""
        total = 0.0
        for field, weight in weights.items():
            value = getattr(inputs, field, 50.0)
            total += value * weight
        return min(100, max(0, total))


def transform_all_data(
    stablecoins: list[Any],
    protocols: list[dict],
    bridges: list[dict],
    current_tvl: float,
    max_historical_tvl: float,
    treasury_10y_rate: float,
    vix: float,
    yield_curve_spread: float,
    tvl_history: list[float] | None = None,
    crypto_equity_corr: float = 0.5,
    regulatory_sentiment: float = 50.0,
) -> TransformedData:
    """
    Transform all raw data into ASRI inputs.

    This is the main entry point for the transform layer.
    """
    transformer = DataTransformer()

    stablecoin_inputs = transformer.transform_stablecoin_risk(
        stablecoins=stablecoins,
        current_tvl=current_tvl,
        max_historical_tvl=max_historical_tvl,
        treasury_10y_rate=treasury_10y_rate,
    )

    defi_inputs = transformer.transform_defi_liquidity_risk(
        protocols=protocols,
        tvl_history=tvl_history,
    )

    contagion_inputs = transformer.transform_contagion_risk(
        protocols=protocols,
        treasury_10y_rate=treasury_10y_rate,
        vix=vix,
        yield_curve_spread=yield_curve_spread,
        bridges=bridges,
        crypto_equity_corr=crypto_equity_corr,
    )

    arbitrage_inputs = transformer.transform_arbitrage_opacity_risk(
        stablecoins=stablecoins,
        protocols=protocols,
        regulatory_sentiment=regulatory_sentiment,
    )

    return TransformedData(
        timestamp=datetime.utcnow(),
        stablecoin_risk=stablecoin_inputs,
        defi_liquidity_risk=defi_inputs,
        contagion_risk=contagion_inputs,
        arbitrage_opacity_risk=arbitrage_inputs,
        raw_metrics={
            'current_tvl': current_tvl,
            'max_historical_tvl': max_historical_tvl,
            'treasury_10y_rate': treasury_10y_rate,
            'vix': vix,
            'yield_curve_spread': yield_curve_spread,
            'num_stablecoins': len(stablecoins),
            'num_protocols': len(protocols),
            'num_bridges': len(bridges),
        }
    )
