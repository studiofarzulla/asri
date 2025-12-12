"""
ASRI Signal Calculator

Computes sub-indices and aggregate ASRI from raw data inputs.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class SubIndexValues:
    """Container for all sub-index values."""

    stablecoin_risk: float
    defi_liquidity_risk: float
    contagion_risk: float
    arbitrage_opacity: float


@dataclass
class ASRIResult:
    """Complete ASRI calculation result."""

    timestamp: datetime
    asri: float
    asri_normalized: float
    sub_indices: SubIndexValues
    alert_level: str


# Sub-index weights (from proposal)
WEIGHTS = {
    "stablecoin_risk": 0.30,
    "defi_liquidity_risk": 0.25,
    "contagion_risk": 0.25,
    "arbitrage_opacity": 0.20,
}

# Alert thresholds
ALERT_THRESHOLDS = {
    "low": 30,
    "moderate": 50,
    "elevated": 70,
    "high": 85,
    "critical": 95,
}


def calculate_stablecoin_risk(
    tvl_current: float,
    tvl_max_historical: float,
    treasury_weight: float,
    total_reserves: float,
    reserve_concentration_hhi: float,
    peg_volatility_30d: float,
) -> float:
    """
    Calculate Stablecoin Concentration & Treasury Exposure Risk sub-index.

    Formula:
    StablecoinRisk = 0.4 × (TVL/MaxTVL) + 0.3 × (Treasury/Reserves) +
                     0.2 × HHI + 0.1 × PegVolatility
    """
    tvl_ratio = tvl_current / tvl_max_historical if tvl_max_historical > 0 else 0
    treasury_ratio = treasury_weight / total_reserves if total_reserves > 0 else 0

    score = (
        0.4 * tvl_ratio
        + 0.3 * treasury_ratio
        + 0.2 * reserve_concentration_hhi
        + 0.1 * peg_volatility_30d
    )
    return min(100, max(0, score * 100))


def calculate_defi_liquidity_risk(
    top10_concentration: float,
    tvl_volatility_30d: float,
    smart_contract_risk_score: float,
    flash_loan_volume_spike: float,
    leverage_ratio_change: float,
) -> float:
    """
    Calculate DeFi Liquidity & Composability Risk sub-index.

    Formula:
    DeFiRisk = 0.35 × Top10Concentration + 0.25 × TVLVolatility +
               0.20 × SCRisk + 0.10 × FlashLoan + 0.10 × LeverageChange
    """
    score = (
        0.35 * top10_concentration
        + 0.25 * tvl_volatility_30d
        + 0.20 * smart_contract_risk_score
        + 0.10 * flash_loan_volume_spike
        + 0.10 * leverage_ratio_change
    )
    return min(100, max(0, score * 100))


def calculate_contagion_risk(
    rwa_growth_rate: float,
    bank_exposure_score: float,
    tradfi_linkage_intensity: float,
    crypto_equity_correlation: float,
    bridge_exploit_frequency: float,
) -> float:
    """
    Calculate Cross-Market Interconnection & Contagion Risk sub-index.

    Formula:
    ContagionRisk = 0.30 × RWAGrowth + 0.25 × BankExposure +
                    0.20 × TradFiLinkage + 0.15 × Correlation + 0.10 × BridgeExploits
    """
    score = (
        0.30 * rwa_growth_rate
        + 0.25 * bank_exposure_score
        + 0.20 * tradfi_linkage_intensity
        + 0.15 * crypto_equity_correlation
        + 0.10 * bridge_exploit_frequency
    )
    return min(100, max(0, score * 100))


def calculate_arbitrage_opacity_risk(
    unregulated_exposure: float,
    multi_issuer_risk: float,
    custody_concentration: float,
    regulatory_sentiment_score: float,
    transparency_score: float,
) -> float:
    """
    Calculate Regulatory Arbitrage & Opacity Risk sub-index.

    Formula:
    ArbitrageRisk = 0.25 × UnregulatedExp + 0.25 × MultiIssuer +
                    0.20 × CustodyConc + 0.15 × RegSentiment + 0.15 × Transparency
    """
    score = (
        0.25 * unregulated_exposure
        + 0.25 * multi_issuer_risk
        + 0.20 * custody_concentration
        + 0.15 * regulatory_sentiment_score
        + 0.15 * (1 - transparency_score)  # Invert: low transparency = high risk
    )
    return min(100, max(0, score * 100))


def calculate_aggregate_asri(sub_indices: SubIndexValues) -> float:
    """
    Calculate aggregate ASRI from sub-indices.

    ASRI = 0.30 × StablecoinRisk + 0.25 × DeFiRisk +
           0.25 × ContagionRisk + 0.20 × ArbitrageRisk
    """
    return (
        WEIGHTS["stablecoin_risk"] * sub_indices.stablecoin_risk
        + WEIGHTS["defi_liquidity_risk"] * sub_indices.defi_liquidity_risk
        + WEIGHTS["contagion_risk"] * sub_indices.contagion_risk
        + WEIGHTS["arbitrage_opacity"] * sub_indices.arbitrage_opacity
    )


def normalize_asri(asri: float, historical_min: float, historical_max: float) -> float:
    """
    Normalize ASRI to 0-100 scale based on historical range.

    Normalized = 100 × (ASRI - min) / (max - min)
    """
    if historical_max == historical_min:
        return 50.0  # Default to midpoint if no range
    return 100 * (asri - historical_min) / (historical_max - historical_min)


def determine_alert_level(asri: float) -> str:
    """Determine alert level based on ASRI value."""
    if asri >= ALERT_THRESHOLDS["critical"]:
        return "critical"
    elif asri >= ALERT_THRESHOLDS["high"]:
        return "high"
    elif asri >= ALERT_THRESHOLDS["elevated"]:
        return "elevated"
    elif asri >= ALERT_THRESHOLDS["moderate"]:
        return "moderate"
    else:
        return "low"


def compute_asri(
    stablecoin_risk: float,
    defi_liquidity_risk: float,
    contagion_risk: float,
    arbitrage_opacity: float,
    historical_min: float = 0,
    historical_max: float = 100,
) -> ASRIResult:
    """
    Compute complete ASRI result from sub-index values.

    Args:
        stablecoin_risk: Stablecoin sub-index (0-100)
        defi_liquidity_risk: DeFi liquidity sub-index (0-100)
        contagion_risk: Contagion sub-index (0-100)
        arbitrage_opacity: Regulatory arbitrage sub-index (0-100)
        historical_min: Historical minimum ASRI for normalization
        historical_max: Historical maximum ASRI for normalization

    Returns:
        ASRIResult with all computed values
    """
    sub_indices = SubIndexValues(
        stablecoin_risk=stablecoin_risk,
        defi_liquidity_risk=defi_liquidity_risk,
        contagion_risk=contagion_risk,
        arbitrage_opacity=arbitrage_opacity,
    )

    asri = calculate_aggregate_asri(sub_indices)
    asri_normalized = normalize_asri(asri, historical_min, historical_max)
    alert_level = determine_alert_level(asri_normalized)

    return ASRIResult(
        timestamp=datetime.utcnow(),
        asri=asri,
        asri_normalized=asri_normalized,
        sub_indices=sub_indices,
        alert_level=alert_level,
    )
