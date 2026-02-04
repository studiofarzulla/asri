"""
Algorithmic Stablecoin Risk Calculator

Addresses Terra/Luna blind spot by incorporating risk factors specific to
algorithmic/crypto-backed stablecoins that fiat-collateralized stablecoins
(USDT, USDC) don't exhibit.

Key insight: UST only depegged AFTER Luna crashed. Peg volatility is a
lagging indicator for algo stablecoins. Instead, we need to track:
- Backing ratio (reserves / circulating supply)
- Backing token volatility (Luna price volatility)
- Supply dilution rate (mint rate relative to backing token)
- Reflexivity risk (correlation between stablecoin flows and backing token)

Reference: Lyons & Viswanath-Natraj (2023) on algorithmic stablecoin death spirals.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class StablecoinType(Enum):
    """Classification of stablecoin backing mechanism."""
    FIAT_COLLATERAL = "fiat-collateral"  # USDT, USDC - backed by USD/T-bills
    CRYPTO_COLLATERAL = "crypto-collateral"  # DAI - overcollateralized by ETH/WBTC
    ALGORITHMIC = "algorithmic"  # UST, FRAX (partial) - backed by native token
    HYBRID = "hybrid"  # Mix of mechanisms (e.g., FRAX)
    UNKNOWN = "unknown"


@dataclass
class AlgorithmicStablecoinMetrics:
    """Risk metrics for a single algorithmic stablecoin."""
    name: str
    symbol: str
    circulating: float
    peg_type: StablecoinType

    # Backing metrics (not all always available)
    backing_ratio: float | None  # reserves / circulating (< 1 = underbacked)
    backing_token_symbol: str | None  # e.g., "LUNA" for UST
    backing_token_volatility_30d: float | None  # annualized vol of backing token
    backing_token_supply_growth_30d: float | None  # % change in backing token supply

    # Derived risk metrics
    backing_ratio_risk: float = 0.0  # 0-100 scale
    collateral_volatility_risk: float = 0.0
    dilution_risk: float = 0.0
    concentration_risk: float = 0.0


@dataclass
class AlgorithmicStablecoinRiskResult:
    """Aggregate algorithmic stablecoin risk output."""
    algo_stablecoin_risk: float  # 0-100 scale
    algo_stablecoin_weight: float  # share of total stablecoin supply
    component_risks: dict[str, float]
    stablecoin_details: list[AlgorithmicStablecoinMetrics]


# Known algorithmic stablecoins and their backing mechanisms
# This is a static mapping; in production would come from DeFi Llama pegType field
KNOWN_ALGORITHMIC_STABLECOINS = {
    # Symbol -> (backing_token, peg_type)
    "UST": ("LUNA", StablecoinType.ALGORITHMIC),
    "USTC": ("LUNC", StablecoinType.ALGORITHMIC),  # Post-crash Terra Classic
    "FRAX": ("FXS", StablecoinType.HYBRID),  # Partially algorithmic
    "FEI": ("TRIBE", StablecoinType.ALGORITHMIC),  # Deprecated
    "BEAN": (None, StablecoinType.ALGORITHMIC),  # Beanstalk
    "FLOAT": ("BANK", StablecoinType.ALGORITHMIC),
    "RAI": ("ETH", StablecoinType.CRYPTO_COLLATERAL),  # Reflex index, not truly algo
    "LUSD": ("ETH", StablecoinType.CRYPTO_COLLATERAL),  # Liquity
    "GHO": ("AAVE", StablecoinType.CRYPTO_COLLATERAL),  # Aave
    "crvUSD": ("CRV", StablecoinType.CRYPTO_COLLATERAL),  # Curve
}

FIAT_COLLATERALIZED = {"USDT", "USDC", "BUSD", "TUSD", "USDP", "GUSD", "PYUSD"}


def classify_stablecoin(symbol: str, peg_type_hint: str | None = None) -> StablecoinType:
    """
    Classify stablecoin by backing mechanism.

    Args:
        symbol: Stablecoin ticker
        peg_type_hint: Optional hint from API (e.g., DeFi Llama pegType field)

    Returns:
        StablecoinType classification
    """
    symbol_upper = symbol.upper()

    # Check known classifications
    if symbol_upper in FIAT_COLLATERALIZED:
        return StablecoinType.FIAT_COLLATERAL

    if symbol_upper in KNOWN_ALGORITHMIC_STABLECOINS:
        return KNOWN_ALGORITHMIC_STABLECOINS[symbol_upper][1]

    # Use API hint if available
    if peg_type_hint:
        hint_lower = peg_type_hint.lower()
        if "fiat" in hint_lower or "usd" in hint_lower:
            return StablecoinType.FIAT_COLLATERAL
        if "algo" in hint_lower:
            return StablecoinType.ALGORITHMIC
        if "crypto" in hint_lower:
            return StablecoinType.CRYPTO_COLLATERAL
        if "hybrid" in hint_lower:
            return StablecoinType.HYBRID

    return StablecoinType.UNKNOWN


def get_backing_token(symbol: str) -> str | None:
    """Get the backing token for a known algorithmic stablecoin."""
    symbol_upper = symbol.upper()
    if symbol_upper in KNOWN_ALGORITHMIC_STABLECOINS:
        return KNOWN_ALGORITHMIC_STABLECOINS[symbol_upper][0]
    return None


def calculate_backing_ratio_risk(backing_ratio: float | None) -> float:
    """
    Calculate risk from backing ratio.

    Risk mapping:
    - backing_ratio >= 1.5: Low risk (0-20)
    - backing_ratio 1.0-1.5: Moderate risk (20-50)
    - backing_ratio 0.8-1.0: Elevated risk (50-80)
    - backing_ratio < 0.8: Critical risk (80-100)

    For algo stablecoins without explicit backing ratio, assume moderate risk.
    """
    if backing_ratio is None:
        return 50.0  # Unknown = moderate risk assumption

    if backing_ratio >= 1.5:
        # Well overcollateralized
        return max(0, 20 - (backing_ratio - 1.5) * 20)
    elif backing_ratio >= 1.0:
        # Adequately collateralized
        return 20 + (1.5 - backing_ratio) * 60  # 20-50
    elif backing_ratio >= 0.8:
        # Undercollateralized but not critical
        return 50 + (1.0 - backing_ratio) * 150  # 50-80
    else:
        # Critically undercollateralized
        return min(100, 80 + (0.8 - backing_ratio) * 100)


def calculate_collateral_volatility_risk(volatility_30d: float | None) -> float:
    """
    Calculate risk from backing token volatility.

    High volatility in backing token = high reflexivity risk.

    Mapping:
    - Vol < 30% (annualized): Low risk (0-30)
    - Vol 30-60%: Moderate risk (30-60)
    - Vol 60-100%: Elevated risk (60-85)
    - Vol > 100%: Critical risk (85-100)

    For reference: ETH vol ~60-80%, Luna pre-crash was ~100%+
    """
    if volatility_30d is None:
        return 50.0  # Unknown = moderate

    vol_pct = volatility_30d * 100  # Assume input is decimal (0.6 = 60%)

    if vol_pct < 30:
        return vol_pct  # Linear 0-30
    elif vol_pct < 60:
        return 30 + (vol_pct - 30)  # 30-60
    elif vol_pct < 100:
        return 60 + (vol_pct - 60) * 0.625  # 60-85
    else:
        return min(100, 85 + (vol_pct - 100) * 0.15)


def calculate_dilution_risk(supply_growth_30d: float | None) -> float:
    """
    Calculate risk from backing token supply dilution.

    Rapid minting of backing token (to defend peg) is a crisis precursor.

    Mapping:
    - Growth < 5% monthly: Low risk (0-20)
    - Growth 5-20%: Moderate risk (20-50)
    - Growth 20-50%: Elevated risk (50-80)
    - Growth > 50%: Critical risk (80-100)

    Note: Luna supply grew ~50,000% during the death spiral.
    """
    if supply_growth_30d is None:
        return 30.0  # Unknown = low-moderate

    growth_pct = supply_growth_30d * 100

    if growth_pct < 5:
        return growth_pct * 4  # 0-20
    elif growth_pct < 20:
        return 20 + (growth_pct - 5) * 2  # 20-50
    elif growth_pct < 50:
        return 50 + (growth_pct - 20)  # 50-80
    else:
        return min(100, 80 + (growth_pct - 50) * 0.4)


def calculate_concentration_risk(
    algo_supply: float,
    total_stablecoin_supply: float,
) -> float:
    """
    Calculate risk from algorithmic stablecoin concentration.

    Higher share of algo stablecoins in total supply = systemic risk.

    Mapping:
    - Share < 5%: Low risk (0-20)
    - Share 5-15%: Moderate risk (20-50)
    - Share 15-30%: Elevated risk (50-80)
    - Share > 30%: Critical risk (80-100)

    Note: At peak, UST was ~10% of total stablecoin supply.
    """
    if total_stablecoin_supply == 0:
        return 0.0

    share = algo_supply / total_stablecoin_supply * 100

    if share < 5:
        return share * 4  # 0-20
    elif share < 15:
        return 20 + (share - 5) * 3  # 20-50
    elif share < 30:
        return 50 + (share - 15) * 2  # 50-80
    else:
        return min(100, 80 + (share - 30) * 0.67)


def calculate_algorithmic_stablecoin_risk(
    stablecoins: list[Any],
    backing_token_data: dict[str, dict] | None = None,
) -> AlgorithmicStablecoinRiskResult:
    """
    Calculate aggregate risk from algorithmic stablecoins.

    Formula:
    AlgoRisk = 0.35 × BackingRatioRisk + 0.30 × CollateralVolRisk +
               0.20 × DilutionRisk + 0.15 × ConcentrationRisk

    Args:
        stablecoins: List of stablecoin data objects with at minimum:
            - symbol: str
            - circulating: float
            - Optional: peg_type, backing_ratio, etc.
        backing_token_data: Optional dict mapping backing token symbols to:
            - volatility_30d: float (annualized)
            - supply_growth_30d: float (decimal, e.g., 0.05 = 5%)

    Returns:
        AlgorithmicStablecoinRiskResult with aggregate risk and components
    """
    if backing_token_data is None:
        backing_token_data = {}

    algo_stablecoins: list[AlgorithmicStablecoinMetrics] = []
    total_supply = 0.0
    algo_supply = 0.0

    for coin in stablecoins:
        symbol = getattr(coin, 'symbol', coin.get('symbol', '')) if hasattr(coin, '__getitem__') else coin.symbol
        circulating = getattr(coin, 'circulating', coin.get('circulating', 0)) if hasattr(coin, '__getitem__') else coin.circulating

        total_supply += circulating

        # Classify stablecoin
        peg_hint = getattr(coin, 'peg_type', None) if hasattr(coin, 'peg_type') else None
        peg_type = classify_stablecoin(symbol, peg_hint)

        # Skip fiat-collateralized stablecoins for algo risk calculation
        if peg_type == StablecoinType.FIAT_COLLATERAL:
            continue

        algo_supply += circulating

        # Get backing token data if available
        backing_token = get_backing_token(symbol)
        backing_data = backing_token_data.get(backing_token, {}) if backing_token else {}

        # Extract metrics
        backing_ratio = getattr(coin, 'backing_ratio', None) if hasattr(coin, 'backing_ratio') else None
        volatility = backing_data.get('volatility_30d')
        supply_growth = backing_data.get('supply_growth_30d')

        # Calculate component risks
        metrics = AlgorithmicStablecoinMetrics(
            name=getattr(coin, 'name', symbol),
            symbol=symbol,
            circulating=circulating,
            peg_type=peg_type,
            backing_ratio=backing_ratio,
            backing_token_symbol=backing_token,
            backing_token_volatility_30d=volatility,
            backing_token_supply_growth_30d=supply_growth,
            backing_ratio_risk=calculate_backing_ratio_risk(backing_ratio),
            collateral_volatility_risk=calculate_collateral_volatility_risk(volatility),
            dilution_risk=calculate_dilution_risk(supply_growth),
        )

        algo_stablecoins.append(metrics)

    # Calculate concentration risk (system-wide)
    concentration_risk = calculate_concentration_risk(algo_supply, total_supply)

    # Calculate supply-weighted aggregate risk
    if algo_supply > 0 and algo_stablecoins:
        weighted_backing_risk = sum(
            m.backing_ratio_risk * m.circulating for m in algo_stablecoins
        ) / algo_supply

        weighted_vol_risk = sum(
            m.collateral_volatility_risk * m.circulating for m in algo_stablecoins
        ) / algo_supply

        weighted_dilution_risk = sum(
            m.dilution_risk * m.circulating for m in algo_stablecoins
        ) / algo_supply
    else:
        weighted_backing_risk = 0.0
        weighted_vol_risk = 0.0
        weighted_dilution_risk = 0.0

    # Aggregate formula
    aggregate_risk = (
        0.35 * weighted_backing_risk +
        0.30 * weighted_vol_risk +
        0.20 * weighted_dilution_risk +
        0.15 * concentration_risk
    )

    algo_weight = algo_supply / total_supply if total_supply > 0 else 0.0

    return AlgorithmicStablecoinRiskResult(
        algo_stablecoin_risk=min(100, max(0, aggregate_risk)),
        algo_stablecoin_weight=algo_weight,
        component_risks={
            "backing_ratio_risk": weighted_backing_risk,
            "collateral_volatility_risk": weighted_vol_risk,
            "dilution_risk": weighted_dilution_risk,
            "concentration_risk": concentration_risk,
        },
        stablecoin_details=algo_stablecoins,
    )


def adjust_scr_for_algorithmic_risk(
    base_scr: float,
    algo_risk_result: AlgorithmicStablecoinRiskResult,
) -> float:
    """
    Adjust Stablecoin Concentration Risk (SCR) to incorporate algorithmic risk.

    New SCR = (1 - algo_weight) × base_SCR + algo_weight × algo_SCR

    where algo_SCR is a blend of the base formula and algorithmic-specific risk:
    algo_SCR = 0.6 × base_component_SCR + 0.4 × algo_risk

    This ensures that when algo stablecoins have high market share,
    their specific risk factors contribute proportionally.
    """
    algo_weight = algo_risk_result.algo_stablecoin_weight

    if algo_weight < 0.01:  # Less than 1% is negligible
        return base_scr

    # Blend base SCR with algo-specific risk
    algo_scr_component = 0.6 * base_scr + 0.4 * algo_risk_result.algo_stablecoin_risk

    # Weight by market share
    adjusted_scr = (1 - algo_weight) * base_scr + algo_weight * algo_scr_component

    return min(100, max(0, adjusted_scr))


def format_algo_risk_table_latex(result: AlgorithmicStablecoinRiskResult) -> str:
    """Format algorithmic stablecoin risk results as LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Algorithmic Stablecoin Risk Components}",
        r"\label{tab:algo_stablecoin_risk}",
        r"\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Component & Risk Score (0--100) \\",
        r"\midrule",
        f"Backing Ratio Risk & {result.component_risks['backing_ratio_risk']:.1f} \\\\",
        f"Collateral Volatility Risk & {result.component_risks['collateral_volatility_risk']:.1f} \\\\",
        f"Dilution Risk & {result.component_risks['dilution_risk']:.1f} \\\\",
        f"Concentration Risk & {result.component_risks['concentration_risk']:.1f} \\\\",
        r"\midrule",
        f"\\textbf{{Aggregate Algo Risk}} & \\textbf{{{result.algo_stablecoin_risk:.1f}}} \\\\",
        f"Algo Stablecoin Market Share & {result.algo_stablecoin_weight * 100:.1f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Aggregate = 0.35$\times$Backing + 0.30$\times$Volatility + 0.20$\times$Dilution + 0.15$\times$Concentration.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    return "\n".join(lines)
