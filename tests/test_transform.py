"""Tests for ASRI data transform layer."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from asri.pipeline.transform import (
    DataTransformer,
    StablecoinRiskInputs,
    DeFiLiquidityRiskInputs,
    ContagionRiskInputs,
    ArbitrageOpacityRiskInputs,
    TransformedData,
    calculate_hhi,
    normalize_to_100,
    normalize_hhi_to_risk,
    transform_all_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coin(symbol: str, circulating: float, peg_deviation: float = 0.001):
    """Create a lightweight stablecoin-like object for testing."""
    return SimpleNamespace(
        symbol=symbol,
        name=symbol,
        circulating=circulating,
        peg_deviation=peg_deviation,
    )


def _protocol(tvl: float, category: str = "DEX", audits: int = 1, change_1d: float = 0.0):
    return {"tvl": tvl, "category": category, "audits": audits, "change_1d": change_1d}


# ===========================================================================
# calculate_hhi
# ===========================================================================


class TestCalculateHHI:
    """Tests for the Herfindahl-Hirschman Index helper."""

    def test_equal_shares(self):
        """Equal shares among n firms → HHI = 10000/n."""
        n = 4
        values = [100.0] * n
        assert calculate_hhi(values) == pytest.approx(10000 / n)

    def test_single_entity(self):
        """Single entity → HHI = 10000 (perfect monopoly)."""
        assert calculate_hhi([500.0]) == pytest.approx(10000.0)

    def test_empty_list(self):
        """Empty list → 0."""
        assert calculate_hhi([]) == 0.0

    def test_all_zeros(self):
        """All zeros → 0 (sum is zero guard)."""
        assert calculate_hhi([0.0, 0.0, 0.0]) == 0.0

    def test_typical_distribution(self):
        """Realistic distribution: one dominant, several small."""
        values = [70.0, 10.0, 10.0, 5.0, 5.0]
        hhi = calculate_hhi(values)
        # 70% share dominates: (0.7)^2*10000 = 4900 + small tails → ~5200
        assert 5000 < hhi < 5500

    def test_two_equal(self):
        """Two equal firms → HHI = 5000."""
        assert calculate_hhi([50.0, 50.0]) == pytest.approx(5000.0)

    def test_absolute_values_irrelevant(self):
        """Only ratios matter, not absolute values."""
        assert calculate_hhi([1, 1, 1]) == pytest.approx(calculate_hhi([1e9, 1e9, 1e9]))


# ===========================================================================
# normalize_to_100
# ===========================================================================


class TestNormalizeTo100:

    def test_basic_scaling(self):
        """Mid-range value maps to 50."""
        assert normalize_to_100(5.0, 0.0, 10.0) == pytest.approx(50.0)

    def test_at_min(self):
        assert normalize_to_100(0.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_at_max(self):
        assert normalize_to_100(10.0, 0.0, 10.0) == pytest.approx(100.0)

    def test_below_min_clamps(self):
        """Values below min clamp to 0."""
        assert normalize_to_100(-5.0, 0.0, 10.0) == 0.0

    def test_above_max_clamps(self):
        """Values above max clamp to 100."""
        assert normalize_to_100(20.0, 0.0, 10.0) == 100.0

    def test_min_equals_max(self):
        """When min == max, return 50 (midpoint default)."""
        assert normalize_to_100(5.0, 5.0, 5.0) == 50.0

    def test_negative_range(self):
        """Works with negative input ranges."""
        assert normalize_to_100(-5.0, -10.0, 0.0) == pytest.approx(50.0)


# ===========================================================================
# normalize_hhi_to_risk
# ===========================================================================


class TestNormalizeHHIToRisk:

    def test_zero_hhi(self):
        assert normalize_hhi_to_risk(0) == pytest.approx(0.0)

    def test_low_competitive(self):
        """HHI 750 → midpoint of low band (0-30)."""
        risk = normalize_hhi_to_risk(750)
        assert 0 < risk < 30

    def test_moderate_boundary(self):
        """HHI 1500 → exactly 30."""
        assert normalize_hhi_to_risk(1500) == pytest.approx(30.0)

    def test_moderate_band(self):
        """HHI 2000 → in 30-60 range."""
        risk = normalize_hhi_to_risk(2000)
        assert 30 < risk < 60

    def test_high_boundary(self):
        """HHI 2500 → exactly 60."""
        assert normalize_hhi_to_risk(2500) == pytest.approx(60.0)

    def test_high_band(self):
        risk = normalize_hhi_to_risk(3750)
        assert 60 < risk < 90

    def test_critical_boundary(self):
        """HHI 5000 → exactly 90."""
        assert normalize_hhi_to_risk(5000) == pytest.approx(90.0)

    def test_monopoly(self):
        """HHI 10000 → exactly 100."""
        assert normalize_hhi_to_risk(10000) == pytest.approx(100.0)

    def test_monotonically_increasing(self):
        """Risk must increase as HHI increases."""
        prev = -1.0
        for hhi in range(0, 10001, 500):
            risk = normalize_hhi_to_risk(hhi)
            assert risk >= prev, f"Non-monotonic at HHI={hhi}"
            prev = risk


# ===========================================================================
# DataTransformer methods
# ===========================================================================


class TestDataTransformerStablecoinRisk:

    def setup_method(self):
        self.transformer = DataTransformer()
        self.coins = [
            _coin("USDT", 80e9, 0.0005),
            _coin("USDC", 30e9, 0.0003),
            _coin("DAI", 5e9, 0.002),
        ]

    def test_returns_correct_type(self):
        result = self.transformer.transform_stablecoin_risk(
            stablecoins=self.coins,
            current_tvl=50e9,
            max_historical_tvl=100e9,
            treasury_10y_rate=4.0,
        )
        assert isinstance(result, StablecoinRiskInputs)

    def test_all_fields_bounded(self):
        result = self.transformer.transform_stablecoin_risk(
            stablecoins=self.coins,
            current_tvl=50e9,
            max_historical_tvl=100e9,
            treasury_10y_rate=4.0,
        )
        assert 0 <= result.tvl_ratio <= 100
        assert 0 <= result.treasury_stress <= 100
        assert 0 <= result.concentration_hhi <= 100
        assert 0 <= result.peg_volatility <= 100

    def test_high_tvl_ratio_lower_risk(self):
        """When current TVL equals max, tvl_risk should be low."""
        result = self.transformer.transform_stablecoin_risk(
            stablecoins=self.coins,
            current_tvl=100e9,
            max_historical_tvl=100e9,
            treasury_10y_rate=3.0,
        )
        assert result.tvl_ratio <= 10  # near-zero risk when at max TVL

    def test_zero_max_historical_tvl(self):
        """Zero max_historical_tvl should not raise."""
        result = self.transformer.transform_stablecoin_risk(
            stablecoins=self.coins,
            current_tvl=50e9,
            max_historical_tvl=0,
            treasury_10y_rate=4.0,
        )
        assert 0 <= result.tvl_ratio <= 100


class TestDataTransformerDeFiLiquidity:

    def setup_method(self):
        self.transformer = DataTransformer()
        self.protocols = [
            _protocol(10e9, "DEX", audits=1, change_1d=2.0),
            _protocol(5e9, "Lending", audits=1, change_1d=-1.0),
            _protocol(3e9, "DEX", audits=0, change_1d=5.0),
            _protocol(1e9, "Yield", audits=1, change_1d=0.5),
        ]

    def test_returns_correct_type(self):
        result = self.transformer.transform_defi_liquidity_risk(self.protocols)
        assert isinstance(result, DeFiLiquidityRiskInputs)

    def test_all_fields_bounded(self):
        result = self.transformer.transform_defi_liquidity_risk(self.protocols)
        for field in ("top10_concentration", "tvl_volatility", "smart_contract_risk",
                      "flash_loan_proxy", "leverage_change"):
            assert 0 <= getattr(result, field) <= 100, f"{field} out of range"

    def test_with_tvl_history(self):
        history = [50e9, 52e9, 48e9, 55e9, 47e9]
        result = self.transformer.transform_defi_liquidity_risk(self.protocols, tvl_history=history)
        assert 0 <= result.tvl_volatility <= 100

    def test_empty_protocols(self):
        """Empty protocol list should not raise."""
        result = self.transformer.transform_defi_liquidity_risk([])
        assert isinstance(result, DeFiLiquidityRiskInputs)

    def test_no_audits_high_risk(self):
        """Protocols with no audits → high smart_contract_risk."""
        unaudited = [_protocol(10e9, audits=0), _protocol(5e9, audits=0)]
        result = self.transformer.transform_defi_liquidity_risk(unaudited)
        assert result.smart_contract_risk == pytest.approx(100.0)


class TestDataTransformerContagionRisk:

    def setup_method(self):
        self.transformer = DataTransformer()
        self.protocols = [
            _protocol(10e9, "DEX"),
            _protocol(2e9, "RWA"),
            _protocol(1e9, "Lending"),
        ]
        self.bridges = [{"name": f"bridge_{i}"} for i in range(20)]

    def test_returns_correct_type(self):
        result = self.transformer.transform_contagion_risk(
            self.protocols, treasury_10y_rate=4.0, vix=20.0,
            yield_curve_spread=0.5, bridges=self.bridges,
        )
        assert isinstance(result, ContagionRiskInputs)

    def test_all_fields_bounded(self):
        result = self.transformer.transform_contagion_risk(
            self.protocols, treasury_10y_rate=4.0, vix=25.0,
            yield_curve_spread=-0.5, bridges=self.bridges,
        )
        for field in ("rwa_growth_rate", "bank_exposure", "tradfi_linkage",
                      "crypto_equity_correlation", "bridge_exploit_frequency"):
            val = getattr(result, field)
            assert 0 <= val <= 100, f"{field}={val} out of range"

    def test_inverted_yield_curve_higher_risk(self):
        """Negative spread → tradfi_linkage should be elevated."""
        normal = self.transformer.transform_contagion_risk(
            self.protocols, 4.0, 20.0, yield_curve_spread=1.0, bridges=self.bridges,
        )
        inverted = self.transformer.transform_contagion_risk(
            self.protocols, 4.0, 20.0, yield_curve_spread=-1.0, bridges=self.bridges,
        )
        assert inverted.tradfi_linkage > normal.tradfi_linkage


class TestDataTransformerArbitrageOpacity:

    def setup_method(self):
        self.transformer = DataTransformer()
        self.coins = [
            _coin("USDT", 80e9),
            _coin("USDC", 30e9),
            _coin("DAI", 5e9),
        ]
        self.protocols = [
            _protocol(10e9, audits=1),
            _protocol(5e9, audits=0),
        ]

    def test_returns_correct_type(self):
        result = self.transformer.transform_arbitrage_opacity_risk(self.coins, self.protocols)
        assert isinstance(result, ArbitrageOpacityRiskInputs)

    def test_all_fields_bounded(self):
        result = self.transformer.transform_arbitrage_opacity_risk(self.coins, self.protocols)
        for field in ("unregulated_exposure", "multi_issuer_risk", "custody_concentration",
                      "regulatory_sentiment", "transparency_score"):
            val = getattr(result, field)
            assert 0 <= val <= 100, f"{field}={val} out of range"


# ===========================================================================
# transform_all_data convenience function
# ===========================================================================


class TestTransformAllData:

    def test_returns_transformed_data(self):
        coins = [_coin("USDT", 80e9), _coin("USDC", 30e9)]
        protocols = [_protocol(10e9), _protocol(5e9, audits=0)]
        bridges = [{"name": "b1"}]

        result = transform_all_data(
            stablecoins=coins,
            protocols=protocols,
            bridges=bridges,
            current_tvl=50e9,
            max_historical_tvl=100e9,
            treasury_10y_rate=4.0,
            vix=20.0,
            yield_curve_spread=0.5,
        )
        assert isinstance(result, TransformedData)
        assert isinstance(result.stablecoin_risk, StablecoinRiskInputs)
        assert isinstance(result.defi_liquidity_risk, DeFiLiquidityRiskInputs)
        assert isinstance(result.contagion_risk, ContagionRiskInputs)
        assert isinstance(result.arbitrage_opacity_risk, ArbitrageOpacityRiskInputs)
        assert result.raw_metrics["num_stablecoins"] == 2

    def test_with_all_optional_params(self):
        coins = [_coin("USDT", 80e9), _coin("USDC", 30e9)]
        protocols = [_protocol(10e9)]
        bridges = []

        result = transform_all_data(
            stablecoins=coins,
            protocols=protocols,
            bridges=bridges,
            current_tvl=50e9,
            max_historical_tvl=100e9,
            treasury_10y_rate=4.0,
            vix=20.0,
            yield_curve_spread=0.5,
            tvl_history=[50e9, 52e9, 48e9],
            crypto_equity_corr=0.7,
            regulatory_sentiment=65.0,
            backing_token_data={"LUNA": {"volatility_30d": 1.2, "supply_growth_30d": 0.5}},
        )
        assert isinstance(result, TransformedData)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:

    def test_hhi_single_zero_among_values(self):
        """Zero values mixed in should be ignored by share calculation."""
        result = calculate_hhi([100.0, 0.0, 100.0])
        assert result == pytest.approx(5000.0)

    def test_normalize_very_large_value(self):
        assert normalize_to_100(1e12, 0, 100) == 100.0

    def test_normalize_very_small_value(self):
        assert normalize_to_100(-1e12, 0, 100) == 0.0

    def test_transformer_zero_tvl_protocols(self):
        """Protocols with zero TVL should be filtered out gracefully."""
        transformer = DataTransformer()
        protocols = [_protocol(0), _protocol(0), _protocol(10e9)]
        result = transformer.transform_defi_liquidity_risk(protocols)
        assert isinstance(result, DeFiLiquidityRiskInputs)

    def test_single_stablecoin(self):
        """Single stablecoin should work (monopoly concentration)."""
        transformer = DataTransformer()
        coins = [_coin("USDT", 80e9)]
        result = transformer.transform_stablecoin_risk(
            stablecoins=coins,
            current_tvl=50e9,
            max_historical_tvl=100e9,
            treasury_10y_rate=4.0,
        )
        # Single stablecoin → HHI=10000 → high concentration risk
        assert result.concentration_hhi > 80
