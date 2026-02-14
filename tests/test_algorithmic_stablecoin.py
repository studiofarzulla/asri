"""Tests for algorithmic stablecoin risk calculator."""

from types import SimpleNamespace

import pytest

from asri.signals.algorithmic_stablecoin import (
    AlgorithmicStablecoinRiskResult,
    StablecoinType,
    adjust_scr_for_algorithmic_risk,
    calculate_algorithmic_stablecoin_risk,
    calculate_backing_ratio_risk,
    calculate_collateral_volatility_risk,
    calculate_concentration_risk,
    calculate_dilution_risk,
    classify_stablecoin,
    get_backing_token,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coin(symbol: str, circulating: float, peg_deviation: float = 0.001):
    return SimpleNamespace(
        symbol=symbol,
        name=symbol,
        circulating=circulating,
        peg_deviation=peg_deviation,
    )


# ===========================================================================
# classify_stablecoin
# ===========================================================================


class TestClassifyStablecoin:

    def test_fiat_collateral_usdt(self):
        assert classify_stablecoin("USDT") == StablecoinType.FIAT_COLLATERAL

    def test_fiat_collateral_usdc(self):
        assert classify_stablecoin("USDC") == StablecoinType.FIAT_COLLATERAL

    def test_fiat_collateral_busd(self):
        assert classify_stablecoin("BUSD") == StablecoinType.FIAT_COLLATERAL

    def test_crypto_collateral_dai(self):
        """DAI is not in FIAT_COLLATERALIZED; it's not in KNOWN_ALGORITHMIC either → UNKNOWN
        unless classified by the mapping. Check actual mapping."""
        # DAI is not in either lookup — should be UNKNOWN unless hinted
        result = classify_stablecoin("DAI")
        assert result == StablecoinType.UNKNOWN

    def test_dai_with_crypto_hint(self):
        assert classify_stablecoin("DAI", peg_type_hint="crypto-collateral") == StablecoinType.CRYPTO_COLLATERAL

    def test_algorithmic_ust(self):
        assert classify_stablecoin("UST") == StablecoinType.ALGORITHMIC

    def test_hybrid_frax(self):
        assert classify_stablecoin("FRAX") == StablecoinType.HYBRID

    def test_unknown_stablecoin(self):
        assert classify_stablecoin("XYZSTABLE") == StablecoinType.UNKNOWN

    def test_case_insensitive(self):
        assert classify_stablecoin("usdt") == StablecoinType.FIAT_COLLATERAL
        assert classify_stablecoin("Ust") == StablecoinType.ALGORITHMIC

    def test_hint_overrides_unknown(self):
        """For an unknown symbol, API hint should determine type."""
        assert classify_stablecoin("NEWSTABLE", "algo-backed") == StablecoinType.ALGORITHMIC
        assert classify_stablecoin("NEWSTABLE", "fiat-usd") == StablecoinType.FIAT_COLLATERAL

    def test_known_crypto_collateral(self):
        """RAI, LUSD, GHO are crypto-collateral in the lookup."""
        assert classify_stablecoin("RAI") == StablecoinType.CRYPTO_COLLATERAL
        assert classify_stablecoin("LUSD") == StablecoinType.CRYPTO_COLLATERAL
        assert classify_stablecoin("GHO") == StablecoinType.CRYPTO_COLLATERAL


# ===========================================================================
# get_backing_token
# ===========================================================================


class TestGetBackingToken:

    def test_ust_luna(self):
        assert get_backing_token("UST") == "LUNA"

    def test_frax_fxs(self):
        assert get_backing_token("FRAX") == "FXS"

    def test_unknown_returns_none(self):
        assert get_backing_token("USDT") is None

    def test_case_insensitive(self):
        assert get_backing_token("ust") == "LUNA"

    def test_bean_no_backing(self):
        """BEAN has None backing token in mapping."""
        assert get_backing_token("BEAN") is None


# ===========================================================================
# calculate_backing_ratio_risk
# ===========================================================================


class TestBackingRatioRisk:

    def test_none_returns_moderate(self):
        assert calculate_backing_ratio_risk(None) == 50.0

    def test_well_overcollateralized(self):
        """Ratio 2.0 → low risk."""
        risk = calculate_backing_ratio_risk(2.0)
        assert 0 <= risk <= 20

    def test_adequate_collateral(self):
        """Ratio 1.25 → moderate range (20-50)."""
        risk = calculate_backing_ratio_risk(1.25)
        assert 20 <= risk <= 50

    def test_at_parity(self):
        """Ratio 1.0 → boundary between adequate and elevated."""
        risk = calculate_backing_ratio_risk(1.0)
        assert risk == pytest.approx(50.0)

    def test_undercollateralized(self):
        """Ratio 0.9 → elevated (50-80)."""
        risk = calculate_backing_ratio_risk(0.9)
        assert 50 <= risk <= 80

    def test_critically_undercollateralized(self):
        """Ratio 0.5 → critical (80-100)."""
        risk = calculate_backing_ratio_risk(0.5)
        assert 80 <= risk <= 100

    def test_zero_ratio(self):
        """Ratio 0.0 → should be 100 or near it."""
        risk = calculate_backing_ratio_risk(0.0)
        assert risk >= 95

    def test_result_bounded(self):
        """Even extreme ratios should stay in 0-100."""
        assert 0 <= calculate_backing_ratio_risk(100.0) <= 100
        assert 0 <= calculate_backing_ratio_risk(0.0) <= 100

    def test_monotonically_decreasing_risk(self):
        """Higher backing ratio → lower risk."""
        prev = 101.0
        for ratio in [0.0, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]:
            risk = calculate_backing_ratio_risk(ratio)
            assert risk <= prev, f"Non-monotonic at ratio={ratio}"
            prev = risk


# ===========================================================================
# calculate_collateral_volatility_risk
# ===========================================================================


class TestCollateralVolatilityRisk:

    def test_none_returns_moderate(self):
        assert calculate_collateral_volatility_risk(None) == 50.0

    def test_low_volatility(self):
        """10% vol → low risk (0-30 range)."""
        risk = calculate_collateral_volatility_risk(0.10)
        assert 0 <= risk <= 30

    def test_moderate_volatility(self):
        """45% vol → moderate (30-60)."""
        risk = calculate_collateral_volatility_risk(0.45)
        assert 30 <= risk <= 60

    def test_high_volatility(self):
        """80% vol → elevated (60-85)."""
        risk = calculate_collateral_volatility_risk(0.80)
        assert 60 <= risk <= 85

    def test_extreme_volatility(self):
        """120% vol → critical (85-100)."""
        risk = calculate_collateral_volatility_risk(1.20)
        assert 85 <= risk <= 100

    def test_zero_volatility(self):
        assert calculate_collateral_volatility_risk(0.0) == pytest.approx(0.0)

    def test_result_bounded(self):
        assert 0 <= calculate_collateral_volatility_risk(5.0) <= 100


# ===========================================================================
# calculate_dilution_risk
# ===========================================================================


class TestDilutionRisk:

    def test_none_returns_low_moderate(self):
        assert calculate_dilution_risk(None) == 30.0

    def test_low_growth(self):
        """2% monthly growth → low risk."""
        risk = calculate_dilution_risk(0.02)
        assert 0 <= risk <= 20

    def test_moderate_growth(self):
        """10% monthly → moderate (20-50)."""
        risk = calculate_dilution_risk(0.10)
        assert 20 <= risk <= 50

    def test_elevated_growth(self):
        """35% → elevated (50-80)."""
        risk = calculate_dilution_risk(0.35)
        assert 50 <= risk <= 80

    def test_extreme_growth(self):
        """100% → critical (80-100)."""
        risk = calculate_dilution_risk(1.0)
        assert 80 <= risk <= 100

    def test_zero_growth(self):
        assert calculate_dilution_risk(0.0) == pytest.approx(0.0)

    def test_result_bounded(self):
        assert 0 <= calculate_dilution_risk(500.0) <= 100


# ===========================================================================
# calculate_concentration_risk
# ===========================================================================


class TestConcentrationRisk:

    def test_zero_total_supply(self):
        assert calculate_concentration_risk(100.0, 0.0) == 0.0

    def test_no_algo_supply(self):
        assert calculate_concentration_risk(0.0, 100e9) == pytest.approx(0.0)

    def test_low_share(self):
        """2% share → low risk (0-20)."""
        risk = calculate_concentration_risk(2e9, 100e9)
        assert 0 <= risk <= 20

    def test_moderate_share(self):
        """10% share → moderate (20-50)."""
        risk = calculate_concentration_risk(10e9, 100e9)
        assert 20 <= risk <= 50

    def test_elevated_share(self):
        """20% share → elevated (50-80)."""
        risk = calculate_concentration_risk(20e9, 100e9)
        assert 50 <= risk <= 80

    def test_critical_share(self):
        """40% share → critical (80-100)."""
        risk = calculate_concentration_risk(40e9, 100e9)
        assert 80 <= risk <= 100

    def test_result_bounded(self):
        assert 0 <= calculate_concentration_risk(100e9, 100e9) <= 100


# ===========================================================================
# calculate_algorithmic_stablecoin_risk (integration)
# ===========================================================================


class TestCalculateAlgorithmicStablecoinRisk:

    def test_all_fiat_returns_zero(self):
        """If only fiat stablecoins, aggregate algo risk should be ~0."""
        coins = [_coin("USDT", 80e9), _coin("USDC", 30e9)]
        result = calculate_algorithmic_stablecoin_risk(coins)
        assert isinstance(result, AlgorithmicStablecoinRiskResult)
        assert result.algo_stablecoin_risk == pytest.approx(0.0)
        assert result.algo_stablecoin_weight == pytest.approx(0.0)

    def test_with_algo_stablecoin(self):
        """Including UST should produce non-zero risk."""
        coins = [
            _coin("USDT", 80e9),
            _coin("USDC", 30e9),
            _coin("UST", 18e9),
        ]
        result = calculate_algorithmic_stablecoin_risk(coins)
        assert result.algo_stablecoin_risk > 0
        assert result.algo_stablecoin_weight > 0

    def test_with_backing_token_data(self):
        """Providing volatile backing data should increase risk."""
        coins = [_coin("USDT", 80e9), _coin("UST", 18e9)]
        calm = calculate_algorithmic_stablecoin_risk(coins)
        volatile = calculate_algorithmic_stablecoin_risk(
            coins,
            backing_token_data={"LUNA": {"volatility_30d": 1.5, "supply_growth_30d": 0.5}},
        )
        assert volatile.algo_stablecoin_risk >= calm.algo_stablecoin_risk

    def test_result_components_present(self):
        coins = [_coin("USDT", 80e9), _coin("UST", 18e9)]
        result = calculate_algorithmic_stablecoin_risk(coins)
        for key in ("backing_ratio_risk", "collateral_volatility_risk",
                     "dilution_risk", "concentration_risk"):
            assert key in result.component_risks

    def test_aggregate_bounded(self):
        coins = [_coin("UST", 100e9)]
        result = calculate_algorithmic_stablecoin_risk(
            coins,
            backing_token_data={"LUNA": {"volatility_30d": 2.0, "supply_growth_30d": 5.0}},
        )
        assert 0 <= result.algo_stablecoin_risk <= 100

    def test_empty_stablecoins(self):
        result = calculate_algorithmic_stablecoin_risk([])
        assert result.algo_stablecoin_risk == pytest.approx(0.0)
        assert result.algo_stablecoin_weight == pytest.approx(0.0)

    def test_multiple_algo_coins(self):
        """Multiple algo/hybrid coins should all contribute."""
        coins = [
            _coin("USDT", 80e9),
            _coin("UST", 18e9),
            _coin("FRAX", 5e9),
            _coin("FEI", 1e9),
        ]
        result = calculate_algorithmic_stablecoin_risk(coins)
        assert len(result.stablecoin_details) == 3  # UST + FRAX + FEI (USDT excluded)


# ===========================================================================
# adjust_scr_for_algorithmic_risk
# ===========================================================================


class TestAdjustSCR:

    def _make_result(self, risk: float, weight: float) -> AlgorithmicStablecoinRiskResult:
        return AlgorithmicStablecoinRiskResult(
            algo_stablecoin_risk=risk,
            algo_stablecoin_weight=weight,
            component_risks={},
            stablecoin_details=[],
        )

    def test_negligible_weight_returns_base(self):
        """Weight < 1% → return base_scr unchanged."""
        result = self._make_result(80.0, 0.005)
        assert adjust_scr_for_algorithmic_risk(50.0, result) == 50.0

    def test_high_weight_blends(self):
        """Significant weight should blend algo risk into SCR."""
        result = self._make_result(90.0, 0.20)
        adjusted = adjust_scr_for_algorithmic_risk(50.0, result)
        # Should be higher than base due to high algo risk
        assert adjusted > 50.0

    def test_low_algo_risk_lowers_scr(self):
        """If algo risk is lower than base, adjusted SCR decreases."""
        result = self._make_result(10.0, 0.20)
        adjusted = adjust_scr_for_algorithmic_risk(50.0, result)
        assert adjusted < 50.0

    def test_result_bounded(self):
        """Output should always be in 0-100."""
        result = self._make_result(100.0, 1.0)
        assert 0 <= adjust_scr_for_algorithmic_risk(100.0, result) <= 100

        result_zero = self._make_result(0.0, 1.0)
        assert 0 <= adjust_scr_for_algorithmic_risk(0.0, result_zero) <= 100

    def test_zero_base_scr(self):
        result = self._make_result(80.0, 0.50)
        adjusted = adjust_scr_for_algorithmic_risk(0.0, result)
        assert adjusted > 0  # algo risk should push it up
        assert adjusted <= 100

    def test_full_algo_weight(self):
        """Weight = 1.0 → SCR = 0.6*base + 0.4*algo_risk."""
        result = self._make_result(100.0, 1.0)
        adjusted = adjust_scr_for_algorithmic_risk(50.0, result)
        expected = 0.6 * 50.0 + 0.4 * 100.0  # = 70
        assert adjusted == pytest.approx(expected)
