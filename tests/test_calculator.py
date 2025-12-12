"""Tests for ASRI signal calculator."""

import pytest

from asri.signals.calculator import (
    WEIGHTS,
    SubIndexValues,
    calculate_aggregate_asri,
    calculate_stablecoin_risk,
    compute_asri,
    determine_alert_level,
    normalize_asri,
)


class TestSubIndexCalculations:
    """Tests for individual sub-index calculations."""

    def test_stablecoin_risk_basic(self):
        """Test basic stablecoin risk calculation."""
        result = calculate_stablecoin_risk(
            tvl_current=100_000_000_000,
            tvl_max_historical=150_000_000_000,
            treasury_weight=50_000_000_000,
            total_reserves=100_000_000_000,
            reserve_concentration_hhi=0.5,
            peg_volatility_30d=0.02,
        )
        assert 0 <= result <= 100
        # With these inputs, expect moderate risk
        assert 30 < result < 70

    def test_stablecoin_risk_zero_division(self):
        """Test stablecoin risk handles zero denominators."""
        result = calculate_stablecoin_risk(
            tvl_current=100,
            tvl_max_historical=0,  # Zero max
            treasury_weight=50,
            total_reserves=0,  # Zero reserves
            reserve_concentration_hhi=0.5,
            peg_volatility_30d=0.02,
        )
        assert 0 <= result <= 100


class TestAggregateASRI:
    """Tests for aggregate ASRI calculation."""

    def test_weights_sum_to_one(self):
        """Verify weights sum to 1.0."""
        assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001

    def test_aggregate_calculation(self):
        """Test aggregate ASRI from sub-indices."""
        sub_indices = SubIndexValues(
            stablecoin_risk=50,
            defi_liquidity_risk=50,
            contagion_risk=50,
            arbitrage_opacity=50,
        )
        result = calculate_aggregate_asri(sub_indices)
        # All at 50 should give 50
        assert abs(result - 50) < 0.001

    def test_aggregate_weighted(self):
        """Test that weights are applied correctly."""
        sub_indices = SubIndexValues(
            stablecoin_risk=100,  # weight 0.30
            defi_liquidity_risk=0,  # weight 0.25
            contagion_risk=0,  # weight 0.25
            arbitrage_opacity=0,  # weight 0.20
        )
        result = calculate_aggregate_asri(sub_indices)
        assert abs(result - 30) < 0.001


class TestNormalization:
    """Tests for ASRI normalization."""

    def test_normalize_midpoint(self):
        """Test normalization at midpoint."""
        result = normalize_asri(50, historical_min=0, historical_max=100)
        assert abs(result - 50) < 0.001

    def test_normalize_extremes(self):
        """Test normalization at extremes."""
        assert normalize_asri(0, 0, 100) == 0
        assert normalize_asri(100, 0, 100) == 100

    def test_normalize_equal_range(self):
        """Test normalization when min equals max."""
        result = normalize_asri(50, 50, 50)
        assert result == 50.0  # Should return midpoint


class TestAlertLevels:
    """Tests for alert level determination."""

    def test_alert_low(self):
        assert determine_alert_level(20) == "low"

    def test_alert_moderate(self):
        assert determine_alert_level(55) == "moderate"

    def test_alert_elevated(self):
        assert determine_alert_level(75) == "elevated"

    def test_alert_high(self):
        assert determine_alert_level(90) == "high"

    def test_alert_critical(self):
        assert determine_alert_level(98) == "critical"


class TestComputeASRI:
    """Tests for complete ASRI computation."""

    def test_compute_returns_result(self):
        """Test that compute_asri returns valid ASRIResult."""
        result = compute_asri(
            stablecoin_risk=60,
            defi_liquidity_risk=55,
            contagion_risk=70,
            arbitrage_opacity=45,
        )
        assert result.asri > 0
        assert result.asri_normalized >= 0
        assert result.alert_level in ["low", "moderate", "elevated", "high", "critical"]
        assert result.timestamp is not None

    def test_compute_all_zero(self):
        """Test ASRI with all zero inputs."""
        result = compute_asri(
            stablecoin_risk=0,
            defi_liquidity_risk=0,
            contagion_risk=0,
            arbitrage_opacity=0,
        )
        assert result.asri == 0
        assert result.alert_level == "low"

    def test_compute_all_max(self):
        """Test ASRI with all max inputs."""
        result = compute_asri(
            stablecoin_risk=100,
            defi_liquidity_risk=100,
            contagion_risk=100,
            arbitrage_opacity=100,
        )
        assert result.asri == 100
        assert result.alert_level == "critical"
