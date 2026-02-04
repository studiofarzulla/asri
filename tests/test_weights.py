"""
Tests for ASRI Weight Derivation Module

Validates PCA, Elastic Net, CRITIC, Entropy weights and comparison.
"""

import numpy as np
import pandas as pd
import pytest

from asri.weights.pca import (
    PCAWeightDeriver,
    derive_pca_weights,
)
from asri.weights.elastic_net import (
    ElasticNetWeightDeriver,
    derive_elastic_net_weights,
)
from asri.weights.critic import (
    CRITICWeightDeriver,
    derive_critic_weights,
)
from asri.weights.entropy import (
    EntropyWeightDeriver,
    derive_entropy_weights,
)
from asri.weights.comparison import (
    compare_weight_methods,
    weight_sensitivity_analysis,
)


class TestPCAWeights:
    """Tests for PCA weight derivation."""
    
    def test_basic_pca(self):
        """Basic PCA should work."""
        np.random.seed(42)
        # Create correlated data
        n = 200
        factor = np.random.randn(n)
        data = pd.DataFrame({
            'A': factor + 0.5 * np.random.randn(n),
            'B': factor + 0.5 * np.random.randn(n),
            'C': 0.5 * factor + 0.5 * np.random.randn(n),
            'D': 0.3 * factor + 0.5 * np.random.randn(n),
        })
        
        deriver = PCAWeightDeriver()
        deriver.fit(data)
        
        weights = deriver.weights
        
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.001
        assert all(w >= 0 for w in weights.values())
    
    def test_pca_variance_explained(self):
        """Highly correlated data should have high PC1 variance."""
        np.random.seed(42)
        n = 200
        # All series driven by same factor
        factor = np.random.randn(n)
        data = pd.DataFrame({
            'A': factor + 0.1 * np.random.randn(n),
            'B': factor + 0.1 * np.random.randn(n),
            'C': factor + 0.1 * np.random.randn(n),
        })
        
        deriver = PCAWeightDeriver()
        deriver.fit(data)
        
        # PC1 should explain most variance
        assert deriver.explained_variance_pc1 > 0.8
    
    def test_convenience_function(self):
        """Convenience function should work."""
        np.random.seed(42)
        data = pd.DataFrame({
            'X': np.random.randn(100),
            'Y': np.random.randn(100),
        })
        
        weights = derive_pca_weights(data)
        
        assert isinstance(weights, dict)
        assert 'X' in weights
        assert 'Y' in weights


class TestElasticNetWeights:
    """Tests for Elastic Net weight derivation."""
    
    def test_basic_elastic_net(self):
        """Basic Elastic Net should work."""
        np.random.seed(42)
        n = 200
        
        # Create data where X1 predicts Y
        data = pd.DataFrame({
            'X1': np.random.randn(n),
            'X2': np.random.randn(n),
            'X3': np.random.randn(n),
        })
        
        deriver = ElasticNetWeightDeriver(forward_window=10)
        deriver.fit(data)
        
        weights = deriver.weights
        
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.001
    
    def test_feature_selection(self):
        """Irrelevant features should get low/zero weight."""
        np.random.seed(42)
        n = 300
        
        # X1 strongly predicts target, X2 is noise
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        
        data = pd.DataFrame({'X1': x1, 'X2': x2})
        
        # Target is negatively related to X1
        target = pd.Series(-x1 + 0.1 * np.random.randn(n))
        
        deriver = ElasticNetWeightDeriver(forward_window=1)
        deriver.fit(data, target)
        
        # X1 should have higher coefficient than X2
        result = deriver.result
        assert abs(result.coefficients['X1']) > abs(result.coefficients['X2']) or \
               result.weights['X1'] >= result.weights['X2']


class TestCRITICWeights:
    """Tests for CRITIC weight derivation."""

    def test_basic_critic(self):
        """Basic CRITIC should work."""
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            'A': np.random.randn(n),
            'B': np.random.randn(n),
            'C': np.random.randn(n),
        })

        deriver = CRITICWeightDeriver()
        deriver.fit(data)

        weights = deriver.weights

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.001
        assert all(w >= 0 for w in weights.values())

    def test_critic_correlated_variables(self):
        """Highly correlated variables should have lower info content."""
        np.random.seed(42)
        n = 200
        # A and B are highly correlated, C is independent
        factor = np.random.randn(n)
        data = pd.DataFrame({
            'A': factor + 0.1 * np.random.randn(n),
            'B': factor + 0.1 * np.random.randn(n),
            'C': np.random.randn(n),  # Independent
        })

        deriver = CRITICWeightDeriver()
        deriver.fit(data)

        result = deriver.result
        # C should have higher information content (less correlated with others)
        assert result.information_content['C'] > result.information_content['A']
        assert result.information_content['C'] > result.information_content['B']

    def test_critic_convenience_function(self):
        """Convenience function should work."""
        np.random.seed(42)
        data = pd.DataFrame({
            'X': np.random.randn(100),
            'Y': np.random.randn(100),
        })

        weights = derive_critic_weights(data)

        assert isinstance(weights, dict)
        assert 'X' in weights
        assert 'Y' in weights
        assert abs(sum(weights.values()) - 1.0) < 0.001


class TestEntropyWeights:
    """Tests for Entropy weight derivation."""

    def test_basic_entropy(self):
        """Basic entropy weighting should work."""
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            'A': np.random.randn(n) + 10,  # Positive values
            'B': np.random.randn(n) + 10,
            'C': np.random.randn(n) + 10,
        })

        deriver = EntropyWeightDeriver()
        deriver.fit(data)

        weights = deriver.weights

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.001
        assert all(w >= 0 for w in weights.values())

    def test_entropy_uniform_vs_varied(self):
        """Uniform distribution should have higher entropy (lower weight)."""
        np.random.seed(42)
        n = 200
        # A is uniform, B has more variance
        data = pd.DataFrame({
            'uniform': np.ones(n) + 0.01 * np.random.randn(n),  # Nearly constant
            'varied': np.random.exponential(2, n),  # More skewed/varied
        })

        deriver = EntropyWeightDeriver()
        deriver.fit(data)

        result = deriver.result
        # Uniform should have higher entropy (more uniform distribution)
        assert result.entropy['uniform'] > result.entropy['varied']
        # Therefore uniform should have lower weight
        assert result.weights['uniform'] < result.weights['varied']

    def test_entropy_handles_negatives(self):
        """Entropy should handle negative values by shifting."""
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            'A': np.random.randn(n) - 5,  # Negative values
            'B': np.random.randn(n) + 5,  # Positive values
        })

        deriver = EntropyWeightDeriver()
        deriver.fit(data)

        weights = deriver.weights

        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_entropy_convenience_function(self):
        """Convenience function should work."""
        np.random.seed(42)
        data = pd.DataFrame({
            'X': np.random.randn(100) + 10,
            'Y': np.random.randn(100) + 10,
        })

        weights = derive_entropy_weights(data)

        assert isinstance(weights, dict)
        assert 'X' in weights
        assert 'Y' in weights
        assert abs(sum(weights.values()) - 1.0) < 0.001


class TestWeightComparison:
    """Tests for weight method comparison."""

    def test_comparison(self):
        """Weight comparison should work."""
        np.random.seed(42)
        n = 200

        data = pd.DataFrame({
            'stablecoin_risk': np.random.randn(n) + 50,
            'defi_liquidity_risk': np.random.randn(n) + 50,
            'contagion_risk': np.random.randn(n) + 50,
            'arbitrage_opacity': np.random.randn(n) + 50,
        })

        comparison = compare_weight_methods(data)

        assert comparison.theoretical_weights is not None
        assert comparison.pca_weights is not None
        assert comparison.elastic_net_weights is not None
        assert comparison.critic_weights is not None
        assert comparison.entropy_weights is not None

        # Correlations should be computed
        assert not np.isnan(comparison.pca_vs_theoretical)
        assert not np.isnan(comparison.critic_vs_theoretical)
        assert not np.isnan(comparison.entropy_vs_theoretical)
    
    def test_sensitivity_analysis(self):
        """Sensitivity analysis should work."""
        np.random.seed(42)
        n = 200
        
        data = pd.DataFrame({
            'A': np.random.randn(n),
            'B': np.random.randn(n),
        })
        target = pd.Series(np.random.randn(n))
        
        weights = {'A': 0.5, 'B': 0.5}
        
        result = weight_sensitivity_analysis(
            data, weights, target,
            perturbation_range=(-0.1, 0.1),
            n_steps=5,
        )
        
        assert result.base_weights == weights
        assert result.performance_matrix is not None
        assert len(result.sensitivity_ranking) == 2


class TestEdgeCases:
    """Edge case tests for all weight methods."""

    def test_all_weights_sum_to_one(self):
        """All weight methods should produce weights summing to 1."""
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            'A': np.random.randn(n) + 10,
            'B': np.random.randn(n) + 10,
            'C': np.random.randn(n) + 10,
        })

        # PCA
        pca_weights = derive_pca_weights(data)
        assert abs(sum(pca_weights.values()) - 1.0) < 0.001

        # Elastic Net
        en_weights = derive_elastic_net_weights(data, forward_window=10)
        assert abs(sum(en_weights.values()) - 1.0) < 0.001

        # CRITIC
        critic_weights = derive_critic_weights(data)
        assert abs(sum(critic_weights.values()) - 1.0) < 0.001

        # Entropy
        entropy_weights = derive_entropy_weights(data)
        assert abs(sum(entropy_weights.values()) - 1.0) < 0.001

    def test_uncorrelated_variables_critic(self):
        """Completely uncorrelated variables should get equal CRITIC info content."""
        np.random.seed(42)
        n = 500
        # Three independent variables with same variance
        data = pd.DataFrame({
            'A': np.random.randn(n),
            'B': np.random.randn(n),
            'C': np.random.randn(n),
        })

        deriver = CRITICWeightDeriver()
        deriver.fit(data)

        result = deriver.result
        # Info content should be similar for all (all uncorrelated)
        info_values = list(result.information_content.values())
        # Allow some variance due to sample correlation
        assert max(info_values) - min(info_values) < 0.5

    def test_identical_columns_entropy(self):
        """Nearly identical columns should have similar entropy."""
        np.random.seed(42)
        n = 200
        base = np.random.randn(n) + 10
        data = pd.DataFrame({
            'A': base + 0.01 * np.random.randn(n),
            'B': base + 0.01 * np.random.randn(n),
        })

        deriver = EntropyWeightDeriver()
        deriver.fit(data)

        result = deriver.result
        # Entropies should be very similar
        entropy_diff = abs(result.entropy['A'] - result.entropy['B'])
        assert entropy_diff < 0.1

    def test_two_variable_case(self):
        """All methods should work with just 2 variables."""
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            'X': np.random.randn(n) + 10,
            'Y': np.random.randn(n) + 10,
        })

        # All should work
        pca_w = derive_pca_weights(data)
        assert len(pca_w) == 2

        critic_w = derive_critic_weights(data)
        assert len(critic_w) == 2

        entropy_w = derive_entropy_weights(data)
        assert len(entropy_w) == 2

    def test_many_variables(self):
        """Methods should scale to many variables."""
        np.random.seed(42)
        n = 200
        k = 10  # 10 variables
        data = pd.DataFrame({
            f'X{i}': np.random.randn(n) + 10 for i in range(k)
        })

        pca_w = derive_pca_weights(data)
        assert len(pca_w) == k
        assert abs(sum(pca_w.values()) - 1.0) < 0.001

        critic_w = derive_critic_weights(data)
        assert len(critic_w) == k
        assert abs(sum(critic_w.values()) - 1.0) < 0.001

        entropy_w = derive_entropy_weights(data)
        assert len(entropy_w) == k
        assert abs(sum(entropy_w.values()) - 1.0) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
