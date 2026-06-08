"""
Unit tests for Decision Tree Classifier (tree.py).

Coverage includes:
- Basic functionality (fit, predict, fit_transform)
- Streaming capability (partial_fit with multiple chunks)
- Edge cases (NaNs, single class, zero variance, empty data)
- Numerical stability
- Tree properties (depth, leaves, node counting)
- Class probability estimation
- Different max_depth and min_samples_leaf configurations

"""

import numpy as np
import pytest
from numcompute_stream.tree import (
    DecisionTreeClassifier,
    _gini_impurity,
    _information_gain,
    _find_best_split,
    TreeNode,
)


# =============================================================================
# Tests for Gini Impurity (Helper Function)
# =============================================================================

class TestGiniImpurity:
    """Test suite for Gini impurity calculation."""
    
    def test_gini_pure_class(self):
        """Gini impurity of pure class should be 0."""
        y = np.array([0, 0, 0, 0])
        assert _gini_impurity(y) == 0.0
    
    def test_gini_two_classes_balanced(self):
        """Gini impurity of balanced binary split should be 0.5."""
        y = np.array([0, 0, 1, 1])
        expected = 1.0 - (0.5**2 + 0.5**2)  # = 0.5
        assert np.isclose(_gini_impurity(y), expected)
    
    def test_gini_three_classes_equal(self):
        """Gini impurity with 3 equal classes."""
        y = np.array([0, 1, 2])
        expected = 1.0 - 3 * (1/3)**2  # = 2/3
        assert np.isclose(_gini_impurity(y), expected)
    
    def test_gini_empty_array(self):
        """Gini impurity of empty array should be 0."""
        y = np.array([])
        assert _gini_impurity(y) == 0.0
    
    def test_gini_with_nans(self):
        """Gini impurity should ignore NaN values."""
        y = np.array([0, 0, np.nan, np.nan])
        expected = _gini_impurity(np.array([0, 0]))
        assert np.isclose(_gini_impurity(y), expected)
    
    def test_gini_all_nans(self):
        """Gini impurity of all-NaN array should be 0."""
        y = np.array([np.nan, np.nan])
        assert _gini_impurity(y) == 0.0


# =============================================================================
# Tests for Information Gain (Helper Function)
# =============================================================================

class TestInformationGain:
    """Test suite for information gain calculation."""
    
    def test_information_gain_perfect_split(self):
        """Information gain for a perfect split should be positive."""
        y_parent = np.array([0, 0, 1, 1])
        y_left = np.array([0, 0])
        y_right = np.array([1, 1])
        gain = _information_gain(y_parent, y_left, y_right)
        assert gain > 0.0
    
    def test_information_gain_no_split(self):
        """Information gain when split doesn't separate classes should be small."""
        y_parent = np.array([0, 0, 1, 1])
        y_left = np.array([0, 0, 1])
        y_right = np.array([1])
        gain = _information_gain(y_parent, y_left, y_right)
        # Should be small but non-zero
        assert 0 <= gain < _information_gain(y_parent, np.array([0, 0]), np.array([1, 1]))
    
    def test_information_gain_empty_child(self):
        """Information gain with empty child should be 0."""
        y_parent = np.array([0, 1])
        y_left = np.array([])
        y_right = np.array([0, 1])
        gain = _information_gain(y_parent, y_left, y_right)
        assert gain == 0.0


class TestCriterionOptions:
    """Test gini vs entropy criteria."""

    def test_entropy_criterion_fit_predict(self):
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])

        clf = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=2
        )

        clf.fit(X, y)

        assert np.array_equal(clf.predict(X), y)

    def test_invalid_criterion_raises_error(self):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(criterion="invalid")

# =============================================================================
# Tests for Basic Fit and Predict
# =============================================================================

class TestBasicFitPredict:
    """Test basic fit and predict functionality."""
    
    def test_fit_simple_dataset(self):
        """Test fitting on simple linearly separable dataset."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier(max_depth=2)
        clf.fit(X, y)
        
        assert clf._is_fitted
        assert clf.tree_ is not None
        assert clf.n_classes_ == 2
        assert clf.n_features_in_ == 2
    
    def test_predict_simple_dataset(self):
        """Test predictions on simple dataset."""
        X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y_train = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier(max_depth=2)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_train)
        
        assert y_pred.shape == (4,)
        assert np.array_equal(y_pred, y_train)
    
    def test_predict_before_fit_raises_error(self):
        """Predict before fit should raise RuntimeError."""
        X = np.array([[0, 0], [1, 1]])
        clf = DecisionTreeClassifier()
        
        with pytest.raises(RuntimeError):
            clf.predict(X)
    
    def test_predict_wrong_n_features_raises_error(self):
        """Predict with wrong number of features should raise ValueError."""
        X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y_train = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        
        X_test = np.array([[0, 0, 0]])  # Wrong number of features
        with pytest.raises(ValueError):
            clf.predict(X_test)
    
    def test_iris_like_dataset(self):
        """Test on iris-like dataset (4 features, 3 classes)."""
        np.random.seed(42)
        X = np.random.randn(30, 4)
        y = np.tile([0, 1, 2], 10)
        
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X, y)
        
        y_pred = clf.predict(X)
        
        assert y_pred.shape == (30,)
        assert set(y_pred).issubset(set(y))


# =============================================================================
# Tests for Streaming (partial_fit)
# =============================================================================

class TestStreamingLearning:
    """Test partial_fit for streaming/incremental learning."""
    
    def test_partial_fit_single_chunk(self):
        """Test partial_fit with single chunk."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.partial_fit(X, y)
        
        assert clf._is_fitted
        assert clf.tree_ is not None
    
    def test_partial_fit_multiple_chunks(self):
        """Test partial_fit with multiple data chunks (streaming scenario)."""
        # Chunk 1
        X1 = np.array([[0, 0], [1, 1]])
        y1 = np.array([0, 0])
        
        # Chunk 2
        X2 = np.array([[2, 2], [3, 3]])
        y2 = np.array([1, 1])
        
        clf = DecisionTreeClassifier(max_depth=2)
        
        # First chunk
        clf.partial_fit(X1, y1)
        assert clf._is_fitted
        
        # Second chunk (update)
        clf.partial_fit(X2, y2)
        assert clf.n_classes_ == 2
        
        # Predict on all data
        X_all = np.vstack([X1, X2])
        y_pred = clf.predict(X_all)
        assert y_pred.shape == (4,)
    
    def test_partial_fit_many_chunks(self):
        """Test partial_fit with many small chunks."""
        np.random.seed(42)
        
        # Generate full dataset
        X_full = np.random.randn(100, 2)
        y_full = (X_full[:, 0] + X_full[:, 1] > 0).astype(int)
        
        # Train with chunks
        clf = DecisionTreeClassifier(max_depth=5)
        
        chunk_size = 10
        for i in range(0, 100, chunk_size):
            X_chunk = X_full[i:i+chunk_size]
            y_chunk = y_full[i:i+chunk_size]
            clf.partial_fit(X_chunk, y_chunk)
        
        # Predictions should work
        y_pred = clf.predict(X_full)
        assert y_pred.shape == (100,)
        
        # Should have non-trivial accuracy on training data
        accuracy = np.mean(y_pred == y_full)
        assert accuracy > 0.5  # Better than random

    def test_partial_fit_accumulates_seen_data(self):
        clf = DecisionTreeClassifier()

        clf.partial_fit(
            np.array([[0], [1]]),
            np.array([0, 0])
        )

        clf.partial_fit(
            np.array([[2], [3]]),
            np.array([1, 1])
        )

        assert clf._X_seen.shape[0] == 4
        assert clf._y_seen.shape[0] == 4


# =============================================================================
# Tests for Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test handling of edge cases and numerical stability."""
    
    def test_single_sample(self):
        """Test fitting with single sample."""
        X = np.array([[1.0, 2.0]])
        y = np.array([0])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        
        y_pred = clf.predict(X)
        assert y_pred[0] == 0
    
    def test_single_class(self):
        """Test fitting with single class."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 0, 0])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        
        y_pred = clf.predict(X)
        assert np.all(y_pred == 0)
    
    def test_zero_variance_feature(self):
        """Test with zero-variance feature."""
        X = np.array([[5, 0], [5, 1], [5, 2], [5, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)  # Should use feature 1
        
        y_pred = clf.predict(X)
        assert y_pred.shape == (4,)
    
    def test_all_nan_feature(self):
        """Test with all-NaN feature."""
        X = np.array([[np.nan, 0], [np.nan, 1], [np.nan, 2], [np.nan, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)  # Should use feature 1
        
        y_pred = clf.predict(X)
        assert y_pred.shape == (4,)
    
    def test_partial_nan_values(self):
        """Test with some NaN values in features."""
        X = np.array([[1.0, 0], [np.nan, 1], [2.0, 2], [3.0, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        
        y_pred = clf.predict(X)
        assert y_pred.shape == (4,)
    
    def test_nan_in_labels(self):
        """Test with NaN in labels."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, np.nan, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)  # Should handle gracefully
        
        y_pred = clf.predict(X)
        assert y_pred.shape == (4,)
    
    def test_large_feature_values(self):
        """Test with large feature values (numerical stability)."""
        X = np.array([[1e10, 1e10], [1e10 + 1, 1e10 + 1], 
                      [2e10, 2e10], [2e10 + 1, 2e10 + 1]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        
        y_pred = clf.predict(X)
        assert np.allclose(y_pred, y)
    
    def test_small_feature_values(self):
        """Test with small (near-zero) feature values."""
        X = np.array([[1e-10, 1e-10], [2e-10, 2e-10],
                      [3e-10, 3e-10], [4e-10, 4e-10]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        
        y_pred = clf.predict(X)
        assert y_pred.shape == (4,)
    
    def test_negative_features(self):
        """Test with negative feature values."""
        X = np.array([[-2, -2], [-1, -1], [1, 1], [2, 2]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        
        y_pred = clf.predict(X)
        assert np.array_equal(y_pred, y)


# =============================================================================
# Tests for Max Depth and Min Samples Controls
# =============================================================================

class TestDepthControls:
    """Test max_depth and min_samples_leaf parameters."""
    
    def test_max_depth_1(self):
        """Test max_depth=1 creates a stump."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(X, y)
        
        depth = clf.get_depth()
        assert depth == 1
    
    def test_max_depth_affects_accuracy(self):
        """Deeper trees should generally fit training data better."""
        np.random.seed(42)
        X = np.random.randn(50, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        clf_shallow = DecisionTreeClassifier(max_depth=1)
        clf_shallow.fit(X, y)
        acc_shallow = np.mean(clf_shallow.predict(X) == y)
        
        clf_deep = DecisionTreeClassifier(max_depth=10)
        clf_deep.fit(X, y)
        acc_deep = np.mean(clf_deep.predict(X) == y)
        
        assert acc_deep >= acc_shallow
    
    def test_min_samples_leaf(self):
        """Test min_samples_leaf prevents small leaves."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier(min_samples_leaf=2)
        clf.fit(X, y)
        
        # All leaves should have at least 2 samples
        def check_leaf_sizes(node):
            if node.is_leaf:
                assert node.samples >= 2
            else:
                check_leaf_sizes(node.left)
                check_leaf_sizes(node.right)
        
        check_leaf_sizes(clf.tree_)


class TestMaxFeatures:
    def test_max_features_sqrt_runs(self):
        np.random.seed(42)

        X = np.random.randn(20, 4)
        y = (X[:, 0] > 0).astype(int)

        clf = DecisionTreeClassifier(
            max_features="sqrt",
            random_state=42
        )

        clf.fit(X, y)

        y_pred = clf.predict(X)

        assert y_pred.shape == (20,)

    def test_max_features_log2_runs(self):
        np.random.seed(42)

        X = np.random.randn(20, 8)
        y = (X[:, 0] > 0).astype(int)

        clf = DecisionTreeClassifier(
            max_features="log2",
            random_state=42
        )

        clf.fit(X, y)

        y_pred = clf.predict(X)

        assert y_pred.shape == (20,)
# =============================================================================
# Tests for Predict Proba
# =============================================================================

class TestPredictProba:
    """Test probability prediction."""
    
    def test_predict_proba_shape(self):
        """Test predict_proba returns correct shape."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        
        assert proba.shape == (4, 2)
    
    def test_predict_proba_sums_to_one(self):
        """Test that probabilities sum to 1 for each sample."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_predict_proba_consistency_with_predict(self):
        """Test that highest probability class matches predict()."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        
        y_pred = clf.predict(X)
        proba = clf.predict_proba(X)
        
        # argmax of probabilities should match predict
        y_pred_from_proba = clf.classes_[np.argmax(proba, axis=1)]
        
        assert np.array_equal(y_pred, y_pred_from_proba)
    
    def test_predict_proba_before_fit_raises_error(self):
        """predict_proba before fit should raise RuntimeError."""
        X = np.array([[0, 0], [1, 1]])
        clf = DecisionTreeClassifier()
        
        with pytest.raises(RuntimeError):
            clf.predict_proba(X)


# =============================================================================
# Tests for Tree Properties
# =============================================================================

class TestTreeProperties:
    """Test tree structure and property methods."""
    
    def test_get_depth(self):
        """Test get_depth method."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier(max_depth=10)
        clf.fit(X, y)
        
        depth = clf.get_depth()
        assert 0 <= depth <= 10
    
    def test_get_n_leaves(self):
        """Test get_n_leaves method."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        
        n_leaves = clf.get_n_leaves()
        assert n_leaves >= 1
    
    def test_single_node_tree(self):
        """Test tree with single node (root is leaf)."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 0, 0])  # All same class
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        
        assert clf.tree_.is_leaf
        assert clf.get_depth() == 0
        assert clf.get_n_leaves() == 1


# =============================================================================
# Tests for Input Validation
# =============================================================================

class TestInputValidation:
    """Test input validation."""
    
    def test_empty_dataset(self):
        X = np.empty((0, 2))
        y = np.array([])

        clf = DecisionTreeClassifier()

        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_invalid_X_shape_1d(self):
        """fit with 1D X should raise ValueError."""
        X = np.array([1, 2, 3])
        y = np.array([0, 0, 1])
        
        clf = DecisionTreeClassifier()
        with pytest.raises(ValueError):
            clf.fit(X, y)
    
    def test_invalid_X_shape_3d(self):
        """fit with 3D X should raise ValueError."""
        X = np.array([[[1, 2], [3, 4]]])
        y = np.array([0])
        
        clf = DecisionTreeClassifier()
        with pytest.raises(ValueError):
            clf.fit(X, y)
    
    def test_invalid_y_shape_2d(self):
        """fit with 2D y should raise ValueError."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[0], [1]])
        
        clf = DecisionTreeClassifier()
        with pytest.raises(ValueError):
            clf.fit(X, y)
    
    def test_mismatched_lengths(self):
        """X and y with different lengths should raise ValueError."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1])
        
        clf = DecisionTreeClassifier()
        with pytest.raises(ValueError):
            clf.fit(X, y)


# =============================================================================
# Tests for Multi-class Classification
# =============================================================================

class TestMultiClass:
    """Test multi-class classification."""
    
    def test_three_class(self):
        """Test with 3 classes."""
        X = np.array([[0, 0], [1, 0], [2, 0],
                      [0, 1], [1, 1], [2, 1],
                      [0, 2], [1, 2], [2, 2]])
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        
        assert clf.n_classes_ == 3
        
        y_pred = clf.predict(X)
        assert set(y_pred).issubset({0, 1, 2})
    
    def test_five_class(self):
        """Test with 5 classes."""
        X = np.random.randn(50, 3)
        y = np.tile(np.arange(5), 10)
        
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X, y)
        
        assert clf.n_classes_ == 5
        
        y_pred = clf.predict(X)
        assert y_pred.shape == (50,)

class TestParameters:
    def test_get_params(self):
        clf = DecisionTreeClassifier(
            max_depth=5,
            criterion="entropy"
        )

        params = clf.get_params()

        assert params["max_depth"] == 5
        assert params["criterion"] == "entropy"

    def test_set_params(self):
        clf = DecisionTreeClassifier()

        clf.set_params(
            max_depth=3,
            criterion="entropy"
        )

        assert clf.max_depth == 3
        assert clf.criterion == "entropy"

# =============================================================================
# Tests for Consistency and Reproducibility
# =============================================================================

class TestConsistency:
    """Test that results are consistent."""
    
    def test_multiple_fit_calls_same_result(self):
        """Multiple fit calls with same data should give same tree."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf1 = DecisionTreeClassifier()
        clf1.fit(X, y)
        y_pred1 = clf1.predict(X)
        
        clf2 = DecisionTreeClassifier()
        clf2.fit(X, y)
        y_pred2 = clf2.predict(X)
        
        assert np.array_equal(y_pred1, y_pred2)
    
    def test_fit_vs_partial_fit_consistency(self):
        """fit() and partial_fit() on same data should give same result."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        clf_fit = DecisionTreeClassifier()
        clf_fit.fit(X, y)
        y_pred_fit = clf_fit.predict(X)
        
        clf_partial = DecisionTreeClassifier()
        clf_partial.partial_fit(X, y)
        y_pred_partial = clf_partial.predict(X)
        
        assert np.array_equal(y_pred_fit, y_pred_partial)


# =============================================================================
# Run all tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])