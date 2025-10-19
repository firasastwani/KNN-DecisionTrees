"""
Sanity Check Tests for HW06: KNN and Decision Trees

This file contains basic tests to verify your code structure and format.
These tests check that your functions exist and return the expected types,
but DO NOT check if your implementation is correct.

Passing these tests means your code won't crash - it doesn't mean it's right!
You should write additional tests to verify your logic is correct.

To run: python test_sanity_check.py
"""

import unittest
import numpy as np
from knn_decision_trees import (
    load_and_prepare_data,
    train_knn_single,
    train_decision_tree_single,
    train_knn_range,
    train_decision_tree_range
)


class TestBasicStructure(unittest.TestCase):
    """Sanity checks for basic function structure."""
    
    def test_load_data_returns_dict(self):
        """Check that load_and_prepare_data returns a dictionary."""
        result = load_and_prepare_data()
        self.assertIsInstance(result, dict, 
                            "load_and_prepare_data should return a dictionary")
    
    def test_load_data_has_required_keys(self):
        """Check that data dictionary has all required keys."""
        result = load_and_prepare_data()
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test']
        
        for key in required_keys:
            self.assertIn(key, result, 
                        f"Data dictionary must contain '{key}' key")
    
    def test_train_knn_single_returns_dict(self):
        """Check that train_knn_single returns a dictionary."""
        data = load_and_prepare_data()
        result = train_knn_single(
            data['X_train'], data['X_test'],
            data['y_train'], data['y_test'], k=5
        )
        self.assertIsInstance(result, dict,
                            "train_knn_single should return a dictionary")
    
    def test_train_tree_single_returns_dict(self):
        """Check that train_decision_tree_single returns a dictionary."""
        data = load_and_prepare_data()
        result = train_decision_tree_single(
            data['X_train'], data['X_test'],
            data['y_train'], data['y_test'], max_depth=5
        )
        self.assertIsInstance(result, dict,
                            "train_decision_tree_single should return a dictionary")
    
    def test_knn_range_returns_lists(self):
        """Check that train_knn_range returns lists of results."""
        data = load_and_prepare_data()
        result = train_knn_range(
            data['X_train'], data['X_test'],
            data['y_train'], data['y_test'],
            k_range=range(1, 4)  # Just test 3 values for speed
        )
        
        self.assertIsInstance(result, dict,
                            "train_knn_range should return a dictionary")
        self.assertIn('accuracies', result,
                     "Result should contain 'accuracies' key")
        self.assertIsInstance(result['accuracies'], list,
                            "Accuracies should be a list")


class TestTask6Requirements(unittest.TestCase):
    """
    Sanity checks for Task 6 - Confusion Matrix and Metrics.
    
    NOTE: You need to implement these functions for Task 6!
    These tests will fail until you complete Task 6.
    """
    
    def test_confusion_matrix_function_exists(self):
        """Check that you've created a confusion matrix plotting function."""
        # Try to import your Task 6 functions
        # Students should name their function something like:
        # plot_confusion_matrix or evaluate_model
        
        try:
            from knn_decision_trees import plot_confusion_matrix
            self.assertTrue(callable(plot_confusion_matrix),
                          "plot_confusion_matrix should be a callable function")
        except ImportError:
            self.fail("Task 6: Create a plot_confusion_matrix function")
    
    def test_evaluate_model_function_exists(self):
        """Check that you've created a model evaluation function."""
        try:
            from knn_decision_trees import evaluate_model
            self.assertTrue(callable(evaluate_model),
                          "evaluate_model should be a callable function")
        except ImportError:
            self.fail("Task 6: Create an evaluate_model function")
    
    def test_metrics_are_valid_range(self):
        """Check that calculated metrics are in valid range [0, 1]."""
        try:
            from knn_decision_trees import evaluate_model
            
            # Get some predictions to test with
            data = load_and_prepare_data()
            result = train_knn_single(
                data['X_train'], data['X_test'],
                data['y_train'], data['y_test'], k=5
            )
            
            # Your evaluate_model should return a dict with metrics
            metrics = evaluate_model(data['y_test'], result['y_pred'], "Test")
            
            # If it returns metrics, check they're in valid range
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.assertGreaterEqual(value, 0,
                            f"{metric_name} should be >= 0")
                        self.assertLessEqual(value, 1,
                            f"{metric_name} should be <= 1")
        except ImportError:
            self.skipTest("Skipping - evaluate_model not implemented yet")
        except Exception as e:
            self.skipTest(f"Skipping - error in evaluate_model: {e}")


class TestTask7Requirements(unittest.TestCase):
    """
    Sanity checks for Task 7 - Dataset Exploration.
    
    NOTE: You need to implement analysis for additional datasets for Task 7!
    """
    
    def test_can_load_alternative_dataset(self):
        """Check that alternative datasets can be loaded."""
        from sklearn.datasets import load_iris
        
        # Students should create a way to load different datasets
        # This just checks that sklearn datasets are accessible
        data = load_iris()
        self.assertIsNotNone(data.data)
        self.assertIsNotNone(data.target)


def run_tests():
    """Run all sanity check tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBasicStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestTask6Requirements))
    suite.addTests(loader.loadTestsFromTestCase(TestTask7Requirements))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("SANITY CHECK SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All sanity checks passed!")
        print("Note: This doesn't mean your code is correct,")
        print("only that it has the right structure.")
    else:
        print("\n⚠️ Some sanity checks failed.")
        print("Fix the structure issues before testing correctness.")
    
    print("="*60)
    
    return result


if __name__ == "__main__":
    run_tests()
