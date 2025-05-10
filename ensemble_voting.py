import numpy as np
from collections import defaultdict
import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

class EnsembleModel:
    def __init__(self, models):
        """
        Initialize ensemble with trained models and identify the best performing model
        
        Args:
            models: list of initialized and trained models
                   [logistic_regression_model, random_forest_model, xgboost_model]
        """
        self.models = models
        self.metrics = {}
        self.best_model = None  # Will store the best performing model

    def evaluate_test(self, X_test, y_test, verbose=True):
        """
        Evaluate ensemble and individual models on test data
        
        Args:
            X_test: Test features
            y_test: True labels
            verbose: Whether to print results
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        results = {
            'ensemble': {'accuracy': 0.0, 'predictions': None},
            'individual': {}
        }
        
        # Collect predictions from all models
        all_preds = []
        for i, model in enumerate(self.models):
            model_results = {}
            try:
                if hasattr(model, 'evaluate_test'):
                    model_results = model.evaluate_test(X_test, y_test, verbose=False)
                else:
                    preds = model.predict(X_test)
                    model_results = {
                        'accuracy': np.mean(preds == y_test),
                        'predictions': preds
                    }
                
                # Ensure we have predictions
                if 'predictions' not in model_results:
                    model_results['predictions'] = model.predict(X_test)
                
                results['individual'][f'model_{i}'] = {
                    'type': model.__class__.__name__,
                    'model_object': model,  # Store the model object
                    **model_results
                }
                all_preds.append(model_results['predictions'])
            except Exception as e:
                print(f"Error evaluating model {i} ({model.__class__.__name__}): {str(e)}")
                continue
        
        if not all_preds:
            raise ValueError("No valid models were evaluated successfully")
        
        # Calculate ensemble predictions (majority vote)
        all_preds = np.array(all_preds)
        ensemble_preds = np.array([np.argmax(np.bincount(row.astype(int))) for row in all_preds.T])
        
        # Calculate ensemble metrics
        results['ensemble']['accuracy'] = np.mean(ensemble_preds == y_test)
        results['ensemble']['predictions'] = ensemble_preds
        results['ensemble']['precision'] = precision_score(y_test, ensemble_preds)
        results['ensemble']['recall'] = recall_score(y_test, ensemble_preds)
        results['ensemble']['f1_score'] = f1_score(y_test, ensemble_preds)

        # Identify best individual model
        best_model_info = None
        best_accuracy = -1
        for model_info in results['individual'].values():
            accuracy = model_info.get('accuracy', model_info.get('test_accuracy', -1))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_info = model_info
        
        if best_model_info:
            self.best_model = best_model_info['model_object']
            if verbose:
                print(f"\nSelected best model: {best_model_info['type']} (Accuracy: {best_accuracy:.4f})")
        
        if verbose:
            self._print_results(results)
        
        self.metrics = results
        return results
    
    def predict(self, X):
        """
        Predict using the best performing model from evaluation
        
        Args:
            X: Feature vector or array (already preprocessed)
            
        Returns:
            Predicted class (0 or 1) or array of predictions
        """
        if self.best_model is None:
            print("Finding best model.")
            test_df = pd.read_csv("DATASET_TEST.csv")
            X_test = test_df.drop(columns=["Label"]).values
            y_test = test_df["Label"].values
            _ = self.evaluate_test(X_test, y_test, verbose=False)
        
        if self.best_model is not None:
            print("Best model found. Making predictions.")
            # Handle single sample prediction
            if len(X.shape) == 1:
                return self.best_model.predict(X.reshape(1, -1))[0]
            # Handle batch prediction
            else:
                return self.best_model.predict(X)
        else:
            raise ValueError("No best model found. Please evaluate the ensemble first.")

    def _print_results(self, results):
        """Helper function to print evaluation results"""
        print("\nEnsemble Evaluation Results:")
        print("=" * 50)
        print(f"Ensemble Accuracy: {results['ensemble']['accuracy']:.4f}")
        print(f"Ensemble Precision: {results['ensemble']['precision']:.4f}")
        print(f"Ensemble Recall: {results['ensemble']['recall']:.4f}")
        print(f"Ensemble F1 Score: {results['ensemble']['f1_score']:.4f}")
        
        print("\nIndividual Model Performance:")
        for model_name, metrics in results['individual'].items():
            print(f"\n{metrics['type']}:")
            
            # Handle accuracy printing safely
            accuracy = metrics.get('accuracy', metrics.get('test_accuracy', None))
            if accuracy is not None:
                print(f"  Accuracy: {accuracy:.4f}")
            else:
                print("  Accuracy: N/A")
            
            # Print other metrics if available
            for metric in ['loss', 'precision', 'recall', 'f1_score']:
                if metric in metrics:
                    print(f"  {metric.capitalize()}: {metrics[metric]:.4f}")

# Example usage
# if __name__ == "__main__":
#     test_df = pd.read_csv("DATASET_TEST.csv")
#     X_test = test_df.drop(columns=["Label"]).values
#     y_test = test_df["Label"].values

#     # Load models
#     from Logistic_Regression_Scratch import LogisticRegressionScratch
#     from random_forest_classifier_scratch import RandomForestScratch
#     from xgboost_scratch import XGBoostScratch

#     log_model = LogisticRegressionScratch(learning_rate=0.1, num_iterations=50)
#     rf_model = RandomForestScratch(n_trees=10, max_depth=5)
#     xgb_model = XGBoostScratch(n_estimators=50, learning_rate=0.1)

#     # Load model parameters
#     log_model.theta = np.load("Trained_Models_From_Scratch/logistic_regression.npy")
#     with open("Trained_Models_From_Scratch/random_forest.pkl", "rb") as f:
#         rf_model.trees = pickle.load(f)
#     with open("Trained_Models_From_Scratch/xgboost.pkl", "rb") as f:
#         xgb_model.trees = pickle.load(f)

#     # Create ensemble model
#     ensemble_model = EnsembleModel(models=[log_model, rf_model, xgb_model])
    
#     # Evaluate ensemble and individual models
#     try:
#         ensemble_results = ensemble_model.evaluate_test(X_test, y_test, verbose=True)
        
#         # Access specific metrics:
#         print("-"*50)
#         print(f"Final Ensemble Accuracy: {ensemble_results['ensemble']['accuracy']:.4f}")
        
#         # Find best model safely
#         best_model = None
#         best_accuracy = -1
#         for model_name, metrics in ensemble_results['individual'].items():
#             accuracy = metrics.get('accuracy', metrics.get('test_accuracy', -1))
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_model = metrics['type']
        
#         if best_model:
#             print(f"Best Individual Model: {best_model} (Accuracy: {best_accuracy:.4f})")
#         else:
#             print("Could not determine best individual model")
#     except Exception as e:
#         print(f"Error during evaluation: {str(e)}")