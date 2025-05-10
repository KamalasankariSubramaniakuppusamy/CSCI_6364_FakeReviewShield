import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

class XGBoostScratch:
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    class DecisionTree:
        def __init__(self, max_depth=3, min_samples_split=2):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.tree = None

        def fit(self, X, residuals):
            data = np.c_[X, residuals]
            self.tree = self._build_tree(data)

        def _build_tree(self, data, depth=0):
            if len(data) <= self.min_samples_split or depth >= self.max_depth:
                return self._to_terminal(data)
            
            split = self._get_best_split(data)
            if split is None:
                return self._to_terminal(data)
            
            left_data = data[data[:, split['index']] < split['value']]
            right_data = data[data[:, split['index']] >= split['value']]
            
            node = {
                'index': split['index'],
                'value': split['value'],
                'left': self._build_tree(left_data, depth+1),
                'right': self._build_tree(right_data, depth+1)
            }
            return node

        def _get_best_split(self, data):
            best_split = None
            min_variance = float('inf')
            
            for feature_idx in range(data.shape[1] - 1):
                unique_values = np.unique(data[:, feature_idx])
                for value in unique_values:
                    left = data[data[:, feature_idx] < value]
                    right = data[data[:, feature_idx] >= value]
                    
                    if len(left) == 0 or len(right) == 0:
                        continue
                        
                    current_variance = (len(left) * np.var(left[:, -1]) + 
                                      len(right) * np.var(right[:, -1]))
                    
                    if current_variance < min_variance:
                        min_variance = current_variance
                        best_split = {
                            'index': feature_idx,
                            'value': value,
                            'left': left,
                            'right': right
                        }
            return best_split

        def _to_terminal(self, data):
            return np.mean(data[:, -1])

        def predict(self, X):
            return np.array([self._predict_tree(x, self.tree) for x in X])

        def _predict_tree(self, x, node):
            if isinstance(node, dict):
                if x[node['index']] < node['value']:
                    return self._predict_tree(x, node['left'])
                else:
                    return self._predict_tree(x, node['right'])
            else:
                return node

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _log_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def _accuracy(self, y_true, y_pred):
        return np.mean(y_true == (y_pred >= 0.5).astype(int))

    def fit(self, X, y, verbose=True):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pred_train = np.zeros_like(y_train, dtype=float)
        pred_val = np.zeros_like(y_val, dtype=float)
        
        for i in range(self.n_estimators):
            grad_train = y_train - self._sigmoid(pred_train)
            
            tree = self.DecisionTree(max_depth=self.max_depth, 
                                   min_samples_split=self.min_samples_split)
            tree.fit(X_train, grad_train)
            
            update_train = tree.predict(X_train)
            pred_train += self.learning_rate * update_train
            
            update_val = tree.predict(X_val)
            pred_val += self.learning_rate * update_val
            
            self.trees.append(tree)
            
            train_prob = self._sigmoid(pred_train)
            val_prob = self._sigmoid(pred_val)
            
            train_loss = self._log_loss(y_train, train_prob)
            val_loss = self._log_loss(y_val, val_prob)
            train_acc = self._accuracy(y_train, train_prob)
            val_acc = self._accuracy(y_val, val_prob)
            
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            
            if verbose:
                print(f"Iter {i+1}/{self.n_estimators}: "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss {val_loss:.4f}")

    def predict_prob(self, X):
        pred = np.zeros(X.shape[0])
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return self._sigmoid(pred)

    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)
    
    def evaluate_test(self, X_test, y_test, verbose=True):
        """Evaluate model performance on test data and print metrics"""
        test_prob = self.predict_prob(X_test)
        test_pred = self.predict(X_test)
        
        test_loss = self._log_loss(y_test, test_prob)
        test_acc = self._accuracy(y_test, test_prob)
        test_precision = precision_score(y_test, test_pred)
        test_recall = recall_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred)
        
        if verbose:
            print("Test Set Metrics:")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            print(f"Test F1 Score: {test_f1:.4f}")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'predictions': test_pred,
            'probabilities': test_prob,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1
        }
