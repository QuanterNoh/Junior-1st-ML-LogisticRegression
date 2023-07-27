import numpy as np
from sklearn.metrics import confusion_matrix

class MyLogisticRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def fit(self, X_data, y_data):
        bias_term = np.ones((X_data.shape[0], 1), dtype=float)
        X = np.concatenate([bias_term, X_data], axis=1)
        
        theta = np.linalg.inv(X.T @ X) @ (X.T @ y_data) # normal_equation
        self.coef_ = theta[1:]
        self.intercept_ = theta[0]


    def predict(self, X_data): 
        linear_hypothesis_func = X_data @ self.coef_ + self.intercept_
        pred = self._sigmoid(linear_hypothesis_func)
        pred_result = [1 if i >= 0.5 else 0 for i in pred]
        return pred_result


    def score(self, X_data, y_data):
        linear_hypothesis_func = X_data @ self.coef_ + self.intercept_
        pred = self._sigmoid(linear_hypothesis_func)
        pred_result = [1 if i >= 0.5 else 0 for i in pred]

        true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_data, pred_result).ravel()
        accuracy_score = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
        return accuracy_score