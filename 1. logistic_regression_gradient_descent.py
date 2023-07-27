import numpy as np
from sklearn.metrics import confusion_matrix

class SimpleLogisticRegression:

    def __init__(self, alpha=0.01, iterations=1000):
        self.alpha = alpha # learning rate
        self.iterations = iterations
        self.coef_ = None 
        self.intercept_ = None 
        self._log_loss = None # log loss

    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    
    def fit(self, X_data, y_data):
        m = len(y_data)

        # W, b initialize
        theta = np.zeros((X_data.shape[1] + 1))
        self.coef_ = theta[1:]
        self.intercept_ = theta[0]

        bias_term = np.ones((X_data.shape[0], 1))
        X = np.concatenate((bias_term, X_data), axis=1)

        # Gradient Descent
        for i in range(self.iterations):
            linear_hypothesis_func = X @ theta
            pred = self._sigmoid(linear_hypothesis_func)
            error = pred - y_data
            self._log_loss = - (1 / m) * np.sum(y_data * np.log(pred) + (1 - y_data) * np.log(1 - pred))

            theta -= self.alpha * (X.T @ error) / len(X)
            self.coef_ = theta[1:]
            self.intercept_ = theta[0]
            print(f'Epoch [{i + 1}/{self.iterations}] | log_loss: {self._log_loss:.4f}')

    
    def predict(self, X_data):
        linear_hypothesis_func = X_data @ self.coef_ + self.intercept_ 
        pred = self._sigmoid(linear_hypothesis_func) 
        pred_result = [1 if i >= 0.5 else 0 for i in pred]
        return np.array(pred_result)


    def score(self, X_data, y_data):
        linear_hypothesis_func = X_data @ self.coef_ + self.intercept_ 
        pred = self._sigmoid(linear_hypothesis_func)
        pred_result = [1 if i >= 0.5 else 0 for i in pred]
        
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_data, pred_result).ravel()
        accuracy_score = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
        return accuracy_score

