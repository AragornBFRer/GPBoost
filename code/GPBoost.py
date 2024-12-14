from scipy.optimize import minimize
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import numpy as np

class GPBoost:
    def __init__(self, learning_rate=0.1, n_iterations=100, boost_type="gradient", 
                 nesterov_accel=False, momentum_sequence=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.boost_type = boost_type
        self.nesterov_accel = nesterov_accel
        self.momentum_sequence = momentum_sequence or [0.5] * n_iterations
        self.base_learners = []

    def loss_function(self, y, F, theta, Sigma):
        residual = y - F
        Sigma += np.eye(Sigma.shape[0]) * 1e-6
        return 0.5 * residual.T @ np.linalg.solve(Sigma, residual) + \
               0.5 * np.log(np.linalg.det(Sigma)) + len(y) * 0.5 * np.log(2 * np.pi)

    def optimize_theta(self, y, F, theta_init, Sigma_func, bounds):
        def objective(theta):
            Sigma = Sigma_func(theta)
            return self.loss_function(y, F, theta, Sigma)
        result = minimize(objective, theta_init, method='L-BFGS-B', bounds=bounds)
        return result.x

    def fit(self, X, y, initial_theta, Sigma_func, base_learner):
        n_samples = len(y)
        theta = initial_theta
        self.F0 = np.mean(y)
        F = np.full(n_samples, self.F0)
        G_prev = None

        for m in range(self.n_iterations):
            bounds = [(1e-5, None)] * len(theta)
            theta = self.optimize_theta(y, F, theta, Sigma_func, bounds)
            Sigma = Sigma_func(theta)

            if self.nesterov_accel:
                G_curr = F.copy()
                if m > 0:
                    F += self.momentum_sequence[m] * (G_curr - G_prev)
                G_prev = G_curr

            residual = y - F
            learner_m = clone(base_learner)
            if self.boost_type == "gradient":
                gradient = np.linalg.solve(Sigma, residual)
                learner_m.fit(X, gradient)
            elif self.boost_type == "newton":
                learner_m.fit(X, residual)
            elif self.boost_type == "hybrid":
                gradient = np.linalg.solve(Sigma, residual)
                learner_m.fit(X, gradient)

            f_m = learner_m.predict(X)
            F += self.learning_rate * f_m
            self.base_learners.append(learner_m)

        self.theta = theta

    def predict(self, X_new, base_learner):
        F = np.full(X_new.shape[0], self.F0)
        for learner in self.base_learners:
            F += self.learning_rate * learner.predict(X_new)
        return F
    