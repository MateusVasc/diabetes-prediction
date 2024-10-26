from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self, C=None, solver=None, max_iter=None):
        if C and solver and max_iter:
            self.model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
        else:
            self.model = LogisticRegression()

    def predict(self, X_train, X_test, y_train):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return y_pred