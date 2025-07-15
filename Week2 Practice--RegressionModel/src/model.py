
class MeanRegressor:
    def fit(self, X, y):
        self.mean = sum(y) / len(y)
    def predict(self, X):
        return [self.mean for _ in X]
    def score(self, X, y):
        y_pred = self.predict(X)
        mean_y = sum(y) / len(y)
        ss_total = sum((yt - mean_y) ** 2 for yt in y)
        ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y, y_pred))
        return 1 - ss_res / ss_total

def build_model(name):
    if name == 'mean':
        return MeanRegressor()
    else:
        raise ValueError("Only 'mean' model is supported.")
