
def get_regression_model_performance(y_true, y_pred):
    n = len(y_true)
    mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n
    mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n
    mean_y = sum(y_true) / n
    ss_total = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    r2 = 1 - ss_res / ss_total if ss_total != 0 else 0.0
    return mse, mae, r2
