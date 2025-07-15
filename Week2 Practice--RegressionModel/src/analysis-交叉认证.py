import os
import numpy as np
import pandas as pd

# —— 路径设置 —— #
HERE = os.path.dirname(__file__)
csv_path = os.path.abspath(os.path.join(HERE, '..', 'data', 'US-pumpkins.csv'))

# —— 数据加载与预处理 —— #
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
df['average_price'] = (df['Low Price'] + df['High Price']) / 2
# 时间特征
df['Year']  = df['Date'].dt.year.astype(float)
df['Month'] = df['Date'].dt.month.astype(float)

# —— 基础特征矩阵 —— #
X_df = pd.concat([
    df[['Year','Month']],
    pd.get_dummies(df['Variety'], prefix='Var')
], axis=1)
X_df = X_df.astype(float)        # **全表转 float**
X = X_df.values                  # (n_samples, n_features)
y = df['average_price'].astype(float).values  # (n_samples,)

# —— 多项式特征生成（degree=2） —— #
def polynomial_features(X):
    n, m = X.shape
    X_sq = X**2
    crosses = []
    for i in range(m):
        for j in range(i+1, m):
            crosses.append((X[:, i] * X[:, j]).reshape(n,1))
    X_cross = np.hstack(crosses) if crosses else np.empty((n,0))
    return np.hstack([X, X_sq, X_cross])

# —— 时间序列交叉验证生成器 —— #
def time_series_cv(n_samples, n_splits=5):
    fold = n_samples // (n_splits + 1)
    for i in range(1, n_splits+1):
        train_end = fold * i
        test_end  = fold * (i+1)
        yield np.arange(0, train_end), np.arange(train_end, test_end)

n = len(y)

# —— 1. 多项式回归 CV —— #
X_poly = polynomial_features(X)
rmses_poly = []
for train_idx, test_idx in time_series_cv(n, 5):
    # 构建设计矩阵：截距 + 特征
    X_train = np.hstack([np.ones((len(train_idx),1)), X_poly[train_idx]]).astype(float)
    y_train = y[train_idx]
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

    X_test  = np.hstack([np.ones((len(test_idx),1)), X_poly[test_idx]]).astype(float)
    y_test  = y[test_idx]
    y_pred  = X_test.dot(beta)

    rmses_poly.append(np.sqrt(np.mean((y_test - y_pred)**2)))

print("多项式回归 交叉验证 RMSE:", np.mean(rmses_poly))

# —— 2. 线性回归 CV —— #
rmses_lin = []
for train_idx, test_idx in time_series_cv(n, 5):
    X_train = np.hstack([np.ones((len(train_idx),1)), X[train_idx]]).astype(float)
    y_train = y[train_idx]
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

    X_test  = np.hstack([np.ones((len(test_idx),1)), X[test_idx]]).astype(float)
    y_test  = y[test_idx]
    y_pred  = X_test.dot(beta)

    rmses_lin.append(np.sqrt(np.mean((y_test - y_pred)**2)))

print("线性回归 交叉验证 RMSE:", np.mean(rmses_lin))
