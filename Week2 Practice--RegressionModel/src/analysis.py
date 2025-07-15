import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 配置：路径自动定位 --- #
# 当前脚本所在目录
HERE = os.path.dirname(__file__)

# 数据文件（位于项目根的 data/US-pumpkins.csv）
csv_path = os.path.abspath(os.path.join(HERE, '..', 'data', 'US-pumpkins.csv'))

# 输出根目录（项目根的 output/）
base_out = os.path.abspath(os.path.join(HERE, '..', 'output'))

# 创建各子目录映射
dirs = {
    'time_series':     os.path.join(base_out, 'time_series'),
    'variety_boxplot': os.path.join(base_out, 'boxplots', 'variety'),
    'monthly_boxplot': os.path.join(base_out, 'boxplots', 'monthly'),
    'city_bar':        os.path.join(base_out, 'bar_charts', 'city'),
    'variety_trends':  os.path.join(base_out, 'line_charts', 'variety'),
    'models':          os.path.join(base_out, 'models'),
}
for path in dirs.values():
    os.makedirs(path, exist_ok=True)

# --- 数据加载与预处理 --- #
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
df['average_price'] = (df['Low Price'] + df['High Price']) / 2
df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
df['Month'] = df['Date'].dt.month

# --- 1. 月度平均价格趋势（时间序列） --- #
monthly_avg = df.groupby('YearMonth')['average_price'].mean()
plt.figure()
monthly_avg.plot()
plt.title('Monthly Average Price Trend')
plt.xlabel('Year-Month')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(dirs['time_series'], 'monthly_average_price_trend.png'))
plt.close()

# --- 2. 不同品种价格箱型图 --- #
plt.figure()
df.boxplot(column='average_price', by='Variety', rot=45)
plt.title('Price Distribution by Variety')
plt.suptitle('')
plt.xlabel('Variety')
plt.ylabel('Average Price')
plt.tight_layout()
plt.savefig(os.path.join(dirs['variety_boxplot'], 'variety_price_boxplot.png'))
plt.close()

# --- 3. 城市平均价格柱状图（前五） --- #
top_cities = df['City Name'].value_counts().nlargest(5).index
city_avg = df[df['City Name'].isin(top_cities)].groupby('City Name')['average_price'].mean()
plt.figure()
city_avg.plot(kind='bar')
plt.title('Average Price of Top 5 Cities by Sales Count')
plt.xlabel('City')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(dirs['city_bar'], 'top5_cities_avg_price.png'))
plt.close()

# --- 4. 月份价格分布箱型图（季节波动） --- #
plt.figure()
df.boxplot(column='average_price', by='Month')
plt.title('Monthly Price Distribution')
plt.suptitle('')
plt.xlabel('Month')
plt.ylabel('Average Price')
plt.tight_layout()
plt.savefig(os.path.join(dirs['monthly_boxplot'], 'monthly_price_boxplot.png'))
plt.close()

# --- 5. 品种随时间变化趋势（多折线） --- #
pivot = df.pivot_table(
    index='YearMonth',
    columns='Variety',
    values='average_price',
    aggfunc='mean'
).fillna(method='ffill')
plt.figure()
for variety in pivot.columns:
    plt.plot(pivot.index, pivot[variety], label=variety)
plt.title('Price Trends by Variety Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(dirs['variety_trends'], 'variety_trends_over_time.png'))
plt.close()

# --- 6. 回归建模（NumPy 最小二乘） --- #
# 构造特征
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
var_dummies = pd.get_dummies(df['Variety'], prefix='Var')
X = pd.concat([df[['Year', 'Month']], var_dummies], axis=1).values
y = df['average_price'].values

# 添加截距项并转为 float
X_design = np.hstack([np.ones((X.shape[0], 1)), X]).astype(float)
y = y.astype(float)

# 最小二乘求解
beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

# 预测并计算 RMSE
y_pred = X_design.dot(beta)
rmse = np.sqrt(np.mean((y - y_pred) ** 2))

# 保存 RMSE 至 models 目录
with open(os.path.join(dirs['models'], 'rmse.txt'), 'w') as f:
    f.write(f"RMSE: {rmse:.4f}\n")

print(f"RMSE (NumPy): {rmse:.4f}")
