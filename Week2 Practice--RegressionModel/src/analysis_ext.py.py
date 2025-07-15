import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# —— 屏蔽 FutureWarning —— #
warnings.filterwarnings('ignore', category=FutureWarning)

# —— 字体配置 —— #
font_path = r'C:\Windows\Fonts\simhei.ttf'  # 根据系统修改为可用中文字体路径
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# —— 路径设置 —— #
HERE = os.path.dirname(__file__)
csv_path = os.path.abspath(os.path.join(HERE, '..', 'data', 'US-pumpkins.csv'))
base_out = os.path.abspath(os.path.join(HERE, '..', 'output'))

dirs = {
    'time_series':     os.path.join(base_out, 'time_series'),
    'variety_boxplot': os.path.join(base_out, 'boxplots', 'variety'),
    'monthly_boxplot': os.path.join(base_out, 'boxplots', 'monthly'),
    'city_bar':        os.path.join(base_out, 'bar_charts', 'city'),
    'variety_trends':  os.path.join(base_out, 'line_charts', 'variety'),
    'models':          os.path.join(base_out, 'models'),
}
for p in dirs.values():
    os.makedirs(p, exist_ok=True)

# —— 数据加载与预处理 —— #
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
df['average_price'] = (df['Low Price'] + df['High Price']) / 2
df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# —— 回归数据准备 —— #
var_dummies = pd.get_dummies(df['Variety'], prefix='Var')
X = pd.concat([df[['Year', 'Month']], var_dummies], axis=1).values
y = df['average_price'].values
X_design = np.hstack([np.ones((X.shape[0], 1)), X]).astype(float)
y = y.astype(float)

# —— 线性回归拟合 —— #
beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
y_pred = X_design.dot(beta)
residuals = (y - y_pred)[np.isfinite(y - y_pred)]
rmse = np.sqrt(np.mean(residuals ** 2))
with open(os.path.join(dirs['models'], 'rmse.txt'), 'w', encoding='utf-8') as f:
    f.write(f"RMSE: {rmse:.4f}\n")
print(f"RMSE: {rmse:.4f}")

monthly_avg = df.groupby('YearMonth')['average_price'].mean()
plt.figure()
monthly_avg.plot()
plt.title('月度平均价格趋势')
plt.xlabel('年-月')
plt.ylabel('平均价格')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(dirs['time_series'], '月度平均价格趋势.png'))
plt.close()


plt.figure()
df.boxplot(column='average_price', by='Variety', rot=45)
plt.title('不同品种价格分布')
plt.suptitle('')
plt.xlabel('品种')
plt.ylabel('平均价格')
plt.tight_layout()
plt.savefig(os.path.join(dirs['variety_boxplot'], '不同品种价格分布.png'))
plt.close()

top_cities = df['City Name'].value_counts().nlargest(5).index
city_avg = df[df['City Name'].isin(top_cities)].groupby('City Name')['average_price'].mean()
plt.figure()
city_avg.plot(kind='bar')
plt.title('销量前五城市平均价格')
plt.xlabel('城市')
plt.ylabel('平均价格')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(dirs['city_bar'], '销量前五城市平均价格.png'))
plt.close()

plt.figure()
df.boxplot(column='average_price', by='Month')
plt.title('各月价格分布')
plt.suptitle('')
plt.xlabel('月份')
plt.ylabel('平均价格')
plt.tight_layout()
plt.savefig(os.path.join(dirs['monthly_boxplot'], '各月价格分布.png'))
plt.close()

pivot = df.pivot_table(
    index='YearMonth', columns='Variety',
    values='average_price', aggfunc='mean'
).ffill()
plt.figure()
for v in pivot.columns:
    plt.plot(pivot.index, pivot[v], label=v)
plt.title('各品种价格趋势')
plt.xlabel('年-月')
plt.ylabel('平均价格')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(dirs['variety_trends'], '各品种价格趋势.png'))
plt.close()

plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, alpha=0.5)
mx = max(np.nanmax(y), np.nanmax(y_pred))
plt.plot([0, mx], [0, mx], 'k--')
plt.title('实际值 vs 预测值')
plt.xlabel('实际平均价格 (元)')
plt.ylabel('预测平均价格 (元)')
plt.tight_layout()
plt.savefig(os.path.join(dirs['models'], '实际值_vs_预测值.png'))
plt.close()

plt.figure()
# 注意这里用 '-' 而不是 Unicode minus
sns.histplot(residuals, bins=30, kde=True)
plt.title('残差分布')
plt.xlabel('残差(实际-预测)')  # 用 ASCII '-' 替换 ‘−’
plt.ylabel('频数')
plt.tight_layout()
plt.savefig(os.path.join(dirs['models'], '残差分布.png'))
plt.close()

# …（后面部分保持不变）…


monthly_std = df.groupby('YearMonth')['average_price'].std()
plt.figure()
monthly_std.plot(marker='o')
plt.title('月度价格波动（标准差）')
plt.xlabel('年-月')
plt.ylabel('价格标准差')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(dirs['time_series'], '月度价格波动_标准差.png'))
plt.close()

# —— 9. 平均价格热力图_月份_品种 —— #
pivot_heat = df.pivot_table(
    index='Month', columns='Variety',
    values='average_price', aggfunc='mean'
)
plt.figure(figsize=(10,6))
sns.heatmap(pivot_heat, cmap='viridis', robust=True)
plt.title('按月份和品种的平均价格热力图')
plt.xlabel('品种')
plt.ylabel('月份')
plt.tight_layout()
plt.savefig(os.path.join(dirs['time_series'], '平均价格热力图_月份_品种.png'))
plt.close()
