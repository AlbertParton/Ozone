from sklearn.preprocessing import  MinMaxScaler
from sklearn.preprocessing import  FunctionTransformer
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import pandas as pd
import numpy as np

# 源数据
xlsx_file = pd.ExcelFile('./2021-2022_O3&气象五参.xlsx')
sheet_names = xlsx_file.sheet_names  # 提取三个表名
data = pd.DataFrame()

# 得到2021和2022年的数据
for sheet in sheet_names[0:2]:
    df = xlsx_file.parse(sheet)
    data = pd.concat([data, df])

# 得到北宅的数据
data['C1705_DATETIME'] = pd.to_datetime(data['C1705_DATETIME'])
data = data[data['C0007_PNAME'] == '北宅街道']

columns_to_interpolate = ['O3VAL', 'QIWENVAL',
                          'FSVAL', 'FXVAL', 'SHIDUVAL', 'YALIVAL']
for column in columns_to_interpolate:
    if column in data.columns:
        #线性填充缺失值
        data[column] = data[column].interpolate(method='linear')
        
# 新维度
data['month'] = data.C1705_DATETIME.dt.month
data['day'] = data.C1705_DATETIME.dt.day
data['hour'] = data.C1705_DATETIME.dt.hour


def sin_transformer(period):
    # 计算sin值

    return FunctionTransformer(lambda x: np.sin(2 * np.pi * x / period))


def cos_transformer(period):
    # 计算cos值

    return FunctionTransformer(lambda x: np.cos(2 * np.pi * x / period))


# 定义一个时间转换实例
cyclic_cossin_transformer = ColumnTransformer(
    transformers=[
        ("month_sin", sin_transformer(12), ["month"]),
        ("month_cos", cos_transformer(12), ["month"]),
        ("day_sin", sin_transformer(7), ["day"]),
        ("day_cos", cos_transformer(7), ["day"]),
        ("hour_sin", sin_transformer(24), ["hour"]),
        ("hour_cos", cos_transformer(24), ["hour"]),
    ],
    remainder=MinMaxScaler(),
)

# 时间数据离散化表达
X_transformed = cyclic_cossin_transformer.fit_transform(
    data[['month', 'day', 'hour']])

X_transformed_df = pd.DataFrame(X_transformed, columns=[
                                'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos'])

data = pd.concat([data.reset_index(), X_transformed_df.reset_index()], axis=1)

data_values = data.values

# 提取特征和目标值
features = data_values[:, 4:]
target = data_values[:, 3]
target = target.astype((float))
target = np.reshape(target, (-1, 1))

train_X = features
train_y = target

# 由于是LSTM模型，需要对数据进行适当的重塑以适应模型输入
# 假设每个样本为一个时间序列，序列长度固定或已知
timesteps = train_X.shape[1]  # 假定第一个维度是样本数，第二个维度是时间序列的长度
train_X_reshaped = train_X.reshape(train_X.shape[0], timesteps, -1)

print("训练集特征形状:", train_X_reshaped.shape)
print("训练集目标形状:", train_y.shape)

# 转换数据类型
train_X_reshaped = train_X_reshaped.astype(np.float64)

# 构建LSTM模型
model = Sequential()  # 建立模型容器
model.add(LSTM(units=64, input_shape=(
    train_X_reshaped.shape[1], train_X_reshaped.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 查看模型结构
print(model.summary())

# 训练模型
model.fit(train_X_reshaped, train_y, epochs=600, batch_size=100, verbose=1)

model.save('北宅.keras')