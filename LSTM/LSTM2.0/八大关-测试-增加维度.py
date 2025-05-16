from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import os
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取模型
model = tf.keras.models.load_model('八大关.keras')
print(model.summary())

# 获取真实值文件
xlsx_file = pd.ExcelFile('7.30预测.xls')
sheet_names = xlsx_file.sheet_names
data = pd.DataFrame()
for sheet in sheet_names:
    df = xlsx_file.parse(sheet)
    data = pd.concat([data, df])

# 得到5月9日起,八大关的臭氧数据
data["TO_CHAR(H.C1705_DATETIME,'YYYY"] = pd.to_datetime(
    data["TO_CHAR(H.C1705_DATETIME,'YYYY"])
data = data[data["TO_CHAR(H.C1705_DATETIME,'YYYY"] >= "2024-05-09 00:00:00"]
data_values = data[data['C0007_PNAME'] == '八大关街道']
data_values = data_values.iloc[:, [2, 8]]
data_values = data_values.interpolate()  # 插值
ori_data = data_values


def calculate_wind_speed(U, V):
    # 计算风速

    return np.sqrt(U**2 + V**2)


def calculate_wind_direction(U, V):
    # 计算风向

    # 确保U和V没有缺失值
    U = np.nan_to_num(U)
    V = np.nan_to_num(V)

    direction = np.arctan2(U, V) * (180 / np.pi)  # 转换为角度
    direction = np.mod(direction + 360, 360)  # 确保风向在0到360度之间

    return direction


def sin_transformer(period):
    # 计算sin值

    return FunctionTransformer(lambda x: np.sin(2 * np.pi * x / period))


def cos_transformer(period):
    # 计算cos值

    return FunctionTransformer(lambda x: np.cos(2 * np.pi * x / period))


# 获取特征文件
folder_path = "./wrfout/"
files = os.listdir(folder_path)
columns_to_convert = ['U10', 'V10', '气温开尔文', 'YALIVAL', 'SHIDUVAL']
features_list = []
date_format = "%Y-%m-%d_%H_%M_%S"

for file in files:

    # 获取文件
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path, delimiter='\s+', header=None, engine='python', skiprows=1,
                       usecols=[0, 1, 2, 3, 5, 6, 7],
                       names=['C1705_DATETIME', '站点', 'U10', 'V10', '气温开尔文', 'YALIVAL', 'SHIDUVAL'])

    # 格式化
    data = data[data['站点'] == 'badaguan']

    for column in columns_to_convert:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        data['FSVAL'] = calculate_wind_speed(data['U10'], data['V10'])
        data['FXVAL'] = calculate_wind_direction(data['U10'], data['V10'])

    data = data[data["C1705_DATETIME"] <= "2024-06-30_23_00_00"]
    for i in range(data.shape[0]):
        ab = data.iloc[i, 0][0:13]
        data.iloc[i, 0] = ab[0:10]+" "+ab[11:13]

    data["C1705_DATETIME"] = pd.to_datetime(data["C1705_DATETIME"])
    data['QIWENVAL'] = data['气温开尔文'] - 273.15

    # 插值
    columns_to_interpolate = ['O3VAL', 'QIWENVAL',
                              'FSVAL', 'FXVAL', 'SHIDUVAL', 'YALIVAL']
    for column in columns_to_interpolate:
        if column in data.columns:
            data[column] = data[column].interpolate(method='linear')

    # 构建特征数据
    features = data.iloc[:, [0, 9, 7, 8, 6, 5]].reset_index()

    # 新维度
    features['month'] = features.C1705_DATETIME.dt.month
    features['day'] = features.C1705_DATETIME.dt.day
    features['hour'] = features.C1705_DATETIME.dt.hour

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
        features[['month', 'day', 'hour']])

    X_transformed_df = pd.DataFrame(X_transformed, columns=[
                                    'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos'])

    features = pd.concat(
        [features.reset_index(), X_transformed_df.reset_index()], axis=1)

    features = features.iloc[:, 2:].values
    features_list.append(features)


result_list = []
for feature in features_list:
    # 开始预测

    this_time = feature[:, 0]
    this_time = this_time.reshape(-1, 1)
    feature = feature[:, 1:]

    # 对数据进行适当的重塑以适应模型输入
    timesteps = feature.shape[1]
    feature = feature.reshape(feature.shape[0], timesteps, -1)
    print("测试集特征形状:", feature.shape)

    # 转换数据类型
    feature = feature.astype(np.float64)

    # 预测
    result = model.predict(feature)
    result_list.append(np.hstack((this_time, result)))


for i in range(8):
    # 开始绘图

    result = result_list[i]

    # 筛选本轮真实值
    beg_time = result[0, 0]
    end_time = pd.to_datetime("2024-06-30 23:00:00")

    y1 = ori_data[ori_data["TO_CHAR(H.C1705_DATETIME,'YYYY"] >= beg_time]
    y2 = ori_data[ori_data["TO_CHAR(H.C1705_DATETIME,'YYYY"] <= end_time]
    y = pd.merge(y1, y2, how='inner')
    y = y.to_numpy()

    # 合并本轮预测值
    ans = result
    k = 1
    while(i+k*8 < 53):
        tem = result_list[i+k*8]
        ans = np.r_[ans, tem]
        k = k+1

        if(result[-1, 0] == pd.to_datetime("2024-06-30 23:00:00")):
            break

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.title("青岛市八大关街道 臭氧浓度 预测-真实比较 pred-%d(每1小时为单位)" % i)
    plt.plot(ans[:, 0], y[:, 1], label='True')
    plt.plot(ans[:, 0], ans[:, 1], label='Predictions')
    plt.xlabel('时间')
    plt.ylabel('臭氧浓度值')
    plt.show

    r = r2_score(y[:, 1], ans[:, 1])
    print("R^2 = %f" % r)

    # 输出到文件
    with open('八大关-预测数据-pred-%d.txt' % i, 'w') as f:

        k = 1
        for line in ans:
            f.write("%s," % (line[1]))
            
            k = k+1
            if(k % 100 == 0):
                f.write("\n")

    with open('八大关-预测时间-pred-%d.txt' % i, 'w') as f:

        k = 1
        for line in ans:
            f.write('"%s",' % (line[0]))
                        
            k = k+1
            if(k % 100 == 0):
                f.write("\n")
