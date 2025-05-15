import math
import chardet
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from datetime import datetime

from datetime import datetime





# data = pd.read_csv('data2_Badaguan.csv')
# data2 = pd.read_csv('O3his2_Badaguan.csv')
# print(len(data.columns), len(data2.columns))

# drop_cols = []
# for col in data.columns:
#     if col in ['date']:
#         continue
#     if data[col].dtype in ['float64', 'bool', 'int64']:
#         continue
#     print(col, data[col].dtype)
#     drop_cols.append(col)
# data = data.drop(columns=drop_cols)
# data2 = data2.drop(columns=drop_cols)

# for col in data.columns:
#     data[col] = data[col].replace(0,0.01)
#
#
#
# data.to_csv('data2_Badaguan.csv', index_label=False, index=False)
# data2.to_csv('O3his2_Badaguan.csv', index_label=False, index=False)
# exit(0)



# f = open('featureImportance.txt', 'r')
# res = []
# for line in f.readlines():
#     # line = line[4:]
#     s1,s2 = line.split(' ')
#     s21,s22 = s2.split(']')
#     s21='\''+s21+'\''
#     s = s1+' '+s21+'],\n'
#     res.append(s)
# f.close()
# f = open('featureImportance.txt', 'w')
# for i in res:
#     f.write(i)
# f.close()

# f = open('BiClsfDatahis_All.csv', 'rb')
# encoding = chardet.detect(f.read())['encoding']
# print(encoding)
# f.close()
# encoding = 'GB2312'
# data = pd.read_csv('BiClsfDatahis_All.csv', encoding = encoding)
# data = data.loc[:, ~data.columns.duplicated()]
# data = data.reset_index(drop=True)
# data['C1705_DATETIME'] = pd.to_datetime(data['C1705_DATETIME'], errors='coerce')
# data['TimeStr'] = data['C1705_DATETIME'].dt.strftime('%Y/%m/%d %H:%M:%S').str[:].astype(int)
# pd.DataFrame(data=data).to_csv('BiClsfDatahis_All_TimeStr.csv')
# exit(0)

# data = pd.read_csv('data_Badaguan.csv')

# def convert_date_to_int(date_str):
#     # date_obj = datetime.strptime(date_str, '%Y-%m-%d %H')
#     date_obj = datetime.strptime(date_str, '%Y/%m/%d %H:%M')
#     formatted_date = date_obj.strftime('%Y%m%H')
#     return int(formatted_date)
# def convert_hour_to_int(date_str):
#     # date_obj = datetime.strptime(date_str, '%Y-%m-%d %H')
#     date_obj = datetime.strptime(date_str, '%Y/%m/%d %H:%M')
#     formatted_date = date_obj.strftime('%Y%m%H')
#     return int(formatted_date)%100
#
# data['dateInt'] = data['date'].apply(convert_date_to_int)
# data['hour'] = data['date'].apply(convert_hour_to_int)
# pd.DataFrame(data=data).to_csv('O3his_Badaguan.csv')

# columns_to_interpolate = ['O3VAL', 'QIWENVAL', 'FSVAL', 'FXVAL', 'SHIDUVAL', 'YALIVAL']
# for column in columns_to_interpolate:
#         if column in data.columns:
#             data[column] = data[column].interpolate(method='linear')
#
# def daypart(hour):
#     if hour in [2,3,4,5]:
#         return "dawn"
#     elif hour in [6,7,8,9]:
#         return "morning"
#     elif hour in [10,11,12,13]:
#         return "noon"
#     elif hour in [14,15,16,17]:
#         return "afternoon"
#     elif hour in [18,19,20,21]:
#         return "evening"
#     else: return "midnight"
#
# def is_weekend_series(series):
#     def is_weekday(date):
#         weekday = date.weekday()
#         return weekday >= 5
#
#     return series.apply(is_weekday)
#
#
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
import numpy as np
from statsmodels.tsa.stl._stl import STL

#
#
# def sin_transformer(period):
#     return FunctionTransformer(lambda x: np.sin(2 * np.pi * x / period))
#
#
# def cos_transformer(period):
#     return FunctionTransformer(lambda x: np.cos(2 * np.pi * x / period))
#
#
# cyclic_cossin_transformer = ColumnTransformer(
#     transformers=[
#         ("month_sin", sin_transformer(12), ["month"]),
#         ("month_cos", cos_transformer(12), ["month"]),
#         ("day_sin", sin_transformer(7), ["day"]),
#         ("day_cos", cos_transformer(7), ["day"]),
#         ("hour_sin", sin_transformer(24), ["hour"]),
#         ("hour_cos", cos_transformer(24), ["hour"]),
#     ],
#     remainder=MinMaxScaler(),
# )
#
# holidays_2024 = [
#     "2024-01-01 00:00:00",  # 元旦
#     "2024-02-10 00:00:00",  # 除夕
#     "2024-02-11 00:00:00",  # 春节（初一）
#     "2024-02-12 00:00:00",  # 春节（初二）
#     "2024-02-13 00:00:00",  # 春节（初三）
#     "2024-02-14 00:00:00",  # 春节（初四）
#     "2024-02-15 00:00:00",  # 春节（初五）
#     "2024-02-16 00:00:00",  # 春节（初六）
#     "2024-04-04 00:00:00",  # 清明节
#     "2024-05-01 00:00:00",  # 劳动节
#     "2024-06-10 00:00:00",  # 端午节
#     "2024-09-17 00:00:00",  # 中秋节
#     "2024-10-01 00:00:00",  # 国庆节
#     "2024-10-02 00:00:00",  # 国庆节
#     "2024-10-03 00:00:00",  # 国庆节
#     "2024-10-04 00:00:00",  # 国庆节
#     "2024-10-05 00:00:00",  # 国庆节
#     "2024-10-06 00:00:00",  # 国庆节
#     "2024-10-07 00:00:00"  # 国庆节
# ]
#
# data['date'] = pd.to_datetime(data['date'])
# data['year'] = data.date.dt.year
# data['month'] = data.date.dt.month
# data['day'] = data.date.dt.day
# data['hour'] = data.date.dt.hour
# X_transformed = cyclic_cossin_transformer.fit_transform(data[['month', 'day', 'hour']])
# X_transformed_df = pd.DataFrame(X_transformed,
#                                 columns=['month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos'])
# data = pd.concat([data, X_transformed_df], axis=1)
#
# data['quarter'] = data['date'].dt.quarter
# data['hour_section'] = (data['hour'] // 6)
# data['diff'] = data['date'] - pd.Timestamp('2021-01-01')
# data['diff1'] = data['date'] - pd.Timestamp('2022-01-01')
# half_point = len(data) // 2
# data['diff_hours'] = np.where(data.index < half_point, data['diff'].values.astype("timedelta64[h]").astype('int'),
#                               data['diff1'].values.astype("timedelta64[h]").astype('int'))
# # print(data['diff_hours'])
#
# data['is_weekend'] = is_weekend_series(data.date)
# data['day_of_year'] = data.date.dt.dayofyear
# data['day_of_week'] = data.date.dt.dayofweek
# data['is_year_start'] = data['date'].dt.is_year_start
# data['is_year_end'] = data['date'].dt.is_year_end
# data['is_quarter_start'] = data['date'].dt.is_quarter_start
# data['is_quarter_end'] = data['date'].dt.is_quarter_end
# data['is_month_start'] = data['date'].dt.is_month_start
# data['is_month_end'] = data['date'].dt.is_month_end
# data['day_to_year_start'] = (data['date'] - pd.to_datetime(data['date'].dt.year,
#                                                                      format='%Y')) / pd.Timedelta(days=1)
# data['is_holiday'] = data['date'].apply(lambda x: 1 if x in holidays_2024 else 0)
# day_of_week = data['date'].dt.dayofweek
# data['day_to_weekend'] = (6 - day_of_week) % 7
#
# to_one_hot = data['date'].dt.day_name()
# days = pd.get_dummies(to_one_hot)
# for column in days.columns:
#     data[column] = days[column]
#
# mask = data['O3VAL'] < 0  # 创建一个布尔掩码，表示负值
# data.loc[mask, 'O3VAL'] *= -1  # 将负值取反
# data1 = data
#
# import numpy as np
# import pandas as pd
# from statsmodels.tsa.seasonal import STL
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
#
# # 补全日期缺失值
# data = pd.read_csv('data_Beizhai.csv')
# # data = pd.read_csv('O3his_Beizhai.csv')
# data['date'] = pd.to_datetime(data['date'])
#
# # dates = pd.date_range(start='2021-01-01 00:00:00', end='2022-12-31 23:00:00', freq='H')
# dates = pd.date_range(start='2024-04-01 00:00:00', end='2024-6-30 22:00:00', freq='H')
# result = pd.DataFrame()
# if len(data['date']) != len(dates):
#     print("fill date")
#     # 创建一个包含所有日期的新 DataFrame
#     new_data = pd.DataFrame({'date': dates})
#     # 合并原始 DataFrame 和新 DataFrame
#     result = new_data.merge(data, on='date', how='left')
#     result = result.loc[:, ~result.columns.duplicated()]
# else:
#     result = data
#     result = result.loc[:, ~result.columns.duplicated()]
# # result.to_csv('tmp.csv')
# # exit(0)
# # 用 0 填充缺失值
# # result.fillna(0, inplace=True)
# # 计算缺失值的上下相邻平均值
# for col in result.columns[1:]:
#     print(col)
#     # print(result[col])
#     for i in range(len(result)):
#         # print(i)
#         # print(result.loc[i, col])
#         if pd.isna(result.loc[i, col]):
#             if i == 0:
#                 result.loc[i, col] = result.loc[i + 1, col]
#             else:
#                 result.loc[i, col] = result.loc[i - 1, col]
#         # print(2)
# data1 = result
#
# ts = pd.Series(scaler.fit_transform(data1['O3VAL'].values.reshape(-1, 1)).flatten(), index=dates)
# # 进行STL分解
# stl = STL(ts)  # seasonal参数表示季节性周期，这里设定为13表示大约一个季度
# result = stl.fit()
#
# # 获取分解后的趋势、季节性和残差
# trend = result.trend.values
# seasonal = result.seasonal.values
# residual = result.resid.values
#
# trend_scaled = scaler.fit_transform(np.array(trend.reshape(-1, 1)))
# seasonal_scaled = scaler.fit_transform(np.array(seasonal.reshape(-1, 1)))
#
# data1['Trend'] = trend_scaled
# data1['Seasonal'] = seasonal_scaled
#
# def add_columns(df):
#     for i in range(1, 11):  # Loop through 1 to 24
#         new_col_name = f"O3VAL_{i}"  # New column name based on index
#         df[new_col_name] = df['O3VAL'].shift(i)
#
# add_columns(data1)
# data1.fillna(0, inplace=True)
#
# drop_cols = []
# for col in data1.columns:
#     if col in ['date']:
#         continue
#     if data1[col].dtype in ['float64', 'bool', 'int64']:
#         continue
#     print(col, data1[col].dtype)
#     drop_cols.append(col)
# data1 = data1.drop(columns=drop_cols)
#
# # data1.to_csv('data1_Beizhai.csv', index_label=False, index=False)
# data1.to_csv('O3his1_Beizhai.csv', index_label=False, index=False)
#
#
# exit(0)














# f = open('data0.csv', 'rb')
# encoding = chardet.detect(f.read())['encoding']
# f.close()
#
# print(encoding)

# data = pd.read_excel('7.30预测.xls').dropna()
# data = pd.read_excel('data1.xls').dropna()
# print(data.columns)

# data = data[data['C0007_PNAME'] == '八大关街道']
# data.to_csv('data_Badaguan.csv')

# print(data['QIWENVAL'])

# data = data[['TO_CHAR(H.C1705_DATETIME,\'YYYY', 'O3']]
# arr = np.array(data)
# print(arr[:,1])

xlsx_file = pd.ExcelFile('./2021-2022_O3&气象五参.xlsx')
sheet_names = xlsx_file.sheet_names
data = pd.DataFrame()
# 得到2021和2022年的数据
df1 = xlsx_file.parse(sheet_names[0])
df2 = xlsx_file.parse(sheet_names[1])
data = pd.concat([df1, df2])

data = data[data['C0007_PNAME'] == '北宅街道']
data = data.loc[:, ~data.columns.duplicated()]
data = data.reset_index(drop=True)
def add_columns(df):
    for i in range(1, 11):  # Loop through 1 to 24
        new_col_name = f"O3VAL_{i}"  # New column name based on index
        df[new_col_name] = df['O3VAL'].shift(i)
add_columns(data)
data.fillna(0, inplace=True)

scaler = MinMaxScaler()
dates = pd.date_range(start='2021-01-01 00:00:00', end='2022-12-31 23:00:00', freq='H')
ts = pd.Series(scaler.fit_transform(data['O3VAL'].values.reshape(-1, 1)).flatten(), index=dates)
# 进行STL分解
stl = STL(ts)  # seasonal参数表示季节性周期，这里设定为13表示大约一个季度
result = stl.fit()

# 获取分解后的趋势、季节性和残差
trend = result.trend.values
seasonal = result.seasonal.values
residual = result.resid.values

trend_scaled = scaler.fit_transform(np.array(trend.reshape(-1, 1)))
seasonal_scaled = scaler.fit_transform(np.array(seasonal.reshape(-1, 1)))

data['Trend'] = trend_scaled
data['Seasonal'] = seasonal_scaled

data.to_csv('O3his_Beizhai.csv', index_label=False, index=False)


exit(0)


# data = pd.read_csv('datahis_hoursincos.csv')

# data = data[data['C0007_PNAME'] == '北宅街道']
data = data.loc[:, ~data.columns.duplicated()]
data = data.reset_index(drop=True)


pred_file = open('LSTM预测结果/八大关-预测结果.txt', 'r')
pred_lstm = []
for line in pred_file.readlines():
    print(line.split(' '))
    if len(line.split(' ')) < 2:
        continue
    y,ho3 = line.split(' ')
    if len(ho3.split('\t')) < 2:
        continue
    h,o3 = ho3.split('\t')
    o3 = o3[:-1]
    o3 = float(o3)
    pred_lstm.append(o3)
print(pred_lstm)
plt.figure()
plt.plot([i for i in range(len(pred_lstm))], pred_lstm)
plt.plot([i for i in range(len(np.array(data['O3VAL'])))], np.array(data['O3VAL']))
plt.show()
exit(0)

# # O3>=160二分类数据
# data['O3_over160'] = data['O3VAL'] >= 160
# pd.DataFrame(data=data).to_csv('BiClsfDatahis_All.csv')
# exit(0)





# 特征工程_sin_cos
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(2 * np.pi * x / period))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(2 * np.pi * x / period))

data['hour'] = data['C1705_DATETIME'].dt.strftime('%Y/%m/%d %H:%M:%S').str[11:13].astype(int)
# data['hour'] = data['TO_CHAR(H.C1705_DATETIME,\'YYYY'].str[11:13].astype(int)

cyclic_cossin_transformer = ColumnTransformer(
    transformers=[
        ("hour_sin", sin_transformer(24), ["hour"]),
        ("hour_cos", cos_transformer(24), ["hour"]),
    ],
    remainder=MinMaxScaler(),
)
print(11)
X_transformed = cyclic_cossin_transformer.fit_transform(data[['hour']])
X_transformed_df = pd.DataFrame(X_transformed,
                                columns=['hour_sin', 'hour_cos'])
datat = pd.concat([data, X_transformed_df], axis=1)
# pd.DataFrame(data=datat).to_csv('datahis_hoursincos.csv')
exit(0)


# STL分解数据
# columns_to_interpolate = ['O3VAL', 'QIWENVAL', 'FSVAL', 'FXVAL', 'SHIDUVAL', 'YALIVAL']
# for column in columns_to_interpolate:
#     if column in data.columns:
#         data[column] = data[column].interpolate(method='linear')
#
# def daypart(hour):
#     if hour in [2, 3, 4, 5]:
#         return "dawn"
#     elif hour in [6, 7, 8, 9]:
#         return "morning"
#     elif hour in [10, 11, 12, 13]:
#         return "noon"
#     elif hour in [14, 15, 16, 17]:
#         return "afternoon"
#     elif hour in [18, 19, 20, 21]:
#         return "evening"
#     else:
#         return "midnight"
#
#
# data['C1705_DATETIME'] = pd.to_datetime(data['C1705_DATETIME'])
# data['year'] = data.C1705_DATETIME.dt.year
# data['month'] = data.C1705_DATETIME.dt.month
# data['day'] = data.C1705_DATETIME.dt.day
# data['hour'] = data.C1705_DATETIME.dt.hour
# # data['weekday'] = data.C1705_DATETIME.dt.weekday
# # data['day_of_Week'] = data.C1705_DATETIME.dt.dayofweek
# to_one_hot = data['C1705_DATETIME'].dt.day_name()
# # second: one hot encode to 7 columns
# days = pd.get_dummies(to_one_hot)
# # display data
# for column in days.columns:
#     data[column] = days[column]
#
#
# raw_dayparts = data['hour'].apply(daypart)
# # one hot encoding
# dayparts = pd.get_dummies(raw_dayparts)
# for column in dayparts.columns:
#     data[column] = dayparts[column]
#
# mask = data['O3VAL'] < 0  # 创建一个布尔掩码，表示负值
# data.loc[mask, 'O3VAL'] *= -1  # 将负值取反
# data1 = data[data['C0007_PNAME'] == '兴城路街道']
# data22 = data[data['C0007_PNAME'] == '北宅街道']
# data33 = data[data['C0007_PNAME'] == '登州路街道']
# data44 = data[data['C0007_PNAME'] == '上马街道']
# data55 = data[data['C0007_PNAME'] == '兴城路街道']
#
# import numpy as np
# import pandas as pd
# from statsmodels.tsa.seasonal import STL
# from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
#
# scaler = MinMaxScaler()
#
# # ------归一化数据---------------
#
# # data1['O3VAL'] = scaler.fit_transform(np.array(data1['O3VAL'].values.reshape(-1, 1)))
# data1['QIWENVAL'] = scaler.fit_transform(np.array(data1['QIWENVAL'].values.reshape(-1, 1)))
# data1['FSVAL'] = scaler.fit_transform(np.array(data1['FSVAL'].values.reshape(-1, 1)))
# data1['FXVAL'] = scaler.fit_transform(np.array(data1['FXVAL'].values.reshape(-1, 1)))
# data1['SHIDUVAL'] = scaler.fit_transform(np.array(data1['SHIDUVAL'].values.reshape(-1, 1)))
# data1['YALIVAL'] = scaler.fit_transform(np.array(data1['YALIVAL'].values.reshape(-1, 1)))
#
# dates = pd.date_range(start='2021-01-01 00:00:00', end='2021-12-31 23:00:00', freq='H')
# ts = pd.Series(data1.loc[0:52559]['O3VAL'].values.reshape(-1, 1).flatten(), index=dates)
# # 进行STL分解
# stl = STL(ts)  # seasonal参数表示季节性周期，这里设定为13表示大约一个季度
# result = stl.fit()
#
#
# # 获取分解后的趋势、季节性和残差
# trend = result.trend.values
# seasonal = result.seasonal.values
# residual = result.resid.values
#
# f = open('tmp0.txt', 'w')
# f1 = open('tmp1.txt', 'w')
# f2 = open('tmp2.txt', 'w')
# f3 = open('tmp3.txt', 'w')
# cnt=0
# for i,j in zip(ts,trend):
#     i=round(i,3)
#     j=round(j,3)
#     s = f"{i},"
#     s1 = f"{j},"
#     cnt+=1
#     if cnt%100 == 0:
#         s += '\n'
#         s1 += '\n'
#     f.write(s)
#     f1.write(s1)
# for i,j in zip(seasonal,residual):
#     i=round(i,3)
#     j=round(j,3)
#     s2 = f"{i},"
#     s3 = f"{j},"
#     cnt+=1
#     if cnt%100 == 0:
#         s2 += '\n'
#         s3 += '\n'
#     f2.write(s2)
#     f3.write(s3)
# f.close()
# f1.close()
# f2.close()
# f3.close()
# exit(0)










# cols = ['TO_CHAR(H.C1705_DATETIME,\'YYYY', 'O3', 'QIWENVAL', 'SHIDUVAL', 'YALIVAL', 'FSVAL']
cols = ['C1705_DATETIME', 'O3VAL', 'QIWENVAL', 'SHIDUVAL', 'YALIVAL', 'FSVAL']

data = data[data['C0007_PNAME'] == '八大关街道']


# 滑动折线图数据
dt = np.array(data[cols[0:]])
file1 = open('tmp1.txt', 'w')
file2 = open('tmp2.txt', 'w')
file3 = open('tmp3.txt', 'w')
file4 = open('tmp4.txt', 'w')
file5 = open('tmp5.txt', 'w')
sum_val = np.array([0,0,0,0,0,0])
for _cnt,i in zip(range(1, len(dt)+1), dt):
    for now_id in range(1,6):
        sum_val[now_id] += i[now_id]
    if _cnt % 24 != 0:
        continue
    # s = '\'' + str(i[0])[:-9] + '\'' + ',\n' # timestamp
    s1 = str(round(sum_val[1]/24, 3)) + ',\n'  # O3
    s2 = str(round(sum_val[2]/24, 3)) + ',\n'  # 气温
    s3 = str(round(sum_val[3]/24, 3)) + ',\n'  # 湿度
    s4 = str(round(sum_val[4]/24 - 1000, 3)) + ',\n'  # 气压-1000
    s5 = str(round(sum_val[5]/24, 3)) + ',\n'  # 风速
    file1.write(s1)
    file2.write(s2)
    file3.write(s3)
    file4.write(s4)
    file5.write(s5)
    sum_val = np.array([0, 0, 0, 0, 0, 0])
file1.close()
file2.close()
file3.close()
file4.close()
file5.close()


# data = pd.DataFrame(df2).dropna()
# data = data[data['C0007_PNAME'] == '八大关街道']
# dt = np.array(data[cols[:2]])
# year = '2022'
# month = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14']
# month_id = 1
# val = 0
# cnt = 0
# timelist = []
# vallist = []
# for i in range(len(data['O3VAL'])):
#     timelimit = year + '-' + month[month_id+1] + '-01 00:00:00'
#     val += dt[i][1]
#     cnt += 1
#     if str(dt[i][0]) > timelimit:
#         if month_id >= 12:
#             break
#         timelist.append(str(month_id)+'月')
#         vallist.append(np.round(val/cnt, 2))
#         cnt = 0
#         val = 0
#         month_id += 1
# timelist.append(month[12] + '月')
# vallist.append(np.round(val/cnt, 2))
# # for i in timelist:
# #     print('\'', i, '\'', sep='', end=', ')
# print('')
# for i in vallist:
#     print(i,end=', ')

# 平行坐标系数据
# dt = np.array(data[cols[1:]])
# file = open('tmp.txt', 'w')
# for i in dt:
#     s = str('['+str(i[0])+','+str(i[1])+','+str(i[2])+','+str(round(i[3]-1000, 2))+','+str(i[4])+']')+',\n'
#     file.write(s)
# file.close()

# dt = np.matrix(data[cols[1:]])
# corr = np.corrcoef(dt.T)
# for i in range(len(corr)):
#     for j in range(len(corr)):
#         print('[', i, ',', j, ',', np.round(corr[i][j],2), ']',sep='', end=', ')
# print('')
# print(cols[1:])





