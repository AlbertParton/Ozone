import math

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

# f = open('data0.csv', 'rb')
# encoding = chardet.detect(f.read())['encoding']
# f.close()
#
# print(encoding)

data = pd.read_excel('7.30预测.xls').dropna()
# data = pd.read_excel('data1.xls').dropna()
# print(data.columns)

data = data[data['C0007_PNAME'] == '兴城路街道']
# print(data['QIWENVAL'])

# data = data[['TO_CHAR(H.C1705_DATETIME,\'YYYY', 'O3']]
# arr = np.array(data)
# print(arr[:,1])

xlsx_file = pd.ExcelFile('./2021-2022_O3&气象五参.xlsx')
sheet_names = xlsx_file.sheet_names
# data = pd.DataFrame()

# 得到2021和2022年的数据
df1 = xlsx_file.parse(sheet_names[0])
df2 = xlsx_file.parse(sheet_names[1])
# data = pd.concat([df1, df2]).dropna()
# data = data.reset_index(drop=True)

# cols = ['TO_CHAR(H.C1705_DATETIME,\'YYYY', 'O3', 'QIWENVAL', 'SHIDUVAL', 'YALIVAL', 'FSVAL']
cols = ['C1705_DATETIME', 'O3VAL', 'QIWENVAL', 'SHIDUVAL', 'YALIVAL', 'FSVAL']
# for i,item in zip(range(len(data['O3VAL'])), data['O3VAL']):
#     if i % 24 != 0:
#         continue
#     # print('\'',item,'\'',sep='',end=', ')
#     # print(item,end = ', ')
#     print(round(item - 1000, 2), end=', ')

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
dt = np.array(data[cols[1:]])
file = open('tmp.txt', 'w')
for i in dt:
    s = str('['+str(i[0])+','+str(i[1])+','+str(i[2])+','+str(round(i[3]-1000, 2))+','+str(i[4])+']')+',\n'
    file.write(s)
file.close()

# dt = np.matrix(data[cols[1:]])
# corr = np.corrcoef(dt.T)
# for i in range(len(corr)):
#     for j in range(len(corr)):
#         print('[', i, ',', j, ',', np.round(corr[i][j],2), ']',sep='', end=', ')
# print('')
# print(cols[1:])


exit(0)

class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)
        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y

    """
    PyCharm Crtl+click nn.LSTM() jump to code of PyTorch:
    Examples::
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)     # 5个时间步，也就是每个时间序列的长度是5,3表示一共有3个时间序列，10表示每个序列在每个时间步的维度是10
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """

    def output_y_hc(self, x, hc):

        y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, hc



def minmaxscaler(x):
    minx = np.amin(x)
    maxx = np.amax(x)
    return (x - minx)/(maxx - minx), (minx, maxx)

def preminmaxscaler(x, minx, maxx):
    return (x - minx)/(maxx - minx)

def unminmaxscaler(x, minx, maxx):
    return x * (maxx - minx) + minx




bchain = np.array(
        [112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
         118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
         114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
         162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
         209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
         272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
         302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
         315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
         318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
         348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
         362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
         342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
         417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
         432.], dtype=np.float32)
bchain = np.array(data['O3'], dtype=np.float32)
bchain = bchain[:, np.newaxis]

inp_dim = 1
out_dim = 1
mid_dim = 18
mid_layers = 4
data_x = bchain[:-1, :]
data_y = bchain[+1:, :]
epoch_cnt = 100
# data_x shape：(143, 1)
# data_y shape：(143, 1)

# train_size = 113
train_size = int(len(bchain) * 34/40)
train_x = data_x[:train_size, :]
train_y = data_y[:train_size, :]
# train_x shape: (113, 1)
# train_y shape: (113, 1)

# 预处理数据  归一化
train_x, train_x_minmax = minmaxscaler(train_x)
train_y, train_y_minmax = minmaxscaler(train_y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 第一种操作，直接把batch_x batch_y这一个序列扔进去
# batch_x = train_x[:, np.newaxis, :]
# batch_y = train_y[:, np.newaxis, :]
# batch_x = torch.tensor(batch_x, dtype=torch.float32, device=device)
# batch_y = torch.tensor(batch_y, dtype=torch.float32, device=device)

# 第二种操作，用滑动窗口的方法构造数据集
train_x_tensor = torch.tensor(train_x, dtype=torch.float32, device=device)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32, device=device)
# 开始构造滑动窗口  40个为1个窗口，step为3
batch_x = list()
batch_y = list()

window_len = 30
for end in range(len(train_x_tensor), window_len, -3):
    batch_x.append(train_x_tensor[end-40:end])
    batch_y.append(train_y_tensor[end-40:end])

# batch_x的shape是(25, 40, 1)  25个时间序列，每个时间序列是40个时间步

from torch.nn.utils.rnn import pad_sequence
batch_x = pad_sequence(batch_x)
batch_y = pad_sequence(batch_y)

# batch_x的shape是(40, 25, 1)   输入模型的时候可以25个时间序列并行处理


# 加载模型
model = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 开始训练
print("Training......")
for e in range(epoch_cnt):
    out = model(batch_x)

    Loss = loss(out, batch_y)

    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print('\rEpoch: {:4}, Loss: {:.5f}'.format(e, Loss.item()), end='')
torch.save(model.state_dict(), './net.pth')
print("\nSave in:", './net.pth')


new_data_x = data_x.copy()
new_data_x[train_size:] = 0

test_len = 40

eval_size = 1
zero_ten = torch.zeros((mid_layers, eval_size, mid_dim), dtype=torch.float32, device=device)

for i in range(train_size, len(new_data_x)):  # 要预测的是i
    test_x = new_data_x[i-test_len:i, np.newaxis, :]
    test_x = preminmaxscaler(test_x, train_x_minmax[0], train_x_minmax[1])
    batch_test_x = torch.tensor(test_x, dtype=torch.float32, device=device)

    if i == train_size:
        test_y, hc = model.output_y_hc(batch_test_x, (zero_ten, zero_ten))
    else:
        test_y, hc = model.output_y_hc(batch_test_x[-2:], hc)
    test_y = model(batch_test_x)
    predict_y = test_y[-1].item()
    predict_y = unminmaxscaler(predict_y, train_x_minmax[0], train_y_minmax[1])
    new_data_x[i] = predict_y


plt.plot(new_data_x, 'r', label='pred')
plt.plot(data_x, 'b', label='real', alpha=0.3)
plt.legend(loc='best')
plt.show()