import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.tools import StandardScaler
from sklearn.metrics import r2_score

df_raw = pd.read_csv('./data/ETT/data_Beizhai.csv')
scaler = StandardScaler()
scaler.fit(df_raw[['O3VAL']].values)

folders = [
'informer_custom_ftM_sl168_ll96_pl168_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1',
]
dates = pd.date_range(start='2024-04-01 00:00:00', end='2024-06-30 22:00:00', freq='H')
print(len(dates))

dates = pd.date_range(start='2024-04-08 00:00:00', end='2024-06-30 00:00:00', freq='H')
f = open('dates730.txt', 'w', encoding='utf-8')
for i in dates[24*7:1824]:
    f.write(f'\'{i}\',')
f.close()
# exit(0)

for _ in range(1):
    folder = folders[_]
    pred = np.load('./results/' + folder + '/pred.npy')
    true = np.load('./results/' + folder + '/true.npy')

    pred = scaler.inverse_transform(pred)
    true = scaler.inverse_transform(true)

    print(pred.shape)
    print(true.shape)

    true_all = []
    pred_all = []
    pred_all_n = [[] for i in range(8)]
    print(pred.shape[0])
    for i in range(0, pred.shape[0], 24*7):
        print(i)
        true_all.extend(list(true[i, :, -1]))
        pred_all.extend(list(pred[i, :, -1]))
        print(f'r2: {r2_score(list(true[i, :, -1]), list(pred[i, :, -1]))}')
        # plt.figure()
        # plt.plot(true[i, :, -1], label='true')
        # plt.plot(pred[i, :, -1], label='pred')
        # plt.legend()
        # plt.show()

    for i in range(0, pred.shape[0], 1):
        pred_all_n[0].extend(list(pred[i, :, -1])[:1])

    for i in range(0, pred.shape[0], 24):
        for j in range(1,8):
            if j>(i/24):
                pred_all_n[j].extend([0 for _ in range(24)])
            else:
                pred_all_n[j].extend(list(pred[i, :, -1])[24*(j-1):24*j])

    plt.figure()
    plt.plot(true_all[24*7:], label='GroundTruth')
    for i in range(3):
        plt.plot(pred_all_n[i][24*7:], label=f'Prediction n-{i}')
    plt.legend()

    mean_pred = np.mean(np.array(pred_all))
    for i in range(8):
        r2 = r2_score(true_all[:1824], pred_all_n[i])
        print(i, r2)

        f = open(f'pred n-{i}.txt', 'w')
        for i in pred_all_n[i][24*7:1824]:
            f.write(f'{np.round(i, 3)},')
        f.close()

    r2 = r2_score(true_all, pred_all)
    print(r2)

    f = open('true.txt', 'w')
    for i in true_all[24*7:1824]:
        f.write(f'{int(np.round(i))},')
    f.close()

plt.show()