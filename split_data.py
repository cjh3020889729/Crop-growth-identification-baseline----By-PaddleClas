import pandas as pd
import numpy as np
import os
from tqdm import tqdm

## 划分比例，可调整
split_rate = 0.8


train_df = pd.read_csv('PaddleClas/dataset/agriculture/train.csv')
train_data = np.asarray(train_df)
np.random.shuffle(train_data)
print('Max Class Id: ', train_data[:,1].max())
print('Class Ids: ', set(train_data[:,1]))
print('Sample x data:\n', train_data[:3])


train_d = train_data[:int(split_rate * len(train_data))]
eval_d = train_data[int(split_rate * len(train_data)):]
print('Train length: ', len(train_d))
print('Eval length: ', len(eval_d))
print('Total length: ', len(train_data))

# 训练集
# 保存train_list.txt
with open('PaddleClas/dataset/agriculture/train_list.txt', 'w+') as f:
    train_tq = tqdm(enumerate(train_d))
    train_tq.set_description_str('-Train Data Saved ({0}/{1})-'.format(0, len(train_d)))
    for ids, i in train_tq:
        if ids + 1 < len(train_d):
            f.write('{0} {1}\n'.format(i[0], i[1]))
        else:
            f.write('{0} {1}'.format(i[0], i[1]))

        train_tq.set_description_str('-Train Data Saved ({0}/{1})-'.format(ids+1, len(train_d)))

# 验证集
# 保存val_list.txt
with open('PaddleClas/dataset/agriculture/val_list.txt', 'w+') as f:
    eval_tq = tqdm(enumerate(eval_d))
    eval_tq.set_description_str('-Train Data Saved ({0}/{1})-'.format(0, len(eval_d)))
    for ids, i in eval_tq:
        if ids + 1 < len(eval_d):
            f.write('{0} {1}\n'.format(i[0], i[1]))
        else:
            f.write('{0} {1}'.format(i[0], i[1]))

        eval_tq.set_description_str('-Train Data Saved ({0}/{1})-'.format(ids+1, len(eval_d)))