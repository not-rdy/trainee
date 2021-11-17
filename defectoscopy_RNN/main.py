import os
import pickle
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from model import CnnLstm
from dl import RNNData
from pipe import train_pipeline
from model_params import batch_size, lr, epochs, n_layers, input_size, hidden_layer_size, output_size, bidirectional, \
    dropout

train_preproc_path = 'C:/Users/rustem.kamilyanov/defectoscopy2/train/preproc'
test_preproc_path = 'C:/Users/rustem.kamilyanov/defectoscopy2/test/preproc'

train_names = os.listdir(train_preproc_path)
test_names = os.listdir(test_preproc_path)

# params
torch.manual_seed(1411)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataloaders list
ids_train = [int(x) for x in np.linspace(start=0, stop=18, num=18, endpoint=True)[:18]]
ids_test = [int(x) for x in np.linspace(start=0, stop=18, num=18, endpoint=True)[:2]]
dataloaders_list = []

for idx_left_train, idx_right_train, idx_left_test, idx_right_test in zip(ids_train[:-1], ids_train[1:], ids_test[:-1],
                                                                          ids_test[1:]):

    os.chdir('C:/Users/rustem.kamilyanov/defectoscopy2/train/preproc')
    with open(f'train_x_{idx_left_train}_{idx_right_train}', 'rb') as x:
        train_x = pickle.load(x)
    with open(f'train_y_{idx_left_train}_{idx_right_train}', 'rb') as y:
        train_y = pickle.load(y)

    os.chdir('C:/Users/rustem.kamilyanov/defectoscopy2/test/preproc')
    with open(f'test_x_{idx_left_test}_{idx_right_test}', 'rb') as x:
        test_x = pickle.load(x)
    with open(f'test_y_{idx_left_test}_{idx_right_test}', 'rb') as y:
        test_y = pickle.load(y)

    dataloader = RNNData(train_x=train_x, train_y=train_y,
                         test_x=test_x, test_y=test_y,
                         batch_size=batch_size)

    dataloaders_list.append(dataloader)

# model and oth
model = CnnLstm(num_layers=n_layers, input_size=input_size,
                 hidden_layer_size=hidden_layer_size, output_size=output_size,
                 bidirectional=bidirectional, dropout=dropout)


optimizer = optim.Adam(model.parameters(), lr=lr)

w0 = (1-0.920936)/0.920936
w1 = (1-0.061986)/0.061986
w2 = (1-0.004744)/0.004744
w3 = (1-0.002846)/0.002846
w4 = (1-0.009488)/0.009488
w = torch.tensor([w0, w1, w2, w3, w4], dtype=torch.float32)
loss_func = nn.CrossEntropyLoss(weight=w)

train_pipeline(model=model, dataloader_list=dataloaders_list, optimizer=optimizer,
               loss_func=loss_func, num_epochs=epochs)
