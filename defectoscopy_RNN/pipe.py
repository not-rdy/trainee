from eval import conf_matrix_part
from model_params import batch_size, lr, epochs, n_layers, input_size, hidden_layer_size, output_size, bidirectional, \
    dropout
from processes import window, step, n_files_train, n_files_test

import torch
import numpy as np
import mlflow
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_pipeline(model, dataloader_list, optimizer, loss_func, num_epochs):
    train_loss_history = []

    class_0 = {'TP': [], 'TN': [], 'FP': [], 'FN': [],
               'precision': [], 'recall': [],
               'f1': []}
    class_1 = {'TP': [], 'TN': [], 'FP': [], 'FN': [],
               'precision': [], 'recall': [],
               'f1': []}
    class_2 = {'TP': [], 'TN': [], 'FP': [], 'FN': [],
               'precision': [], 'recall': [],
               'f1': []}
    class_3 = {'TP': [], 'TN': [], 'FP': [], 'FN': [],
               'precision': [], 'recall': [],
               'f1': []}
    class_4 = {'TP': [], 'TN': [], 'FP': [], 'FN': [],
               'precision': [], 'recall': [],
               'f1': []}

    get_test_list = [dl.get_test() for dl in dataloader_list]

    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('CNN_RNN_defectoscopy')

    with mlflow.start_run():

        for idx_epoch in range(num_epochs):

            model.train()

            get_train_list = [dl.get_train() for dl in dataloader_list]

            loss_batch = []
            # train
            for train in get_train_list:
                for x, y in tqdm(train, desc=f'[Epoch #{idx_epoch}]', total=len(train)):
                    optimizer.zero_grad()
                    y_hat = model(x.to(device))
                    loss = loss_func(y_hat, y.reshape(-1).to(device))
                    loss.backward()
                    optimizer.step()
                    loss_batch.append(loss.item())

            train_loss_epoch = sum(loss_batch) / len(loss_batch)
            train_loss_history.append(train_loss_epoch)

            # Eval

            # counting metrics
            conf_matrix_list = []
            for train in get_train_list:
                conf_matrix_p = conf_matrix_part(model, train)
                conf_matrix_list.append(conf_matrix_p)
            conf_matrix_all_train = sum(conf_matrix_list)
            print(f'train Confusion Matrix: \n {conf_matrix_all_train}')

            conf_matrix_list = []
            for test in get_test_list:
                conf_matrix_p = conf_matrix_part(model, test)
                conf_matrix_list.append(conf_matrix_p)
            conf_matrix_all_test = sum(conf_matrix_list)
            print(f'test Confusion Matrix: \n {conf_matrix_all_test}')
            # class 0
            TP = conf_matrix_all_test[0, 0]
            FN = conf_matrix_all_test[0, 1] + conf_matrix_all_test[0, 2] + \
                 conf_matrix_all_test[0, 3] + conf_matrix_all_test[0, 4]
            FP = conf_matrix_all_test[1, 0] + conf_matrix_all_test[2, 0] + \
                 conf_matrix_all_test[3, 0] + conf_matrix_all_test[4, 0]
            TN = conf_matrix_all_test[1, 1] + conf_matrix_all_test[1, 2] + \
                 conf_matrix_all_test[1, 3] + conf_matrix_all_test[1, 4] + \
                 conf_matrix_all_test[2, 1] + conf_matrix_all_test[2, 2] + \
                 conf_matrix_all_test[2, 3] + conf_matrix_all_test[2, 4] + \
                 conf_matrix_all_test[3, 1] + conf_matrix_all_test[3, 2] + \
                 conf_matrix_all_test[3, 3] + conf_matrix_all_test[3, 4] + \
                 conf_matrix_all_test[4, 1] + conf_matrix_all_test[4, 2] + \
                 conf_matrix_all_test[4, 3] + conf_matrix_all_test[4, 4]
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            class_0['TP'].append(TP)
            class_0['FN'].append(FN)
            class_0['FP'].append(FP)
            class_0['TN'].append(TN)
            class_0['precision'].append(precision)
            class_0['recall'].append(recall)
            class_0['f1'].append((precision * recall) / (precision + recall))
            # class 1
            TP = conf_matrix_all_test[1, 1]
            FN = conf_matrix_all_test[1, 0] + conf_matrix_all_test[1, 2] + \
                 conf_matrix_all_test[1, 3] + conf_matrix_all_test[1, 4]
            FP = conf_matrix_all_test[0, 1] + conf_matrix_all_test[2, 1] + \
                 conf_matrix_all_test[3, 1] + conf_matrix_all_test[4, 1]
            TN = conf_matrix_all_test[0, 0] + conf_matrix_all_test[0, 2] + \
                 conf_matrix_all_test[0, 3] + conf_matrix_all_test[0, 4] + \
                 conf_matrix_all_test[2, 0] + conf_matrix_all_test[2, 2] + \
                 conf_matrix_all_test[2, 3] + conf_matrix_all_test[2, 4] + \
                 conf_matrix_all_test[3, 0] + conf_matrix_all_test[3, 2] + \
                 conf_matrix_all_test[3, 3] + conf_matrix_all_test[3, 4] + \
                 conf_matrix_all_test[4, 0] + conf_matrix_all_test[4, 2] + \
                 conf_matrix_all_test[4, 3] + conf_matrix_all_test[4, 4]
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            class_1['TP'].append(TP)
            class_1['FN'].append(FN)
            class_1['FP'].append(FP)
            class_1['TN'].append(TN)
            class_1['precision'].append(precision)
            class_1['recall'].append(recall)
            class_1['f1'].append((precision * recall) / (precision + recall))
            # class 2
            TP = conf_matrix_all_test[2, 2]
            FN = conf_matrix_all_test[2, 0] + conf_matrix_all_test[2, 1] + \
                 conf_matrix_all_test[2, 3] + conf_matrix_all_test[2, 4]
            FP = conf_matrix_all_test[0, 2] + conf_matrix_all_test[1, 2] + \
                 conf_matrix_all_test[3, 2] + conf_matrix_all_test[4, 2]
            TN = conf_matrix_all_test[0, 0] + conf_matrix_all_test[0, 1] + \
                 conf_matrix_all_test[0, 3] + conf_matrix_all_test[0, 4] + \
                 conf_matrix_all_test[1, 0] + conf_matrix_all_test[1, 1] + \
                 conf_matrix_all_test[1, 3] + conf_matrix_all_test[1, 4] + \
                 conf_matrix_all_test[3, 0] + conf_matrix_all_test[3, 1] + \
                 conf_matrix_all_test[3, 3] + conf_matrix_all_test[3, 4] + \
                 conf_matrix_all_test[4, 0] + conf_matrix_all_test[4, 1] + \
                 conf_matrix_all_test[4, 3] + conf_matrix_all_test[4, 4]
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            class_2['TP'].append(TP)
            class_2['FN'].append(FN)
            class_2['FP'].append(FP)
            class_2['TN'].append(TN)
            class_2['precision'].append(precision)
            class_2['recall'].append(recall)
            class_2['f1'].append((precision * recall) / (precision + recall))
            # class 3
            TP = conf_matrix_all_test[3, 3]
            FN = conf_matrix_all_test[3, 0] + conf_matrix_all_test[3, 1] + \
                 conf_matrix_all_test[3, 2] + conf_matrix_all_test[3, 4]
            FP = conf_matrix_all_test[0, 3] + conf_matrix_all_test[1, 3] + \
                 conf_matrix_all_test[2, 3] + conf_matrix_all_test[4, 3]
            TN = conf_matrix_all_test[0, 0] + conf_matrix_all_test[0, 1] + \
                 conf_matrix_all_test[0, 2] + conf_matrix_all_test[0, 4] + \
                 conf_matrix_all_test[1, 0] + conf_matrix_all_test[1, 1] + \
                 conf_matrix_all_test[1, 2] + conf_matrix_all_test[1, 4] + \
                 conf_matrix_all_test[2, 0] + conf_matrix_all_test[2, 1] + \
                 conf_matrix_all_test[2, 2] + conf_matrix_all_test[2, 4] + \
                 conf_matrix_all_test[4, 0] + conf_matrix_all_test[4, 1] + \
                 conf_matrix_all_test[4, 2] + conf_matrix_all_test[4, 4]
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            class_3['TP'].append(TP)
            class_3['FN'].append(FN)
            class_3['FP'].append(FP)
            class_3['TN'].append(TN)
            class_3['precision'].append(precision)
            class_3['recall'].append(recall)
            class_3['f1'].append((precision * recall) / (precision + recall))
            # class 4
            TP = conf_matrix_all_test[4, 4]
            FN = conf_matrix_all_test[4, 0] + conf_matrix_all_test[4, 1] + \
                 conf_matrix_all_test[4, 2] + conf_matrix_all_test[4, 3]
            FP = conf_matrix_all_test[0, 4] + conf_matrix_all_test[1, 4] + \
                 conf_matrix_all_test[2, 4] + conf_matrix_all_test[3, 4]
            TN = conf_matrix_all_test[0, 0] + conf_matrix_all_test[0, 1] + \
                 conf_matrix_all_test[0, 2] + conf_matrix_all_test[0, 3] + \
                 conf_matrix_all_test[1, 0] + conf_matrix_all_test[1, 1] + \
                 conf_matrix_all_test[1, 2] + conf_matrix_all_test[1, 3] + \
                 conf_matrix_all_test[2, 0] + conf_matrix_all_test[2, 1] + \
                 conf_matrix_all_test[2, 2] + conf_matrix_all_test[2, 3] + \
                 conf_matrix_all_test[3, 0] + conf_matrix_all_test[3, 1] + \
                 conf_matrix_all_test[3, 2] + conf_matrix_all_test[3, 3]
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            class_4['TP'].append(TP)
            class_4['FN'].append(FN)
            class_4['FP'].append(FP)
            class_4['TN'].append(TN)
            class_4['precision'].append(precision)
            class_4['recall'].append(recall)
            class_4['f1'].append((precision * recall) / (precision + recall))

            metrics_for_each_class = {'Class_0': class_0, 'Class_1': class_1,
                                      'Class_2': class_2, 'Class_3': class_3, 'Class_4': class_4}

            # трекаем метрики
            # loss
            mlflow.log_metric(key='train_loss_epoch', value=train_loss_epoch, step=idx_epoch)
            # metrics for each class
            for class_name in metrics_for_each_class.keys():
                for metric_name in metrics_for_each_class[class_name].keys():
                    mlflow.log_metric(key=f'{class_name} {metric_name}',
                                      value=metrics_for_each_class[class_name][metric_name][idx_epoch],
                                      step=idx_epoch)
            # f1_mean for all classes
            f1_mean = np.mean([class_0['f1'][idx_epoch], class_1['f1'][idx_epoch],
                               class_2['f1'][idx_epoch], class_3['f1'][idx_epoch], class_4['f1'][idx_epoch]])
            mlflow.log_metric(key='f1_mean', value=float(f1_mean), step=idx_epoch)

            # трекаем параметры
            params = {'batch_size': batch_size, 'lr': lr, 'num_epochs': epochs, 'n_layers': n_layers,
                      'input_size': input_size, 'hidden_layer_size': hidden_layer_size,
                      'output_size': output_size, 'bidirectional': bidirectional, 'dropout': dropout,
                      'window': window, 'step': step, 'n_files_train': n_files_train, 'n_files_test': n_files_test}
            mlflow.log_params(params=params)

        mlflow.end_run()
