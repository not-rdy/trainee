from eval import eval_model
from model_params import batch_size, lr, epochs, n_layers, input_size, hidden_layer_size, output_size, bidirectional, \
    dropout
from processes import window, step, n_files_train, n_files_test

import torch
import mlflow
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_pipeline(model, dataloader_list, optimizer, loss_func, num_epochs):
    train_loss_history = []

    metrics = {'class_0': None, 'class_2': None,
               'class_3': None, 'class_5': None, 'class_6': None,
               'val_loss_epoch': None}

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
            test_loss_part = []
            class_rec_0 = []
            class_rec_2 = []
            class_rec_3 = []
            class_rec_5 = []
            class_rec_6 = []

            for test in get_test_list:
                metrics_part = eval_model(model, test, loss_func)
                test_loss_part.append(metrics_part['val_loss_part'])
                class_rec_0.append(metrics_part['class_0'])
                class_rec_2.append(metrics_part['class_2'])
                class_rec_3.append(metrics_part['class_3'])
                class_rec_5.append(metrics_part['class_5'])
                class_rec_6.append(metrics_part['class_6'])

            metrics['val_loss_epoch'] = round(sum(test_loss_part)/len(test_loss_part), ndigits=2)
            metrics['class_0'] = round(sum(class_rec_0)/len(class_rec_0), ndigits=2)
            metrics['class_2'] = round(sum(class_rec_2) / len(class_rec_2), ndigits=2)
            metrics['class_3'] = round(sum(class_rec_3) / len(class_rec_3), ndigits=2)
            metrics['class_5'] = round(sum(class_rec_5) / len(class_rec_5), ndigits=2)
            metrics['class_6'] = round(sum(class_rec_6) / len(class_rec_6), ndigits=2)

            # трекаем метрики
            # avrg_train_loss = round(sum(train_loss_history) / len(train_loss_history), ndigits=2)
            # avrg_test_loss = round(sum(metrics['val_loss_part']) / len(metrics['val_loss_part']), ndigits=2)
            mlflow.log_metric(key='train_loss_epoch', value=train_loss_epoch, step=idx_epoch)
            mlflow.log_metrics(metrics=metrics, step=idx_epoch)
            # mlflow.log_metrics(metrics={'avrg_train_loss': avrg_train_loss, 'avrg_test_loss': avrg_test_loss})

            # трекаем параметры
            params = {'batch_size': batch_size, 'lr': lr, 'num_epochs': epochs, 'n_layers': n_layers,
                      'input_size': input_size, 'hidden_layer_size': hidden_layer_size,
                      'output_size': output_size, 'bidirectional': bidirectional, 'dropout': dropout,
                      'window': window, 'step': step, 'n_files_train': n_files_train, 'n_files_test': n_files_test}
            mlflow.log_params(params=params)

        mlflow.end_run()
