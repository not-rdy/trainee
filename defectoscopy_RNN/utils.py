import pickle
import multiprocessing
import numpy as np
import torch
from scipy.sparse import load_npz


class DataParts:

    def __init__(self, window, step, names_train, names_test, train_signal_pth, test_signal_pth, train_markup_pth,
                 test_markup_pth, num_processes):

        self.window = window
        self.step = step
        self.train_names = names_train
        self.test_names = names_test
        self.train_signal_path = train_signal_pth
        self.test_signal_path = test_signal_pth
        self.train_markup_path = train_markup_pth
        self.test_markup_path = test_markup_pth
        self.num_processes = num_processes

    def _preproc(self, start, end, names, signal_path, markup_path, parts_name):
        parts_x = []
        parts_y = []
        names = names[start:end]
        for i in range(len(names)):
            name = names[i]
            signals = []

            # считываем npz файл, приводим к виду (N, 642, 2)
            temp_signals = load_npz(signal_path + '/' + name).toarray()
            temp_signals = np.stack([temp_signals[:, :642], temp_signals[:, 642:]], axis=2)

            # для каждой матрицы (642, 2) попарно берем max,
            # получаем общий вектор (642, )
            for matrix in temp_signals:
                temp = []
                for a, b in zip(matrix[:, 0], matrix[:, 1]):
                    temp.append(max(a, b) / 15)
                signals.append(temp)

            # из полученных векторов формируем массив array (N, 642)
            temp_signals = np.array(signals)

            # считываем лейблы
            temp_labels = load_npz(markup_path + '/' + name).toarray()

            # объединяем полученный массив array с лейблами, получаем массив (N, 643)
            # в котором последний столбец - лейблы
            df_train = np.concatenate([temp_signals, temp_labels.reshape(-1, 1)], axis=1)

            # скользящим окном шириной 500 и шагом 100 разбиваем полученный массив на части
            for idx in range(0, len(df_train) - self.window, self.step):
                idx_left = idx
                idx_right = idx + self.window

                if len(np.unique(df_train[idx_left:idx_right, -1])) == 1:
                    x = df_train[idx_left:idx_right, :-1]
                    y = df_train[idx_left:idx_right, -1][0]
                    parts_x.append(torch.tensor(x, dtype=torch.float))
                    parts_y.append(torch.tensor(y, dtype=torch.long))
                else:
                    if len(np.unique(df_train[idx_left:idx_right, -1])) == 2 and \
                            (0 in np.unique(df_train[idx_left:idx_right, -1])):
                        x = df_train[idx_left:idx_right, :-1]
                        y = df_train[idx_left:idx_right, -1][1]
                        parts_x.append(torch.tensor(x, dtype=torch.float))
                        parts_y.append(torch.tensor(y, dtype=torch.long))
                    else:
                        if len(np.unique(df_train[idx_left:idx_right, -1])) > 2:
                            # parts_x.append(df_train[idx_left:idx_right, :-1])
                            # parts_y.append(1)
                            pass

            for ids, label in enumerate(parts_y):
                if label.item() == 2:
                    parts_y[ids] = torch.tensor(1, dtype=torch.long)
                elif label.item() == 3:
                    parts_y[ids] = torch.tensor(2, dtype=torch.long)
                elif label.item() == 5:
                    parts_y[ids] = torch.tensor(3, dtype=torch.long)
                elif label.item() == 6:
                    parts_y[ids] = torch.tensor(4, dtype=torch.long)

            file_name = parts_name + '_x_' + str(start) + '_' + str(end)
            with open(file_name, 'wb') as fp:
                pickle.dump(parts_x, fp)

            file_name = parts_name + '_y_' + str(start) + '_' + str(end)
            with open(file_name, 'wb') as fp:
                pickle.dump(parts_y, fp)

    def __process_maker_train(self, start, end, num):
        ctx = multiprocessing.get_context('spawn')
        processes_list = []
        l1 = [int(x) for x in np.linspace(start=start, stop=end, num=num, endpoint=True)]
        l2 = [int(x) for x in np.linspace(start=start, stop=end, num=num, endpoint=True)]
        for idx_left, idx_right in zip(l1[:-1], l2[1:]):
            processes_list.append(ctx.Process(target=self._preproc, args=(idx_left, idx_right,
                                                                          self.train_names,
                                                                          self.train_signal_path,
                                                                          self.train_markup_path,
                                                                          'train', )))
        return processes_list

    def __process_maker_test(self, start, end, num):
        ctx = multiprocessing.get_context('spawn')
        processes_list = []
        l1 = [int(x) for x in np.linspace(start=start, stop=end, num=num, endpoint=True)]
        l2 = [int(x) for x in np.linspace(start=start, stop=end, num=num, endpoint=True)]
        for idx_left, idx_right in zip(l1[:-1], l2[1:]):
            processes_list.append(ctx.Process(target=self._preproc, args=(idx_left, idx_right,
                                                                          self.test_names,
                                                                          self.test_signal_path,
                                                                          self.test_markup_path,
                                                                          'test', )))
        return processes_list

    def make_parts(self, n_files_train, n_files_test):

        # train parts
        train_processes_list = self.__process_maker_train(start=0, end=n_files_train, num=self.num_processes)
        ids = list(range(0, 19, 6))
        for idx_left, idx_right in zip(ids[:-1], ids[1:]):
            processes = train_processes_list[idx_left:idx_right]
            for proc in processes:
                proc.start()
            for proc in processes:
                proc.join()

        # test parts
        test_processes_list = self.__process_maker_test(start=0, end=n_files_test, num=self.num_processes)
        ids = list(range(0, 19, 6))
        for idx_left, idx_right in zip(ids[:-1], ids[1:]):
            processes = test_processes_list[idx_left:idx_right]
            for proc in processes:
                proc.start()
            for proc in processes:
                proc.join()
