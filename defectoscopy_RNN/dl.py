from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SignalLabelDatset(Dataset):
    def __init__(self, signals, labels):
        super(SignalLabelDatset, self).__init__()
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


class RNNData(object):
    def __init__(self, train_x, train_y, test_x, test_y, batch_size):
        self.batch_size = batch_size
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def get_train(self):
        dataset = SignalLabelDatset(signals=self.train_x, labels=self.train_y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def get_test(self):
        dataset = SignalLabelDatset(signals=self.test_x, labels=self.test_y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
