import os
from utils import DataParts
os.chdir('C:/Users/rustem.kamilyanov/defectoscopy2')

train_signal_path = 'C:/Users/rustem.kamilyanov/defectoscopy2/train/signal'
train_markup_path = 'C:/Users/rustem.kamilyanov/defectoscopy2/train/markup'

test_signal_path = 'C:/Users/rustem.kamilyanov/defectoscopy2/test/signal'
test_markup_path = 'C:/Users/rustem.kamilyanov/defectoscopy2/test/markup'

train_names = os.listdir(train_signal_path)
test_names = os.listdir(test_signal_path)

# train, test
window = 500
step = 100

if __name__ == '__main__':
    parts_maker = DataParts(window=window, step=step, names_train=train_names, names_test=test_names,
                            train_signal_pth=train_signal_path, test_signal_pth=test_signal_path,
                            train_markup_pth=train_markup_path, test_markup_pth=test_markup_path,
                            num_processes=18)
    parts_maker.make_parts(n_files_train=902, n_files_test=50)
