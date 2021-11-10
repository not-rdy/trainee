import os

train_preproc_path = 'C:/Users/rustem.kamilyanov/defectoscopy2/train/preproc'
test_preproc_path = 'C:/Users/rustem.kamilyanov/defectoscopy2/test/preproc'

train_names = os.listdir(train_preproc_path)
test_names = os.listdir(test_preproc_path)

print(train_names)