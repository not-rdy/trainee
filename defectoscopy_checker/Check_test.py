from unittest import TestCase, main
from check import Check
import pandas as pd
import numpy as np
from os import chdir

chdir('C:/Users/rustem.kamilyanov/trainee/defectoscopy')
#chdir('/home/rustem/trainee/defectoscopy/')

# 1)
before1 = pd.read_csv('markup before feedback/2e5808cebb24e3377a4facb05af24c35318ad68b.csv', sep=';')
after1 = pd.read_csv('markup after feedback/2e5808cebb24e3377a4facb05af24c35318ad68b.csv',   sep=';')

# 2)
before2 = pd.read_csv('markup before feedback/2e5808cebb24e3377a4facb05af24c35318ad68b.csv', sep=';')
after2 = before2.copy()
after2.loc[0, 'start_coord'] = 12345
after2.loc[1, 'end_coord'] = 12345
after2.loc[:, 'right_border_content'] = 'да_123_коммент'

# 3)
before3 = pd.read_csv('markup after feedback/2e5808cebb24e3377a4facb05af24c35318ad68b.csv',   sep=';')
after3 = before3.copy()
after3.loc[0, 'right_border_content'] = 'НЕ ЗНАЮ_23_шумы'
after3.loc[3, 'right_border_content'] = 'wat?_89_шумы'

# 4)
before4 = pd.read_csv('markup after feedback/2e5808cebb24e3377a4facb05af24c35318ad68b.csv',   sep=';')
after4 = before4.copy()
after4.loc[1, 'right_border_content'] = 'нет_23ABCD///_шумы'

# 5)
before5 = pd.read_csv('markup after feedback/2e5808cebb24e3377a4facb05af24c35318ad68b.csv',   sep=';')
after5 = before5.copy()
after5.loc[3, 'right_border_content'] = 'Конец дефектной зоны'
after5.loc[2, 'right_border_content'] = np.nan

# 6)
before6 = pd.read_csv('markup before feedback/2e5808cebb24e3377a4facb05af24c35318ad68b.csv', sep=';')
after6 = before6.copy()
after6.loc[0, ['start_coord', 'end_coord']] = 12345, 6789
after6.loc[3, ['start_coord', 'end_coord']] = 1111, 2222
after6.loc[:, 'right_border_content'] = 'да_123_коммент'
before6 = before6.drop(0)
before6 = before6.drop(3).reset_index(drop=True)


# определяем экземпляр класса Check
fun = Check()

# 1) для тестирования на наличие удаленных областей:                           before1, after1
# 2) для тестирования на наличие областей в которых были изменены координаты:  before2, after2
# 3) для тестирования формата комментария ('на вторичный контроль'):           before3, after3
# 4) для тестирования формата комментария ('номера каналов'):                  before4, after4
# 5) для тестирования формата комментария (наличие/отсутствие):                before5, after5
# 6) для тестирования на наличие добавленных новых областей:                   before6, after6


class CheckTest(TestCase):
    def test_del_area(self):
        self.assertEqual(fun.checkup(before1, after1).iloc[0, 0], 'error!')
        self.assertEqual(fun.checkup(before1, after1).iloc[0, 1], (2082290, 2082936))
        self.assertEqual(fun.checkup(before1, after1).iloc[0, 2], 'Область удалена')
        self.assertEqual(fun.checkup(before1, after1).iloc[1, 0], 'error!')
        self.assertEqual(fun.checkup(before1, after1).iloc[1, 1], (2099073, 2100116))
        self.assertEqual(fun.checkup(before1, after1).iloc[1, 2], 'Область удалена')

    def test_change_coords(self):
        self.assertEqual(fun.checkup(before2, after2).iloc[0, 0], 'error!')
        self.assertEqual(fun.checkup(before2, after2).iloc[0, 1], (1, 12345, 2045054))
        self.assertEqual(fun.checkup(before2, after2).iloc[0, 2], 'Изменена левая координата')
        self.assertEqual(fun.checkup(before2, after2).iloc[1, 0], 'error!')
        self.assertEqual(fun.checkup(before2, after2).iloc[1, 1], (1, 2061477, 12345))
        self.assertEqual(fun.checkup(before2, after2).iloc[1, 2], 'Изменена правая координата')

    def test_second_control(self):
        self.assertEqual(fun.checkup(before3, after3).iloc[0, 0], 'error!')
        self.assertEqual(fun.checkup(before3, after3).iloc[0, 1], (1, 2043811, 2045054))
        self.assertEqual(fun.checkup(before3, after3).iloc[0, 2], 'Комментарий не по формату')
        self.assertEqual(fun.checkup(before3, after3).iloc[1, 0], 'error!')
        self.assertEqual(fun.checkup(before3, after3).iloc[1, 1], (1, 2157325, 2158171))
        self.assertEqual(fun.checkup(before3, after3).iloc[1, 2], 'Комментарий не по формату')

    def test_channel_num(self):
        self.assertEqual(fun.checkup(before4, after4).iloc[0, 0], 'error!')
        self.assertEqual(fun.checkup(before4, after4).iloc[0, 1], (1, 2061477, 2062122))
        self.assertEqual(fun.checkup(before4, after4).iloc[0, 2], 'Комментарий не по формату')

    def test_add_new_area(self):
        self.assertEqual(fun.checkup(before6, after6).iloc[0, 0], 'warning!')
        self.assertEqual(fun.checkup(before6, after6).iloc[0, 1], (12345, 6789))
        self.assertEqual(fun.checkup(before6, after6).iloc[0, 2], 'Добавлена новая область')
        self.assertEqual(fun.checkup(before6, after6).iloc[1, 0], 'warning!')
        self.assertEqual(fun.checkup(before6, after6).iloc[1, 1], (1111, 2222))
        self.assertEqual(fun.checkup(before6, after6).iloc[1, 2], 'Добавлена новая область')



if __name__ == '__main__':
    main()