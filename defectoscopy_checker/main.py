from check import Check
import pandas as pd
from os import chdir

chdir('C:/Users/rustem.kamilyanov/trainee/defectoscopy')
#chdir('/home/rustem/trainee/defectoscopy/')

before1 = pd.read_csv('markup before feedback/2e5808cebb24e3377a4facb05af24c35318ad68b.csv', sep=';')
before2 = pd.read_csv('markup before feedback/6b558455d9c71b411cc1ae7125721ffd3f11f50c.csv', sep=';')
before3 = pd.read_csv('markup before feedback/28bc11c3c40bae15175b493a1b3da0593524be0a.csv', sep=';')
before4 = pd.read_csv('markup before feedback/38e234257d7da8f4f4128bcb33add2b0fcc90313.csv', sep=';')

after1 = pd.read_csv('markup after feedback/2e5808cebb24e3377a4facb05af24c35318ad68b.csv', sep=';')
after2 = pd.read_csv('markup after feedback/6b558455d9c71b411cc1ae7125721ffd3f11f50c.csv', sep=';')
after3 = pd.read_csv('markup after feedback/28bc11c3c40bae15175b493a1b3da0593524be0a.csv', sep=';')
after4 = pd.read_csv('markup after feedback/38e234257d7da8f4f4128bcb33add2b0fcc90313.csv', sep=';')


# определяем экземпляр класса Check
fun = Check()

print(fun.checkup(before4, after4), '\n')
