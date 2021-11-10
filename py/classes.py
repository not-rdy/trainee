import pandas as pd


class Recommendation:
    from os import chdir
    chdir('C:/Users/rustem.kamilyanov/trainee/rec')

    def rec_top(self, new_user_features, n_top):
        from pickle import load 
        from numpy import array, std
        from pandas import read_csv, Series, get_dummies
        from scipy.spatial import distance

        # добавляем нового юзера в таблицу ко всем юзерам
        users = read_csv('features_clust.csv')
        users = users.drop(['Unnamed: 0', 'smoking', 'name', 'Unnamed: 0.1'], axis=1)
        users = users.drop('clust', axis=1)
        users.rename(columns={'sex': 'gender'}, inplace=True)
        
        id_new = 'new'
        new_user = Series(new_user_features, name=id_new)
        users = users.append(new_user)
        
        # переводим таблицу с юзерами в дамми
        users = get_dummies(users, columns=['gender', 'classes', 'disability', 'preferential', 'job', 'city',
                                            'marit_status', 'education', 'childs', 'networks'], drop_first=True)
        
        # стандартизируем возраст, цену, время
        users['age'] = [(i - users['age'].mean()) / std(users['age']) for i in users['age']]
        users['price'] = [(i - users['price'].mean()) / std(users['price']) for i in users['price']]
        users['time'] = [(i - users['time'].mean()) / std(users['time']) for i in users['time']]
        
        # подгружаем обученную модель
        m = load(open('kMeans.sav', 'rb'))
        
        # получаем метку кластера для нового юзера
        new_label = m.predict(array(users.loc[id_new, :]).reshape(1, -1))
        
        # подгружаем табличку user-item сгруппированную по кластерам
        user_item_by_clust = read_csv('user_item_by_clust.csv', index_col='clust')

        # Расстояния до центров кластеров
        c0 = m.cluster_centers_[0]
        c1 = m.cluster_centers_[1]
        c2 = m.cluster_centers_[2]
        print(' Расстояние до центра кластера 0: ', distance.euclidean(c0, users.loc[id_new, :]), '\n',
              'Расстояние до центра кластера 1: ', distance.euclidean(c1, users.loc[id_new, :]), '\n',
              'Расстояние до центра кластера 2: ', distance.euclidean(c2, users.loc[id_new, :]), '\n',
              'Пассажир отнесен к кластеру: ', new_label[0])

        # выделяем топ товаров по кластеру к которому относится новый юзер
        rec = user_item_by_clust.iloc[new_label, :].T.sort_values(by=int(new_label), ascending=False)[0:n_top].index
        print(" Топ "+str(n_top)+" товаров/услуг по кластеру "+str(new_label[0])+": " + '\n',
              list(rec))

    def rec_als(self, user_purchases, N=5):
        from pickle import load

        # подгружаем предобученную модель
        m_als = load(open('ALS.sav', 'rb'))

        # преобразуем входной лист в csr
        from scipy.sparse import csr_matrix
        user_purchases = csr_matrix(user_purchases).tocsr()

        # рекомендация для нового юзера
        rec = m_als.recommend(0,
                              user_purchases,
                              N=N,
                              filter_already_liked_items=False,
                              recalculate_user=True)
        # получаем индексы рекомендуемых товаров
        idx_rec_items = list(map(lambda x: x[0], rec))

        # выводим перечень товаров
        item_list = pd.read_csv('item_list.csv')
        rec_items = item_list.iloc[idx_rec_items, :].values
        print("Топ "+str(N)+" товаров услуг для пассажира: ", '\n',
              list(map(lambda x: x[0], rec_items)))
