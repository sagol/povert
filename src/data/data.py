import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

class Data():

    def __init__(self):
        self.country_df_train = pd.DataFrame()
        self.country_df_test = pd.DataFrame()
        self.categorical_list = []
        self.float_list = []
        self.train_file_name = None
        self.test_file_name = None

    def split_data(self, size=0.8, n_splits=1, random_state=1, balance=False, df=None):
        if not isinstance(df, pd.DataFrame):
            train = self.country_df_train
        else:
            train = df            
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=1-size, random_state=random_state)
        splits = []
        for train_index, validate_index in sss.split(train, train.poor):
            df_train = train.iloc[train_index]
            if balance:
                df_train = self.resample(df_train)
            splits.append((df_train, train.iloc[validate_index]))
        return splits

    def _rename_col(self):
        train_columns = self.country_df_train.columns
        train_new_columns = [x if x == 'poor' or x == 'country' else '{0}_{1}'.format(self.country, 
                             train_columns.get_loc(x)) for x in train_columns]
        self.country_df_train.columns=train_new_columns
        self.col_maping = dict(zip(train_columns, train_new_columns))
        self.col_maping_reverse = dict(zip(train_new_columns, train_columns))

        self.country_df_test.rename(columns=self.col_maping, inplace=True)
        
    
    def del_nonunique(self, df):
        cols = list(df)
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        print('Cols to drop:', cols_to_drop)
        return df.drop(cols_to_drop, axis=1)

    def category_float_search(self, countries=['B'], cat_types=['object'], fi_types=['float64', 'int64']):
        categorical_list = list(self.country_df_train[self.col_common_list].select_dtypes(
                include=cat_types).columns)
        if self.country not in countries:
            return categorical_list, list(self.country_df_train[self.col_common_list].select_dtypes(include=fi_types).columns)
        float_list = []
        scaler = StandardScaler()
        print('float list length: ', len(list(self.country_df_test.select_dtypes(include=fi_types).columns)))
        for i in list(self.country_df_test[self.col_common_list].select_dtypes(include=fi_types).columns):
            self.country_df_train[i].fillna(self.country_df_train[i].median(), inplace=True)
            self.country_df_test[i].fillna(self.country_df_test[i].median(), inplace=True)
            value_set = set(self.country_df_test[i].unique()).union(set(self.country_df_train[i].unique()))
            if len(value_set) <= 5:
                categorical_list.append(i)
            else:
                self.country_df_train[i] = scaler.fit_transform(self.country_df_train[i].reshape(-1, 1))
                self.country_df_test[i] = scaler.transform(self.country_df_test[i].reshape(-1, 1))
                float_list.append(i)
        print('float list length: ', len(sorted(float_list)))
        return sorted(categorical_list), sorted(float_list)
            
    def set_file_names(self, files_dict):
        self.train_file_name = files_dict.get('train')
        self.test_file_name = files_dict.get('test')

    def set_country(self, country):
        self.country = country
        print('Country: ', self.country)
        
    def load(self, load=True):
        self.country_df_train = self.del_nonunique(pd.read_csv(self.train_file_name, index_col='id'))
        self.country_df_test = self.del_nonunique(pd.read_csv(self.test_file_name, index_col='id'))
        
        if not load:
            self._rename_col()
        self.col_common_list = \
            sorted(list(set(self.country_df_train.columns).intersection(self.country_df_test.columns)))
        self.categorical_list, self_float_list = self.category_float_search()
        return True

    def save(self, files_dict, poor=True):
        train = self.get_train()
        if poor:
            train = pd.concat([train[0], train[1]], axis=1)
        else:
            train = train[0]
        train.to_csv(files_dict.get('train'), index=True, mode='w')
        test = self.get_test()
        test.to_csv(files_dict.get('test'), index=True, mode='w')        
        return True

    def resample(self, df):
        df_majority = df[self.country_df_train.poor==False]
        df_minority = df[self.country_df_train.poor==True]

        df_minority_upsampled = resample(df_minority, 
                                         replace=True,
                                         n_samples=df_majority.shape[0],
                                         random_state=1)        
        return pd.concat([df_majority, df_minority_upsampled])        
        
    
    def get_train(self, balance=False):
        if balance:
            train = self.resample(self.country_df_train)
            return train[self.col_common_list], train['poor']    
        return self.country_df_train[self.col_common_list], self.country_df_train['poor']
    
    def get_train_valid(self, n_splits=1, balance=False):
        splits = self.split_data(n_splits=n_splits, balance=balance)
        return [((x[self.col_common_list], x.poor),(y[self.col_common_list], y.poor)) for x,y in splits]
    
    def get_test(self):
        return self.country_df_test[self.col_common_list]
    
    def get_cat_list(self):        
        return self.categorical_list
    
    def get_float_list(self):
        return self.float_list

class DataInd(Data):

    def __init__(self):
        super().__init__()
            
    def get_poor(self, df):
        return df['poor'].reset_index()[['id', 'poor']].drop_duplicates().set_index('id')
    
    def summarize(self, df):
        count = df.copy().groupby(level=0).sum()
        res_df = pd.concat({'sum': count}, axis=1)
        res_df.columns = ['{0}_{1}'.format(i[0], i[1]) for i in res_df.columns]
        res_df = res_df.reindex(index=df.index.get_level_values(0))
        res_df = res_df[~res_df.index.duplicated(keep='first')]
        print('summarized size df: ', res_df.shape)
        return res_df
            
    def _get_id_list(self, df):
        return list(OrderedDict.fromkeys(df.index.get_level_values(0)))
    
    def count_iid(self, df):
        s = df.index.get_level_values(0).value_counts()
        return s.reindex(index = self._get_id_list(df)).to_frame('iid_cnt')
    
    def count_neg_poz(self, df):
        res_df = df.select_dtypes(include=['float64','int64','int8'])
        res_df = res_df.groupby(level=0).apply(lambda c: c.apply(
                lambda x: pd.Series([(x < 0).sum(), (x >= 0).sum()])).unstack())
        res_df.columns = ['{0}_{1}'.format(i[0], i[1]) for i in res_df.columns]  
        print('count_neg_poz size df: ', res_df.shape)
        return res_df.reindex(index = self._get_id_list(df))
    
    def count_unique_categories(self, df, iid=True):
        res_df = df.groupby(level=0).apply(lambda c: c.apply(lambda x: pd.Series([len((x).unique())])))
        res_df.index = res_df.index.droplevel(1)
        res_df.columns = ['{0}_{1}'.format('cat_n', i) for i in res_df.columns]
        print('count_unique_categories size df: ', res_df.shape)
        res_df = res_df.reindex(index = self._get_id_list(df))
        if iid:
            div_df = res_df.div(self.count_iid(df)['iid_cnt'], axis=0)
            div_df.columns = ['{0}_{1}'.format('div_cat_iid', i) for i in res_df.columns]
            res_df = pd.concat([res_df, div_df], axis=1)
        return res_df
    
    
    def load(self, load=True, cat_enc=False):

        self.country_df_train = self.del_nonunique(pd.read_csv(self.train_file_name, index_col=['id','iid']))
        self.country_df_test = self.del_nonunique(pd.read_csv(self.test_file_name, index_col=['id','iid']))

        if not load:
            self._rename_col()
            self.col_common_list = sorted(list(set(self.country_df_train.columns).intersection(
                        self.country_df_test.columns)))

            self.categorical_list, self_float_list = self.category_float_search(countries=['A', 'B', 'C'])

            if cat_enc:
                for header in self.categorical_list:
                    self.country_df_train[header] = self.country_df_train[header].astype('category').cat.codes
                    self.country_df_test[header] = self.country_df_test[header].astype('category').cat.codes
            
            self.country_df_train = pd.concat([self.get_poor(self.country_df_train),
                                               self.count_iid(self.country_df_train),
                                               self.count_neg_poz(self.country_df_train),
                                               self.summarize(self.country_df_train),
                                               self.count_unique_categories(self.country_df_train),
                                              ], axis=1)
            self.country_df_test = pd.concat([self.count_iid(self.country_df_test),
                                              self.count_neg_poz(self.country_df_test),
                                              self.summarize(self.country_df_test),
                                              self.count_unique_categories(self.country_df_test),
                                             ], axis=1)

        self.col_common_list = sorted(list(set(self.country_df_train.columns).intersection(self.country_df_test.columns)))
        self.categorical_list, self_float_list = self.category_float_search(countries=['A', 'B', 'C'])


        print('indiv train shape: ', self.country_df_train.shape)
        print('indiv test shape: ', self.country_df_test.shape)
        return True

class DataConcat(Data):

    def __init__(self):
        self.data_hh_train = pd.DataFrame()
        self.data_hh_test = pd.DataFrame()        
        self.data_indiv_train = pd.DataFrame()
        self.data_indiv_test = pd.DataFrame()
        super().__init__()
    
    def set_file_names(self, files_dict):
         self.hh_train_file_name = files_dict.get('train_hh')
         self.hh_test_file_name = files_dict.get('test_hh')
         self.ind_train_file_name = files_dict.get('train_ind')
         self.ind_test_file_name = files_dict.get('test_ind')

    def load(self, load=True, cat_enc=False):
        
        if load:
            self.country_df_train = self.del_nonunique(pd.read_csv(self.train_file_name, index_col=['id','iid']))
            self.country_df_test = self.del_nonunique(pd.read_csv(self.test_file_name, index_col=['id','iid']))
        else:
            data_ind = DataInd()
            data_ind.set_country(self.country)
            data_ind.set_file_names({'train': self.ind_train_file_name, 'test': self.ind_test_file_name})

            data_hh = Data()
            data_hh.set_country(self.country)
            data_hh.set_file_names({'train': self.hh_train_file_name, 'test': self.hh_test_file_name})

            if data_ind.load(load=False, cat_enc=cat_enc) and data_hh.load(load=False):
                self.country_df_test = data_hh.country_df_test.join(data_ind.country_df_test)    
                self.country_df_train = data_hh.country_df_train.join(data_ind.country_df_train.drop('poor', axis=1))    

        self.col_common_list = sorted(list(set(self.country_df_test.columns).intersection(self.country_df_train.columns)))
        self.categorical_list, self_float_list = self.category_float_search(countries=['B'])
            
        print('train:', self.country_df_train.shape)
        print('test:', self.country_df_test.shape)

        return True
