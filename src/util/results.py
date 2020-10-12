import pandas as pd
import os
from util.file import create_if_not_exist
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np


class Result:
    default_column_names = ['dataset', 'task', 'method', 'acc', 'time']
    def __init__(self, column_names=default_column_names):
        self.columns = column_names
        self.df = pd.DataFrame(columns=column_names)

    @classmethod
    def load(cls, path, create_if_not_exists=True):
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t', index_col=0)
            res = Result(df.columns.values)
            res.df = df
        elif create_if_not_exists:
            print('no file found in {}, returning a new one'.format(path))
            res = Result(cls.default_column_names)
        else: raise ValueError('File does not exist')
        return res

    def add(self, index=None, **colum_value):
        colum_value = colum_value.copy()
        assert set(colum_value.keys()).issubset(set(self.columns)), 'unknown columns'
        for col in self.columns:
            if col not in colum_value:
                colum_value[col]='-'
        data = [colum_value[k] for k in self.columns]
        if index:
            new = pd.DataFrame(data=[data], index=[index], columns=self.columns)
        else:
            new = pd.DataFrame(data=[data], columns=self.columns)
        self.df = self.df.append(new, ignore_index=False)

    def __str__(self):
        return str(self.df)

    def __contains__(self, index):
        return index in self.df.index

    def dump(self, path):
        create_if_not_exist(os.path.dirname(path))
        self.df.to_csv(path, sep='\t')

    def pivot(self, index=['dataset','task'], values='acc', aggfunc=np.mean, grand_totals=False):
        pv = pd.pivot_table(self.df, values=values, index=index, columns=['method'], aggfunc=aggfunc)
        print(pv)
        if grand_totals:
            print('\nGrand Totals')
            pv = pd.pivot_table(self.df, values=values, index=['dataset'], columns=['method'], aggfunc=aggfunc)
            print(pv)
            pv = pd.pivot_table(self.df, values=values, columns=['method'], aggfunc=aggfunc)
            print(pv)


