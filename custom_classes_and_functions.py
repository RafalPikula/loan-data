import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import recall_score





class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Transformer that drops specified columns.
    """

    def __init__(self, column_list):
        self.column_list = column_list        
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.column_list, axis=1, inplace=False)


class ColumnSelectorByType(BaseEstimator, TransformerMixin):
    """
    Transformer that selects columns of specified types.
    """

    def __init__(self, column_type_list):
        self.column_type_list = column_type_list
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.select_dtypes(include=self.column_type_list)


class MissingThresholdIndicator(BaseEstimator, TransformerMixin):
    """
    Transformer that appends binary indicators to the data set for the columns 
    with more than the threshold (%) of missing values, where missing corresponds to 1.
    """

    def __init__(self, threshold):
        self.threshold = threshold
        return None

    def fit(self, X, y=None):
        self.columns_to_transform = list(X.columns[X.isnull().mean() > self.threshold])     
        return self


    def transform(self, X):
        W = X.copy()
        if self.columns_to_transform:
            V = W[self.columns_to_transform].copy()
            V = V.isnull().astype(np.int8).astype('category')
            # Adds a suffix to column names
            V.columns = list(pd.Series(V.columns).apply(lambda x: x + '_MISSING')) 
            W = pd.concat([W, V], axis=1)              
        return W


class RareCategoriesMerger(BaseEstimator, TransformerMixin):
    """
    Transformer that groups rarely occuring categories with at most threshold 
    occurences into a single category, provided there are at least 2 such categories.
    """

    def __init__(self, threshold, replace_value='rare_value'):
        self.threshold = threshold
        self.replace_value = replace_value
        return None

    def fit(self, X, y=None):
        self.categories_to_keep = []

        for column in X.columns:
            col_val_count = X[column].value_counts()
            to_keep = col_val_count > self.threshold

            if len(col_val_count) - (to_keep).sum() > 1:
                self.categories_to_keep.append([column, 
                                                list(to_keep[to_keep == True].index)])
            else:
                self.categories_to_keep.append([column, list(to_keep.index)])

        self.categories_to_keep = dict(self.categories_to_keep)
        return self

    def transform(self, X):
        W = X.copy()

        for column, categories in self.categories_to_keep.items():
            W.loc[~W[column].isin(categories), column] = self.replace_value        
        return W

class SimpleImputerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper to SimpleImputer that returns a DataFrame with column names 
    instead of a numpy array
    """

    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.simple_imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        return None

    def fit(self, X, y=None):
        self.simple_imputer.fit(X, y)
        self.columns = X.columns
        self.dtypes = X.dtypes
        return self

    def transform(self, X):
        W = pd.DataFrame(self.simple_imputer.transform(X), columns=self.columns)
        #preserve dtypes
        for column in W.columns:
            W[column] = W[column].astype(self.dtypes[column])
        return W


class StandardScalerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper to StandardScaler that returns a DataFrame with column names 
    instead of a numpy array
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.standard_scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        return None

    def fit(self, X, y=None):
        self.standard_scaler.fit(X, y)
        self.columns = X.columns
        return self

    def transform(self, X):
        W = pd.DataFrame(self.standard_scaler.transform(X), columns = self.columns)
        return W

class VarianceThresholdWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper to VarianceThreshold that returns a DataFrame with column names 
    instead of a numpy array
    """

    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.variance_threshold = VarianceThreshold(threshold=self.threshold)
        return None

    def fit(self, X, y=None):
        self.variance_threshold.fit(X, y)
        self.columns = X.columns[X.var() >= self.threshold]
        return self

    def transform(self, X):
        W = pd.DataFrame(self.variance_threshold.transform(X), columns = self.columns)
        return W

    
# Custom scoring function
def recall_recall_product_score(y_true, y_pred):
    return recall_score(y_true, y_pred) * recall_score(1 - y_true, 1 - y_pred)