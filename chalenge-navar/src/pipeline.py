import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler




'''ordem de aplicação das transformações no df
1 - Data Cleaning:
2 - Remove unwanted features numerical
3 - Remove unwanted features categorical
4 - Feature scaling in numerical features
5 - Concat two features set
6 - Feature selction
7 - Drop Nan again'''


SELECTED_FEATURES = ['total_mediaids',
                     'age_without_access',
                    'idade',
                    'mobile_web_time',
                    'video_info_time_spent_0_5',
                    'total_dependents',
                    'total_active_dependents',
                    'total_cancels',
                    'month_subs',
                    'assinatura_age']

PAYMENT_TYPE = ['BOLETO WEB',
                'CARTAO DE CREDITO',
                'DEBITO AUTOMATICO',
                'IN APP PURCHASE']


class DataCleaning(BaseEstimator, TransformerMixin):
    def fit(self, df):
        return self
        
    def transform(self, df):
        df.drop(columns=['cidade'])
        df.drop(columns=['estado'])
        
        obj = df.select_dtypes('object').columns
        df[obj] = df[obj].apply(lambda x: x.astype('str'))

        median = df.median() # option 3
        df_clean = df.fillna(median, inplace=False)
        
        df_clean = df_clean.fillna({'sexo':'1'}, inplace=False)

        
        #get categorical and numerical features
        categorical_features = df_clean[['sexo', 'tipo_de_cobranca']]
        numerical_features = df_clean.drop(columns=['sexo', 
                                                    'tipo_de_cobranca'])
        
        
        #hot-encoding in categorical features
        sexo_1hot = categorical_features.sexo.map({'F':0, 'M':1})
        cobranca_1hot = pd.get_dummies(categorical_features.tipo_de_cobranca)
        
        #add a missing columns in test set with default value equal to 0
        for payment in set(PAYMENT_TYPE) - set(cobranca_1hot.columns):
            cobranca_1hot[payment] = 0
            
        #ensure the order of column in the test set is in the same order than in train set
        categorical_features_1hot = pd.concat([df_clean['week'], sexo_1hot, cobranca_1hot], axis=1)
        df_buffer = categorical_features_1hot.loc[categorical_features_1hot['week'] <17]
        categorical_features_unique = df_buffer.groupby(['user']).agg('max')
        
        #AVERAGE OF THE NUMERICAL FEATURE
        df_buffer = numerical_features.loc[numerical_features['week']<17]
        numerical_features_average = df_buffer.groupby(['user']).agg('mean')
        
        return {'numerical': numerical_features_average, 'categorical': categorical_features_unique}


#remove a coluna 'week' dos data sets numerico e categorico
class RemoveFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features='week'):
        self.features = features
        
    def fit(self, df):
        return self
    
    def transform(self, df):
        return {'numerical': df['numerical'].drop(columns=self.features),
               'categorical': df['categorical'].drop(columns=self.features)}



class FeatureScaling(BaseEstimator, TransformerMixin):
    def __init__(self, type='std'):
        self.type = type
        
    def fit(self, df):
        self._scaler = StandardScaler().fit(df['numerical'])
        return self
    
    def transform(self, df):
        if self.type =='std':
            df_std = self._scaler.transform(df['numerical'])
            df_std = pd.DataFrame(data=df_std,
                                 columns=df['numerical'].columns,
                                 index=df['numerical'].index)
            
            return {'numerical': df_std, 'categorical': df['categorical']}



class MergeFeature(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        return pd.concat([df['numerical'], df['categorical']], axis=1)


class DropNaN(TransformerMixin):
    def fit(self, df):
        return self
    
    def transform(self, df):
        return df.dropna()



class FeatureSelection(TransformerMixin):
    def __init__(self, features = SELECTED_FEATURES):
        self.features = features
        
    def fit(self, df):
        return self
    
    def transform(self, df):
        return df[self.features]



class GetLabels(TransformerMixin):
    def fit(self, df_user, df_features):
        return self
    
    def transform(self, df_user, df_features):
        df_user_clean = df_user.loc[df_features.index.unique()]
        return df_user_clean.loc[df_features.index]

