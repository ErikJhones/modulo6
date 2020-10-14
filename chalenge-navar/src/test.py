import joblib
import pandas as pd
from os import path
from train import load_data
from decouple import config as cfg
from pipeline import GetLabels
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DIR_NAME = path.dirname(__file__)

MODELS_FOLDER = path.join(DIR_NAME, 'models') 
print(MODELS_FOLDER)
EXPERIMENT_NAME = 'exp_01_default'
#TRANSFORMER_NAME_FEATURE = 'tf_std_user_info_v0.1.pkl'
#TRANSFORMER_NAME_LABEL = 'tf_std_user_status_v0.1.pkl'
#MODEL_NAME = 'model_mlp_user_info_v0.1.pkl'

TRANSFORMER_NAME_FEATURE = cfg('TRANSFORMER_NAME_FEATURE', cast=str)
TRANSFORMER_NAME_LABEL = cfg('TRANSFORMER_NAME_LABEL', cast=str)
MODEL_NAME = cfg('MODEL_NAME', cast=str)

X, y = load_data()

#load models
tf_feature = joblib.load(path.join(MODELS_FOLDER, EXPERIMENT_NAME, TRANSFORMER_NAME_FEATURE))
tf_label = joblib.load(path.join(MODELS_FOLDER, EXPERIMENT_NAME, TRANSFORMER_NAME_LABEL))
model = joblib.load(path.join(MODELS_FOLDER,  EXPERIMENT_NAME, MODEL_NAME))

X_tf = tf_feature.transform(X)
print(X_tf)
y_tf = GetLabels().transform(y, X_tf)

X_train, X_test , y_train, y_test = train_test_split(X_tf, y_tf, train_size=0.8, random_state=42)
y_hat = model.predict(X_test)

print(f'accuracy score {accuracy_score(y_test, y_hat)}')