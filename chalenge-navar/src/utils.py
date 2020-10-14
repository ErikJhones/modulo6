from os import path
import joblib
from decouple import config as cfg
import numpy as np

#variable global
DIR_NAME = path.dirname(__file__)
MODELS_FOLDER = path.join(DIR_NAME, 'models') 
print(MODELS_FOLDER)
EXPERIMENT_NAME = path.join(MODELS_FOLDER, 'exp_01_default')

TRANSFORMER_NAME_FEATURE = cfg('TRANSFORMER_NAME_FEATURE', cast=str)
MODEL_NAME = cfg('MODEL_NAME', cast=str)

def load_models():
    #load models
    tf_feature = joblib.load(path.join(EXPERIMENT_NAME, TRANSFORMER_NAME_FEATURE))
    model = joblib.load(path.join(  EXPERIMENT_NAME, MODEL_NAME))

    return model, tf_feature

def check_inputs(input):
    print(input)

    #check if list
    if type(input) == list:
        if len(input) == 4:
            return np.array(input).reshape(1,-1)

    else:
        return 205
    pass