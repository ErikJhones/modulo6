from os import path
import joblib
from decouple import config as cfg
import numpy as np
import json 

#variable global
DIR_NAME = path.dirname(__file__)
MODELS_FOLDER = path.join(DIR_NAME, 'models') 
EXPERIMENT_NAME = path.join(MODELS_FOLDER, 'exp_01_default')

TRANSFORMER_NAME = cfg('TRANSFORMER_NAME', cast=str)
MODEL_NAME = cfg('MODEL_NAME', cast=str)

def load_models():
    #load models
    tf = joblib.load(path.join(MODELS_FOLDER, EXPERIMENT_NAME, TRANSFORMER_NAME))
    model = joblib.load(path.join(MODELS_FOLDER,  EXPERIMENT_NAME, MODEL_NAME))

    return model, tf

def check_inputs(input):
    print(input)

    #check if list
    if type(input) == list:
        if len(input) == 4:
            return np.array(input).reshape(1,-1)

    else:
        return 205
    pass

def convert_string_to_list(string_recieve):
    
    # Converting string to list 
    res = json.loads(string_recieve) 
    
    return res 