import joblib
from os import path
from train import load_data
from sklearn.metrics import accuracy_score

DIR_NAME = path.dirname(__file__)
MODELS_FOLDER = path.join(DIR_NAME, 'models') 
EXPERIMENT_NAME = path.join(MODELS_FOLDER, 'exp_01_default')
TRANSFORMER_NAME = 'tf_std_default_v0.1.pkl'
MODEL_NAME = 'model_mlp_default_v0.1.pkl'

X, y = load_data()

#load models
tf = joblib.load(path.join(MODELS_FOLDER, EXPERIMENT_NAME, TRANSFORMER_NAME))
model = joblib.load(path.join(MODELS_FOLDER,  EXPERIMENT_NAME, MODEL_NAME))

X_tf = tf.transform(X)
y_hat = model.predict(X_tf)

print(f'accuracy score {accuracy_score(y, y_hat)}')