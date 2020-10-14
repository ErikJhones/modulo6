import joblib
import argparse
import numpy as np
import pandas as pd
from pipeline import *
from os import path, mkdir

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split

DIR_NAME = path.dirname(__file__)

def load_data2():
    
    ### load the data set
    user_info = pd.read_csv(path.join('..','data','weekly-infos-before-test.csv'))
    user_status = pd.read_csv(path.join('..', 'data', 'user-status-shrink.csv'))

    return user_info, user_status

def load_data():
    
    ### load the data set
    user_info = pd.read_csv(path.join('..','data','weekly-infos-shrink.csv'), index_col=0)
    user_status = pd.read_csv(path.join('..', 'data', 'user-status-shrink.csv'), index_col=0)

    return user_info, user_status

def transform():

    #building a pipeline to preprocessing the model
    pipeline_X = Pipeline([('cleaner', DataCleaning()),
                        ('remover', RemoveFeatures()),
                        ('scaler', FeatureScaling()),
                        ('merger', MergeFeature()),
                        ('selector', FeatureSelection()),
                        ('droper', DropNaN())])
    
    pipeline_y = Pipeline([('labels', GetLabels())])

    return pipeline_X, pipeline_y

def train(args):
    X, y = load_data()

    # transformations
    tf_features, tf_labels = transform()

    clf = MLPClassifier(hidden_layer_sizes=(500), 
                    max_iter=100, 
                    alpha=1e-4, 
                    activation='relu',
                    solver='adam', 
                    verbose=10, 
                    random_state=1,
                    learning_rate_init=.1)

    
    #preprocessing the dataset
    X_tf = tf_features.fit_transform(X)
    y_tf = GetLabels().transform(y, X_tf)

    X_tf, X_test, y_tf, y_test = train_test_split(X_tf, y_tf, train_size=0.8, random_state=42)

    X_tf = X_tf.values
    y_tf = y_tf.values
    
    clf.fit(X_tf, y_tf)

    print(f'Training set score: {clf.score(X_tf, y_tf)}')

    dump_folder = path.join(args['output_folder'], args['experiment_name'])
    print(dump_folder)

    #save models
    if not path.exists(dump_folder):
        mkdir(dump_folder)

    #dump model
    filename = 'model_mlp_{}_v0.1.pkl'.format(args['model_feature_name_tag'])
    joblib.dump(clf, filename=path.join(dump_folder, filename))

    #dump transform features
    filename = 'tf_std_{}_v0.1.pkl'.format(args['model_feature_name_tag'])
    joblib.dump(tf_features, filename=path.join(dump_folder, filename))

    #dump transform labels
    filename = 'tf_std_{}_v0.1.pkl'.format(args['model_label_name_tag'])
    joblib.dump(tf_labels, filename=path.join(dump_folder, filename))
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iris classifierr training 0.0.1")
    parser.add_argument("--experiment_name", default = 'exp_01_default' ,type=str)
    parser.add_argument("--output_folder", default=path.join(DIR_NAME, "models"), type=str)
    parser.add_argument("--model_feature_name_tag", default= 'user_infos',  type=str)
    parser.add_argument("--model_label_name_tag", default= 'user_statuss', type=str)
    args = vars(parser.parse_args())

    train(args)

