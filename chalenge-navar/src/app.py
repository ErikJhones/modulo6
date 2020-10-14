from typing import Optional
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from os import path
from decouple import config as cfg #para variavel de ambiente
import argparse, joblib
from utils import load_models, check_inputs
from train import load_data2, transform
import numpy as np
from sklearn.model_selection import train_test_split

DIR_NAME = path.dirname(__file__)
DATA_FOLDER = path.join(DIR_NAME, 'models', 'exp_01_default') 
DATA_CSV = cfg('DATA_CSV', cast=str)

#load models
model, tf_feature = load_models()
datas = pd.read_csv(path.join(DATA_FOLDER, DATA_CSV))

class Item(BaseModel):
    #classe: Optional[str] = None
    user: Optional[str] = 'None'
    week: Optional[int] = 7
    total_sessions: Optional[int] = 4
    total_mediaids: Optional[int] = 7
    total_days: Optional[int] = 3
    total_played: Optional[float] = 130.450
    max_played_time: Optional[float] = 58.280
    age_without_access: Optional[int] = -285
    sexo: Optional[str] = 'M'
    idade: Optional[float] = 42
    cidade: Optional[str] = 'None'
    estado: Optional[str] = 'None'
    android_app_time: Optional[float] = 0.0
    ios_app_time: Optional[float] = 0.0
    tv_app_time: Optional[float] = 0.0
    mobile_web_time: Optional[float] = 0.0
    desktop_web_time: Optional[float] = 0.0
    time_spent_on_news: Optional[float] = 0.0
    time_spent_on_humor: Optional[float] = 0.0
    time_spent_on_series: Optional[float] = 0.0
    time_spent_on_novelas: Optional[float] = 32.689267
    time_spent_on_special: Optional[float] = 0.0
    time_spent_on_varieties: Optional[float] = 0.0
    time_spent_on_sports: Optional[float] = 0.0
    time_spent_on_realities: Optional[float] = 0.0
    time_spent_on_disclosure: Optional[float] = 0.0
    time_spent_on_archived: Optional[float] = 0.0
    time_spent_on_subscribed_content: Optional[float] = 120.002617
    time_spent_on_free_content: Optional[float] = 0.0
    time_spent_on_grade: Optional[float] = 109.735817
    video_info_excerpt_time: Optional[float] = 0.0
    video_info_extra_time: Optional[float] = 0.0
    video_info_episode_time: Optional[float] = 74.162383
    video_info_time_spent_0_5: Optional[float] = 0.0
    video_info_time_spent_5_15: Optional[float] = 0.0
    video_info_time_spent_15_30: Optional[float] = 0.0
    video_info_time_spent_30_60:  Optional[float] = 47.607417
    video_info_time_spent_60mais: Optional[float] = 0.0
    total_dependents: Optional[int] = 0
    total_active_dependents: Optional[int] = 0
    total_played_for_dependents: Optional[float] = 0.0
    tipo_de_cobranca: Optional[str] = 'CARTAO DE CREDITO'
    total_cancels: Optional[int] = 0
    month_subs: Optional[int] = 7
    assinatura_age: Optional[float] = 433.0


class Lep(BaseModel):
    item: dict

app = FastAPI()

@app.post("/class/")
async def create_item(item:Item):
    dic = dict(item)

    col = ['user', 'week', 'total_sessions', 'total_mediaids', 'total_days',
       'total_played', 'max_played_time', 'age_without_access', 'sexo',
       'idade', 'cidade', 'estado', 'android_app_time', 'ios_app_time',
       'tv_app_time', 'mobile_web_time', 'desktop_web_time',
       'time_spent_on_news', 'time_spent_on_humor', 'time_spent_on_series',
       'time_spent_on_novelas', 'time_spent_on_special',
       'time_spent_on_varieties', 'time_spent_on_sports',
       'time_spent_on_realities', 'time_spent_on_disclosure',
       'time_spent_on_archived', 'time_spent_on_subscribed_content',
       'time_spent_on_free_content', 'time_spent_on_grade',
       'video_info_excerpt_time', 'video_info_extra_time',
       'video_info_episode_time', 'video_info_time_spent_0_5',
       'video_info_time_spent_5_15', 'video_info_time_spent_15_30',
       'video_info_time_spent_30_60', 'video_info_time_spent_60mais',
       'total_dependents', 'total_active_dependents',
       'total_played_for_dependents', 'tipo_de_cobranca', 'total_cancels',
       'month_subs', 'assinatura_age']

    feature = pd.DataFrame(dic, columns=col, index=[0])
    #X, y = load_data2()
    X_train, X_test = train_test_split(datas, train_size=0.8, random_state=42)

    X = pd.concat([X_test, feature], axis = 0,ignore_index=True)
    X.set_index('user', inplace=True)
    
    X_tf = tf_feature.transform(X)
    
    y_hat = model.predict(X_tf.tail(1))

    return str(y_hat)

@app.post("/items/")
async def create_item(lep:Lep):
    col = ['user', 'week', 'total_sessions', 'total_mediaids', 'total_days',
       'total_played', 'max_played_time', 'age_without_access', 'sexo',
       'idade', 'cidade', 'estado', 'android_app_time', 'ios_app_time',
       'tv_app_time', 'mobile_web_time', 'desktop_web_time',
       'time_spent_on_news', 'time_spent_on_humor', 'time_spent_on_series',
       'time_spent_on_novelas', 'time_spent_on_special',
       'time_spent_on_varieties', 'time_spent_on_sports',
       'time_spent_on_realities', 'time_spent_on_disclosure',
       'time_spent_on_archived', 'time_spent_on_subscribed_content',
       'time_spent_on_free_content', 'time_spent_on_grade',
       'video_info_excerpt_time', 'video_info_extra_time',
       'video_info_episode_time', 'video_info_time_spent_0_5',
       'video_info_time_spent_5_15', 'video_info_time_spent_15_30',
       'video_info_time_spent_30_60', 'video_info_time_spent_60mais',
       'total_dependents', 'total_active_dependents',
       'total_played_for_dependents', 'tipo_de_cobranca', 'total_cancels',
       'month_subs', 'assinatura_age']

    x = pd.DataFrame(lep.item, columns=col, index=[0])
    X, y = load_data2()
    X_train, X_test = train_test_split(X, train_size=0.8, random_state=42)

    X = pd.concat([X_test, x], axis = 0,ignore_index=True )
    X.set_index('user', inplace=True)
    
    X_tf = tf_feature.transform(X)
    
    y_hat = model.predict(X_tf.tail(1))

    return str(y_hat)