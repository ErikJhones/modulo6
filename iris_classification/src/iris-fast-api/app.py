'''from typing import Optional

from os import path
from decouple import config as cfg #para variavel de ambiente
import argparse, joblib
from utils import load_models, check_inputs
from train import load_data
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi import FastAPI, Request

app = FastAPI()



@app.get("/")
def root():
    a = "a"
    b = "b" + a
    return {"hello world": b}

@app.post("/predicit/")
def create_item():
    x = check_inputs(Request.json['features'])
    return 'predicit'

@app.get('/predict')
def predict():
    if request.method == 'POST':
        #x = check_inputs(request.json['features'])
        
        x = np.array(request.json['features']).reshape(1,-1)
        y_hat = model.predict(tf.transform(x))

        return jsonify(output={"y_hat": y_hat.tolist()}, status=200, message="Model Working")
    return {"hello world": b}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)'''


from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from os import path
from decouple import config as cfg #para variavel de ambiente
import argparse, joblib
from utils import load_models, check_inputs
from train import load_data
import numpy as np

#load models
model, tf = load_models()

class Item(BaseModel):
    features: list
    classe: Optional[str] = None


app = FastAPI()

@app.post("/items/")
async def create_item(item: Item):
    x = np.array(item.features).reshape(1,-1)
    y_hat = model.predict(tf.transform(x))
    item.classe = str(y_hat)

    return item