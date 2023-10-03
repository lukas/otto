# Load a model from an artifact, and then server, logging the results to a streamtable

import os
from contextlib import asynccontextmanager

from pydantic import BaseModel
import weave
from weave.monitoring import StreamTable
from fastapi import FastAPI

import util
import model_basic


stream_table = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global stream_table

    model_uri =os.environ['MODEL_URI']
    weave_model_uri = util.get_weave_uri_from_wandb_uri(model_uri)
    model = weave.storage.get(weave_model_uri)

    stream_table_name = os.environ['STREAM_TABLE_NAME']
    stream_table = StreamTable(stream_table_name)

    yield

app = FastAPI(lifespan=lifespan)


class Item(BaseModel):
    example: str

@app.post("/predict")
def predict(item: Item):
    result = weave.use(model.predict(item.example))
    record = {
        'model': model,
        'example': item.example,
        'result': result,
    }
    stream_table.log(record)
    return {'prediction': result}