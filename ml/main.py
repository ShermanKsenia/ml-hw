from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pickle
import uvicorn
import pandas as pd
import io

app = FastAPI()
cat_features = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
fill_values = {
    'mileage': 19.369999999999997,
    'engine': 1248.0,
    'max_power': 81.86,
    'torque': 150.0,
    'seats': 5.0,
    'max_torque_rpm': 2800.0
 }

with open("./models/model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open("./models/ohe_hot_encoder.pkl", 'rb') as ohe_file:
    ohe = pickle.load(ohe_file)

with open("./models/scaler.pkl", 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    torque: float
    seats: int
    max_torque_rpm: float

class Items(BaseModel):
    objects: List[Item]

def prepare_data(data):
    data = data.fillna(fill_values)
    data_cat = data[cat_features]
    ohe_cats = ohe.transform(data_cat)
    ohe_cats = pd.DataFrame(ohe_cats.toarray(), columns=[cat for cats in ohe.categories_ for cat in cats[1:]])
    data_cat = pd.concat([data.reset_index(), ohe_cats.reset_index()], axis=1)
    data_cat = data_cat.drop(columns='index')
    data_cat = data_cat.select_dtypes(include='number')
    data_cat.columns = [str(col) for col in data_cat.columns]

    data_scaled = scaler.transform(data_cat)

    return data_scaled

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame(item.model_dump(), index=[0])
    data_scaled = prepare_data(data)
    prediction = model.predict(data_scaled)
    return prediction[0]

@app.post("/predict_items")
def predict_items(csvfile: UploadFile = File(...)) -> StreamingResponse:
    data = pd.read_csv(csvfile.file, index_col=0)
    data_scaled = prepare_data(data)
    prediction = model.predict(data_scaled)
    data['prediction'] = prediction

    stream = io.StringIO()
    data.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )
    
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    
    return response

if __name__ == '__main__':
    uvicorn.run('main:app', port=8000, reload=True) #host='127.0.0.1',