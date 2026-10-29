
from fastapi import FastAPI

app = FastAPI()

@app.get('/create')
def create():
    return "UPDATE"