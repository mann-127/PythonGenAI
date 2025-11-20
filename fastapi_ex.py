 
from fastapi import FastAPI
from pydantic import BaseModel
 
# Create app object
app = FastAPI()
 
# --------------- Simple GET Example ----------------
@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI 🎉"}
 
@app.get("/hello/{name}")
def say_hello(name: str):
    return {"message": f"Hello, {name}!"}
 
# --------------- POST Example with JSON ----------------
class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True
 
@app.post("/items/")
def create_item(item: Item):
    return {
        "status": "success",
        "data": item.dict()
    }
 
# --------------- Query Parameters ----------------
@app.get("/search/")
def search_items(q: str, limit: int = 5):
    return {"query": q, "limit": limit, "results": [f"Item {i}" for i in range(limit)]}
