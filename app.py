from fastapi import FastAPI
import pandas as pd

# 상대 경로로 CSV 파일 읽기
df = pd.read_csv('data/모니카데이터_전자제품.csv')
print(df.head())

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, Heroku!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
