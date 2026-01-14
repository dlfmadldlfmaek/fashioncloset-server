# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

@app.on_event("startup")
def load_routes():
    from api.recommend import router
    app.include_router(router)
