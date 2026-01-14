# main.py
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok"}


# 🔥 핵심: recommend 라우터를 startup 이후에 로드
@app.on_event("startup")
def load_routes():
    from api.recommend import router as recommend_router
    app.include_router(recommend_router)
