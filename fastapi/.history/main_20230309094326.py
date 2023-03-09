from fastapi import ai

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}