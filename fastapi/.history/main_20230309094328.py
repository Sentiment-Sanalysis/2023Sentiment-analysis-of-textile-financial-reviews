from fastapi import fi

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}