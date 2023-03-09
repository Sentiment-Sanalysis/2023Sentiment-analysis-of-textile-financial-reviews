from fastapi import fi

app = fi()


@app.get("/")
async def root():
    return {"message": "Hello World"}