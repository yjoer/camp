# ruff: noqa: S104
import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def hello() -> dict:
  return {"hello": "world"}


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=3000, log_level="warning")
