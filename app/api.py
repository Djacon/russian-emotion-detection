from fastapi import FastAPI, Request, HTTPException
from .inference import predict_emotions

app = FastAPI()

# You can choose your own website hosts
allowed_hosts = [
    "127.0.0.1",
    "djacon.github.io"
]


@app.post('/predict_emotion')
async def predict(request: Request, text: str):
    client_host = request.client.host

    if client_host not in allowed_hosts:
        raise HTTPException(status_code=403, detail="Доступ запрещен")

    return predict_emotions(text)
