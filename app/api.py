from fastapi import FastAPI

app = FastAPI()


@app.get('/', tags=['Root'])
async def hello():
    return {'hello': 'you success deploy guys..'}
