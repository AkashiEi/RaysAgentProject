from fastapi import FastAPI,APIRouter
from src.config import setting
# from src.router import api_router
import uvicorn
from src.entity.retrieval_api import create_app as CA
from retrieval_app import ensure_retrieval_alive
import requests
import time
from loguru import logger

ensure_retrieval_alive()

retrieval_router = APIRouter()

@retrieval_router.get('/emb/start',summary='手动启动retrieval模型')
def start():
    retrieval_host = "127.0.0.1" if setting.retrieval_host == "0.0.0.0" else setting.retrieval_host
    EMBEDDING_URL = f"http://{retrieval_host}:{setting.retrieval_port}"
    r = requests.get(EMBEDDING_URL+f"/{setting.retrieval_prefix}/v1/shutdown",proxies={"http": None, "https": None}, timeout=1)

    if r.status_code == 200 and r.json().get("status") == 1:
        logger.info(f"{setting.retrieval_prefix}模型已关闭")
        time.sleep(1)
        ensure_retrieval_alive()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=setting.project_name,
        description=setting.project_description,
        version=setting.project_version,
    )
    register_routes(app)

    return app


def register_routes(app: FastAPI) -> None:
    """Register routes to the FastAPI application."""
    app.include_router(retrieval_router, prefix=f"/{setting.server_prefix}")

app = create_app()

if __name__ == '__main__':
    uvicorn.run(app,host=setting.server_host,port=setting.server_port,workers=setting.server_workers)