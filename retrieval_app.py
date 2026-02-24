import subprocess
import requests
import time
from src.config import setting
import sys
from src.entity.retrieval_api import create_app as CA,load_embedding_model
import uvicorn
from loguru import logger

retrieval_host = "127.0.0.1" if setting.retrieval_host == "0.0.0.0" else setting.retrieval_host
app = CA()

def is_embedding_alive() -> bool:
    EMBEDDING_URL = f"http://{retrieval_host}:{setting.retrieval_port}/{setting.retrieval_prefix}/v1/health/embedding"
    logger.info(EMBEDDING_URL)
    try:
        r = requests.get(EMBEDDING_URL,proxies={"http": None, "https": None}, timeout=1)
        return r.status_code == 200 and r.json().get("status") == 1
    except Exception:
        return False

def is_reranker_alive() -> bool:
    RERANKER_URL = f"http://{retrieval_host}:{setting.retrieval_port}/{setting.retrieval_prefix}/v1/health/reranker"
    logger.info(RERANKER_URL)
    try:
        r = requests.get(RERANKER_URL,proxies={"http": None, "https": None}, timeout=1)
        return r.status_code == 200 and r.json().get("status") == 1
    except Exception:
        return False

def start_retrieval_server():
    logger.info("retrieval 服务未启动，正在启动...")
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "retrieval_app:app",
            "--host", setting.retrieval_host,
            "--port", str(setting.retrieval_port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def ensure_retrieval_alive(timeout=120):
    if not is_embedding_alive() or not is_reranker_alive():
        start_retrieval_server()

    # start = time.time()
    # while time.time() - start < timeout:
    #     if is_embedding_alive():
    #         logger.info("Embedding 服务已就绪")
    #         return
    #     time.sleep(1)
    while True :
        if is_embedding_alive() and is_reranker_alive():
            logger.info("retrieval 服务已就绪")
            return

    raise RuntimeError("Embedding 服务启动失败")

if __name__ == "__main__":
    uvicorn.run(app, host=setting.retrieval_host, port=setting.retrieval_port,workers=setting.retrieval_workers)