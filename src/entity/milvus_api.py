from pymilvus import MilvusClient
from src.config import setting
from loguru import logger
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import requests

class searchRequest(BaseModel):
    collection_name: str
    query: str
    top_k: int = 10
    output_fields: List[str]

milvus_client = MilvusClient(
    uri=setting.milvus_uri,
    token=setting.milvus_token
)

milvus_router = APIRouter()

@milvus_router.get("/collections")
async def list_collections():
    try:
        collections = milvus_client.list_collections()
        return {"status": 1, "collections": collections}
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail="Failed to list collections")

@milvus_router.post("/search")
async def search(request: searchRequest):
    query_vector = requests.post(
                "http://127.0.0.1:26226/retrieval/v1/embedding/normal",
                json={"context": request.query},
                proxies={"http": None, "https": None}).json()["data"][0]["embedding"]
    try:
        results = milvus_client.search(
            collection_name=request.collection_name,
            data=[query_vector],
            anns_field="vector",
            limit=request.top_k,
            output_fields=request.output_fields
        )
        return {"status": 1, "results": results}
    except Exception as e:
        logger.error(f"Failed to search collection: {e}")
        raise HTTPException(status_code=500, detail="Failed to search collection")
    
