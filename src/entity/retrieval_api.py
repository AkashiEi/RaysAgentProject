from fastapi import APIRouter,FastAPI,HTTPException
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM,Qwen2Tokenizer
from src.config import setting
from pydantic import BaseModel
from loguru import logger
from typing import Literal,List,Union
import uvicorn
from datetime import datetime
import threading
import time
import os
from contextlib import asynccontextmanager
import asyncio

class ApiResp(BaseModel):
    """
    status:状态 -1失败，1成功
    data:返回内容
    """
    status:Literal[-1, 1]
    data: dict | None = None

class RerankContext(BaseModel):
    query: str
    doc: List[str]
    top_k : int = 5

class ApiRequ(BaseModel):
    context:Union[str, RerankContext]

class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]

class UsageInfo(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    model: str
    data: List[EmbeddingItem]
    usage: UsageInfo

class RerankData(BaseModel):
    object: str = "reranking"
    index: int
    score: float
    document: str

class RerankResponse(BaseModel):
    object: str = "list"
    data: List[RerankData]
    model: str
    usage: UsageInfo

if torch.cuda.is_available():
    logger.info("检测到GPU，优先使用GPU进行模型推理")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    logger.info("检测到Apple Silicon GPU，优先使用MPS进行模型推理")
    device = torch.device("mps")
    torch.mps.empty_cache()
else:
    logger.info("未检测到GPU，使用CPU进行模型推理")
    device = torch.device("cpu")

api_router = APIRouter()
embedding_path = setting.embedding_path
embedding_name = setting.embedding_model
logger.info("读取EMBEDDING模型位置")
reranker_path = setting.reranker_path
reranker_name = setting.reranker_model
logger.info("读取RERANKER模型位置")

_embedding_tokenizer = None
_embedding_model = None
EMBEDDING_MODEL_STATUS = "Fail"

def load_embedding_model():
    global _embedding_tokenizer, _embedding_model,EMBEDDING_MODEL_STATUS
    try :
        if _embedding_tokenizer is None:
            EMBEDDING_MODEL_STATUS = "Loading"
            _embedding_tokenizer = AutoTokenizer.from_pretrained(
                embedding_name, 
                cache_dir=embedding_path, 
                trust_remote_code=True, 
                use_fast=True
            )
            _embedding_model = AutoModel.from_pretrained(
                embedding_name, 
                cache_dir=embedding_path
            ).to(device)
            EMBEDDING_MODEL_STATUS = "Ready"
            logger.info("完成加载EMBEDDING模型")
        return _embedding_tokenizer, _embedding_model,EMBEDDING_MODEL_STATUS
    except Exception:
        EMBEDDING_MODEL_STATUS = "Fail"

_reranker_tokenizer = None
_reranker_model = None
RERANKER_MODEL_STATUS = "Fail"

def load_reranker_model():
    global _reranker_tokenizer, _reranker_model,RERANKER_MODEL_STATUS
    try :
        if _reranker_tokenizer is None:
            RERANKER_MODEL_STATUS = "Loading"
            _reranker_tokenizer = AutoTokenizer.from_pretrained(
                reranker_name, 
                cache_dir=reranker_path, 
                trust_remote_code=True, 
                use_fast=True
            )
            _reranker_model = AutoModelForCausalLM.from_pretrained(
                reranker_name, 
                cache_dir=reranker_path
            ).to(device).eval()
            RERANKER_MODEL_STATUS = "Ready"
            logger.info("完成加载RERANKER模型")
        return _reranker_tokenizer, _reranker_model,RERANKER_MODEL_STATUS
    except Exception:
        RERANKER_MODEL_STATUS = "Fail"

# 后台加载模型
embedding_model_ready = False
async def load_embedding_background():
    global model_ready
    try:
        logger.info("开始后台加载 embedding 模型")
        load_embedding_model()
        model_ready = True
        logger.info("embedding 模型加载完成")
    except Exception:
        logger.exception("embedding 模型加载失败")
        model_ready = False

rereanker_model_ready = False
async def load_reranker_background():
    global rereanker_model_ready
    try:
        logger.info("开始后台加载 reranker 模型")
        load_reranker_model()
        rereanker_model_ready = True
        logger.info("reranker 模型加载完成")
    except Exception:
        logger.exception("reranker 模型加载失败")
        rereanker_model_ready = False

# 解读embedding向量
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


@api_router.post('/v1/embedding/normal',
                 response_model=EmbeddingResponse,
                 summary='对文本内容进行向量化操作',
                 responses={500: {"description": "Embedding error",
                                  "content": {
                                      "application/json": {
                                          "example": {
                                              "error": {
                                                  "message": "embedding failed",
                                                  "type": "embedding_error",
                                                  "code": "embedding_failed"
                            }}}}}})
async def embedding_(reuqest:ApiRequ):
    global EMBEDDING_MODEL_STATUS
    context = reuqest.context
    if isinstance(context, str):
         logger.info(f"文本：{context} 开始向量化")
    else:
        return HTTPException(
        status_code=500,
        detail={
            "error": {
                "message": "Invalid context format for embedding",
                "type": "embedding_error",
                "code": "embedding_failed"
            }
        }
    )
    try :
        embedding_tokenizer, embedding_model,EMBEDDING_MODEL_STATUS = load_embedding_model()
        if context is None:
            return {
                "status":-1,
                "data":None
            }
        max_length = 8192

        # Tokenize the input texts
        batch_dict = embedding_tokenizer(
            context,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        batch_dict.to(embedding_model.device)

        outputs = embedding_model(**batch_dict)

        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embedding_vector = embeddings[0].detach().cpu().tolist()
        logger.info(f"文本：{context} 完成向量化")
        return {
            "object": "list",
            "model": embedding_name,
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": embedding_vector
                }
            ],
            "usage": {
                "prompt_tokens": int(batch_dict["attention_mask"].sum().item()),
                "total_tokens": int(batch_dict["attention_mask"].sum().item())
            }
        }
    except Exception as e:
        logger.error(str(e))
        return HTTPException(
        status_code=500,
        detail={
            "error": {
                "message": str(e),
                "type": "embedding_error",
                "code": "embedding_failed"
            }
        }
    )

@api_router.get('/v1/health/embedding',response_model=ApiResp,summary='检测embedding接口存活')
async def health_check():
    global EMBEDDING_MODEL_STATUS
    if EMBEDDING_MODEL_STATUS in ["Loading","Ready"]:
        return {
            "status": 1,
            "data": {
                "embedding":embedding_name,
                "service": "retrieval-api",
                "EMBEDDING_MODEL_STATUS":EMBEDDING_MODEL_STATUS,
                "alive": True,
                "time": datetime.now().isoformat()
            }}
    else :
        return {
            "status": -1,
            "data": {
                "embedding":None,
                "service": "retrieval-api",
                "EMBEDDING_MODEL_STATUS":EMBEDDING_MODEL_STATUS,
                "alive": False,
                "time": datetime.now().isoformat()
            }}

# reranker部分
token_true_id, token_false_id,prefix_tokens, suffix_tokens,task = None, None, None, None, None
def rerank_comsume():
    global _reranker_tokenizer, _reranker_model, token_true_id, token_false_id,prefix_tokens, suffix_tokens,task
    token_false_id = _reranker_tokenizer.convert_tokens_to_ids("no")
    token_true_id = _reranker_tokenizer.convert_tokens_to_ids("yes")
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = _reranker_tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = _reranker_tokenizer.encode(suffix, add_special_tokens=False)
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    return token_false_id, token_true_id, prefix_tokens, suffix_tokens,task
        

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs):
    global _reranker_tokenizer, _reranker_model, token_true_id, token_false_id,prefix_tokens, suffix_tokens
    max_length = 8192
    inputs = _reranker_tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = _reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(_reranker_model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    global _reranker_tokenizer, _reranker_model, token_true_id, token_false_id
    batch_scores = _reranker_model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

@api_router.post('/v1/reranker/normal',summary='reranker接口',response_model=RerankResponse,responses={500: {"description": "Reranker error",
                                  "content": {
                                      "application/json": {
                                          "example": {
                                              "error": {
                                                  "message": "reranker failed",
                                                  "type": "reranker_error",
                                                  "code": "reranker_failed"
                            }}}}}})
async def reranker_(reuqest:ApiRequ):
    global RERANKER_MODEL_STATUS
    context = reuqest.context
    if isinstance(context, RerankContext):
        logger.info(f"query：{context.query} 开始rerank")
    else:
            return HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Invalid context format for reranker",
                    "type": "reranker_error",
                    "code": "reranker_failed"
                }
            }
        )
    try :
        query = context.query
        docs = context.doc
        top_k = context.top_k
        if context is None:
            return {
                "status":-1,
                "data":None
            }
        _reranker_tokenizer, _reranker_model,RERANKER_MODEL_STATUS = load_reranker_model()
        token_false_id, token_true_id, prefix_tokens, suffix_tokens,task = rerank_comsume()
        pairs = []
        for doc in docs:
            instruction = format_instruction(task, query=query, doc=doc)
            pairs.append(instruction)

        inputs = process_inputs(pairs)
        scores = compute_logits(inputs)

        result = []
        result = sorted(
            [
                {
                    "object": "reranker",
                    "index": i,
                    "score": float(score),
                    "document": doc
                }
                for i, (score, doc) in enumerate(zip(scores, docs))
            ],
            key=lambda x: x["score"],
            reverse=True
        )

        return {
            "object": "list",
            "model": reranker_name,
            "data": result[:top_k],
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        }
    except Exception as e:
        logger.error(str(e))
        return HTTPException(
        status_code=500,
        detail={
            "error": {
                "message": str(e),
                "type": "reranker_error",
                "code": "reranker_failed"
            }
        }
    )

@api_router.get("/v1/health/reranker",response_model=ApiResp,summary='检测reranker接口存活')
async def health_check_reranker():
    global RERANKER_MODEL_STATUS
    if RERANKER_MODEL_STATUS in ["Loading","Ready"]:
        return {
            "status": 1,
            "data": {
                "reranker":reranker_name,
                "service": "retrieval-api",
                "RERANKER_MODEL_STATUS":RERANKER_MODEL_STATUS,
                "alive": True,
                "time": datetime.now().isoformat()
            }}
    else :
        return {
            "status": -1,
            "data": {
                "reranker":None,
                "service": "retrieval-api",
                "RERANKER_MODEL_STATUS":RERANKER_MODEL_STATUS,
                "alive": False,
                "time": datetime.now().isoformat()
            }}


@api_router.get("/v1/shutdown")
def shutdown():
    def _shutdown():
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()

    return {
        "status": 1,
        "message": "Embedding service is shutting down"
    }

@api_router.get("/v1/satrtup")
def satrtup():
    if EMBEDDING_MODEL_STATUS != "Ready" or EMBEDDING_MODEL_STATUS != "Loading":
        load_embedding_model()
    if RERANKER_MODEL_STATUS != "Ready" or RERANKER_MODEL_STATUS != "Loading":
        load_reranker_model()
    
    return {
        "status": 1,
        "message": "Embedding and Reranker service is started up"
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Retrieval 服务启动，开始加载模型")
    asyncio.create_task(load_embedding_background())
    asyncio.create_task(load_reranker_background())
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="retrieval",description="语义检索",version="0.1",lifespan=lifespan)

    app.include_router(api_router,prefix=f"/{setting.retrieval_prefix}",tags=["retrieval"])

    return app
