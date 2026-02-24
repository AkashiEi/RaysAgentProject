from typing import List, Optional, Literal
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel
from openai import OpenAI
from src.config import setting
from langchain_openai import ChatOpenAI

class ChatMessage(ChatCompletionMessage):
    # 消息的角色
    role: Literal["system", "user", "assistant", "function"]

# openai的聊天请求
class ChatCompletionRequest(BaseModel):
    # 消息列表
    messages: List[ChatMessage]
    # 模型
    model: Optional[str] = None

# 普通模型
llm_base_client = OpenAI(api_key=setting.llm_api_key, base_url=setting.llm_base_url)
# 思考模型
llm_reason_client = OpenAI(api_key=setting.llm_api_key, base_url=setting.llm_reason_url)

# Agent
langChainClient = ChatOpenAI(
    api_key=setting.llm_api_key,
    base_url=setting.llm_base_url,
    model =setting.llm_chat_model,
    stream_usage=True,
    temperature=1.0)