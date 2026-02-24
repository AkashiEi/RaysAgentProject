import json
import os
from dotenv import load_dotenv
from typing import Optional,Dict,Any
from pydantic_settings import SettingsConfigDict, BaseSettings
from pathlib import Path

load_dotenv()

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent
PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"

class Setting(BaseSettings):
    model_config = SettingsConfigDict(validate_default=False)
    # 服务配置
    project_name: str
    project_description: str
    project_version: str
    server_host: Optional[str] = '0.0.0.0'
    server_port: int
    server_prefix: str
    server_workers: int
    # 大模型配置
    llm_base_url: str
    llm_reason_url: str
    llm_chat_model: str
    llm_reason_model: str
    llm_api_key: str
    # 数据库配置
    database_url: str
    database_port: int
    database_user: str
    database_password: str
    database_name: str

    # 获取skills根目录
    skills_path: str

    # Embedding model
    embedding_path: str
    embedding_model: str

    # Reranker model
    reranker_path: str
    reranker_model: str

    retrieval_host: Optional[str] = '0.0.0.0'
    retrieval_port: int
    retrieval_prefix: str
    retrieval_workers: int


    
    @property
    def workspace_root(self) -> Path:
        """Get the workspace root directory"""
        return WORKSPACE_ROOT

setting = Setting()
