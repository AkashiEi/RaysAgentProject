from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config import setting
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config import setting
from loguru import logger
from pymilvus import MilvusClient
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
import pymysql

# MySQL数据库连接配置
engine = create_engine(f"mysql+pymysql://{setting.database_user}:{quote_plus(setting.database_password)}@{setting.database_url}:{setting.database_port}/{setting.database_name}?charset=utf8mb4", pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db(name):
    db = SessionLocal()
    logger.info(f"{name}-Database session created")
    try:
        logger.info(f"{name}-Database connection successful")
        return db
    finally:
        db.close()

def get_milvus_client():
    try:
        client = MilvusClient(uri=setting.milvus_uri,token=setting.milvus_token)
        logger.info("Milvus client created successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to create Milvus client: {e}")
        raise

def get_agent_db_saver():
    conn = pymysql.connect(
        host=setting.agent_db_url,
        port=setting.agent_db_port,
        user=setting.agent_db_user,
        password=setting.agent_db_password,
        database=setting.agent_db_name,
        charset='utf8mb4',
        autocommit=True
    )
    checkpointer = PyMySQLSaver(conn)
    checkpointer.setup()
    logger.info("Agent database saver created successfully")
    return checkpointer
