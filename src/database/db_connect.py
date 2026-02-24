from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config import setting
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config import setting
from loguru import logger

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