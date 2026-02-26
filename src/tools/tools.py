from abc import ABC, abstractmethod
from typing import Any, List, Dict, Union, Optional
from pydantic import BaseModel
from src.database import get_db,get_milvus_client
import json
import re
from loguru import logger
from src.config import setting
from src.entity.request import ChatMessage, llm_base_client, llm_reason_client,langChainClient
from sqlalchemy import text
from langchain.tools import tool
import requests


def loadMainSkills(path: str):
    """
    加载文件夹下的所有技能文件及内容
    Args:
        path (str): 技能文件夹路径
    """
    import os

    skills = {}
    if not os.path.isdir(path):
        raise ValueError(f"The provided path '{path}' is not a valid directory.")
    
    for filename in os.listdir(path):
        if filename.endswith(".md"):
            file_path = os.path.join(path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    skills[filename] = content
                    logger.info(f"Loaded skill file: {filename}")
            except Exception as e:
                logger.error(f"Failed to load skill file '{filename}': {e}")
    
    if not skills:
        logger.warning(f"No skill files found in directory: {path}")
    
    return str(skills)

@tool
def loadSkills(path: str):
    """
    加载文件夹下的所有技能文件及内容
    Args:
        path (str): 技能文件夹路径
    """
    import os

    skills = {}
    if not os.path.isdir(path):
        raise ValueError(f"The provided path '{path}' is not a valid directory.")
    
    for filename in os.listdir(path):
        if filename.endswith(".md"):
            file_path = os.path.join(path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    skills[filename] = content
                    logger.info(f"Loaded skill file: {filename}")
            except Exception as e:
                logger.error(f"Failed to load skill file '{filename}': {e}")
    
    if not skills:
        logger.warning(f"No skill files found in directory: {path}")
    
    return {"result": skills}

@tool
def DatabaseTool(query: str,name:str):
    """执行数据库查询的工具
    Args:
        query (str): 要执行的SQL查询语句
        name (str): 对目前所进行的事情做简单描述
    """
    db_connection = get_db(name)
    if not query:
        raise ValueError("No query provided for database execution.")
    try :
        result = db_connection.execute(text(query))
        rows = result.fetchall()
        message = {"result": [row._asdict() for row in rows]}
        logger.info(f"Database query executed successfully: {query[:50]}...")
        return message
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        message = {"result": str(e)}
        return message
    
@tool
def SQLGenerationTool(table_name:str,user_request:str,user_model:str="chat"):
    """根据表结构和用户需求生成SQL查询语句的工具
    Args:
        table_name (str): 表结构信息
        user_request (str): 用户的查询需求
        user_model (str): 使用的大模型类型，默认为"chat"，可选"reason"
    """
    try:
        if not table_name or not user_request:
            raise ValueError("Both table_name and user_request must be provided.")
        prompt = f"""请根据以下的表结构和用户需求，生成对应的SQL查询语句。\n表结构: {table_name}\n用户需求: {user_request}\n注意：最后需要以```sql```格式返回sql语句，sql语句中不需要有换行，遇到复杂需求时需要有思考过程,且sql语句中需要去重。重点规则：允许WITH子句，但进行危险关键字检查，禁止出现UPDATE、DELETE、INSERT、DROP、TRUNCATE、ALTER、CREATE等操作，只允许查询操作。如果生成的SQL语句中包含危险操作，请重新生成只包含查询的SQL语句。"""
        # 调用大模型生成SQL语句
        print(prompt)
        model_client = llm_base_client if user_model == "chat" else llm_reason_client
        model_name = setting.llm_chat_model if user_model == "chat" else setting.llm_reason_model
        logger.info(f"Using model {model_name} for SQL generation.")
        resp = model_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个SQL生成专家，擅长根据用户需求生成高效的SQL查询语句。"},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            temperature=0.2
        )
        full_reasoning = []
        full_response = []
        reson_satrt = 0
        anwser_start = 0
        for chunk in resp:
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                reson_satrt += 1
                if reson_satrt == 1:
                    print("\n\n【Tools】SQL语句生成工具-Thinking",end="\n", flush=True)
                reasoning_fragment = chunk.choices[0].delta.reasoning_content
                full_reasoning.append(reasoning_fragment)
                print(f"{reasoning_fragment}",end="", flush=True)

            if chunk.choices[0].delta.content is not None:
                anwser_start += 1
                if anwser_start == 1:
                    print("\n\n【Tools】SQL语句生成工具-Answer",end="\n", flush=True)
                content_fragment = chunk.choices[0].delta.content
                print(f"{content_fragment}",end="", flush=True)
                full_response.append(content_fragment)
        SQLGenerationTool_response = "".join(full_response)
        logger.info(f"SQL generation response: {SQLGenerationTool_response}")
        sql_match = re.findall(r"```sql(.*?)```", SQLGenerationTool_response, re.DOTALL)
        # print(sql_match)
        if not sql_match:
            # 如果没有匹配到 SQL 语句，则返回错误
            raise ValueError("Failed to generate SQL query. No SQL found in the response.")
        sql_query = sql_match[-1].strip()+';'
        if sql_query.lower().startswith("select"):
            pass
        else:
            raise ValueError("Generated SQL is not a SELECT statement.")
        message = {"result": {"query":sql_query}}
        return message
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        message = {"result": str(e)}
    return message


@tool
def pokemonStat(IV:int,EV:int,base_stat:int,level:int,nature:str,stat_type:str):
    """
    基于等级、个体值、努力值、宝可梦种族值计算宝可梦的实际能力值
    Args:
        IV (int): 个体值,取值范围0-31
        EV (int): 努力值，取值范围0-255
        base_stat (int): 基础种族值
        level (int): 宝可梦等级
        nature (float): 性格修正，取值0.9/1.0/1.1
        stat_type (str): 计算的种族值类型，"HP"-HP种族值，其他为非HP种族值
    """
    logger.info(f"Pokemon Stat {stat_type} IV:{IV} EV:{EV} baseStat:{base_stat} nature:{nature}")
    try:
        nature = float(nature)
        if nature not in [0.9, 1.0, 1.1]:
            print(f"[Warning] nature不在有效范围: {nature}, 使用默认值1.0")
            nature = 1.0
    except Exception as e:
        print(f"[Warning] nature参数异常: {nature}, 使用默认值1.0")
        nature = 1.0

    if stat_type.lower() == "hp":
        actual_stat = ((2 * base_stat + IV + (EV // 4)) * level) // 100 + level + 10
    else:
        actual_stat = (((2 * base_stat + IV + (EV // 4)) * level) // 100 + 5) * nature
    logger.info(f"Pokemon Stat {stat_type} actualStat:{actual_stat}")
    return {"result": {"actual_stat": int(actual_stat)}}

@tool
def milvus_search(collection_name:str,query:str,output_fields:List[str],top_k:int=5):
    """
    基于Milvus向量数据库的相似度搜索工具
    Args:
        collection_name (str): Milvus中的集合名称
        query (List[float]): 查询的向量表示
        top_k (int): 返回的最相似结果数量，默认为5,
        output_fields (List[str]): 需要返回的字段列表
    """
    try:
        query_vector = requests.post(
                f"http://{setting.retrieval_host}:{setting.retrieval_port}/{setting.retrieval_prefix}/v1/embedding/normal",
                json={"context": query},proxies={"http": None, "https": None}).json()["data"][0]["embedding"]
        milvus_client = get_milvus_client()
        results = milvus_client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="vector",
            limit=top_k,
            output_fields=output_fields
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Milvus search failed: {e}")
        return {"result": str(e)}