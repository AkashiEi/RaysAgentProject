from src.tools.tools import DatabaseTool,SQLGenerationTool,loadSkills,pokemonStat,loadMainSkills,milvus_search
from loguru import logger
from src.config import setting
from src.entity.request import ChatMessage, llm_base_client, llm_reason_client,langChainClient
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from src.database.db_connect import get_agent_db_saver


# 加载skills文件夹下的所有技能
skills_prompt = """
你是一个具备工具调用能力的智能体。

当问题涉及以下情况时，必须直接调用对应工具，而不是输出文本判断：

1. 使用向量数据库查询文本信息时 → 调用 milvus_search 查询对应的小说内容
2. 使用结构化数据库查询数据时 → 调用 DatabaseTool
3. 需要生成 SQL语句时 → 调用 SQLGenerationTool
4. 需要读取技能规则 → 调用 loadSkills
5. 需要计算宝可梦属性数值 → 调用 pokemonStat

禁止：
- 输出 requires_skill
- 仅做判断不执行
- 直接回答小说剧情问题

当满足触发条件时，必须直接发起 tool_call。
不要输出普通文本。

当需要访问向量数据库和结构化数据库时：
步骤1：必须先调用 loadSkills 加载 data_search_flow.md
步骤2：解析其中定义的字段名和查询流程
步骤3：按照文档中的字段名调用 milvus_search或DatabaseTool，获取结果后继续后续步骤
步骤4：禁止猜测字段名，必须严格按照文档中的字段名调用工具，当查询结构化数据时，必须先使用loadSkills获取对应的表结构,再调用 SQLGenerationTool 生成 SQL语句，最后调用 DatabaseTool 执行查询。
如果未读取 data_search_flow.md，禁止调用 milvus_search和DatabaseTool。

持有以下skills目录\n"""+loadMainSkills(setting.skills_path)


def RayAgent():
    checkpoint = get_agent_db_saver()
    RayAgent = create_agent(
        model=langChainClient,
        tools=[loadSkills,DatabaseTool,SQLGenerationTool,pokemonStat,milvus_search],
        system_prompt=skills_prompt,
        checkpointer=checkpoint)
    return RayAgent

class PrintMessagesCallback(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("====== 即将发送给模型的 messages ======")
        for i, m in enumerate(messages[0]):
            print(f"[{i}] {m.type}: {m.content}")
        print("=====================================")