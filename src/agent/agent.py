from src.tools.tools import DatabaseTool,SQLGenerationTool,loadSkills,pokemonStat,loadMainSkills
from loguru import logger
from src.config import setting
from src.entity.request import ChatMessage, llm_base_client, llm_reason_client,langChainClient
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler


# 加载skills文件夹下的所有技能
skills_prompt = """你是一个【技能使用判定器】，不是直接回答问题的助手。

你的唯一任务是判断：  
【用户问题是否必须使用 skills（技能文件）才能完成】。

判定原则（严格遵守）：

1. 如果问题满足以下任意一条，必须使用技能：
   - 涉及结构化数据查询（数值、范围、统计、等级、属性、参数）
   - 涉及数据库 / 表 / 字段 / SQL / 查询流程
   - 涉及需要遵循“固定步骤 / 固定流程 / 标准化查找方式”的任务
   - 技能文件中可能定义了专用流程或规则

2. 如果问题满足以下全部条件，禁止使用技能：
   - 纯概念解释
   - 常识性知识
   - 不依赖内部数据或流程
   - 不要求精确数值或范围

3. 你【不能假设】自己知道技能内容，
   技能只能通过 loadSkills 工具加载后才能使用。

输出要求（非常重要）：
- 只允许输出 JSON
- 不要解释原因
- 不要回答用户问题

持有以下skills目录\n"""+loadMainSkills(setting.skills_path)


def PokemonAgent():
    PokemonAgent = create_agent(
        model=langChainClient,
        tools=[loadSkills,DatabaseTool,SQLGenerationTool,pokemonStat],
        system_prompt=skills_prompt)
    return PokemonAgent

class PrintMessagesCallback(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("====== 即将发送给模型的 messages ======")
        for i, m in enumerate(messages[0]):
            print(f"[{i}] {m.type}: {m.content}")
        print("=====================================")