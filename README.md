# RayAgentProject

```
RaysAgentProject/          # 项目根目录
├── .env.example           # 环境变量示例文件
├── .gitignore             # Git 忽略文件配置
├── LICENSE                # 项目许可证文件
├── README.md              # 项目说明文档（更新日志/功能说明）
├── app.py                 # 项目主应用入口
├── retrieval_app.py       # 检索相关应用入口
├── milvus/                # Milvus 向量库相关模块
│   └── milvus_collections.py  # Milvus 集合（表）操作逻辑
└── src/                   # 核心源代码目录
    ├── __init__.py        # 包初始化文件
    ├── agent/             # 智能体核心逻辑模块
    ├── config/            # 项目配置模块（配置项/常量等）
    ├── database/          # 数据库交互模块
    ├── entity/            # 实体类定义模块
    ├── router/            # 接口路由模块（如 FastAPI 路由）
    ├── skills/            # Skills Agent 核心模块
    └── tools/             # 工具函数/第三方工具封装模块
```

**2026/02/27**

1、优化skills文件内容及格式

2、为milvus_search功能添加rerank重排文本

**2026/02/06**

基本实现embedding + milvus + llm 实现RAG agent

**2026/02/25**

- 初步实现通过LlamaIndex + Qwen3-Embedding-0.6B 写入Milvus向量库，及相关搜索接口输出

**2026/02/24**

- 1、通过FastAPI实现基础的embedding、reranker模型运行；

- 2、通过LangChian 1.0构建基础的Skills Agent。


**todo**

- 1、基于LangChain实现多轮对话能力

- 2、微调Qwen3-0.6B模型为NL2SQL，实现快速的生成sql语句

- 3、优化RAG agent

- 4、将Agent改为FastAPI接口调用模式
