# 数据查询核心规则（Agent 必须优先加载）
## 一、通用规则
1. 禁止猜测字段名，所有查询字段必须严格匹配本文件定义；
2. 结构化数据库查询流程：loadSkills(对应表tableList.md) → SQLGenerationTool → DatabaseTool；
3. 向量数据库查询流程：loadSkills(milvus/collection/collection_list.md) → milvus_search。

## 二、字段定义
### 2.1 向量查询（milvus_search）
- 核心字段：
  - id: int（主键）
  - content: str（小说文本内容）
  - vector: list[float]（文本向量，维度1024）
  - volume: str（小说卷名）

### 2.2 结构化查询（DatabaseTool）
- 通用字段：
  - uuid（唯一标识）