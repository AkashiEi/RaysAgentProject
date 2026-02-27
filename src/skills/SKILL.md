# 智能体技能清单
## 技能分类及文件说明
| 技能文件路径 | 用途 | 关联工具 |
|--------------|------|----------|
| src/skills/pokemon/nameRule.md | 宝可梦属性命名规则、数值计算逻辑 | pokemonStat |
| src/skills/data_search/data_search_flow.md | 向量/结构化查询的核心流程、字段定义 | milvus_search / DatabaseTool / SQLGenerationTool |
| src/skills/database/SystemInfo/tableList.md | 系统信息表列表 | DatabaseTool / SQLGenerationTool |
| src/skills/database/UserInfo/tableList.md | 用户信息表列表 | DatabaseTool / SQLGenerationTool |
| src/skills/milvus/collection_list.md | Milvus 向量集合列表 | milvus_search |

## 工具调用优先级
1. 所有数据查询类问题，必须先加载 src/skills/data_search/data_search_flow.md；
2. 宝可梦属性计算加载 src/skills/pokemon/nameRule.md；
3. 结构化数据库查询需加载对应表的 src/skills/database/SystemInfo/tableList.md和src/skills/database/UserInfo/tableList.md；
4. 向量查询需加载 src/skills/milvus/collection_list.md。