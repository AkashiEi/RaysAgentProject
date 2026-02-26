### 数据查找流程

1、通过skill获取到当前持有的表结构

2、基于表结构生成对应的sql语句,当涉及语句复杂时使用reason模型，当语句不复杂时使用chat模型

3、使用sql语句在数据库重进行查找

4、涉及需要计算能力值的时候，需要使用对应的tool进行计算

数据库表结构
 - 作用：告知目前数据库中存在表资源
 - skill所在位置：src/skills/database

### 知识库数据获取流程

1、通过skills获取当前milvus数据库持有的主题以及表结构

2、通过milvus_search获取相关文本内容

Milvus数据库信息
 - 作用：存放相关自有知识库
 - skill所在位置：src/skills/milvus