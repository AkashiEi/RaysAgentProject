# 用户宝可梦持有表
## 基础信息
数据库表名：get_pokemon_info
表文件路径：src/skills/database/UserInfo/userPokemonInfo/idLink.md

## 核心字段（含关联表指引）
| 字段名       | 含义（关联表指引）|
|--------------|--------------------------|
| uuid         | 唯一标识（无关联表）|
| user_id      | 用户id（无关联表）|
| pokemon_id   | 宝可梦id（关联：宝可梦基本信息表） |
| nature_id    | 性格id（关联：性格基本信息表）|
| ball_id      | 精灵球id（关联：精灵球基本信息表） |
| ability_id   | 特性id（关联：特性基本信息表）|
| IV           | 个体值JSON（无关联表）|
| basePoints   | 努力值JSON（无关联表）|
| create_time_ | 创建时间（无关联表）|

# 说明：仅提供字段名、核心含义及关联表指引，无JSON结构/额外规则