### 字段对应关系

| 外键名 | 字段名 | 所属表 | 关联表 | 关联字段 | ON DELETE | ON UPDATE |
|--------|--------|--------|--------|----------|-----------|-----------|
| 宝可梦id | pokemon_id | pokemon | pokemon_base_info | uuid | CASCADE | CASCADE |
| 性格id | nature_id | pokemon | nature_info | uuid | CASCADE | CASCADE |
| 特性id | ability_id | pokemon | ability_base_info | uuid | CASCADE | CASCADE |
| 球种id | ball_id | pokemon | monster_ball_info | uuid | CASCADE | CASCADE |
