# GKMSUINet

学園アイドルマスター H.I.F 出牌网络。

## 安装

```
pip install .
pip install ".[yaml]"      # master 是 YAML 时
```

## 使用

```python
import json
from gkms_uinet import Policy

policy = Policy("model/gkms_uinet.onnx", "path/to/master")
obs = json.load(open("examples/obs.json", encoding="utf-8"))
out = policy.decide(obs)

out["action"]            # ["play", 2] / ["drink", 0] / ["end"]
out["probs"]             # 与 obs["legal"] 逐项对应
out["score_quantiles"]   # 最终分数预测 {"10%", "25%", "50%", "75%", "90%"}
```

`Policy` 创建时读一次 master，之后每次 `decide` 只做推理。

## 性能

留出的考试中，与对比对象打同一局（同卡组、同考试种子），比较平均分：

| 对比对象 | 局数 | 对比对象平均分 | Ver1.1 平均分 | Ver1.1 / 对比对象 |
|---|---|---|---|---|
| 玩家手打（排行榜玩家的实录） | 18 | 1,908,212 | 2,136,020 | 111.9% |
| RNG 未来视（已知随机数与抽牌顺序的搜索） | 276 | 1,157,429 | 925,467 | 80.0% |
| Ver1.0-zukunftstrachtig | 276 | 866,518 | 925,467 | 106.8% |
| ver0.1-beta5 | 276 | 843,792 | 925,467 | 109.7% |
| ver0.1-beta4 | 276 | 831,773 | 925,467 | 111.3% |
| ver0.1-beta3 | 276 | 822,806 | 925,467 | 112.5% |
| ver0.1-beta1 | 276 | 479,967 | 925,467 | 192.8% |

CPU 4 线程每次 `decide` 约 7.5 ms。

## master

目录内放以下 9 张表，`.yaml`（如 gakumasu-diff）或 `.jsonl` 均可：

`ProduceCard` `ProduceCardCustomize` `ProduceCardGrowEffect` `ProduceCardSearch` `ProduceExamEffect` `ProduceExamTrigger` `ProduceExamStatusEnchant` `ProduceDrink` `ProduceDrinkEffect`

## obs

每次轮到选择时按当前画面填写。格式见 `gkms_uinet/data/uiobs.v1.schema.json`，示例见 `examples/obs.json`。

| 字段 | 内容 |
|---|---|
| `exam` | `produce`（`produce-007` 選抜試験，`produce-008` 本戦）、`step`（`AUDITION_MID1` / `AUDITION_MID2` / `AUDITION_FINAL`）、`plan`（2 センス，3 ロジック，4 アノマリー）、`turn`、`limit`（总回合数，含已获得的追加回合）、`types`（每回合的属性：1 Vo，2 Da，3 Vi）、`permil`（Vo / Da / Vi 倍率，千分比）、`crit`（Vo / Da / Vi 审查基准） |
| `num` | `score`、`stamina`、`max_stamina`、`block`（元気）、`playable`（本回合还能出几张）、`plays`（本回合已出几张） |
| `stance` | `[指针, 段数]`，指针为 `Preservation` / `OverPreservation` / `Leisurely` / `Concentration` / `FullPower`，没有时为 `[null, 0]` |
| `status` | 状态图标，每项 `[种类, 数值, 剩余回合]`，见下表 |
| `ench` | 持续效果图标（P 道具、回忆、卡的持续效果），每项 `[ProduceExamStatusEnchant id, 剩余次数, 剩余回合]` |
| `hand` | 手牌，按画面顺序 |
| `deck` `grave` `hold` `lost` | 山札、捨て札、保留、除外，顺序不限 |
| `played` | 本回合已打出的牌，按打出顺序 |
| `drinks` | 持有的 ProduceDrink id，按画面顺序 |
| `legal` | 现在可选的操作：`["play", 手牌序号]`、`["drink", 饮料序号]`、`["end"]` |

卡：`{"id": ProduceCard id, "u": 当前强化次数, "c": [[ProduceCardCustomize id, 次数]], "g": [考试中获得的 ProduceCardGrowEffect id], "p": 能否打出 1/0}`。`c`、`g` 没有时省略，`p` 只在手牌上写。

无期限、无次数限制写 `-1`。

| 种类 | 图标 | 数值 |
|---|---|---|
| `lesson_buff` | 集中 | 数值 |
| `review` | 好印象 | 数值 |
| `aggressive` | やる気 | 数值 |
| `full_power_point` | 全力値 | 点数 |
| `enthusiastic` | 熱意 | 数值 |
| `anti_debuff` | 低下状態無効 | 次数 |
| `parameter_buff` | 好調 | 回合数（剩余回合同） |
| `parameter_buff_multiple` | 絶好調 | 1 |
| `stance_lock` | 指針固定 | 1 |
| `lesson_debuff` | 集中減少 / 弱気 | 数值，每层一项 |
| `stamina_consumption_down` | 消費体力減少 | 1 |
| `stamina_consumption_add` | 消費体力増加 | 1 |
| `stamina_consumption_down_fix` | 消費体力削減 | 数值 |
| `stamina_consumption_add_fix` | 消費体力追加 | 数值，每层一项 |
| `block_restriction` | 元気増加無効 | 1 |
| `block_add_down` | 元気増加量減少（回合） | 1 |
| `block_add_down_fix` | 元気増加量減少（数值） | 数值 |
| `stamina_recover_restriction` | 体力回復無効 | 1 |
| `stamina_recover_add` | 体力回復量増加 | 数值 |
| `buff_consumption_add` | 強化状態コスト増加 | 1 |
| `slump` | スランプ | 1 |
| `gimmick_parameter_debuff` | 審査ギミック：パラメータ低下 | 1 |
| `panic` | パニック | 1 |
| `card_draw_down` | 手札減少 | 张数，剩余回合 1 |
| `play_card_limit` | スキルカード使用制限 | 1，每个限制一项 |
| `lesson_parameter_multiple` | パラメータ上昇量増加 | 比例（0.1 = +10%），每层一项 |
| `lesson_parameter_multiple_down` | パラメータ上昇量減少 | 比例，每层一项 |
| `lesson_buff_multiple` | 集中効果増加 | 比例，每层一项 |
| `review_multiple` | 好印象効果増加 | 比例，每层一项 |
| `review_count_add` | 好印象強化 | 数值，每层一项 |
| `full_power_lesson_multiple_add` | 全力の効果増加 | 比例，每层一项 |
| `concentration_lesson_multiple_add` | 強気の効果増加 | 比例，每层一项 |
| `full_power_point_add` | 全力値獲得量増加 | 数值，每层一项 |
| `enthusiastic_add` | 熱意獲得量増加 | 数值，每层一项 |
| `lesson_parameter_depend` | パラメータ上昇量が好印象 / やる気に依存 | 1 |
| `lesson_buff_gain_multiple` | 集中獲得量増加 | 比例，每层一项 |
| `parameter_buff_gain_multiple` | 好調獲得量増加 | 比例，每层一项 |
| `aggressive_gain_multiple` | やる気獲得量増加 | 比例，每层一项 |
| `review_gain_multiple` | 好印象獲得量増加 | 比例，每层一项 |
| `lesson_buff_gain_fix` | 集中獲得量 +n | n，每层一项 |
| `aggressive_gain_fix` | やる気獲得量 +n | n，每层一项 |
| `parameter_buff_gain_fix` | 好調獲得量 +n | n，每层一项 |
| `enthusiastic_gain_permil` | 熱意増加 | 千分比，每层一项 |
| `extra_turn` | 追加ターン | 回合数 |
