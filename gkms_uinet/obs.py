"""The names a screen observation (UIObs v1) uses."""

SCHEMA = "uiobs.v1"
INF = -1                                   # "no limit" for turns and uses

STATUS_KINDS = (
    "lesson_buff",                    # 集中
    "review",                         # 好印象
    "aggressive",                     # やる気
    "full_power_point",               # 全力値
    "enthusiastic",                   # 熱意
    "anti_debuff",                    # 低下状態無効 (count)
    "parameter_buff",                 # 好調 (turns)
    "parameter_buff_multiple",        # 絶好調 (turns)
    "stance_lock",                    # 指針固定 (turns)
    "lesson_debuff",                  # 集中減少 / 弱気 value
    "stamina_consumption_down",       # 消費体力減少 (turns)
    "stamina_consumption_add",        # 消費体力増加 (turns)
    "stamina_consumption_down_fix",   # 消費体力削減 (value)
    "stamina_consumption_add_fix",    # 消費体力追加 (value, layers)
    "block_restriction",              # 元気増加無効 (turns)
    "block_add_down",                 # 元気増加量減少 (turns)
    "block_add_down_fix",             # 元気増加量減少 (value)
    "stamina_recover_restriction",    # 体力回復無効 (turns)
    "stamina_recover_add",            # 体力回復量増加
    "buff_consumption_add",           # 強化状態コスト増加 (turns)
    "slump",                          # スランプ (turns)
    "gimmick_parameter_debuff",       # 審査ギミック: パラメータ低下 (turns)
    "panic",                          # パニック (turns)
    "card_draw_down",                 # 手札減少 (this turn)
    "play_card_limit",                # スキルカード使用制限 (one per limit)
    "lesson_parameter_multiple",      # パラメータ上昇量増加 (layer)
    "lesson_parameter_multiple_down", # パラメータ上昇量減少 (layer)
    "lesson_buff_multiple",           # 集中効果増加 (layer)
    "review_multiple",                # 好印象効果増加 (layer)
    "review_count_add",               # 好印象強化 (layer)
    "full_power_lesson_multiple_add", # 全力の効果増加 (layer)
    "concentration_lesson_multiple_add",  # 強気の効果増加 (layer)
    "full_power_point_add",           # 全力値獲得量増加 (layer)
    "enthusiastic_add",               # 熱意獲得量増加 (layer)
    "lesson_parameter_depend",        # パラメータ上昇量が好印象/やる気に依存 (turns)
    "timed_gain",                     # 時限の獲得量増加 of a kind not listed below (layer)
    "extra_turn",                     # 追加ターン
    "lesson_buff_gain_multiple",      # 集中獲得量増加 (multiplier)
    "parameter_buff_gain_multiple",   # 好調獲得量増加
    "aggressive_gain_multiple",       # やる気獲得量増加
    "review_gain_multiple",           # 好印象獲得量増加
    "lesson_buff_gain_fix",           # 集中獲得量 +n
    "aggressive_gain_fix",            # やる気獲得量 +n
    "parameter_buff_gain_fix",        # 好調獲得量 +n
    "enthusiastic_gain_permil",       # 熱意増加 (permil)
)
STANCES = ("Preservation", "OverPreservation", "Leisurely", "Concentration", "FullPower")
STEPS = ("AUDITION_MID1", "AUDITION_MID2", "AUDITION_FINAL")
PRODUCES = ("produce-007", "produce-008")
ORIGINS = ("pool", "idol", "support", "trouble")

