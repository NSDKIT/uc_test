"""
保安制約付き発電機起動停止計画（SCUC）の例題 — 3エリア版・蓄電池付き

- エリア1: 発電機 G1〜G10（10機）、蓄電池1台、純負荷の一部
- エリア2: 発電機 G11〜G20（10機）、蓄電池1台、純負荷の一部
- エリア3: 発電機 G21〜G30（10機）、蓄電池1台、純負荷の残り
- 連系線: 直列トポロジー エリア1—エリア2—エリア3（2本の連系線、DC潮流）
  - 連系線1-2: 潮流 = エリア1の発電 − エリア1の純負荷
  - 連系線2-3: 潮流 = (エリア1+エリア2の発電) − (エリア1+エリア2の純負荷)
- 制約: 各連系線の潮流が ±F_max 以内
- 蓄電池: 各エリアに1台。SOC ダイナミクス・充放電上下限・充放電排他制約。
"""

import os
import random
import html
import base64
import io
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pulp import (
    LpMinimize,
    LpProblem,
    LpVariable,
    lpSum,
    LpBinary,
    LpContinuous,
    LpStatus,
    value,
)

# ---------------------------------------------------------------------------
# 0. 対話形式でのパラメータ入力
# ---------------------------------------------------------------------------

def input_int(prompt, default, min_val, max_val):
    """整数入力（範囲チェック・デフォルト付き）"""
    while True:
        s = input(prompt).strip() or str(default)
        try:
            v = int(s)
            if min_val <= v <= max_val:
                return v
            print(f"  → {min_val}〜{max_val} の値を入力してください。")
        except ValueError:
            print("  → 整数を入力してください。")

def input_scenario(prompt, n_days, min_v, max_v):
    """各日の曲線タイプをカンマ区切りで入力"""
    while True:
        s = input(prompt).strip()
        try:
            vals = [int(x.strip()) for x in s.split(",")]
            if len(vals) != n_days:
                print(f"  → {n_days}日分の値をカンマ区切りで入力してください（例: 1,2,3）。")
                continue
            if all(min_v <= v <= max_v for v in vals):
                return vals
            print(f"  → 各値は {min_v}〜{max_v} の範囲で入力してください。")
        except ValueError:
            print("  → 整数をカンマ区切りで入力してください。")

def input_int_list(prompt, default_list, n_items, min_v, max_v):
    """[a,b,c] 形式（カンマ区切り）の整数リスト入力。空入力はデフォルト。"""
    while True:
        s = input(prompt).strip()
        if not s:
            return list(default_list)
        try:
            vals = [int(x.strip()) for x in s.split(",")]
            if len(vals) != n_items:
                print(f"  → {n_items}個の値をカンマ区切りで入力してください（例: 1,2,3）。")
                continue
            if all(min_v <= v <= max_v for v in vals):
                return vals
            print(f"  → 各値は {min_v}〜{max_v} の範囲で入力してください。")
        except ValueError:
            print("  → 整数をカンマ区切りで入力してください。")

def input_pct_list(prompt, default_list, n_items):
    """[a,b,c] 形式（カンマ区切り）の%リスト入力。合計=100を要求。空入力はデフォルト。"""
    while True:
        s = input(prompt).strip()
        if not s:
            vals = list(default_list)
        else:
            try:
                vals = [float(x.strip()) for x in s.split(",")]
            except ValueError:
                print("  → 数値をカンマ区切りで入力してください。")
                continue
        if len(vals) != n_items:
            print(f"  → {n_items}個の値をカンマ区切りで入力してください（例: 40,35,25）。")
            continue
        total = sum(vals)
        if abs(total - 100.0) > 1e-6:
            print(f"  → 合計が100になるように入力してください（現在 {total:g}）。")
            continue
        if any(v < 0 for v in vals):
            print("  → 0以上で入力してください。")
            continue
        return vals

print("=" * 60)
print("保安制約付きUC（SCUC）3エリア パラメータ入力")
print("=" * 60)
DAYS = input_int("解析対象日数（日） [1-7, 省略=1]: ", 1, 1, 7)

RENEW_MODE = 1
if DAYS >= 2:
    print("日ごとの再エネ条件: 1=全日同じ, 2=日ごとランダム, 3=シナリオ（日別に指定）")
    RENEW_MODE = input_int("モード [1-3, 省略=1]: ", 1, 1, 3)

print("解析対象季節: 1=春秋, 2=夏, 3=冬")
SEASON = input_int("季節 [1-3, 省略=1]: ", 1, 1, 3)

AREAS = ["Area1", "Area2", "Area3"]
N_AREAS = 3

print("太陽光発電曲線: 1=快晴日, 2=準快晴日, 3=曇天日, 4=雨天日, 5=ランプアップ, 6=ランプダウン")
print("風力発電曲線: 1=風が強い日, 2=風が弱い日, 3=午前中が強い日, 4=午後が強い日")
print("※ エリア別設定は [エリア1,エリア2,エリア3] の順で入力（例: 1,2,3）")

# エリア別の曲線タイプ（日ごとに変化する場合は内部で DAYS 分に展開）
solar_types_by_area_day = [[1] * DAYS for _ in range(N_AREAS)]
wind_types_by_area_day = [[1] * DAYS for _ in range(N_AREAS)]

if DAYS == 1 or RENEW_MODE == 1:
    # デフォルト: エリアごとに曲線を変える（Area1=快晴/風強, Area2=準快晴/午前強, Area3=曇天/午後強）
    solar_by_area = input_int_list("太陽光曲線 [3個, 省略=1,2,3]: ", [1, 2, 3], N_AREAS, 1, 6)
    wind_by_area = input_int_list("風力曲線   [3個, 省略=1,3,4]: ", [1, 3, 4], N_AREAS, 1, 4)
    for a in range(N_AREAS):
        solar_types_by_area_day[a] = [solar_by_area[a]] * DAYS
        wind_types_by_area_day[a] = [wind_by_area[a]] * DAYS
elif RENEW_MODE == 2:
    random.seed(42)
    for a in range(N_AREAS):
        solar_types_by_area_day[a] = [random.randint(1, 6) for _ in range(DAYS)]
        wind_types_by_area_day[a] = [random.randint(1, 4) for _ in range(DAYS)]
    print("【ランダム割当】")
    for a in range(N_AREAS):
        print(f"  {AREAS[a]} solar:", solar_types_by_area_day[a], "wind:", wind_types_by_area_day[a])
else:
    # RENEW_MODE == 3
    print(f"各日（{DAYS}日分）の曲線タイプをエリア別に入力してください。")
    for a in range(N_AREAS):
        solar_types_by_area_day[a] = input_scenario(f"  {AREAS[a]} 太陽光(1-6) 例: 1,2,3: ", DAYS, 1, 6)
        wind_types_by_area_day[a] = input_scenario(f"  {AREAS[a]} 風力(1-4)   例: 1,2,3: ", DAYS, 1, 4)
    print("【シナリオ】")
    for a in range(N_AREAS):
        print(f"  {AREAS[a]} solar:", solar_types_by_area_day[a], "wind:", wind_types_by_area_day[a])

# 連系線容量 (MW): 各連系線（1-2, 2-3）の送電量の絶対値上限（共通）
F_MAX = input_int("連系線容量 F_max (MW) [1-500, 省略=250]: ", 250, 1, 500)

ALLOW_SOLAR_CURTAIL = input_int("太陽光余剰（出力抑制）を許容する？ 1=許容, 2=許容しない [省略=1]: ", 1, 1, 2)

# 需要・再エネのエリア別配分（デフォルトでエリアごとに差をつける）
# 需要: 40/35/25%、太陽光設備: 35/40/25%、風力設備: 30/40/30%
load_share_pct = [40.0, 35.0, 25.0]
solar_cap_pct = [35.0, 40.0, 25.0]
wind_cap_pct = [30.0, 40.0, 30.0]
print()

# ---------------------------------------------------------------------------
# 1. データ定義（季節・太陽光・風力の1日パターン）
# ---------------------------------------------------------------------------

HOURS_PER_DAY = 24
DEMAND_1D_BY_SEASON = {
    1: [
        360, 340, 320, 310, 320, 360, 440, 560, 640, 680, 700, 720,
        700, 680, 660, 680, 720, 760, 800, 760, 700, 600, 500, 400,
    ],
    2: [
        400, 380, 370, 360, 380, 440, 560, 720, 840, 920, 960, 1000,
        960, 920, 880, 900, 940, 980, 1000, 960, 880, 760, 640, 500,
    ],
    3: [
        560, 600, 640, 620, 600, 640, 720, 800, 840, 860, 880, 900,
        880, 860, 840, 860, 900, 960, 1000, 980, 920, 840, 760, 640,
    ],
}

SOLAR_1D_BY_TYPE = {
    1: [0,0,0,0,0,5,30,80,140,200,240,260,260,240,200,140,80,30,5,0,0,0,0,0],
    2: [0,0,0,0,0,3,18,50,90,130,160,170,170,160,130,90,50,18,3,0,0,0,0,0],
    3: [0,0,0,0,0,2,10,30,50,70,80,85,85,80,70,50,30,10,2,0,0,0,0,0],
    4: [0,0,0,0,0,0,5,15,25,35,40,42,42,40,35,25,15,5,0,0,0,0,0,0],
    5: [0,0,0,0,0,0,0,0,0,20,60,100,140,180,220,260,260,200,100,20,0,0,0,0],
    6: [0,0,0,0,0,50,120,180,240,260,260,260,220,180,140,80,40,10,0,0,0,0,0,0],
}

WIND_1D_BY_TYPE = {
    1: [80,85,82,78,75,80,85,90,88,85,82,80,78,82,85,88,90,85,82,80,78,82,85,80],
    2: [15,18,16,14,12,15,18,20,18,16,14,12,14,16,18,20,18,16,14,15,14,16,18,15],
    3: [70,75,80,85,90,95,90,85,80,70,60,50,40,35,30,35,40,45,50,45,40,35,30,25],
    4: [25,30,35,40,45,50,55,60,65,70,75,80,85,90,85,80,75,70,65,60,55,50,45,40],
}

T = HOURS_PER_DAY * DAYS
TIME = range(1, T + 1)

# 需要（全体）: 季節の1日パターンを日数分繰り返し × 需要スケール 3 倍
demand_1d = DEMAND_1D_BY_SEASON[SEASON]
DEMAND_SCALE = 3.0
DEMAND = [int(round(d * DEMAND_SCALE)) for d in (demand_1d * DAYS)[:T]]

# 太陽光は 1 倍で 1 回のみ実行

# 需要のエリア別プロファイル（24h、各時刻で3エリアの合計=1）: 形をエリアごとに変える
# Area1=夜間多め, Area2=昼間多め, Area3=夕方多め
_w1 = [1.25, 1.22, 1.20, 1.18, 1.15, 1.10, 0.95, 0.85, 0.80, 0.78, 0.78, 0.80,
       0.82, 0.82, 0.82, 0.85, 0.88, 0.92, 0.95, 0.98, 1.00, 1.05, 1.12, 1.18]
_w2 = [0.70, 0.72, 0.75, 0.78, 0.82, 0.88, 1.05, 1.20, 1.28, 1.32, 1.32, 1.28,
       1.25, 1.25, 1.25, 1.22, 1.18, 1.12, 1.08, 1.05, 1.02, 0.95, 0.88, 0.78]
_w3 = [1.05, 1.06, 1.05, 1.04, 1.03, 1.02, 1.00, 0.95, 0.92, 0.90, 0.90, 0.92,
       0.93, 0.93, 0.93, 0.93, 0.94, 0.96, 0.97, 0.97, 0.98, 1.00, 1.00, 1.04]
_denom = [_w1[h] + _w2[h] + _w3[h] for h in range(24)]
DEMAND_PROFILE_BY_AREA = [
    [_w1[h] / _denom[h] for h in range(24)],
    [_w2[h] / _denom[h] for h in range(24)],
    [_w3[h] / _denom[h] for h in range(24)],
]

# 需要（エリア別）: 全体需要 × エリア別プロファイル（形の差） × 配分のスケール
# 配分 load_share_pct も反映するため、プロファイルと配分の両方で按分する
DEMAND_BY_AREA = {AREAS[a]: [] for a in range(N_AREAS)}
for t in range(T):
    d_total = DEMAND[t]
    h = (t % 24)
    # プロファイル（形）と配分（合計1）を組み合わせ: プロファイルで按分した後、配分で再スケール
    p1 = DEMAND_PROFILE_BY_AREA[0][h] * (load_share_pct[0] / 100.0)
    p2 = DEMAND_PROFILE_BY_AREA[1][h] * (load_share_pct[1] / 100.0)
    p3 = DEMAND_PROFILE_BY_AREA[2][h] * (load_share_pct[2] / 100.0)
    s = p1 + p2 + p3
    if s <= 0:
        s = 1.0
    d1 = int(round(d_total * p1 / s))
    d2 = int(round(d_total * p2 / s))
    d3 = d_total - d1 - d2
    DEMAND_BY_AREA[AREAS[0]].append(d1)
    DEMAND_BY_AREA[AREAS[1]].append(d2)
    DEMAND_BY_AREA[AREAS[2]].append(d3)

# 風力は太陽光スケールに依存しないので1回だけ構築（ループ外で参照用に1スケール分）
WIND_AVAIL_BY_AREA = {AREAS[a]: [] for a in range(N_AREAS)}
for a in range(N_AREAS):
    w_scale = wind_cap_pct[a] / 100.0
    for d in range(DAYS):
        w_type = wind_types_by_area_day[a][d]
        WIND_AVAIL_BY_AREA[AREAS[a]].extend([x * w_scale for x in WIND_1D_BY_TYPE[w_type]])
    WIND_AVAIL_BY_AREA[AREAS[a]] = WIND_AVAIL_BY_AREA[AREAS[a]][:T]

# 発電機セット
# 1エリアあたり10機のテンプレート（各エリアで同じ構成を10機ずつ配置）
GEN_TEMPLATE = [
    {"Pmin": 50, "Pmax": 200, "no_load": 500, "cost_per_mwh": 3000, "startup": 8000, "MUT": 4, "MDT": 4, "RU": 80, "RD": 80},
    {"Pmin": 30, "Pmax": 100, "no_load": 200, "cost_per_mwh": 3500, "startup": 4000, "MUT": 4, "MDT": 4, "RU": 50, "RD": 50},
    {"Pmin": 80, "Pmax": 300, "no_load": 800, "cost_per_mwh": 2500, "startup": 12000, "MUT": 4, "MDT": 4, "RU": 100, "RD": 100},
    {"Pmin": 20, "Pmax": 80, "no_load": 100, "cost_per_mwh": 4000, "startup": 2000, "MUT": 4, "MDT": 4, "RU": 40, "RD": 40},
    {"Pmin": 40, "Pmax": 150, "no_load": 350, "cost_per_mwh": 3200, "startup": 6000, "MUT": 4, "MDT": 4, "RU": 60, "RD": 60},
    {"Pmin": 25, "Pmax": 90, "no_load": 180, "cost_per_mwh": 3600, "startup": 3500, "MUT": 4, "MDT": 4, "RU": 45, "RD": 45},
    {"Pmin": 60, "Pmax": 250, "no_load": 600, "cost_per_mwh": 2700, "startup": 10000, "MUT": 4, "MDT": 4, "RU": 90, "RD": 90},
    {"Pmin": 35, "Pmax": 120, "no_load": 250, "cost_per_mwh": 3400, "startup": 4500, "MUT": 4, "MDT": 4, "RU": 55, "RD": 55},
    {"Pmin": 45, "Pmax": 180, "no_load": 420, "cost_per_mwh": 2900, "startup": 7000, "MUT": 4, "MDT": 4, "RU": 70, "RD": 70},
    {"Pmin": 15, "Pmax": 60, "no_load": 80, "cost_per_mwh": 4200, "startup": 1500, "MUT": 4, "MDT": 4, "RU": 30, "RD": 30},
]
GENS_PER_AREA = 10
# エリア1: G1〜G10, エリア2: G11〜G20, エリア3: G21〜G30（各エリア10機）
G_AREA1 = [f"G{i+1}" for i in range(GENS_PER_AREA)]
G_AREA2 = [f"G{i+11}" for i in range(GENS_PER_AREA)]
G_AREA3 = [f"G{i+21}" for i in range(GENS_PER_AREA)]
G = G_AREA1 + G_AREA2 + G_AREA3
GEN_DATA = {}
for area_idx, area_gens in enumerate([G_AREA1, G_AREA2, G_AREA3]):
    for i, g in enumerate(area_gens):
        GEN_DATA[g] = dict(GEN_TEMPLATE[i])

U0 = {g: 0 for g in G}
MIN_COMMITTED = 3  # 各エリアで常にこの機数以上稼働

# 蓄電池パラメータ（各エリア1台、同一仕様）
BATT_E_CAP = 200.0       # 容量 [MWh]
BATT_P_MAX = 50.0        # 最大充放電 [MW]（充電・放電とも同値）
BATT_ETA_C = 0.95        # 充電効率
BATT_ETA_D = 0.95        # 放電効率
BATT_SOC_MIN_FRAC = 0.1  # SOC 下限（容量比）
BATT_SOC_MAX_FRAC = 0.9  # SOC 上限（容量比）
BATT_E_INITIAL_FRAC = 0.5  # 初期・終端 SOC（容量比、サイクル維持）

print("【3エリア構成】 トポロジー: Area1 — Area2 — Area3")
print(f"  エリア1: 発電機 {G_AREA1}")
print(f"  エリア2: 発電機 {G_AREA2}")
print(f"  エリア3: 発電機 {G_AREA3}")
print(f"  連系線 1-2, 2-3: 各 ±{F_MAX} MW")
print(f"  最低稼働機数: 各エリアで {MIN_COMMITTED} 機以上")
print(f"  蓄電池: 各エリア {BATT_E_CAP} MWh, ±{BATT_P_MAX} MW, ηc={BATT_ETA_C}, ηd={BATT_ETA_D}, SOC {BATT_SOC_MIN_FRAC*100:.0f}〜{BATT_SOC_MAX_FRAC*100:.0f}%")
print(f"  太陽光余剰（出力抑制）: {'許容' if ALLOW_SOLAR_CURTAIL == 1 else '許容しない（全量利用）'}")
print(f"  需要: エリア別プロファイル+配分 {load_share_pct[0]:.0f}/{load_share_pct[1]:.0f}/{load_share_pct[2]:.0f}%")
print(f"  太陽光設備: {solar_cap_pct[0]:.0f}/{solar_cap_pct[1]:.0f}/{solar_cap_pct[2]:.0f}%, 風力: {wind_cap_pct[0]:.0f}/{wind_cap_pct[1]:.0f}/{wind_cap_pct[2]:.0f}%")
print("  曲線タイプ（1日目）:")
print(f"    Solar: {[solar_types_by_area_day[a][0] for a in range(N_AREAS)]}  (Area1,2,3)")
print(f"    Wind : {[wind_types_by_area_day[a][0] for a in range(N_AREAS)]}  (Area1,2,3)")
print("【発電機 最低運転時間(MUT)・最低停止時間(MDT)】 全機 MUT=4 h, MDT=4 h")
print("【発電機 最大出力 Pmax (MW)】 エリア1:", [GEN_DATA[g]["Pmax"] for g in G_AREA1])
print("  エリア2:", [GEN_DATA[g]["Pmax"] for g in G_AREA2], "  エリア3:", [GEN_DATA[g]["Pmax"] for g in G_AREA3])
max_total_capacity = sum(GEN_DATA[g]["Pmax"] for g in G)
max_demand = max(DEMAND)
# print(f"【需要スケール】       {DEMAND_SCALE} 倍")
# print(f"【太陽光スケール】     {SOLAR_SCALE} 倍")
# print(f"【最大合計発電機出力】 {max_total_capacity} MW")
# print(f"【最大総需要】         {max_demand} MW")
# print()

# ---------------------------------------------------------------------------
# 2. SCUC を 1 回実行（太陽光 1 倍）
# ---------------------------------------------------------------------------
runs = []
ADJ_LOAD_FRAC = 0.10
ADJ_RES_FRAC = 0.25

# 太陽光（エリア別）: 曲線タイプを連結し、設備配分でスケール（1 倍）
SOLAR_AVAIL_BY_AREA = {AREAS[a]: [] for a in range(N_AREAS)}
for a in range(N_AREAS):
    s_scale = solar_cap_pct[a] / 100.0
    for d in range(DAYS):
        s_type = solar_types_by_area_day[a][d]
        SOLAR_AVAIL_BY_AREA[AREAS[a]].extend([x * s_scale for x in SOLAR_1D_BY_TYPE[s_type]])
    SOLAR_AVAIL_BY_AREA[AREAS[a]] = SOLAR_AVAIL_BY_AREA[AREAS[a]][:T]
SOLAR_AVAIL_TOTAL = [sum(SOLAR_AVAIL_BY_AREA[a][t] for a in AREAS) for t in range(T)]
WIND_AVAIL_TOTAL = [sum(WIND_AVAIL_BY_AREA[a][t] for a in AREAS) for t in range(T)]
ADJ_REQUIRED_BY_AREA = [
    [
        DEMAND_BY_AREA[AREAS[a]][t] * ADJ_LOAD_FRAC
        + (SOLAR_AVAIL_BY_AREA[AREAS[a]][t] + WIND_AVAIL_BY_AREA[AREAS[a]][t]) * ADJ_RES_FRAC
        for t in range(T)
    ]
    for a in range(N_AREAS)
]
ADJ_REQUIRED = [
    DEMAND[t] * ADJ_LOAD_FRAC + (SOLAR_AVAIL_TOTAL[t] + WIND_AVAIL_TOTAL[t]) * ADJ_RES_FRAC
    for t in range(T)
]

# ---------------------------------------------------------------------------
# 2b. 最適化モデル（SCUC）
# ---------------------------------------------------------------------------
prob = LpProblem("SCUC_ThreeArea", LpMinimize)

u = LpVariable.dicts("u", (G, TIME), cat=LpBinary)
v = LpVariable.dicts("v", (G, TIME), cat=LpBinary)
w = LpVariable.dicts("w", (G, TIME), cat=LpBinary)
P = LpVariable.dicts("P", (G, TIME), lowBound=0, cat=LpContinuous)
f12 = LpVariable.dicts("f12", TIME, lowBound=-F_MAX, upBound=F_MAX, cat=LpContinuous)
f23 = LpVariable.dicts("f23", TIME, lowBound=-F_MAX, upBound=F_MAX, cat=LpContinuous)
solar_used = LpVariable.dicts("solar_used", (AREAS, TIME), lowBound=0, cat=LpContinuous)
# 蓄電池: SOC [MWh], 充電・放電 [MW], 放電モードフラグ (1=放電)
E_batt = LpVariable.dicts("E_batt", (AREAS, TIME), lowBound=0, cat=LpContinuous)
P_charge = LpVariable.dicts("P_charge", (AREAS, TIME), lowBound=0, upBound=BATT_P_MAX, cat=LpContinuous)
P_discharge = LpVariable.dicts("P_discharge", (AREAS, TIME), lowBound=0, upBound=BATT_P_MAX, cat=LpContinuous)
delta_batt = LpVariable.dicts("delta_batt", (AREAS, TIME), cat=LpBinary)

prob += lpSum(
    GEN_DATA[g]["cost_per_mwh"] * P[g][t] + GEN_DATA[g]["no_load"] * u[g][t] + GEN_DATA[g]["startup"] * v[g][t]
    for g in G for t in TIME
), "TotalCost"

for t in TIME:
    # 太陽光利用量（出力抑制ありの場合は 0〜avail、なしの場合は avail 固定）
    for a in AREAS:
        prob += solar_used[a][t] <= SOLAR_AVAIL_BY_AREA[a][t - 1], f"SolarUsedMax_{a}_{t}"
        if ALLOW_SOLAR_CURTAIL != 1:
            prob += solar_used[a][t] == SOLAR_AVAIL_BY_AREA[a][t - 1], f"SolarMustTake_{a}_{t}"

    # エリア別需給バランス（潮流・蓄電池込み: 放電は供給、充電は需要）
    prob += (
        lpSum(P[g][t] for g in G_AREA1)
        + WIND_AVAIL_BY_AREA["Area1"][t - 1]
        + solar_used["Area1"][t]
        + P_discharge["Area1"][t]
        - P_charge["Area1"][t]
        - DEMAND_BY_AREA["Area1"][t - 1]
        - f12[t]
        == 0,
        f"AreaBalance_1_{t}",
    )
    prob += (
        lpSum(P[g][t] for g in G_AREA2)
        + WIND_AVAIL_BY_AREA["Area2"][t - 1]
        + solar_used["Area2"][t]
        + P_discharge["Area2"][t]
        - P_charge["Area2"][t]
        - DEMAND_BY_AREA["Area2"][t - 1]
        + f12[t]
        - f23[t]
        == 0,
        f"AreaBalance_2_{t}",
    )
    prob += (
        lpSum(P[g][t] for g in G_AREA3)
        + WIND_AVAIL_BY_AREA["Area3"][t - 1]
        + solar_used["Area3"][t]
        + P_discharge["Area3"][t]
        - P_charge["Area3"][t]
        - DEMAND_BY_AREA["Area3"][t - 1]
        + f23[t]
        == 0,
        f"AreaBalance_3_{t}",
    )

# 蓄電池: SOC ダイナミクス E(t) = E(t-1) + ηc*P_charge(t) - P_discharge(t)/ηd
E_min = BATT_E_CAP * BATT_SOC_MIN_FRAC
E_max = BATT_E_CAP * BATT_SOC_MAX_FRAC
E_initial = BATT_E_CAP * BATT_E_INITIAL_FRAC
E_final = E_initial
for a in AREAS:
    t = 1
    prob += (
        E_batt[a][t] == E_initial + BATT_ETA_C * P_charge[a][t] - P_discharge[a][t] / BATT_ETA_D,
        f"BattSOC_{a}_{t}",
    )
    for t in range(2, T + 1):
        prob += (
            E_batt[a][t] == E_batt[a][t - 1] + BATT_ETA_C * P_charge[a][t] - P_discharge[a][t] / BATT_ETA_D,
            f"BattSOC_{a}_{t}",
        )
    for t in TIME:
        prob += E_batt[a][t] >= E_min, f"BattSOCmin_{a}_{t}"
        prob += E_batt[a][t] <= E_max, f"BattSOCmax_{a}_{t}"
    prob += E_batt[a][T] >= E_final, f"BattSOCfinal_{a}"
# 充放電排他: 放電時 delta=1, 充電時 delta=0
for a in AREAS:
    for t in TIME:
        prob += P_discharge[a][t] <= BATT_P_MAX * delta_batt[a][t], f"BattDischMax_{a}_{t}"
        prob += P_charge[a][t] <= BATT_P_MAX * (1 - delta_batt[a][t]), f"BattChargeMax_{a}_{t}"

# 調整力: エリア別時刻別（各エリアの headroom >= 当該エリアの所要調整力）
for a, area_gens in enumerate([G_AREA1, G_AREA2, G_AREA3]):
    for t in TIME:
        prob += (
            lpSum(GEN_DATA[g]["Pmax"] * u[g][t] - P[g][t] for g in area_gens) >= ADJ_REQUIRED_BY_AREA[a][t - 1],
            f"Adjustment_{AREAS[a]}_{t}",
        )

# 最低稼働機数: 各エリアで常に MIN_COMMITTED 機以上稼働
for t in TIME:
    prob += lpSum(u[g][t] for g in G_AREA1) >= MIN_COMMITTED, f"MinCommitted_Area1_{t}"
    prob += lpSum(u[g][t] for g in G_AREA2) >= MIN_COMMITTED, f"MinCommitted_Area2_{t}"
    prob += lpSum(u[g][t] for g in G_AREA3) >= MIN_COMMITTED, f"MinCommitted_Area3_{t}"

# 連系線容量（変数の上下限制約として適用済み）: -F_MAX <= f12,f23 <= F_MAX

for g in G:
    pmin, pmax = GEN_DATA[g]["Pmin"], GEN_DATA[g]["Pmax"]
    for t in TIME:
        prob += P[g][t] >= pmin * u[g][t], f"Pmin_{g}_{t}"
        prob += P[g][t] <= pmax * u[g][t], f"Pmax_{g}_{t}"

for g in G:
    for t in TIME:
        u_prev = U0[g] if t == 1 else u[g][t - 1]
        prob += u[g][t] - u_prev == v[g][t] - w[g][t], f"StateChange_{g}_{t}"

for g in G:
    MUT = GEN_DATA[g]["MUT"]
    for s in TIME:
        end = min(s + MUT - 1, T)
        if end - s + 1 >= MUT:
            prob += lpSum(u[g][t] for t in range(s, end + 1)) >= MUT * v[g][s], f"MUT_{g}_{s}"

for g in G:
    MDT = GEN_DATA[g]["MDT"]
    for s in TIME:
        end = min(s + MDT - 1, T)
        if end - s + 1 >= MDT:
            prob += lpSum(1 - u[g][t] for t in range(s, end + 1)) >= MDT * w[g][s], f"MDT_{g}_{s}"

for g in G:
    RU, RD = GEN_DATA[g]["RU"], GEN_DATA[g]["RD"]
    for t in TIME:
        if t >= 2:
            prob += P[g][t] - P[g][t - 1] <= RU, f"RampUp_{g}_{t}"
            prob += P[g][t - 1] - P[g][t] <= RD, f"RampDown_{g}_{t}"

# ---------------------------------------------------------------------------
# 3. 求解と結果表示
# ---------------------------------------------------------------------------

try:
    prob.solve()
except Exception as e:
    print(f"  求解エラー - {e}")
    runs.append({"scale": 1.0, "status": "Error", "cond_b64": None, "result_b64": None, "cost": None})
else:
    if prob.status != 1:
        print(f"  最適解なし ({LpStatus[prob.status]})")
        runs.append({"scale": 1.0, "status": LpStatus[prob.status], "cond_b64": None, "result_b64": None, "cost": None})
    else:
        print(f"  OK, 総コスト {value(prob.objective):,.0f} 円")

        # 潮流の結果（連系線1-2, 2-3）
        flow_12_vals = [value(f12[t]) for t in TIME]
        flow_23_vals = [value(f23[t]) for t in TIME]
        solar_used_vals_by_area = {a: [value(solar_used[a][t]) for t in TIME] for a in AREAS}
        solar_used_total = [sum(solar_used_vals_by_area[a][i] for a in AREAS) for i in range(T)]
        solar_avail_total = list(SOLAR_AVAIL_TOTAL)
        wind_total = list(WIND_AVAIL_TOTAL)
        time_display = list(TIME)[:HOURS_PER_DAY]

        # 各エリア需給バランス
        gen1_vals = [sum(value(P[g][t]) for g in G_AREA1) for t in TIME]
        gen2_vals = [sum(value(P[g][t]) for g in G_AREA2) for t in TIME]
        gen3_vals = [sum(value(P[g][t]) for g in G_AREA3) for t in TIME]
        P_disch_vals = {a: [value(P_discharge[a][t]) for t in TIME] for a in AREAS}
        P_ch_vals = {a: [value(P_charge[a][t]) for t in TIME] for a in AREAS}
        bal1_res = [
            gen1_vals[t - 1]
            + WIND_AVAIL_BY_AREA["Area1"][t - 1]
            + solar_used_vals_by_area["Area1"][t - 1]
            + P_disch_vals["Area1"][t - 1]
            - P_ch_vals["Area1"][t - 1]
            - DEMAND_BY_AREA["Area1"][t - 1]
            - flow_12_vals[t - 1]
            for t in TIME
        ]
        bal2_res = [
            gen2_vals[t - 1]
            + WIND_AVAIL_BY_AREA["Area2"][t - 1]
            + solar_used_vals_by_area["Area2"][t - 1]
            + P_disch_vals["Area2"][t - 1]
            - P_ch_vals["Area2"][t - 1]
            - DEMAND_BY_AREA["Area2"][t - 1]
            + flow_12_vals[t - 1]
            - flow_23_vals[t - 1]
            for t in TIME
        ]
        bal3_res = [
            gen3_vals[t - 1]
            + WIND_AVAIL_BY_AREA["Area3"][t - 1]
            + solar_used_vals_by_area["Area3"][t - 1]
            + P_disch_vals["Area3"][t - 1]
            - P_ch_vals["Area3"][t - 1]
            - DEMAND_BY_AREA["Area3"][t - 1]
            + flow_23_vals[t - 1]
            for t in TIME
        ]
        max_abs_bal1 = max(abs(x) for x in bal1_res)
        max_abs_bal2 = max(abs(x) for x in bal2_res)
        max_abs_bal3 = max(abs(x) for x in bal3_res)
        max_abs_f12 = max(abs(x) for x in flow_12_vals)
        max_abs_f23 = max(abs(x) for x in flow_23_vals)
        cost_fuel = sum(GEN_DATA[g]["cost_per_mwh"] * value(P[g][t]) for g in G for t in TIME)
        cost_noload = sum(GEN_DATA[g]["no_load"] * value(u[g][t]) for g in G for t in TIME)
        cost_start = sum(GEN_DATA[g]["startup"] * value(v[g][t]) for g in G for t in TIME)

        hours = list(TIME)
        demand_arr = np.array(DEMAND, dtype=float)
        solar_avail_arr = np.array(solar_avail_total, dtype=float)
        solar_arr = np.array(solar_used_total, dtype=float)
        wind_arr = np.array(wind_total, dtype=float)
        flow_12_arr = np.array(flow_12_vals)
        flow_23_arr = np.array(flow_23_vals)

        def _arr(x):
            return np.array(x, dtype=float)
        demand_by_area = {a: _arr(DEMAND_BY_AREA[a]) for a in AREAS}
        solar_used_by_area = {a: _arr(solar_used_vals_by_area[a]) for a in AREAS}
        solar_avail_by_area = {a: _arr(SOLAR_AVAIL_BY_AREA[a]) for a in AREAS}
        wind_by_area = {a: _arr(WIND_AVAIL_BY_AREA[a]) for a in AREAS}
        thermal_by_area = {"Area1": _arr(gen1_vals), "Area2": _arr(gen2_vals), "Area3": _arr(gen3_vals)}
        net_load_by_area = {a: demand_by_area[a] - solar_used_by_area[a] - wind_by_area[a] for a in AREAS}
        G_BY_AREA = [G_AREA1, G_AREA2, G_AREA3]
        p_by_area = [np.array([[value(P[g][t]) for t in TIME] for g in G_BY_AREA[a]]) for a in range(N_AREAS)]
        u_by_area = [np.array([[int(value(u[g][t])) for t in TIME] for g in G_BY_AREA[a]]) for a in range(N_AREAS)]
        reserve_by_area = [
            (np.array([GEN_DATA[g]["Pmax"] for g in G_BY_AREA[a]])[:, np.newaxis] * u_by_area[a]).sum(axis=0)
            for a in range(N_AREAS)
        ]
        thermal_by_area_arr = [p_by_area[a].sum(axis=0) for a in range(N_AREAS)]
        adjustment_by_area = [reserve_by_area[a] - thermal_by_area_arr[a] for a in range(N_AREAS)]
        adj_required_by_area = [np.array(ADJ_REQUIRED_BY_AREA[a]) for a in range(N_AREAS)]

        _colors_all = [plt.cm.tab10(i % 10) for i in range(len(G))]
        colors_by_area = [[_colors_all[G.index(g)] for g in G_BY_AREA[a]] for a in range(N_AREAS)]
        supply_target_by_area = {
            "Area1": demand_by_area["Area1"] + flow_12_arr,
            "Area2": demand_by_area["Area2"] + (flow_23_arr - flow_12_arr),
            "Area3": demand_by_area["Area3"] - flow_23_arr,
        }
        tie_receive_area1 = -flow_12_arr
        tie_receive_area2_from1 = flow_12_arr
        tie_receive_area2_from3 = -flow_23_arr
        tie_receive_area3 = flow_23_arr

        # 条件図（エリア別需要・太陽光・風力プロファイル）
        hours_24 = list(range(1, min(25, T + 1)))
        d_a1 = [DEMAND_BY_AREA["Area1"][t - 1] for t in hours_24]
        d_a2 = [DEMAND_BY_AREA["Area2"][t - 1] for t in hours_24]
        d_a3 = [DEMAND_BY_AREA["Area3"][t - 1] for t in hours_24]
        s_a1 = [SOLAR_AVAIL_BY_AREA["Area1"][t - 1] for t in hours_24]
        s_a2 = [SOLAR_AVAIL_BY_AREA["Area2"][t - 1] for t in hours_24]
        s_a3 = [SOLAR_AVAIL_BY_AREA["Area3"][t - 1] for t in hours_24]
        w_a1 = [WIND_AVAIL_BY_AREA["Area1"][t - 1] for t in hours_24]
        w_a2 = [WIND_AVAIL_BY_AREA["Area2"][t - 1] for t in hours_24]
        w_a3 = [WIND_AVAIL_BY_AREA["Area3"][t - 1] for t in hours_24]
        fig_cond, axc = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        axc[0].plot(hours_24, d_a1, "-", color="C0", linewidth=1.2, label="Area1")
        axc[0].plot(hours_24, d_a2, "-", color="C2", linewidth=1.2, label="Area2")
        axc[0].plot(hours_24, d_a3, "-", color="C4", linewidth=1.2, label="Area3")
        axc[0].set_ylabel("Demand (MW)")
        axc[0].legend(loc="upper right", fontsize=8)
        axc[0].grid(True, alpha=0.3)
        axc[0].set_title("Demand by area", fontsize=11)
        axc[1].plot(hours_24, s_a1, "-", color="C0", linewidth=1.2, label="Area1")
        axc[1].plot(hours_24, s_a2, "-", color="C2", linewidth=1.2, label="Area2")
        axc[1].plot(hours_24, s_a3, "-", color="C4", linewidth=1.2, label="Area3")
        axc[1].set_ylabel("Solar available (MW)")
        axc[1].set_title("Solar available by area", fontsize=11)
        axc[1].legend(loc="upper right", fontsize=8)
        axc[1].grid(True, alpha=0.3)
        axc[2].plot(hours_24, w_a1, "-", color="C0", linewidth=1.2, label="Area1")
        axc[2].plot(hours_24, w_a2, "-", color="C2", linewidth=1.2, label="Area2")
        axc[2].plot(hours_24, w_a3, "-", color="C4", linewidth=1.2, label="Area3")
        axc[2].set_ylabel("Wind available (MW)")
        axc[2].set_xlabel("Time (h)")
        axc[2].set_title("Wind available by area", fontsize=11)
        axc[2].legend(loc="upper right", fontsize=8)
        axc[2].grid(True, alpha=0.3)
        fig_cond.tight_layout()
        buf_cond = io.BytesIO()
        fig_cond.savefig(buf_cond, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig_cond)
        buf_cond.seek(0)
        cond_b64 = base64.b64encode(buf_cond.read()).decode("ascii")

        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        fig = plt.figure(figsize=(max(15, T * 0.22), 14))
        gs = GridSpec(5, 4, figure=fig, width_ratios=[1, 1, 1, 0.35])
        axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(5)]
        axes_legend = [fig.add_subplot(gs[r, 3]) for r in range(5)]

        def _vline_day(ax):
            if DAYS > 1:
                for d in range(1, DAYS):
                    ax.axvline(d * HOURS_PER_DAY + 0.5, color="gray", linestyle=":", alpha=0.7)

        for col, area_name in enumerate(AREAS):
            a = col
            ax = axes[0][col]
            d_a = demand_by_area[area_name]
            s_u = solar_used_by_area[area_name]
            s_av = solar_avail_by_area[area_name]
            w_a = wind_by_area[area_name]
            th_a = thermal_by_area[area_name]
            nl_a = net_load_by_area[area_name]
            supply_target = supply_target_by_area[area_name]
            ax.fill_between(hours, 0, d_a, alpha=0.3, color="C0", label="Demand")
            ax.fill_between(hours, 0, s_u, alpha=0.5, color="gold", label="Solar (used)")
            if ALLOW_SOLAR_CURTAIL == 1:
                ax.plot(hours, s_av, "--", color="goldenrod", linewidth=0.8, alpha=0.9, label="Solar (avail)")
            ax.fill_between(hours, s_u, s_u + w_a, alpha=0.5, color="skyblue", label="Wind")
            ax.plot(hours, nl_a, "-", color="C3", linewidth=1.2, label="Net load")
            ax.plot(hours, th_a, "o-", color="C1", markersize=2, label="Thermal")
            ax.plot(hours, supply_target, "k--", linewidth=1.0, alpha=0.9, label="Demand+NetExport (=Supply)")
            ax.set_ylabel("Power (MW)")
            ax.set_title(f"{area_name}  Supply-Demand")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, T + 0.5)
            _vline_day(ax)
            ax = axes[1][col]
            supply_target = supply_target_by_area[area_name]
            if col == 0:
                tie_receive_arr = tie_receive_area1
            elif col == 1:
                tie_receive_arr = tie_receive_area2_from1 + tie_receive_area2_from3
            else:
                tie_receive_arr = tie_receive_area3
            bottom = np.zeros_like(tie_receive_arr, dtype=float)
            ax.fill_between(hours, bottom, tie_receive_arr, alpha=0.6, color="purple", label="Tie receive")
            bottom = np.array(tie_receive_arr, dtype=float)
            ax.fill_between(hours, bottom, bottom + s_u, alpha=0.6, color="gold", label="Solar")
            bottom = bottom + s_u
            ax.fill_between(hours, bottom, bottom + w_a, alpha=0.6, color="skyblue", label="Wind")
            bottom = bottom + w_a
            for i, g in enumerate(G_BY_AREA[a]):
                top = bottom + p_by_area[a][i]
                ax.fill_between(hours, bottom, top, alpha=0.85, color=colors_by_area[a][i], label=g)
                bottom = top
            ax.plot(hours, d_a, "k-", linewidth=1.2, label="Demand")
            ax.plot(hours, supply_target, "k--", linewidth=1.0, alpha=0.9, label="Demand+NetExport (=Supply)")
            ax.plot(hours, nl_a, "k:", linewidth=0.8, alpha=0.8, label="Net load")
            ax.set_ylabel("Generation (MW)")
            ax.set_title(f"{area_name}  Stacked")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, T + 0.5)
            _vline_day(ax)
            ax = axes[2][col]
            u_a = u_by_area[a]
            n_g = u_a.shape[0]
            im = ax.imshow(u_a, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, extent=[0.5, T + 0.5, n_g, 0])
            ax.set_yticks(np.arange(n_g - 0.5, -0.5, -1))
            ax.set_yticklabels(G_BY_AREA[a])
            ax.set_ylabel("Generator")
            ax.set_title(f"{area_name}  Unit Commitment")
            ax.set_xlim(0.5, T + 0.5)
            if DAYS > 1:
                for d in range(1, DAYS):
                    ax.axvline(d * HOURS_PER_DAY - 0.5, color="gray", linewidth=0.8)
            ax = axes[3][col]
            adj_a = adjustment_by_area[a]
            req_adj_a = adj_required_by_area[a]
            ax.fill_between(hours, req_adj_a, adj_a, where=(adj_a >= req_adj_a), alpha=0.3, color="blue")
            ax.plot(hours, req_adj_a, "--", color="gray", linewidth=1.0, label="Required")
            ax.plot(hours, adj_a, "-", color="blue", linewidth=1.0, label="Actual")
            ax.set_ylabel("Adjustment (MW)")
            ax.set_title(f"{area_name}  Adjustment")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            ax.set_xlim(0.5, T + 0.5)
            _vline_day(ax)
            ax = axes[4][col]
            ax.axhspan(-F_MAX, F_MAX, alpha=0.15, color="gray")
            ax.axhline(F_MAX, color="red", linestyle="--", linewidth=0.8, alpha=0.8)
            ax.axhline(-F_MAX, color="red", linestyle="--", linewidth=0.8, alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.5)
            if col == 0:
                ax.plot(hours, tie_receive_area1, "-", color="navy", linewidth=1.2, label="Tie receive")
            elif col == 1:
                ax.plot(hours, tie_receive_area2_from1, "-", color="navy", linewidth=1.0, label="Receive from Area1")
                ax.plot(hours, tie_receive_area2_from3, "-", color="darkgreen", linewidth=1.0, label="Receive from Area3")
            else:
                ax.plot(hours, tie_receive_area3, "-", color="darkgreen", linewidth=1.2, label="Tie receive")
            ax.set_ylabel("Tie receive (MW)\n(positive=import)")
            ax.set_title(f"{area_name}  Tie-Line")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, T + 0.5)
            _vline_day(ax)
        for col in range(3):
            axes[4][col].set_xlabel("Time (h)")
        step = max(1, (T // 24)) * 6
        xticks = list(range(1, T + 1, step))
        for row in range(5):
            for col in range(3):
                axes[row][col].set_xticks(xticks)
                axes[row][col].set_xticklabels(xticks)
        ax = axes_legend[0]
        ax.axis("off")
        h0 = [
            Line2D([0], [0], color="black", linewidth=1.5, label="Demand"),
            Line2D([0], [0], color="gray", linestyle="--", linewidth=1.2, label="Demand+NetExport"),
            Patch(facecolor="gold", edgecolor="gray", label="Solar"),
            Patch(facecolor="skyblue", edgecolor="gray", label="Wind"),
            Line2D([0], [0], color="C3", linewidth=1.2, label="Net load"),
            Line2D([0], [0], color="C1", linewidth=1, label="Thermal"),
        ]
        if ALLOW_SOLAR_CURTAIL == 1:
            h0.insert(3, Line2D([0], [0], color="goldenrod", linestyle="--", linewidth=0.8, label="Solar (avail)"))
        ax.legend(handles=h0, loc="center", fontsize=7, frameon=True)
        ax = axes_legend[1]
        ax.axis("off")
        ax.legend(handles=[
            Patch(facecolor="purple", edgecolor="gray", alpha=0.6, label="Tie receive"),
            Patch(facecolor="gold", edgecolor="gray", label="Solar"),
            Patch(facecolor="skyblue", edgecolor="gray", label="Wind"),
            Patch(facecolor="darkorange", edgecolor="gray", label="Thermal"),
            Line2D([0], [0], color="black", linewidth=1.2, label="Demand"),
            Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="Demand+NetExport"),
            Line2D([0], [0], color="black", linestyle=":", linewidth=0.8, label="Net load"),
        ], loc="center", fontsize=7, frameon=True)
        ax = axes_legend[2]
        ax.axis("off")
        ax.legend(handles=[
            Patch(facecolor="green", edgecolor="gray", label="ON (1)"),
            Patch(facecolor="red", edgecolor="gray", label="OFF (0)"),
        ], loc="center", fontsize=7, frameon=True)
        ax = axes_legend[3]
        ax.axis("off")
        ax.legend(handles=[
            Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="Required"),
            Line2D([0], [0], color="blue", linewidth=1, label="Actual"),
        ], loc="center", fontsize=7, frameon=True)
        ax = axes_legend[4]
        ax.axis("off")
        ax.legend(handles=[
            Line2D([0], [0], color="navy", linewidth=1.2, label="Tie receive / from Area1"),
            Line2D([0], [0], color="darkgreen", linewidth=1.2, label="Tie receive / from Area3"),
            Line2D([0], [0], color="red", linestyle="--", linewidth=0.8, label=f"±F_max ({F_MAX} MW)"),
        ], loc="center", fontsize=7, frameon=True)
        plt.tight_layout()
        buf_result = io.BytesIO()
        fig.savefig(buf_result, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf_result.seek(0)
        result_b64 = base64.b64encode(buf_result.read()).decode("ascii")
        runs.append({
            "scale": 1.0,
        "status": LpStatus[prob.status],
        "cond_b64": cond_b64,
        "result_b64": result_b64,
        "cost": value(prob.objective),
        "cost_fuel": cost_fuel,
        "cost_noload": cost_noload,
        "cost_start": cost_start,
        "max_abs_bal1": max_abs_bal1,
        "max_abs_bal2": max_abs_bal2,
        "max_abs_bal3": max_abs_bal3,
        "max_abs_f12": max_abs_f12,
        "max_abs_f23": max_abs_f23,
        "solar_avail_total": list(solar_avail_total),
        "solar_used_total": list(solar_used_total),
        "adjustment_by_area": [np.array(adjustment_by_area[a]) for a in range(N_AREAS)],
        "adj_required_by_area": [np.array(adj_required_by_area[a]) for a in range(N_AREAS)],
    })

print("SCUC 完了。HTMLレポートを生成します。")

# ---------------------------------------------------------------------------
# 5. HTML レポート（手法 | 条件 | 結果 | 考察）
# ---------------------------------------------------------------------------
ts = datetime.now().strftime("%y%m%d%H%M%S")
cond = f"{DAYS}d_s{SEASON}_m{RENEW_MODE}"
cond += f"_load{int(load_share_pct[0])}{int(load_share_pct[1])}{int(load_share_pct[2])}"
cond += f"_scap{int(solar_cap_pct[0])}{int(solar_cap_pct[1])}{int(solar_cap_pct[2])}"
cond += f"_wcap{int(wind_cap_pct[0])}{int(wind_cap_pct[1])}{int(wind_cap_pct[2])}"
cond += f"_solarA{''.join(str(solar_types_by_area_day[a][0]) for a in range(N_AREAS))}"
cond += f"_windA{''.join(str(wind_types_by_area_day[a][0]) for a in range(N_AREAS))}"
if DAYS > 1 and RENEW_MODE != 1:
    cond += "_vary"
cond += f"_curt{1 if ALLOW_SOLAR_CURTAIL == 1 else 0}"
cond += f"_scuc_f{F_MAX}"
os.makedirs("results", exist_ok=True)

def _build_report_html(runs):
    season_name = {1: "Spring/Autumn", 2: "Summer", 3: "Winter"}.get(SEASON, str(SEASON))
    method_html = r"""
    <h2>Simulation method (数式)</h2>
    <p>3エリア SCUC: 決定変数・目的関数・制約を以下で定義する。</p>
    <p><strong>集合・添字:</strong> エリア \(a \in \{1,2,3\}\)、発電機 \(g \in \mathcal{G}\)、時刻 \(t \in \mathcal{T}\)。\(\mathcal{G}_a\) はエリア \(a\) の発電機集合。</p>
    <p><strong>目的関数（最小化）:</strong></p>
    \[
    \min \sum_{t \in \mathcal{T}} \sum_{g \in \mathcal{G}} \left( c_g^{\mathrm{fuel}} P_{g,t} + c_g^{\mathrm{noload}} u_{g,t} + c_g^{\mathrm{start}} v_{g,t} \right)
    \]
    <p><strong>需給バランス（エリア別）:</strong></p>
    \[
    \sum_{g \in \mathcal{G}_1} P_{g,t} + W_{1,t} + S_{1,t} = D_{1,t} + f_{12,t}
    \quad,\quad
    \sum_{g \in \mathcal{G}_2} P_{g,t} + W_{2,t} + S_{2,t} = D_{2,t} - f_{12,t} + f_{23,t}
    \quad,\quad
    \sum_{g \in \mathcal{G}_3} P_{g,t} + W_{3,t} + S_{3,t} = D_{3,t} - f_{23,t}
    \]
    <p>（\(W_{a,t}, S_{a,t}\): 風力・太陽光利用量、\(D_{a,t}\): 需要、\(f_{12,t}, f_{23,t}\): 連系線潮流。正の方向は 1→2, 2→3。）</p>
    <p><strong>連系線制約:</strong></p>
    \[
    -F_{\max} \le f_{12,t} \le F_{\max}
    \quad,\quad
    -F_{\max} \le f_{23,t} \le F_{\max}
    \]
    <p><strong>調整力（エリア別）:</strong></p>
    \[
    \sum_{g \in \mathcal{G}_a} \overline{P}_g u_{g,t} - \sum_{g \in \mathcal{G}_a} P_{g,t} \ge A_{a,t}^{\mathrm{req}}
    \]
    <p><strong>最低稼働機数:</strong> \(\displaystyle \sum_{g \in \mathcal{G}_a} u_{g,t} \ge N_{\min}\)（各 \(a,t\)）。</p>
    <p><strong>その他:</strong> \(P_{g,t} \in [\underline{P}_g u_{g,t}, \overline{P}_g u_{g,t}]\)、ランプ・MUT/MDT・スタートアップ \(v_{g,t}\) の論理制約は通常のSCUCと同様。</p>
    <p><strong>蓄電池（各エリア）:</strong> 需給に \(P_{\mathrm{disch},a,t} - P_{\mathrm{ch},a,t}\) を追加。SOC: \(E_{a,t} = E_{a,t-1} + \eta_c P_{\mathrm{ch},a,t} - P_{\mathrm{disch},a,t}/\eta_d\)、上下限・初期/終端値・充放電排他（同時不可）を制約。</p>
    """
    # 条件タブ・結果タブを各 run ごとに生成
    cond_panels = []
    result_panels = []
    tab_buttons_cond = []
    tab_buttons_result = []
    for i, run in enumerate(runs):
        idx = i + 1
        scale = run.get("scale", idx)
        tab_buttons_cond.append(f'<li><button type="button" role="tab" id="tab-cond-{idx}" aria-selected="false">条件</button></li>')
        tab_buttons_result.append(f'<li><button type="button" role="tab" id="tab-result-{idx}" aria-selected="false">結果</button></li>')
        cond_b64 = run.get("cond_b64")
        result_b64 = run.get("result_b64")
        status = run.get("status", "—")
        cost = run.get("cost")
        cost_str = f"{cost:,.0f}" if cost is not None else "—"
        cond_img = f'<img src="data:image/png;base64,{cond_b64}" alt="Condition" class="cond-fig result-fig" />' if cond_b64 else "<p>条件図なし（未求解または失敗）</p>"
        result_img = f'<img src="data:image/png;base64,{result_b64}" alt="Result" class="result-fig" />' if result_b64 else "<p>結果図なし（未求解または失敗）</p>"
        cond_panels.append(f'''
    <div id="panel-cond-{idx}" class="tab-panel" role="tabpanel">
      <h2>条件</h2>
      <p class="cond-caption">Season: {season_name}, Days: {DAYS}, F_max: {F_MAX} MW.</p>
      <div class="cond-fig-wrap">{cond_img}</div>
    </div>''')
        result_panels.append(f'''
    <div id="panel-result-{idx}" class="tab-panel" role="tabpanel">
      <h2>結果</h2>
      <p class="result-summary"><strong>Status:</strong> {status} &nbsp; <strong>Total cost:</strong> {cost_str} JPY</p>
      <div class="result-fig-wrap">{result_img}</div>
    </div>''')

    # 考察: 結果サマリ（倍数なし）
    rows = []
    for run in runs:
        cost = run.get("cost")
        cost_str = f"{cost:,.0f}" if cost is not None else "—"
        mf12 = run.get("max_abs_f12")
        mf23 = run.get("max_abs_f23")
        util_f12 = (mf12 / F_MAX * 100) if (F_MAX and mf12 is not None) else 0
        util_f23 = (mf23 / F_MAX * 100) if (F_MAX and mf23 is not None) else 0
        mf12_s = f"{mf12:.1f}" if mf12 is not None else "—"
        mf23_s = f"{mf23:.1f}" if mf23 is not None else "—"
        savail = run.get("solar_avail_total") or []
        sused = run.get("solar_used_total") or []
        curt = sum(max(0, (savail[t] if t < len(savail) else 0) - (sused[t] if t < len(sused) else 0)) for t in range(T))
        avail_sum = sum(savail) if savail else 0
        curt_pct = (curt / avail_sum * 100) if avail_sum else 0
        rows.append(f"      <tr><td>{cost_str}</td><td>{mf12_s}</td><td>{util_f12:.0f}%</td><td>{mf23_s}</td><td>{util_f23:.0f}%</td><td>{curt:.1f}</td><td>{curt_pct:.1f}%</td></tr>")
    table_rows = "\n".join(rows)
    consideration_html = f"""
    <h2>考察（Consideration）</h2>
    <h3>結果サマリ</h3>
    <div class="scroll-wrap">
    <table class="data-table">
      <thead><tr><th>Total cost (JPY)</th><th>|F12| max (MW)</th><th>F12 util %</th><th>|F23| max (MW)</th><th>F23 util %</th><th>Solar curt (MW)</th><th>Curt %</th></tr></thead>
      <tbody>
{table_rows}
      </tbody>
    </table>
    </div>
    <h3>まとめ</h3>
    <p>上表と「結果」タブの図を参照し、連系線利用率・太陽光抑制率などを確認できる。</p>
    """

    tab_list = (
        '<li><button type="button" role="tab" id="tab-method" aria-selected="true" class="active">シミュレーション手法</button></li>\n      '
        + "\n      ".join(tab_buttons_cond) + "\n      "
        + "\n      ".join(tab_buttons_result) + "\n      "
        + '<li><button type="button" role="tab" id="tab-consideration" aria-selected="false">考察</button></li>'
    )
    panels = (
        '<div id="panel-method" class="tab-panel active" role="tabpanel">' + method_html + "</div>\n"
        + "".join(cond_panels) + "\n"
        + "".join(result_panels) + "\n"
        + '<div id="panel-consideration" class="tab-panel" role="tabpanel">' + consideration_html + "</div>"
    )
    n = len(runs)
    tab_ids_js = "['method', " + ", ".join(f"'cond-{j+1}'" for j in range(n)) + ", " + ", ".join(f"'result-{j+1}'" for j in range(n)) + ", 'consideration']"
    onclick_js = []
    onclick_js.append("document.getElementById('tab-method').onclick = function() { show('method'); };")
    for j in range(n):
        idx = j + 1
        onclick_js.append(f"document.getElementById('tab-cond-{idx}').onclick = function() {{ show('cond-{idx}'); }};")
        onclick_js.append(f"document.getElementById('tab-result-{idx}').onclick = function() {{ show('result-{idx}'); }};")
    onclick_js.append("document.getElementById('tab-consideration').onclick = function() { show('consideration'); };")

    html_content = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SCUC 3-Area Report</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js" async></script>
  <style>
    * { box-sizing: border-box; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #222; }
    .container { max-width: 1200px; margin: 0 auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); overflow: hidden; }
    .tab-panel .cond-fig-wrap { margin: 16px 0; text-align: center; }
    .tab-panel .cond-fig { max-width: 100%; height: auto; display: block; margin: 0 auto; border: 1px solid #ccc; border-radius: 8px; }
    .tab-panel .cond-caption { color: #555; font-size: 0.95rem; margin-bottom: 12px; }
    #panel-method { overflow-x: auto; }
    #panel-method .MathJax { overflow-x: auto; overflow-y: visible; }
    .tabs { display: flex; flex-wrap: wrap; background: #1a1a2e; padding: 0; margin: 0; list-style: none; }
    .tabs li { min-width: 0; }
    .tabs button { padding: 10px 12px; border: none; background: transparent; color: #eee; font-size: 0.9rem; cursor: pointer; transition: background .2s; white-space: nowrap; }
    .tabs button:hover { background: rgba(255,255,255,0.1); }
    .tabs button.active { background: #16213e; color: #7dd3fc; font-weight: 600; }
    .tab-panel { display: none; padding: 24px; }
    .tab-panel.active { display: block; }
    .tab-panel h2 { margin-top: 0; color: #1a1a2e; border-bottom: 2px solid #7dd3fc; padding-bottom: 8px; }
    .tab-panel h3 { margin-top: 24px; color: #16213e; }
    .info-table, .data-table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.9rem; }
    .info-table th, .info-table td, .data-table th, .data-table td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
    .info-table th, .data-table th { background: #16213e; color: #fff; }
    .data-table td { text-align: right; }
    .data-table td:first-child, .data-table th:first-child { text-align: left; }
    .scroll-wrap { overflow-x: auto; margin: 12px 0; }
    .result-summary { margin-bottom: 16px; font-size: 1.05rem; }
    .result-fig-wrap { margin-top: 8px; text-align: center; }
    .result-fig { max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 8px; display: block; margin: 0 auto; }
  </style>
</head>
<body>
  <div class="container">
    <ul class="tabs" role="tablist">
      """ + tab_list + """
    </ul>
    """ + panels + """
  </div>
  <script>
    (function() {
      var tabs = """ + tab_ids_js + """;
      function show(id) {
        tabs.forEach(function(key) {
          var on = key === id;
          var tabEl = document.getElementById('tab-' + key);
          var panelEl = document.getElementById('panel-' + key);
          if (tabEl) tabEl.classList.toggle('active', on);
          if (panelEl) panelEl.classList.toggle('active', on);
        });
      }
      """ + "\n      ".join(onclick_js) + """
    })();
  </script>
</body>
</html>"""
    return html_content

_report_html = _build_report_html(runs)
_html_path = os.path.join("results", f"{ts}_{cond}.html")
with open(_html_path, "w", encoding="utf-8") as f:
    f.write(_report_html)
print(f"HTMLレポートを保存しました: {_html_path}")
