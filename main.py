"""
発電機起動停止計画（ユニット・コミットメント）の例題

調査書に基づく数理モデル:
- 目的関数: 燃料費 + 無負荷コスト + 起動コストの最小化
- 制約: 需給バランス、予備力、出力上下限、最低運転/停止時間、ランプ速度
- 3変量モデル: u(状態), v(起動), w(停止)
- 対話形式で解析日数・季節・太陽光・風力曲線を指定可能
"""

import os
import random
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

print("=" * 60)
print("発電機起動停止計画（UC） パラメータ入力")
print("=" * 60)
DAYS = input_int("解析対象日数（日） [1-7, 省略=1]: ", 1, 1, 7)

# 複数日のときだけ、はじめにモードを選択
RENEW_MODE = 1  # 1日または未指定時は「全日同じ」
if DAYS >= 2:
    print("日ごとの再エネ条件: 1=全日同じ, 2=日ごとランダム, 3=シナリオ（日別に指定）")
    RENEW_MODE = input_int("モード [1-3, 省略=1]: ", 1, 1, 3)

print("解析対象季節: 1=春秋, 2=夏, 3=冬")
SEASON = input_int("季節 [1-3, 省略=1]: ", 1, 1, 3)

# 太陽光・風力の指定: 1日または「全日同じ」のときだけ単一曲線を入力
solar_types_per_day = None
wind_types_per_day = None
if DAYS == 1 or RENEW_MODE == 1:
    print("太陽光発電曲線: 1=快晴日, 2=準快晴日, 3=曇天日, 4=雨天日, 5=ランプアップ, 6=ランプダウン")
    SOLAR_TYPE = input_int("太陽光曲線 [1-6, 省略=1]: ", 1, 1, 6)
    print("風力発電曲線: 1=風が強い日, 2=風が弱い日, 3=午前中が強い日, 4=午後が強い日")
    WIND_TYPE = input_int("風力曲線 [1-4, 省略=1]: ", 1, 1, 4)
else:
    SOLAR_TYPE = None
    WIND_TYPE = None

# シナリオ（日別指定）のときは、各日の曲線をここで入力
if RENEW_MODE == 3:
    print(f"各日（{DAYS}日分）の曲線タイプをカンマ区切りで入力してください。")
    solar_types_per_day = input_scenario(
        f"  太陽光(1-6) 例: {','.join(str((i % 6) + 1) for i in range(DAYS))}: ",
        DAYS, 1, 6,
    )
    wind_types_per_day = input_scenario(
        f"  風力(1-4) 例: {','.join(str((i % 4) + 1) for i in range(DAYS))}: ",
        DAYS, 1, 4,
    )
    print("【シナリオ】太陽光:", solar_types_per_day, " 風力:", wind_types_per_day)
print()

# ---------------------------------------------------------------------------
# 1. データ定義（季節・太陽光・風力の1日パターン）
# ---------------------------------------------------------------------------

# 季節別 1日24時間の需要パターン (MW) — 初期値を2倍
HOURS_PER_DAY = 24
DEMAND_1D_BY_SEASON = {
    1: [  # 春秋
        360, 340, 320, 310, 320, 360, 440, 560, 640, 680, 700, 720,
        700, 680, 660, 680, 720, 760, 800, 760, 700, 600, 500, 400,
    ],
    2: [  # 夏（冷房で日中のピークが高い）
        400, 380, 370, 360, 380, 440, 560, 720, 840, 920, 960, 1000,
        960, 920, 880, 900, 940, 980, 1000, 960, 880, 760, 640, 500,
    ],
    3: [  # 冬（暖房で朝夕のピーク）
        560, 600, 640, 620, 600, 640, 720, 800, 840, 860, 880, 900,
        880, 860, 840, 860, 900, 960, 1000, 980, 920, 840, 760, 640,
    ],
}

# 太陽光発電 1日24時間パターン (MW) — 想定設備容量ベース
SOLAR_1D_BY_TYPE = {
    1: [0,0,0,0,0,5,30,80,140,200,240,260,260,240,200,140,80,30,5,0,0,0,0,0],   # 快晴日
    2: [0,0,0,0,0,3,18,50,90,130,160,170,170,160,130,90,50,18,3,0,0,0,0,0],   # 準快晴日
    3: [0,0,0,0,0,2,10,30,50,70,80,85,85,80,70,50,30,10,2,0,0,0,0,0],         # 曇天日
    4: [0,0,0,0,0,0,5,15,25,35,40,42,42,40,35,25,15,5,0,0,0,0,0,0],           # 雨天日
    5: [0,0,0,0,0,0,0,0,0,20,60,100,140,180,220,260,260,200,100,20,0,0,0,0], # ランプアップ
    6: [0,0,0,0,0,50,120,180,240,260,260,260,220,180,140,80,40,10,0,0,0,0,0,0], # ランプダウン
}

# 風力発電 1日24時間パターン (MW)
WIND_1D_BY_TYPE = {
    1: [80,85,82,78,75,80,85,90,88,85,82,80,78,82,85,88,90,85,82,80,78,82,85,80],  # 風が強い日
    2: [15,18,16,14,12,15,18,20,18,16,14,12,14,16,18,20,18,16,14,15,14,16,18,15],  # 風が弱い日
    3: [70,75,80,85,90,95,90,85,80,70,60,50,40,35,30,35,40,45,50,45,40,35,30,25],  # 午前中が強い日
    4: [25,30,35,40,45,50,55,60,65,70,75,80,85,90,85,80,75,70,65,60,55,50,45,40],  # 午後が強い日
}

# 解析期間: T 時間 = 24 * 日数
T = HOURS_PER_DAY * DAYS
TIME = range(1, T + 1)

# 日ごとの太陽光・風力の曲線タイプを決定（モード3は入力済み）
if RENEW_MODE == 1:
    solar_types_per_day = [SOLAR_TYPE] * DAYS
    wind_types_per_day = [WIND_TYPE] * DAYS
elif RENEW_MODE == 2:
    random.seed(42)
    solar_types_per_day = [random.randint(1, 6) for _ in range(DAYS)]
    wind_types_per_day = [random.randint(1, 4) for _ in range(DAYS)]
    print("【ランダム割当】太陽光:", solar_types_per_day, " 風力:", wind_types_per_day)
# RENEW_MODE == 3: solar_types_per_day, wind_types_per_day は対話で入力済み

# 需要は季節の1日パターンを日数分繰り返し
demand_1d = DEMAND_1D_BY_SEASON[SEASON]
DEMAND = (demand_1d * DAYS)[:T]

# 太陽光・風力は日ごとの曲線を連結
SOLAR = []
WIND = []
for d in range(DAYS):
    SOLAR.extend(SOLAR_1D_BY_TYPE[solar_types_per_day[d]])
    WIND.extend(WIND_1D_BY_TYPE[wind_types_per_day[d]])
SOLAR = SOLAR[:T]
WIND = WIND[:T]
# 火力が賄う純負荷（需要 − 太陽光 − 風力）
NET_LOAD = [max(0, d - s - w) for d, s, w in zip(DEMAND, SOLAR, WIND)]

# 予備力の所要量：最大需要 × 3%（全時刻共通）
RESERVE_FRACTION = 0.03  # 最大需要に対する率
reserve_value = max(1, int(max(DEMAND) * RESERVE_FRACTION))
RESERVE = [reserve_value] * T

# 所要調整力（需要10% + 再エネ25%）・確保調整力＝Σ(定格出力−計画出力)、稼働機のみ
ADJ_LOAD_FRAC = 0.10
ADJ_RES_FRAC = 0.25
ADJ_REQUIRED = [DEMAND[t] * ADJ_LOAD_FRAC + (SOLAR[t] + WIND[t]) * ADJ_RES_FRAC for t in range(T)]

# 発電機セット（10機）
G = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"]

# 発電機パラメータ（MUT, MDT は全機 4 h で統一）— 初期値1倍、スケールなし
GEN_DATA = {
    "G1": {"Pmin": 50, "Pmax": 200, "no_load": 500, "cost_per_mwh": 3000, "startup": 8000, "MUT": 4, "MDT": 4, "RU": 80, "RD": 80},
    "G2": {"Pmin": 30, "Pmax": 100, "no_load": 200, "cost_per_mwh": 3500, "startup": 4000, "MUT": 4, "MDT": 4, "RU": 50, "RD": 50},
    "G3": {"Pmin": 80, "Pmax": 300, "no_load": 800, "cost_per_mwh": 2500, "startup": 12000, "MUT": 4, "MDT": 4, "RU": 100, "RD": 100},
    "G4": {"Pmin": 20, "Pmax": 80, "no_load": 100, "cost_per_mwh": 4000, "startup": 2000, "MUT": 4, "MDT": 4, "RU": 40, "RD": 40},
    "G5": {"Pmin": 40, "Pmax": 150, "no_load": 350, "cost_per_mwh": 3200, "startup": 6000, "MUT": 4, "MDT": 4, "RU": 60, "RD": 60},
    "G6": {"Pmin": 25, "Pmax": 90, "no_load": 180, "cost_per_mwh": 3600, "startup": 3500, "MUT": 4, "MDT": 4, "RU": 45, "RD": 45},
    "G7": {"Pmin": 60, "Pmax": 250, "no_load": 600, "cost_per_mwh": 2700, "startup": 10000, "MUT": 4, "MDT": 4, "RU": 90, "RD": 90},
    "G8": {"Pmin": 35, "Pmax": 120, "no_load": 250, "cost_per_mwh": 3400, "startup": 4500, "MUT": 4, "MDT": 4, "RU": 55, "RD": 55},
    "G9": {"Pmin": 45, "Pmax": 180, "no_load": 420, "cost_per_mwh": 2900, "startup": 7000, "MUT": 4, "MDT": 4, "RU": 70, "RD": 70},
    "G10": {"Pmin": 15, "Pmax": 60, "no_load": 80, "cost_per_mwh": 4200, "startup": 1500, "MUT": 4, "MDT": 4, "RU": 30, "RD": 30},
}

# 初期状態: 時刻0では全機停止
U0 = {g: 0 for g in G}

# 起動停止維持時間・最大合計出力・最大需要の表示
print("【発電機 最低運転時間(MUT)・最低停止時間(MDT)】")
for g in G:
    print(f"  {g}: MUT={GEN_DATA[g]['MUT']} h, MDT={GEN_DATA[g]['MDT']} h")
print("【発電機 最大出力 Pmax (MW)】")
for g in G:
    print(f"  {g}: Pmax={GEN_DATA[g]['Pmax']} MW", end="  ")
print()
max_total_capacity = sum(GEN_DATA[g]["Pmax"] for g in G)
max_demand = max(DEMAND)
print(f"【最大合計発電機出力】 {max_total_capacity} MW")
print(f"【最大総需要】         {max_demand} MW")
print()

# ---------------------------------------------------------------------------
# 2. 最適化モデル（MILP）
# ---------------------------------------------------------------------------

prob = LpProblem("UnitCommitment", LpMinimize)

# 決定変数
# u[g,t]: 1=稼働, 0=停止
u = LpVariable.dicts("u", (G, TIME), cat=LpBinary)
# v[g,t]: 1=時刻tで起動
v = LpVariable.dicts("v", (G, TIME), cat=LpBinary)
# w[g,t]: 1=時刻tで停止
w = LpVariable.dicts("w", (G, TIME), cat=LpBinary)
# P[g,t]: 発電出力 (MW)
P = LpVariable.dicts("P", (G, TIME), lowBound=0, cat=LpContinuous)

# --- 目的関数: 総運用コストの最小化 ---
# 燃料費(線形) + 無負荷コスト + 起動コスト（停止コストは省略）
prob += lpSum(
    GEN_DATA[g]["cost_per_mwh"] * P[g][t] + GEN_DATA[g]["no_load"] * u[g][t] + GEN_DATA[g]["startup"] * v[g][t]
    for g in G for t in TIME
), "TotalCost"

# --- 需給バランス制約: 火力の合計 = 純負荷（需要−太陽光−風力） ---
for t in TIME:
    prob += lpSum(P[g][t] for g in G) == NET_LOAD[t - 1], f"PowerBalance_{t}"

# --- 予備力制約: 稼働機の定格出力の合計 >= 所要予備力 ---
# 確保予備力 = Σ_g (Pmax_g * u_{g,t})（稼働機の定格合計）
for t in TIME:
    prob += (
        lpSum(GEN_DATA[g]["Pmax"] * u[g][t] for g in G) >= RESERVE[t - 1],
        f"Reserve_{t}",
    )

# --- 調整力制約: 確保調整力 >= 所要調整力（所要＝需要10%+再エネ25%、確保＝Σ(定格−計画出力)・稼働機のみ）---
for t in TIME:
    prob += (
        lpSum(GEN_DATA[g]["Pmax"] * u[g][t] - P[g][t] for g in G) >= ADJ_REQUIRED[t - 1],
        f"Adjustment_{t}",
    )

# --- 出力上下限: Pmin*u <= P <= Pmax*u ---
for g in G:
    pmin, pmax = GEN_DATA[g]["Pmin"], GEN_DATA[g]["Pmax"]
    for t in TIME:
        prob += P[g][t] >= pmin * u[g][t], f"Pmin_{g}_{t}"
        prob += P[g][t] <= pmax * u[g][t], f"Pmax_{g}_{t}"

# --- 3変量の関係: u_{g,t} - u_{g,t-1} = v_{g,t} - w_{g,t} ---
for g in G:
    for t in TIME:
        u_prev = U0[g] if t == 1 else u[g][t - 1]
        prob += u[g][t] - u_prev == v[g][t] - w[g][t], f"StateChange_{g}_{t}"

# --- 最低運転時間 (MUT): 起動したら MUT 時間は運転継続 ---
# 若し v[g,s]=1 なら sum_{t=s}^{s+MUT-1} u[g,t] >= MUT
for g in G:
    MUT = GEN_DATA[g]["MUT"]
    for s in TIME:
        end = min(s + MUT - 1, T)
        if end - s + 1 >= MUT:
            prob += lpSum(u[g][t] for t in range(s, end + 1)) >= MUT * v[g][s], f"MUT_{g}_{s}"

# --- 最低停止時間 (MDT): 停止したら MDT 時間は停止継続 ---
# 若し w[g,s]=1 なら sum_{t=s}^{s+MDT-1} (1 - u[g,t]) >= MDT
for g in G:
    MDT = GEN_DATA[g]["MDT"]
    for s in TIME:
        end = min(s + MDT - 1, T)
        if end - s + 1 >= MDT:
            prob += lpSum(1 - u[g][t] for t in range(s, end + 1)) >= MDT * w[g][s], f"MDT_{g}_{s}"

# --- ランプ制約 ---
# 増加: P_{g,t} - P_{g,t-1} <= RU_g (t>=2). 起動時(t=1)は Pmin〜Pmax のみで制限
# 減少: P_{g,t-1} - P_{g,t} <= RD_g
for g in G:
    RU, RD = GEN_DATA[g]["RU"], GEN_DATA[g]["RD"]
    for t in TIME:
        if t >= 2:
            prob += P[g][t] - P[g][t - 1] <= RU, f"RampUp_{g}_{t}"
            prob += P[g][t - 1] - P[g][t] <= RD, f"RampDown_{g}_{t}"

# ---------------------------------------------------------------------------
# 3. 求解と結果表示
# ---------------------------------------------------------------------------

# 利用可能なソルバーで求解（CBC は PuLP に同梱、Gurobi/CPLEX は別途）
try:
    prob.solve()
except Exception as e:
    print("求解でエラーが発生しました:", e)
    print("モデルを unit_commitment_model.lp に出力しました。")
    prob.writeLP("unit_commitment_model.lp")
    raise

print("=" * 60)
print("発電機起動停止計画（ユニット・コミットメント） 求解結果")
print("=" * 60)
print(f"ステータス: {LpStatus[prob.status]}")
if prob.status != 1:
    print("最適解が得られていません。モデルを unit_commitment_model.lp に出力しました。")
    prob.writeLP("unit_commitment_model.lp")
    exit(1)
print(f"総コスト: {value(prob.objective):,.0f} 円")
print()

# 時刻別 稼働機と出力（1日分のみ表示）
print("【時刻別 稼働状態 u と 発電出力 P (MW)】（1日目のみ）")
time_display = list(TIME)[:HOURS_PER_DAY]
print("時刻", " ".join(f"{t:3d}" for t in time_display))
for g in G:
    u_vals = [int(value(u[g][t])) for t in time_display]
    p_vals = [value(P[g][t]) for t in time_display]
    print(f"  {g} u ", " ".join(f"{x:3d}" for x in u_vals))
    print(f"     P ", " ".join(f"{x:5.0f}" for x in p_vals))
if T > HOURS_PER_DAY:
    print(f"  ... 2日目〜{DAYS}日目は省略")
print()

# 需給確認（先頭24時間のみ表示、複数日は要約）
print("【需給確認】（時刻=1〜24のみ表示）")
print("時刻  需要  純負荷  火力合計  太陽光  風力  予備力(MW)")
for t in list(TIME)[:HOURS_PER_DAY]:
    total_p = sum(value(P[g][t]) for g in G)
    total_cap = sum(GEN_DATA[g]["Pmax"] * value(u[g][t]) for g in G)
    reserve_mw = total_cap - total_p
    print(f" {t:2d}   {DEMAND[t-1]:4d}   {NET_LOAD[t-1]:4d}    {total_p:6.0f}   {SOLAR[t-1]:4d}  {WIND[t-1]:4d}   {reserve_mw:6.0f}")
if T > HOURS_PER_DAY:
    print(f" ... （2日目以降は省略、全{T}時間で需給バランス・予備力を満足）")
print()

# 起動・停止イベント
print("【起動 v / 停止 w イベント】")
for g in G:
    v_vals = [int(value(v[g][t])) for t in TIME]
    w_vals = [int(value(w[g][t])) for t in TIME]
    v_times = [t for t in TIME if v_vals[t - 1] > 0.5]
    w_times = [t for t in TIME if w_vals[t - 1] > 0.5]
    print(f"  {g}: 起動時刻 {v_times}, 停止時刻 {w_times}")
print()

# コスト内訳（簡易）
cost_fuel = sum(GEN_DATA[g]["cost_per_mwh"] * value(P[g][t]) for g in G for t in TIME)
cost_noload = sum(GEN_DATA[g]["no_load"] * value(u[g][t]) for g in G for t in TIME)
cost_start = sum(GEN_DATA[g]["startup"] * value(v[g][t]) for g in G for t in TIME)
print("【コスト内訳】")
print(f"  燃料費(線形): {cost_fuel:,.0f} 円")
print(f"  無負荷コスト: {cost_noload:,.0f} 円")
print(f"  起動コスト:   {cost_start:,.0f} 円")
print("=" * 60)

# ---------------------------------------------------------------------------
# 4. 計画立案結果の図（PNG）出力 → results/YYMMDDHHMMSS_解析条件.png
# ---------------------------------------------------------------------------

hours = list(TIME)
demand_arr = np.array(DEMAND)
solar_arr = np.array(SOLAR)
wind_arr = np.array(WIND)
net_load_arr = np.array(NET_LOAD)
# 各機の出力 [時間]
p_by_gen = np.array([[value(P[g][t]) for t in TIME] for g in G])
# 稼働状態 [機, 時間]
u_by_gen = np.array([[int(value(u[g][t])) for t in TIME] for g in G])
total_thermal = p_by_gen.sum(axis=0)
# 予備力: 稼働機の定格出力の合計（確保予備力）
total_capacity = (np.array([GEN_DATA[g]["Pmax"] for g in G])[:, np.newaxis] * u_by_gen).sum(axis=0)
actual_reserve = total_capacity  # 予備力 = 稼働機の定格合計
reserve_required = np.array(RESERVE)
# 調整力: 確保＝稼働機の(定格−計画出力)合計、所要＝需要10%+再エネ25%
adj_required = np.array(ADJ_REQUIRED)
actual_adjustment = total_capacity - total_thermal  # 確保調整力 = 稼働機の(定格−計画出力)合計

# カラーマップ用（10機分の色）
cmap_stacked = plt.cm.tab10
colors_stacked = [cmap_stacked(i % 10) for i in range(len(G))]

fig, axes = plt.subplots(5, 1, figsize=(max(12, T * 0.15), 14), sharex=True)

# (1) Demand, RES, net load, thermal
ax1 = axes[0]
ax1.fill_between(hours, 0, demand_arr, alpha=0.3, color="C0", label="Demand")
ax1.fill_between(hours, 0, solar_arr, alpha=0.5, color="gold", label="Solar")
ax1.fill_between(hours, solar_arr, np.array(solar_arr) + np.array(wind_arr), alpha=0.5, color="skyblue", label="Wind")
ax1.plot(hours, net_load_arr, "-", color="C3", linewidth=1.5, label="Net load (Demand - RES)")
ax1.plot(hours, total_thermal, "o-", color="C1", markersize=3, label="Thermal total")
ax1.set_ylabel("Power (MW)")
ax1.set_title("Supply-Demand Balance (Demand, Solar, Wind, Net Load, Thermal)")
ax1.legend(loc="upper left", fontsize=7)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.5, T + 0.5)
if DAYS > 1:
    for d in range(1, DAYS):
        ax1.axvline(d * HOURS_PER_DAY + 0.5, color="gray", linestyle=":", alpha=0.7)

# (2) 再エネをはじめに図示し、その上に火力を積み上げ。Net load=点線、Demand=実線
ax2 = axes[1]
ax2.fill_between(hours, 0, solar_arr, alpha=0.6, color="gold", label="Solar")
ax2.fill_between(hours, solar_arr, np.array(solar_arr) + np.array(wind_arr), alpha=0.6, color="skyblue", label="Wind")
renew_base = np.array(solar_arr) + np.array(wind_arr)
bottom = renew_base.copy()
for i, g in enumerate(G):
    top = bottom + p_by_gen[i]
    ax2.fill_between(hours, bottom, top, alpha=0.85, color=colors_stacked[i], label=g)
    bottom = top
ax2.plot(hours, demand_arr, "k-", linewidth=1.5, label="Demand")
ax2.plot(hours, net_load_arr, "k--", linewidth=1.2, alpha=0.9, label="Net load")
ax2.set_ylabel("Generation (MW)")
ax2.set_title("Solar + Wind + Thermal by Unit (Stacked)")
ax2.legend(loc="upper right", ncol=2, fontsize=7)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.5, T + 0.5)
if DAYS > 1:
    for d in range(1, DAYS):
        ax2.axvline(d * HOURS_PER_DAY + 0.5, color="gray", linestyle=":", alpha=0.7)

# (3) Unit commitment heatmap
ax3 = axes[2]
im = ax3.imshow(
    u_by_gen, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
    extent=[0.5, T + 0.5, len(G), 0],
)
ax3.set_yticks(np.arange(len(G) - 0.5, -0.5, -1))
ax3.set_yticklabels(G)
ax3.set_ylabel("Generator")
ax3.set_title("Unit Commitment (Green=ON, Red=OFF)")
ax3.set_xlim(0.5, T + 0.5)
if DAYS > 1:
    for d in range(1, DAYS):
        ax3.axvline(d * HOURS_PER_DAY - 0.5, color="gray", linewidth=0.8)

# (4) 予備力制約
ax4 = axes[3]
ax4.fill_between(hours, reserve_required, actual_reserve, where=(actual_reserve >= reserve_required), alpha=0.3, color="green", label="Reserve margin (actual − required)")
ax4.plot(hours, reserve_required, "--", color="gray", linewidth=1.2, label="Reserve required")
ax4.plot(hours, actual_reserve, "-", color="green", linewidth=1.2, label="Reserve actual (Σ rated, committed)")
ax4.set_ylabel("Reserve (MW)")
ax4.set_title("Reserve Constraint (actual ≥ required)")
ax4.legend(loc="upper right", fontsize=7)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(bottom=0)
ax4.set_xlim(0.5, T + 0.5)
if DAYS > 1:
    for d in range(1, DAYS):
        ax4.axvline(d * HOURS_PER_DAY + 0.5, color="gray", linestyle=":", alpha=0.7)

# (5) 調整力制約
ax5 = axes[4]
ax5.fill_between(hours, adj_required, actual_adjustment, where=(actual_adjustment >= adj_required), alpha=0.3, color="blue", label="Adjustment margin (actual − required)")
ax5.plot(hours, adj_required, "--", color="gray", linewidth=1.2, label="Adjustment required (load×10% + RES×25%)")
ax5.plot(hours, actual_adjustment, "-", color="blue", linewidth=1.2, label="Adjustment actual (Σ(rated−planned), committed)")
ax5.set_ylabel("Adjustment (MW)")
ax5.set_xlabel("Time (h)")
ax5.set_title("Adjustment Force Constraint (actual ≥ required)")
ax5.legend(loc="upper right", fontsize=7)
ax5.grid(True, alpha=0.3)
ax5.set_ylim(bottom=0)
ax5.set_xlim(0.5, T + 0.5)
if DAYS > 1:
    for d in range(1, DAYS):
        ax5.axvline(d * HOURS_PER_DAY + 0.5, color="gray", linestyle=":", alpha=0.7)

step = max(1, (T // 24)) * 6
xticks = list(range(1, T + 1, step))
for ax in axes:
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)

plt.tight_layout()
# 解析年月日 YYMMDDHHMMSS ＋ 解析条件でファイル名を生成
ts = datetime.now().strftime("%y%m%d%H%M%S")
cond = f"{DAYS}d_s{SEASON}_m{RENEW_MODE}"
if DAYS == 1 or RENEW_MODE == 1:
    cond += f"_solar{SOLAR_TYPE}_wind{WIND_TYPE}"
elif RENEW_MODE == 2:
    cond += "_rand"
else:
    cond += "_solar" + "".join(map(str, solar_types_per_day)) + "_wind" + "".join(map(str, wind_types_per_day))
os.makedirs("results", exist_ok=True)
out_path = os.path.join("results", f"{ts}_{cond}.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"計画結果の図を保存しました: {out_path}")
