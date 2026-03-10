# Unit Commitment (UC) / Security-Constrained UC (SCUC)

発電機起動停止計画（ユニット・コミットメント）の例題です。単一エリア版（UC）と **5 エリア連系版（SCUC）** を実装しています。

## リポジトリ構成

| ファイル | 説明 |
|----------|------|
| `main.py` | 単一エリア UC：需要・太陽光・風力・10 機の火力。需給バランス・予備力・MUT/MDT・ランプ制約。結果は PNG で `results/` に出力。 |
| `main_scuc.py` | **5 エリア SCUC（蓄電池付き）**：隣接エリア連系線（4 本）付き。需給・太陽光・風力はエリア別。求解後に **HTML レポート**（手法・条件・結果・考察のタブ）を `results/` に出力。 |
| `LOG.md` | 会話ログ・変更履歴（SCUC の機能追加や HTML レポートの仕様メモ）。 |
| `requirements.txt` | Python 依存パッケージ。 |

## 必要な環境

- Python 3.8+
- PuLP（CBC ソルバー同梱）、NumPy、Matplotlib

```bash
pip install -r requirements.txt
```

または:

```bash
pip install pulp numpy matplotlib
```

## 実行方法

### 単一エリア UC

```bash
python main.py
```

対話で日数・季節・太陽光・風力曲線などを指定します。計画結果の PNG が `results/` に保存されます。

### 5 エリア SCUC（推奨）

```bash
python main_scuc.py
```

対話で日数・季節・負荷配分・太陽光・風力の設備割合や曲線タイプなどを指定します。

- **出力**: `results/` に HTML レポート（タイムスタンプ＋条件名のファイル名）が生成されます。
- **レポート内容**:
  - **Simulation method**: 目的関数・需給バランス・連系線制約・調整力などの数式（MathJax）
  - **Conditions**: 条件図（Demand by area, Solar available by area, Wind available by area）
  - **Results**: 求解結果（Status, Total cost）と 5 パネル図（Supply-Demand, Stacked, Unit Commitment, Adjustment, Tie-Line）
  - **Consideration**: 結果サマリ表（コスト・最大連系線潮流の利用率・太陽光抑制率など）とまとめ

ブラウザで該当 HTML を開くとタブ切り替えで閲覧できます。

## モデル概要（SCUC）

- **5 エリア**: Area1〜Area5。各エリアで需要・太陽光・風力が定義され、需給バランスを満たしつつ連系線で融通。
- **連系線**: 直列トポロジー（Area1—Area2—Area3—Area4—Area5）。各線の潮流は ±F_max 以内。
- **蓄電池**: 各エリア 1 台。SOC（充電状態）ダイナミクス、SOC 上下限、充放電の同時実行禁止（排他）を含む。
- **目的関数**: 燃料費＋無負荷コスト＋起動コストの最小化。
- **制約**: エリア別需給、連系線容量、調整力（headroom）、最低稼働機数、Pmin/Pmax、ランプ、MUT/MDT、スタートアップ論理。

## 対話パラメータ（共通）

実行時に日数・季節・再エネ曲線タイプ・負荷配分（SCUC）などを入力します。Enter でデフォルトを使用できます。詳細は実行時のプロンプトを参照してください。

## 参考文献

「発電機起動停止計画（ユニット・コミットメント）の定義、目的、および数理モデル化に関する包括的調査報告書」に基づく定式化を採用しています。
