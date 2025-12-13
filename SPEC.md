
-----

# ACE Active Carver v2: Technical Specification (SPEC.md)

## 1\. プロジェクト概要 (Project Overview)

### 1.1. ミッション

\*\*「システムに疎い実験家でも、対象の原子系を入力するだけで、On-the-flyで理論計算（MLIP構築）ができる自律型システム」\*\*を構築する。

### 1.2. コア・フィロソフィー

  * **Simple is Robust:** 複雑怪奇なマイクロサービス構成（v1）を廃止し、保守性と堅牢性を最優先したモノリス構成へ回帰する。
  * **Foundation First:** MACEの事前学習モデルを最大限活用し、ゼロからの学習ではなく「Fine-tuning」による高速な立ち上がりを実現する。
  * **Autonomous:** 初期構造生成から探索、学習までを人間の介在なし（Zero-shot start）で完遂可能にする。

-----

## 2\. システムアーキテクチャ (System Architecture)

通信オーバーヘッドと依存関係地獄を排除するため、役割を明確に分離した「2コンテナ構成」を採用する。

### 2.1. Container A: The Brain (Main Application)

システムの中枢であり、第一原理計算以外の全てのロジックを担う。

  * **Role:** オーケストレーション、構造生成、MLIP推論・学習、MD実行。
  * **Base Image:** `pytorch/pytorch:2.x-cuda` base + Python 3.10+
  * **Key Libraries:**
      * `ASE` (Atomic Simulation Environment): 原子操作の共通言語。
      * `MACE` (mace-torch): メインのMLIPエンジン。
      * `LAMMPS` (via Python interface): 高速MD実行用（ASE Calculatorとしてラップ）。
      * `Hydra` / `Pydantic`: 統一設定管理。
  * **Hardware:** NVIDIA GPU (Training/Inferenceに必須)。

### 2.2. Container B: The Oracle (DFT Worker)

依存関係が特殊な第一原理計算エンジンのみを隔離する。

  * **Role:** 教師データ生成（Energy, Forces, Stress）。
  * **Engine:** Quantum ESPRESSO (QE)。
  * **Interface:** Container A から `ase.calculators.espresso` を経由し、SSH または Socket 通信でジョブを受け付ける。計算完了後、結果をファイルシステム（Volume共有）経由で返す。

### 2.3. ディレクトリ構造 (Directory Structure)

```text
/
├── config.yaml             # 統合設定ファイル (Hydra)
├── data/                   # 学習データ、ポテンシャルモデル、MD軌跡
├── docker-compose.yml      # 2つのサービス (app, qe) の定義
└── src/
    ├── core/               # オーケストレーション、State管理
    ├── interfaces.py       # 抽象クラス定義 (Strategy Pattern)
    ├── potentials/         # MACE, (Future: Linear MLIP) ラッパー
    ├── generators/         # 初期構造生成 (mlip_struc_generator 移植)
    ├── carvers/            # BoxCarver ロジック
    └── workflows/          # Active Learning Loop の実装
```

-----

## 3\. アルゴリズムとコアロジック (Core Logic)

### 3.1. ポテンシャル戦略 (Potential Strategy)

将来的な拡張性（Linear MLIP対応）を見越し、Strategy Patternで実装するが、**Cycle 1ではMACEに特化する。**

  * **Engine:** **MACE (Multi-ACE)**
  * **Learning Strategy:** **Standard Fine-tuning**
      * Foundation Model (e.g., `MACE-MP-0`) をロードし、全エネルギー ($E_{total}$) に対して追加学習を行う。
      * **Delta Learning禁止:** MACE使用時は物理的整合性を破壊するため、LJなどのベースライン減算は行わない。（ただし孤立原子エネルギー $E_0$ の減算は必須）。
  * **Interface:** `AbstractPotential`
      * `train(dataset)`
      * `predict(atoms)`
      * `get_uncertainty(atoms)`

### 3.2. 不確かさ推定 (Uncertainty Quantification)

On-the-fly学習のボトルネックとなる計算コストを抑えるため、Committee（アンサンブル）方式は採用しない。

  * **Method:** **D-Optimality (Last Layer Uncertainty)**
      * MACEの最終層直前の「不変特徴量 (Invariant Features)」を抽出する。
      * 学習データの特微量から情報行列（または共分散行列の逆行列）$C^{-1}$ を更新・保持する。
      * 推論時、新データの特徴量ベクトル $v$ に対し、マハラノビス距離 $u = v^T C^{-1} v$ を計算し、これを不確かさスコアとする。
  * **Performance:** モデル1つで計算可能であり、推論コスト増は軽微（\~1.1倍）。未知の化学環境への感度が高い。

### 3.3. 初期構造生成 (Seed Generation)

「データがないとポテンシャルが作れない」問題を解決するため、`mlip_struc_generator` のロジックを統合する。

  * **Module:** `src/generators`
  * **Logic:**
      * ユーザー入力（組成）に基づき、以下の構造を自動生成する。
          * **Bulk:** 乱れた超胞、各種結晶構造。
          * **Surface/Interface:** 真空層を持つスラブ構造。
          * **Cluster:** 孤立クラスター。
      * これらを **MACE-MP-0 (Pre-trained)** でスクリーニングし、明らかに非物理的な構造を除外した後、DFT計算に投入する。

### 3.4. 領域切り出し (Box Carving)

MD中に検出された「高不確かさ領域」を、第一原理計算可能なサイズに切り出す。

  * **Class:** `BoxCarver`
  * **Logic:**
    1.  **Detection:** 不確かさが閾値を超えた原子 $i$ を特定。
    2.  **Definition:** 原子 $i$ を中心とする $L_x \times L_y \times L_z$ の直方体領域を定義。
    3.  **Extraction:** 周期境界条件 (PBC) を考慮し、ボックス内の原子を抽出。
    4.  **PBC Reset:** 切り出したクラスターに対し、十分な真空層を持つ新しいセル、またはボックスサイズに合わせたPBCを設定する。
    5.  **Constraints (Phase 2):** 将来的には、共有結合の切断回避や、イオン電荷の中性維持を行うロジックを注入可能にする。

-----

## 4\. ワークフロー (Workflow)

システムは以下のステートマシンとして動作する。

### Phase 0: Initialization (Cold Start)

1.  **Config Load:** ユーザー設定（元素、温度、使用リソース）を読み込む。
2.  **Seed Gen:** `StructureGenerator` が初期構造プールを生成。
3.  **Labelling:** `Oracle (QE)` でDFT計算を実行。
4.  **First Training:** 得られたデータで MACE Foundation Model を Fine-tune。

### Phase 1: Exploration (The Loop)

1.  **MD Run:** 学習済みモデルでMDシミュレーション（LAMMPS/ASE）を実行。
2.  **Monitoring:** ステップ毎に `Uncertainty` を監視。
3.  **Trigger:** 閾値超過、またはモデルの破綻（原子間距離の異常接近など）を検知したらMDを中断。

### Phase 2: Active Learning

1.  **Carving:** 破綻箇所周辺を `BoxCarver` で切り出し。
2.  **Labelling:** 切り出した構造を `Oracle` に送信し、正解エネルギー・力を取得。
      * *Check:* DFTが収束しない場合は、その構造を破棄し、MDを少し前のステップから別乱数で再開。
3.  **Retraining:** 新データをデータセットに追加し、MACEを再学習（Incremental Learning）。
4.  **Resume:** 新モデルで Phase 1 へ戻る。

-----

## 5\. データ管理と設定 (Data & Config)

### 5.1. Configuration

`config.yaml` 一つで全パラメータを制御する。

```yaml
project:
  name: "FePt_Alloy_Run1"
  gpu_id: 0

generation:
  elements: ["Fe", "Pt"]
  supercell_size: [3, 3, 3]

potential:
  arch: "mace"
  model_path: "medium"  # Foundation model size
  uncertainty: "d_opt"

active_learning:
  threshold: 0.15      # Uncertainty threshold
  carving_box: [10.0, 10.0, 10.0]
  md_steps: 10000
```

### 5.2. State Management

複雑なDBサーバーを立てず、ファイルシステムベースで管理する。

  * `experiments/{name}/gen_0/`: 初期データ
  * `experiments/{name}/cycle_{N}/`: N回目のループ結果
      * `model_cycle_{N}.pt`: その時点のモデル
      * `candidates.xyz`: 切り出された候補構造
      * `training_set.xyz`: 累積学習データ

-----

## 6\. 開発ロードマップと役割分担

### Phase 1: Foundation (Current Target)

  * モノリスDocker環境の構築。
  * MACE (Fine-tune) + D-Optimality の実装。
  * `mlip_struc_generator` の統合。
  * 基本的な `BoxCarver` の実装。

### Phase 2: Chemistry Aware

  * Carving時の化学的ルール（Stoichiometry）適用。
  * Linear MLIP (Delta Learning) オプションの追加。

### Phase 3: UI & Scale

  * 進捗可視化ダッシュボード。
  * 複数GPU対応。

-----

### AI Agents への申し送り事項

  * **To AI-CODER (Jules):**

      * 君の敵は「複雑さ」だ。コードは短く、モジュールは疎結合に。
      * `interfaces.py` で抽象化を徹底せよ。MACE以外のポテンシャルが来てもコードが壊れないように。
      * エラーハンドリングは「ログを吐いて落ちる」のではなく、「リトライするか、そのデータを捨てて進む」タフな設計にせよ。

  * **To AI-AUDITOR:**

      * コードの「可読性」と「型安全性」を徹底的に監視せよ。
      * 「動くけど遅い」コード（例：Pythonでのforループ多用）は許すな。NumPy/Torchのベクトル演算を強制せよ。
      * Dockerビルドが通らないコミットは万死に値する。
