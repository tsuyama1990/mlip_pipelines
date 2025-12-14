# ACE Active Carver v2: Cycle 3 Master Specification

**Version:** 2.0 (Final)
**Strategy:** "Local Integration" + "Interactive LAMMPS Controller"
**Status:** 🚀 **Ready for Implementation**

## 0\. プロジェクト定義

### 0.1. ミッション

**「現実の物理（Real Physics）と脳（MACE）を、LAMMPSという身体で接続する」**

Cycle 3の目的は、ローカル環境において「構造探索」と「学習」の完全な自律ループを構築することである。
探索エンジンには標準的なASE MDではなく、将来の拡張性（複雑な変形・破壊、Pacemakerへの移行）を見据えて **LAMMPS Python Interface** を採用する。

### 0.2. アーキテクチャ概要

  * **No Docker:** ユーザーのローカル環境にある `pw.x` (Quantum ESPRESSO) と `lammps` (Python module) を直接利用する。
  * **Interactive Controller:** PythonがLAMMPSを起動し、`fix external` コマンドを通じて MACEの力場と不確かさ判定を注入する。
  * **MVP Scope:** 圧力制御（NPT）は実装せず、体積一定（NVT）または強制変形（Deform）のみをサポートする。

-----

## 1\. ディレクトリ構造

フラットな構成（Flat Layout）を採用し、`src` ディレクトリをパッケージルートとする。

```text
ace_active_carver_v2/
├── config.yaml              # 統合設定ファイル (LAMMPSスクリプト含む)
├── pyproject.toml           # 依存関係定義
├── scripts/
│   ├── download_sssp.py     # SSSP自動ダウンローダー
│   └── check_env.py         # 環境診断ツール
├── src/
│   ├── engines/             # ★ New
│   │   └── lammps_mace.py   # LAMMPS Interactive Driver
│   ├── oracles/
│   │   └── qe_oracle.py     # Real QE Wrapper
│   ├── core/
│   │   ├── orchestrator.py  # The Loop Manager
│   │   └── ...
│   └── ...
└── tests/
    ├── unit/                # Logic Tests (Mock everything)
    └── integration/         # Smoke Tests (Requires binaries)
```

-----

## 2\. コンポーネント詳細仕様

### 2.1. The Driver: `src/engines/lammps_mace.py`

MACEをLAMMPSの外部ポテンシャルとして機能させるためのブリッジ。

  * **Class:** `LammpsMaceDriver`
  * **依存:** `lammps` (Python module), `ase`, `numpy`
  * **Callback Mechanism (`_callback`):**
      * LAMMPSの `fix external pf/callback` から毎ステップ呼び出される。
      * **Force Injection:** MACE (GPU推奨) で計算した力をLAMMPS配列 `f` に書き込む。
      * **Uncertainty Interrupt:** `potential.get_uncertainty()` が閾値を超えた場合、Python例外 `UncertaintyInterrupt` を送出して **シミュレーションを即時停止** させる。
      * **Constraint:** MVPでは **Stress (Virial) の受け渡しは行わない**。
  * **Execution:**
      * ユーザー定義のLAMMPSコマンド（`fix nvt`, `fix deform` 等）を受け取り、注入する。

### 2.2. The Real Oracle: `src/oracles/qe_oracle.py`

Cycle 2のMockを置き換える、本物のDFT計算クラス。

  * **Inheritance:** `AbstractOracle`
  * **Execution:**
      * `ase.calculators.espresso.Espresso` を使用。
      * **必須:** 必ず `tempfile.TemporaryDirectory` 内で実行し、終了時にゴミファイルを削除すること。
  * **Physics Automation:**
      * **Cutoffs:** SSSP (Precision v1.3.0) JSONから、系に含まれる元素の最大推奨値を自動設定。
      * **K-Points:** セルサイズに基づき、指定された密度（Density）からメッシュを自動計算。クラスター（非周期）の場合はGamma点 `(1,1,1)` を強制。
  * **Robustness:**
      * SCF収束失敗時は `OracleComputationError` を送出する（システム全体をクラッシュさせない）。

### 2.3. The Orchestrator: `src/core/orchestrator.py`

自律ループの指揮者。

  * **Phase 0: Bootstrapping**
      * 初期構造生成 → QE計算 → MACE初期学習。
  * **Phase 1: Exploration (Interactive MD)**
      * `LammpsMaceDriver` を起動。
      * `UncertaintyInterrupt` が発生するまでMDを継続。
  * **Phase 2: Active Learning**
      * 中断された構造から `BoxCarver` で不安定領域を切り出し。
      * `QeOracle` で計算（失敗したら当該データを破棄してMD再開）。
      * `MacePotential` をFine-tuning。
      * 閾値 ($u_{max}$) を更新して Phase 1 へ戻る。

-----

## 3\. 設定ファイル (`config.yaml`)

```yaml
system:
  device: "cuda"
  dft_command: "mpirun -np 4 pw.x" # ユーザー環境に依存
  sssp_dir: "data/sssp"

exploration:
  # LAMMPS制御用スクリプト (pair_styleは自動設定されるので不要)
  lammps_script: |
    fix 1 all nvt temp 300 300 0.1
    # fix 2 all deform 1 x erate 0.001 remap x  <-- 破壊シミュレーション用

active_learning:
  threshold_ratio: 1.5  # u_max の何倍で止めるか
  carving_skin: 2.0
```

-----

## 4\. テスト計画

### 4.1. Unit Tests (Logic Verification)

物理計算を行わず、ロジックのみを検証する。

  * **`test_lammps_mace.py`:**
      * `lammps` モジュールを `MagicMock` で偽装。
      * Callbackを手動で呼び出し、「不確かさが高い時に例外が飛ぶか」「力が正しく渡されるか」を検証。
  * **`test_qe_oracle.py`:**
      * `subprocess` をMock化。
      * 生成されたQE入力ファイル（`espresso.pwi`）の中身（Cutoff, K-points）が正しいか検証。

### 4.2. Integration Test (Smoke Test)

  * **`test_real_loop.py`:**
      * 実際の `pw.x` と `lammps` を使用。
      * 対象: Si ダイマー等の極小系。
      * 目的: 「エラーで落ちずに2サイクル回るか」の確認のみ。

-----

## 5\. ロードマップ (Cycle 3実装順序)

1.  **Infrastructure:** SSSPダウンローダーと `check_env.py` の作成。
2.  **Engine:** `LammpsMaceDriver` の実装（Mockテスト含む）。
3.  **Oracle:** `QeOracle` の実装（Mockテスト含む）。
4.  **Core:** `Orchestrator` で全てを結合。
5.  **Integration:** 実機でのループテスト。

-----

**Approved by:** Antigravity (Lead Auditor)
**Date:** 2025-12-14