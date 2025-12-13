# ACE Active Carver v2: Cycle 2 Specification (SPEC-CYCLE2.md)

## 0\. Cycle 2 プロジェクト定義

### 0.1. ミッション

**「Dockerはまだ使うな。ロジックを完成させよ。」**

Cycle 2の目的は、システムの中枢である「学習・推論・不確かさ推定」のロジックをローカル環境（Conda/Venv）で完全に動作させることである。インフラ（Docker/Kubernetes）の構築はCycle 3へ先送りし、まずはPythonコードとしての完成度と数学的な正当性を確立する。

### 0.2. スコープ (In-Scope)

1.  **Project Skeleton:** プロジェクト構成の刷新とインターフェース定義。
2.  **The Brain (MACE Wrapper):** MACE-MP-0 をロードし、Fine-tuning、推論、そして **D-Optimality による不確かさ推定** を行うモジュールの実装。
3.  **The Surgeon (Box Carver):** 指定された原子を中心に、PBCを考慮して構造を切り出すジオメトリエンジンの実装。
4.  **The Mock Oracle:** QE（Quantum ESPRESSO）を使わず、ダミーのエネルギー・力を返すMockクラスの実装。
5.  **Integration Test:** 上記を組み合わせ、疑似的な能動学習ループがエラーなく回ることを確認する。

### 0.3. スコープ外 (Out-of-Scope)

  * Docker / Docker Compose 環境構築。
  * 実際の Quantum ESPRESSO の実行。
  * GUI / Dashboard 実装。

-----

## 1\. アーキテクチャとディレクトリ構造

Cycle 1の成果物 `mlip_struc_generator` を取り込みつつ、以下の構成でモノリス・アプリケーションを構築する。

```text
ace_active_carver_v2/
├── pyproject.toml           # 依存関係定義 (mace-torch, ase, torch, numpy)
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py        # Pydantic Settings or Hydra
│   ├── interfaces.py        # Abstract Base Classes (Strategy Pattern)
│   ├── generators/          # Cycle 1 成果物 (Structure Generator)
│   ├── potentials/
│   │   ├── __init__.py
│   │   └── mace_impl.py     # ★Cycle 2 Main Task: MACE & Uncertainty
│   ├── carvers/
│   │   ├── __init__.py
│   │   └── box_carver.py    # ★Cycle 2 Main Task: Geometry Logic
│   └── oracles/
│       ├── __init__.py
│       └── mock_oracle.py   # ★Cycle 2 Main Task: Fake DFT
└── tests/
    ├── test_mace_wrapper.py
    ├── test_box_carver.py
    └── test_workflow_mock.py
```

-----

## 2\. インターフェース定義 (`src/interfaces.py`)

依存関係逆転の原則（DIP）に従い、具体的な実装（MACEやQE）に依存しない抽象クラスを定義せよ。

### 2.1. `AbstractPotential`

```python
class AbstractPotential(ABC):
    @abstractmethod
    def train(self, training_data: list[ase.Atoms]) -> None:
        """学習データを受け取り、内部モデルを更新する"""
        pass

    @abstractmethod
    def predict(self, atoms: ase.Atoms) -> tuple[float, np.ndarray, np.ndarray]:
        """エネルギー, 力, ストレスを返す"""
        pass

    @abstractmethod
    def get_uncertainty(self, atoms: ase.Atoms) -> np.ndarray:
        """
        原子ごとの不確かさを返す。
        Return: (N_atoms,) array of float (0.0 ~ 1.0 ideally normalized)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass
```

### 2.2. `AbstractOracle`

```python
class AbstractOracle(ABC):
    @abstractmethod
    def compute(self, atoms: ase.Atoms) -> ase.Atoms:
        """
        第一原理計算を実行し、Energy, Forces, Stress が付与されたAtomsを返す。
        計算失敗時は OracleComputationError を送出すること。
        """
        pass
```

-----

## 3\. コンポーネント詳細仕様

### 3.1. The Brain: `MacePotential` (`src/potentials/mace_impl.py`)

本プロジェクトの最難関パートである。MACEの公式実装 (`mace-torch`) をラップする。

#### A. 初期化とロード

  * `foundation_model_path` (例: "medium", "large" or local path) を指定してロードする。
  * `device` (cuda/cpu) を適切に処理する。Cycle 2では `cpu` での動作確認も必須とする（CI用）。

#### B. Fine-tuning (`train`)

  * Zero-shotではなく、ロードしたFoundation Modelの **Readout Layer（最終層）** のみを再学習するオプションを用意すること（高速化のため）。
  * 全層学習（Full Fine-tuning）もconfigで切り替え可能にせよ。
  * **Loss Function:** Energy, Forces, Stress の重み付き和。
  * **Optimizer:** AdamW。

#### C. 不確かさ推定 (Uncertainty Quantification: D-Optimality)

MACEはデフォルトでは不確かさを出力しない。以下のアルゴリズムを実装せよ。

1.  **特徴量抽出:**
      * MACEモデルの最終層（Readoutの手前）の原子特徴量ベクトル $\phi_i \in \mathbb{R}^D$ (通常 D=128 or 256) を抽出するフックを仕込む。
2.  **共分散行列の更新:**
      * 学習データセットに含まれる全原子の特徴量を用いて、共分散行列（デザイン行列の積） $C$ を計算・保持する。
      * $C = \sum_{n \in Dataset} \sum_{i \in Atoms} \phi_{n,i} \phi_{n,i}^T + \lambda I$
      * $\lambda$: 正則化項（小さな値）。
3.  **スコア計算 (`get_uncertainty`):**
      * 推論対象の原子 $j$ の特徴量 $\phi_j$ に対し、以下のマハラノビス距離を計算する。
      * $u_j = \phi_j^T C^{-1} \phi_j$
      * $C^{-1}$ は学習完了時に一度だけ計算し、推論時は行列ベクトル積のみを行う（高速化）。
      * **最大値正規化:** $u_j$ の絶対値は直感的でないため、学習データ内での最大値 $u_{max}$ で割るなどしてスケーリングしてもよい。

### 3.2. The Surgeon: `BoxCarver` (`src/carvers/box_carver.py`)

不安定な領域のみを切り出してOracleに投げるための幾何操作クラス。

  * **入力:** `atoms` (親構造), `center_index` (中心原子のID), `box_size` (例: 10.0 Å)。
  * **ロジック:**
    1.  `center_index` の原子を原点またはセル中心に持ってくる（PBC考慮）。
    2.  $x, y, z$ 各軸について $\pm box\_size/2$ の範囲にある原子を抽出する。
    3.  **PBC Wrap:** 切り出した原子団に対し、新しいセル（十分な真空層を持つ非周期セル、またはボックスサイズに合わせた周期セル）を定義する。
    4.  **Sanity Check:** 切り出した結果、原子数が1個だけ等の異常系を検知する。

### 3.3. The Mock Oracle: `MockOracle` (`src/oracles/mock_oracle.py`)

開発速度を落とさないためのダミー。

  * **機能:** 入力された構造に対し、適当な（しかし再現性のある）値を返す。
      * Energy: `num_atoms * (-4.0) + noise`
      * Forces: ランダムベクトル (magnitude \< 1.0 eV/A)
  * **遅延シミュレーション:** `time.sleep(1.0)` などを入れ、非同期処理のテストができるようにする。

-----

## 4\. 依存関係と環境設定 (`pyproject.toml`)

以下のライブラリバージョンを厳守する設定ファイルを作成せよ。

```toml
[project]
name = "ace_active_carver_v2"
version = "0.2.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "mace-torch>=0.3.4",  # 最新安定版を確認せよ
    "ase>=3.22.1",
    "numpy<2.0.0",        # mace/torchとの互換性のためv1系推奨
    "scipy",
    "pydantic",
    "hydra-core"
]

[tool.ruff]
# Cycle 1と同様の厳格な設定
```

-----

## 5\. 監査およびテスト要件 (Instruction to Auditor)

Auditorは以下の観点でPRを検査せよ。

### 5.1. 数学的妥当性チェック

  * **D-Optimalityの実装:** 共分散行列 $C$ の計算において、バイアス項や正規化項が適切か。逆行列計算 `torch.linalg.inv` or `pinv` が特異行列（Singular Matrix）で落ちないよう対策されているか（`try-except` または `jitter` の付加）。
  * **MACEの勾配:** 推論時、Forcesを取得するために `loss.backward()` 相当の処理あるいは `torch.autograd.grad` が正しく使われているか。`create_graph=True` の要否を確認せよ。

### 5.2. ソフトウェア品質チェック

  * **Type Hinting:** 行列演算の入出力（Shapes）がコメントやDocstringで明記されているか。例: `Features: [batch, n_atoms, n_features]`
  * **Mockの分離:** テストコードが `MockOracle` を使っており、誤って外部通信や重い計算を走らせていないか。

### 5.3. 必須テストケース

Coderに以下の `pytest` を実装させよ。

1.  **`test_mace_uncertainty`**:
      * 同じ構造を2回入力したら、全く同じ不確かさスコアが出るか（決定性）。
      * 学習データに含まれる構造（既知）はスコアが低く、極端に歪ませた構造（未知）はスコアが高くなる傾向が出るか。
2.  **`test_box_carver_pbc`**:
      * セルの端にある原子を中心にして切り出した際、反対側の原子が正しく含まれるか（PBC Wrapping）。
3.  **`test_mock_loop`**:
      * Generate -\> Train (Mock) -\> Predict -\> Uncertainty Check の1サイクルがエラーなく完走するか。

-----

## 6\. 開発フローの指示

1.  **Step 1:** `src` ディレクトリ構造と `interfaces.py` を作成。
2.  **Step 2:** `MockOracle` と `BoxCarver` を実装し、単体テストをパスさせる。
3.  **Step 3 (Heavy):** `MacePotential` を実装。まずは学習・推論のみ。
4.  **Step 4 (Math):** `MacePotential` に不確かさ推定機能を追加。
5.  **Step 5:** 結合テスト。

以上。
Cycle 2終了条件は、\*\*「ローカル環境で `pytest` がAll Greenになり、かつ不確かさ推定の挙動が物理的に納得できるものであること」\*\*とする。