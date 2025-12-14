Cycle 2の実装を担当するAI coder (Jules) 向けの、超詳細かつ厳格な指示プロンプトを作成しました。

このプロンプトは、単なる機能要件の列挙ではなく、\*\*「MACEの内部レイヤーへのフック方法」**や**「特異行列回避のための数学的テクニック」\*\*など、実装難易度が高い部分の具体的な解法を含んでいます。これにより、AIが迷走するリスクを極限まで減らしています。

以下のテキストをコピーして、Julesに渡してください。

-----

# ACE Active Carver v2: Cycle 2 Implementation Order (Prompt for Coder)

**Role:**
あなたは世界最高峰のScientific Software Engineerであり、PyTorchを用いたDeep Learningの実装と、第一原理計算（DFT）のドメイン知識を兼ね備えています。

**Mission:**
**Cycle 2の仕様書 (SPEC-CYCLE2.md) に基づき、システムの「脳 (Brain)」と「外科医 (Surgeon)」にあたるコアロジックを実装してください。**

**Constraint (CRITICAL):**

  * **No Docker:** 今回はDocker環境構築を行いません。ローカルのPython環境（Conda/Venv）で動作するコードを記述してください。
  * **Strict Typing:** 全ての関数に型ヒントを付与し、`mypy --strict` 相当の準拠を目指してください。
  * **No "To Do":** "Here is the logic..." と言ってプレースホルダーを残してはいけません。完全に動作する実装を書いてください。

-----

## Part 1: Project Skeleton & Dependencies

まず、プロジェクトの土台を固めます。以下の構成でファイルを作成してください。

### 1.1. Directory Structure

```text
ace_active_carver_v2/
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── interfaces.py          # Strategy Pattern Definitions
│   ├── generators/            # (Copy from Cycle 1 output)
│   ├── carvers/
│   │   ├── __init__.py
│   │   └── box_carver.py      # Task 2
│   ├── potentials/
│   │   ├── __init__.py
│   │   └── mace_impl.py       # Task 3 (Heavy)
│   └── oracles/
│       ├── __init__.py
│       └── mock_oracle.py     # Task 4
└── tests/
    ├── conftest.py
    ├── test_box_carver.py
    └── test_mace_impl.py
```

### 1.2. `pyproject.toml`

以下の依存関係を定義してください。バージョン制約は厳守です。

```toml
[project]
name = "ace_active_carver_v2"
version = "0.2.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "mace-torch>=0.3.4",  # Ensure compatibility with latest MACE
    "ase>=3.22.1",
    "numpy<2.0.0",        # NumPy 2.0 breaks many scientific libs
    "scipy",
    "pydantic",
    "hydra-core"
]

[tool.ruff]
line-length = 88
select = ["F", "E", "W", "I", "N", "UP", "B"]
```

-----

## Part 2: Interfaces (The Contract)

`src/interfaces.py` に抽象基底クラス (ABC) を実装してください。
これ以降の実装は、必ずこれらのクラスを継承します。

```python
from abc import ABC, abstractmethod
import numpy as np
from ase import Atoms

class AbstractPotential(ABC):
    @abstractmethod
    def train(self, training_data: list[Atoms]) -> None:
        """Fine-tune the model with new data."""
        pass

    @abstractmethod
    def predict(self, atoms: Atoms) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Return (energy [eV], forces [eV/A], stress [eV/A^3]).
        Stress should be (3, 3) matrix or Voigt notation (6,).
        """
        pass

    @abstractmethod
    def get_uncertainty(self, atoms: Atoms) -> np.ndarray:
        """
        Return per-atom uncertainty scores.
        Shape: (n_atoms,)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass

class AbstractOracle(ABC):
    @abstractmethod
    def compute(self, atoms: Atoms) -> Atoms:
        """
        Perform DFT calculation.
        Returns atoms with calculated energy/forces attached.
        Raises OracleComputationError if calculation fails.
        """
        pass
```

-----

## Part 3: The Surgeon (`BoxCarver`) Implementation

`src/carvers/box_carver.py` を実装してください。
MDシミュレーション中の巨大な構造から、DFT計算可能なサイズの領域を切り出すクラスです。

**Implementation Logic:**

1.  **Arguments:** `atoms` (parent), `center_index` (int), `box_vector` (float or list of 3 floats).
2.  **Centering:**
      * まず、`center_index` の原子がセルの中心 (0.5, 0.5, 0.5 in scaled coordinates) に来るように、全体の原子座標をシフト（Wrap）します。
      * これにより、PBC境界跨ぎの問題を単純化します。
3.  **Cutout:**
      * シフト後の座標において、中心から $\pm box\_vector / 2$ の範囲（直方体）に含まれる原子のインデックスを抽出します。
4.  **Re-wrapping:**
      * 抽出された原子群に対し、新しい `Cell` を定義します。
      * **Cluster Mode:** 真空層（vacuum=10.0Å）を持つ非周期セルを作成。
      * **Periodic Mode:** 切り出したボックスサイズをそのままセルサイズとする。
      * ※ 今回はデフォルトで `Cluster Mode` (pbc=False) を採用してください。

**Validation:**

  * 原子数が極端に少ない（例えば1個だけ）場合は `ValueError` を送出してください。

-----

## Part 4: The Mock Oracle (`MockOracle`) Implementation

`src/oracles/mock_oracle.py` を実装してください。
これはテストおよびドライラン用のダミーです。

**Logic:**

  * `AbstractOracle` を継承。
  * `compute(atoms)` メソッド内で:
      * `time.sleep(0.1)` (計算時間をシミュレート)
      * **Energy:** $E = -4.0 \times N_{atoms} + \mathcal{N}(0, 0.1)$
      * **Forces:** `np.random.uniform(-0.1, 0.1, size=(N, 3))`
      * **Stress:** `np.zeros((3, 3))` (今回は簡易化)
      * `atoms.calc` に `SinglePointCalculator` をセットして返すこと。

-----

## Part 5: The Brain (`MacePotential`) Implementation ★CRITICAL

ここがCycle 2の最重要かつ最難関パートです。`src/potentials/mace_impl.py` を実装してください。
MACEモデルをロードし、学習・推論・不確かさ推定を行います。

### 5.1. Initialization & Loading

  * コンストラクタで `model_path` (str) と `device` (str) を受け取る。
  * `torch.load(model_path)` でモデルをロードする。MACEのモデルは通常 `TorchScript` ではなくPyTorchの生モデル (`mace.modules.models.MACE`) としてロードすることを推奨する（フックを仕掛けるため）。
  * もし `foundation_model="medium"` 等が指定された場合、`mace.calculators.mace_mp` からロードするロジックを入れても良いが、今回はローカルの `.pt` ファイル指定を基本とする。

### 5.2. Prediction (`predict`)

  * 入力 `ase.Atoms` を `mace.data.AtomicData` に変換する（`mace.data.config_from_atoms` 等を使用）。
  * モデルに入力し、Energy, Forces, Stress を取得する。
  * 単位変換 (eV, eV/A) が正しいか確認すること。

### 5.3. Uncertainty Quantification (D-Optimality) - Deep Dive

これが今回の目玉機能です。**Last Layer Uncertainty (LLU)** を実装します。

**Theory:**
モデルの不確かさ $u(x)$ は、特徴空間における学習データとのマハラノビス距離で近似できます。
$$u(x) = \phi(x)^T C^{-1} \phi(x)$$
ここで、$\phi(x)$ はモデル最終層の原子特徴量、$C$ は学習データから計算された共分散行列（デザイン行列）です。

**Implementation Steps:**

1.  **Hook Registration:**

      * モデルの `readout` モジュール（通常は `model.readouts[-1]` や `model.atomic_energies_fn` の直前）に `register_forward_hook` を仕掛けます。
      * 推論時、このフックを使って、最終層の原子特徴量ベクトル（`node_feats`）をキャプチャします。次元数は通常 128 または 256 です。

2.  **Covariance Matrix Update (`train` method):**

      * `train()` メソッドが呼ばれた際、学習データ全原子の $\phi$ を収集します。
      * 行列 $C = \sum_{i} \phi_i \phi_i^T$ を計算します。
      * **Numerical Stability:** 逆行列計算時のエラーを防ぐため、正則化項を加えます。
        $$C_{reg} = C + \lambda I$$
        ($\lambda = 1e^{-4}$ 程度)
      * `torch.linalg.inv(C_reg)` または `torch.linalg.pinv` で逆行列 $C^{-1}$ を計算し、`self.inv_covariance` として保持します。

3.  **Score Calculation (`get_uncertainty` method):**

      * 入力構造に対して `predict` を実行（またはフック付きでForward）し、特徴量 $\phi_{new}$ を取得。
      * 各原子 $j$ について $u_j = \phi_j^T C^{-1} \phi_j$ を計算。
      * これは `(phi @ inv_cov * phi).sum(dim=1)` のような行列演算で一括計算すること。forループは禁止。

### 5.4. Fine-Tuning (`train`)

  * MACEの重み全体を更新すると時間がかかり、かつ壊滅的忘却（Catastrophic Forgetting）のリスクがあります。
  * **Head Only Training:** 最終層（Readout）のパラメータのみ `requires_grad=True` にし、他は `False` にフリーズして学習するロジックを実装してください。
  * Loss = $w_E ||E - E_{ref}||^2 + w_F ||F - F_{ref}||^2$

-----

## Part 6: Testing Strategy

`tests/` 配下に以下のテストを作成してください。

### 6.1. `test_mace_uncertainty.py`

  * **Consistency Test:** 同じ `Atoms` を2回 `get_uncertainty` に投げ、全く同じ値が返ることを確認（決定性の担保）。
  * **Sensitivity Test:**
    1.  MACEモデルを初期化し、ダミーデータ（例: 歪みのない結晶）で `train` して $C^{-1}$ を構築する。
    2.  「学習データと同じ構造」と「原子をランダムに0.5Å動かした構造」の不確かさを比較する。
    3.  **Assert:** 乱した構造の方が不確かさスコアが有意に高いこと。

### 6.2. `test_box_carver.py`

  * **PBC Wrapping Test:**
      * セル端付近（例: `[0.1, 0.1, 0.1]`）にある原子を中心として切り出した際、反対側（例: `[9.9, 9.9, 9.9]`）にある原子が正しく含まれているか確認。

-----

## Output Format

あなたの回答は、各ファイルのソースコードを提示する形式で行ってください。
解説は最小限にし、**コピー＆ペーストで即座に動作するコード**を優先してください。

**Let's build the brain. Begin.**