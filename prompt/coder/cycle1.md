# PROMPT FOR AI-CODER: Cycle 1 Implementation

あなたは計算科学と機械学習（MLIP: Machine Learning Interatomic Potentials）に精通したシニアPythonバックエンドエンジニアです。
以下の詳細設計書に基づき、**「MACEを用いた結晶構造最適化パイプライン」のMVP（Minimum Viable Product）** を実装してください。

## 1\. プロジェクト定義と制約事項

### 1.1 プロジェクト概要

  - **名称**: `mlip-struc-gen-local`
  - **目的**: 既存の構造生成ロジック（外部レポジトリ）と、MACEポテンシャルによる構造最適化を組み合わせた、ローカル（WSL/Linux）で動作するPythonパイプラインを構築する。
  - **将来展望**: 最終的にはFastAPI/Streamlitを用いたWebアプリ化を目指すため、**「ロジック（Core）」と「インターフェース（CLI/Web）」の完全な分離**を徹底する。

### 1.2 技術スタックと制約

  - **OS**: WSL (Ubuntu 22.04+)
  - **言語**: Python 3.10+
  - **コンテナ技術**: **使用禁止 (No Docker)**。`venv` による仮想環境のみ使用。
  - **コアライブラリ**:
      - `ase` (Atomic Simulation Environment): データ構造の標準として使用。
      - `mace-torch`: 構造最適化用のポテンシャルエンジン。
      - `pydantic` & `pydantic-settings`: 堅牢な設定管理。
      - `loguru`: ログ管理。
  - **外部資産**: `MLIP_STRUC_GEN` レポジトリを `git submodule` として取り込み、ラッパー経由で利用する。

-----

## 2\. ディレクトリ構成と環境構築

以下のディレクトリ構成を厳密に守って構築してください。

```text
mlip-struc-gen-local/
├── .gitignore               # .venv, .env, data/, __pycache__ 等を除外
├── .gitmodules              # git submodule 設定（自動生成）
├── pyproject.toml           # または requirements.txt
├── README.md                # セットアップ手順（特にTorch/MACEの順序）
├── main_cli.py              # デバッグ・動作確認用CLIエントリーポイント
└── src/
    ├── __init__.py
    ├── config/              # 設定定義
    │   ├── __init__.py
    │   └── settings.py      # Pydanticモデル
    ├── external/            # 外部レポジトリ格納場所
    │   ├── __init__.py
    │   └── mlip_struc_gen/  # (git submodule: 構造生成ロジック)
    ├── core/                # アプリケーションコアロジック
    │   ├── __init__.py
    │   ├── calculators/     # ASE Calculatorファクトリー
    │   │   ├── __init__.py
    │   │   └── mace_factory.py
    │   ├── engines/         # 計算実行エンジン
    │   │   ├── __init__.py
    │   │   └── relaxer.py   # 構造最適化ロジック
    │   ├── generators/      # 構造生成アダプター
    │   │   ├── __init__.py
    │   │   └── adapter.py   # 外部レポジトリのラッパー
    │   └── utils/
    │       ├── __init__.py
    │       ├── io.py        # ファイル入出力
    │       └── logging.py   # Loguru設定
```

### 依存関係の定義 (`requirements.txt`)

MACEとPyTorchの依存関係は非常にセンシティブです。以下の順序とバージョン指定を推奨するコメントを含めてください。

1.  **PyTorch**: ホストのCUDAバージョン（例: 11.8 or 12.1）に合わせた `torch`, `torchvision`。
2.  **MACE**: `mace-torch` (最新安定版)。
3.  **ASE**: `ase>=3.23.0`
4.  **Utilities**: `pydantic`, `pydantic-settings`, `python-dotenv`, `loguru`, `numpy`, `scipy`

-----

## 3\. 実装詳細仕様

### 3.1 設定管理 (`src/config/settings.py`)

`pydantic-settings` を使用し、環境変数（`.env`）とデフォルト値を管理します。

  * **クラス `MACESettings`**:
      * `model_path`: str (デフォルト: "medium", またはローカルパス)
      * `device`: str (デフォルト: "cuda", バリデーションで "cuda" or "cpu" をチェック)
      * `default_dtype`: str ("float64" 推奨)
  * **クラス `RelaxationSettings`**:
      * `fmax`: float (力の収束判定, default: 0.01 eV/A)
      * `steps`: int (最大ステップ数, default: 200)
      * `optimizer`: str (default: "LBFGS")
  * **クラス `GeneratorSettings`**:
      * `target_element`: str (例: "Si")
      * `supercell_size`: int (例: 2)
      * その他、外部レポジトリが必要とするパラメータ。
  * **クラス `Settings`**:
      * 上記すべてを包含し、`output_dir` (Path) などを管理。

### 3.2 外部レポジトリのアダプター (`src/core/generators/adapter.py`)
https://github.com/tsuyama1990/mlip_struc_generator.git
をインポートして使います。
`src/external/mlip_struc_gen` 内にあるコードを直接修正せず、インポートして利用します。

1.  **パス解決**: ファイル冒頭で `sys.path.append` を使い、`src/external` 配下をモジュール検索パスに追加してください。
2.  **ラッパークラス**: `ExternalGeneratorAdapter` クラスを実装します。
      * `__init__(self, settings: GeneratorSettings)`
      * `generate(self) -> ase.Atoms`:
          * 外部レポジトリ内の関数を呼び出す。
          * もし外部コードがファイル出力しか行わない場合、一時ディレクトリ (`tempfile`) を作成してそこで実行させ、生成されたファイルを `ase.io.read` で読み込んで `Atoms` オブジェクトとして返す処理を実装する。
          * **重要**: 外部コードの「クセ（標準出力へのprintや、sys.argv依存）」をここで吸収し、メインロジックには影響させないこと。

### 3.3 MACE Calculator ファクトリー (`src/core/calculators/mace_factory.py`)

MACEモデルのロードを抽象化します。

  * **関数 `get_mace_calculator(settings: MACESettings) -> Calculator`**:
      * `mace.calculators.MACECalculator` を初期化して返す。
      * **デバイス管理**: `torch.cuda.is_available()` をチェック。設定が "cuda" なのにGPUがない場合は警告ログを出して "cpu" にフォールバックするか、エラーを出すこと。
      * **モデルロード**: `model_path` が "small/medium/large" の場合は `mace_mp` (Materials Project Foundation Model) をロードするロジックにする。

### 3.4 構造最適化エンジン (`src/core/engines/relaxer.py`)

物理計算の中核です。

  * **クラス `StructureRelaxer`**:
      * `__init__(self, settings: Settings)`: ロガーの初期化とCalculatorの準備。
      * `run(self, atoms: ase.Atoms, run_id: str) -> Dict`:
        1.  入力 `atoms` のコピーを作成。
        2.  `atoms.calc` に MACE Calculator をセット。
        3.  **Pre-Optimization**: 初期エネルギーと最大の力（Force）を計算しログ出力。
        4.  `ase.optimize.LBFGS` (または設定されたOptimizer) をセットアップ。
        5.  **Trajectory保存**: ステップごとの構造をメモリストリームまたは一時ファイルに記録する設定を行う。
        6.  `opt.run(fmax=settings.relax.fmax, steps=settings.relax.steps)` を実行。
        7.  実行中の例外（SCF収束エラー等）を `try-except` で捕捉。
        8.  **Post-Optimization**: 最終エネルギー、ステップ数、収束フラグを取得。
        9.  結果を辞書形式（`final_structure`, `trajectory`, `energy`, `forces` 等）で返す。

### 3.5 ロギングとファイルIO (`src/core/utils/`)

  * **logging.py**:
      * `loguru` を使用。
      * コンソールには `INFO` レベル、ファイル（`data/output/{run_id}/app.log`）には `DEBUG` レベルで出力する設定関数を作る。
  * **io.py**:
      * `save_results(result_dict, output_dir)`:
          * 最終構造を `.xyz` (extxyz形式, エネルギー・力情報付き) と `.cif` で保存。
          * 計算条件（`Settings`の内容）を `config.json` として保存。
          * 計算結果サマリー（エネルギー、収束可否など）を `results.json` として保存。

-----

## 4\. CLIエントリーポイント (`main_cli.py`)

全体の動作確認用スクリプトです。

1.  **初期化**: 設定 (`Settings`) のロード。
2.  **ディレクトリ準備**: 実行時刻に基づき `data/output/YYYYMMDD_HHMMSS/` を作成。
3.  **ロガー設定**: 作成したディレクトリにログを出力設定。
4.  **構造生成**: `ExternalGeneratorAdapter` を呼び出し、初期構造（`Atoms`）を取得。
      * 取得後、必要であれば `atoms.rattle(0.05)` 等で少し構造を乱し、最適化の余地を作る。
5.  **構造最適化**: `StructureRelaxer` を初期化し、`run()` を実行。
6.  **保存**: 結果をファイルに書き出す。
7.  **完了表示**: 処理にかかった時間と最終エネルギーを表示。

-----

## 5\. コーディングガイドライン（品質基準）

コードを生成する際は、以下の基準を厳守してください。

1.  **Type Hinting**: すべての関数の引数と戻り値に型ヒントを記述すること（`ase.Atoms`, `pathlib.Path` 等）。
2.  **Docstrings**: クラスと主要なメソッドには Google Style の docstring を記述すること。
3.  **Pure Python**: `os.system()` や `subprocess` で無理やりコマンドを叩くのではなく、可能な限りライブラリのPython APIを使用すること。
4.  **Error Handling**:
      * MACEモデルが見つからない場合。
      * CUDAメモリ不足（OOM）。
      * 外部レポジトリのインポートエラー。
        これらを想定し、ユーザーに分かりやすいエラーメッセージを表示して終了すること。

-----

## 6\. 実装のステップ

以下の順序でコードブロックを提示してください。

1.  実装環境はuv, `pyproject.toml`とセットアップコマンド（submodule含む）。
2.  `src/config/settings.py`
3.  `src/core/utils/logging.py` & `io.py`
4.  `src/core/calculators/mace_factory.py`
5.  `src/core/generators/adapter.py` (※外部コードがないと仮定し、モックあるいはインポートの枠組みを実装)
6.  `src/core/engines/relaxer.py`
7.  `main_cli.py`
