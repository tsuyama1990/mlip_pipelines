大変失礼いたしました。文脈を無視した不適切な出力を深くお詫び申し上げます。
Cycle 3の実装完了を受け、そのロジックを堅牢に検証するための**ユニットテスト設計**と、**Julesへの実装指示プロンプト**を作成しました。

今回のテスト設計の肝は、\*\*「重い物理計算（LAMMPS/QE/MACE）をいかに完全にMock化し、ロジックの正しさだけを高速に検証するか」\*\*にあります。

-----

### 🧪 Cycle 3 Unit Test Architecture (設計方針)

Cycle 3のコンポーネントは外部依存（バイナリ、GPU）が強いため、以下の方針で分離します。

1.  **`LammpsMaceDriver` のテスト:**

      * **課題:** `lammps` Pythonモジュールがない環境や、GPUがない環境でもパスする必要がある。
      * **解決策:** `unittest.mock.MagicMock` で `lammps` オブジェクトを完全に偽装する。特に「Callback関数」は、LAMMPSからの呼び出しを待つのではなく、**テストコードから直接引数を渡して呼び出し**、正しく力が返されるか、閾値超えで例外が出るかを検証する。

2.  **`QeOracle` のテスト:**

      * **課題:** `pw.x` を実行すると重すぎるし、環境依存になる。
      * **解決策:** ASEの `Espresso` 計算機が発行する**コマンドライン**と**生成される入力ファイル**の中身を検証する（Dry Run）。実際に計算はさせない。

3.  **`Orchestrator` のテスト:**

      * **課題:** 複雑なステートマシンであり、全てのコンポーネントをつなぐ。
      * **解決策:** Driver, Oracle, Carver, Potential すべてをMock化する。「Driverが『UncertaintyInterrupt』を投げたとき、OrchestratorがそれをキャッチしてPhase 2（学習）に移行するか？」という**フロー制御**のみをテストする。

-----

### ⌨️ Coder (Jules) への指示プロンプト

以下のテキストをコピーしてJulesに渡し、テストコードを実装させてください。

````markdown
# Cycle 3: Unit Test Implementation Specification

**Role:**
You are the Lead QA Engineer.
Now that the Cycle 3 implementation is complete, your mission is to implement the **Unit Test Suite**.

**Core Philosophy:**
**"Test the Logic, Mock the Physics."**
Unit tests must run **instantly (< 1s)**, without GPU, and without external binaries (`pw.x`, `lmp`).
We assume the external tools work; we verify only that your Python code orchestrates them correctly.

**Directory Structure:**
```text
tests/
├── unit/
│   ├── engines/
│   │   └── test_lammps_mace.py   # Target: src/engines/lammps_mace.py
│   ├── oracles/
│   │   └── test_qe_oracle.py     # Target: src/oracles/qe_oracle.py
│   └── core/
│       └── test_orchestrator.py  # Target: src/core/orchestrator.py
└── integration/
    └── test_real_loop.py         # (Keep existing Integration Test)
````

**Requirements by Module:**

-----

### 1\. `tests/unit/engines/test_lammps_mace.py`

**Challenge:** Test the driver without `lammps` installed and without a GPU.

  * **Mocking:**

      * Use `unittest.mock.MagicMock` to replace `lammps.lammps`.
      * Mock `driver.potential` (MacePotential) to return deterministic forces/uncertainty.

  * **Test Case 1: Callback - Normal Step**

      * **Setup:** Mock potential returns `uncertainty = 0.0` (safe).
      * **Action:** Manually call `driver._callback(caller=None, ntimestep=1, nlocal=2, tag=[1,2], x=[...], f=[0,0...])`.
      * **Assert:**
          * The `f` array passed in is populated with forces from the potential.
          * `potential.predict_forces` was called.
          * No exception is raised.

  * **Test Case 2: Callback - Uncertainty Interrupt**

      * **Setup:** Mock potential returns `uncertainty = 10.0` (above threshold).
      * **Action:** Call `driver._callback(...)`.
      * **Assert:** `UncertaintyInterrupt` is raised.
      * **Assert:** The exception object contains the `Atoms` and the correct uncertainty value.

  * **Test Case 3: LAMMPS Configuration**

      * **Action:** Call `driver.run_md(atoms, lammps_script="fix 1...")`.
      * **Assert:**
          * `mock_lammps.command` was called with "pair\_style none".
          * `mock_lammps.set_fix_external_callback` was called.
          * `mock_lammps.commands_string` was called with the user script.

-----

### 2\. `tests/unit/oracles/test_qe_oracle.py`

**Challenge:** Test QE generation without running `pw.x`.

  * **Mocking:**

      * Mock `subprocess.Popen` (used by ASE) to prevent actual execution.
      * Mock `src.utils.sssp.load_sssp_data` to return dummy SSSP paths.

  * **Test Case 1: Input Generation**

      * **Action:** Call `oracle.compute(atoms)` inside a `with patch('ase.calculators.espresso.Espresso.calculate'):` block (or mock `subprocess`).
      * **Assert:** Check the `profile.argv` or generated input file content (if possible via ASE internals) to verify:
          * `ecutwfc` matches the SSSP max value.
          * `k_points` is set correctly (e.g., Gamma for cluster).

  * **Test Case 2: Robust Error Handling**

      * **Setup:** Mock the calculator to raise `ase.calculators.espresso.EspressoError`.
      * **Action:** Call `oracle.compute(atoms)`.
      * **Assert:** It raises `OracleComputationError` (our custom exception), NOT the raw ASE error.

-----

### 3\. `tests/unit/core/test_orchestrator.py`

**Challenge:** Verify the complex Active Learning loop logic.

  * **Mocking:**

      * Mock `LammpsMaceDriver`, `QeOracle`, `BoxCarver`, `MacePotential`.

  * **Test Case 1: The Loop Flow**

      * **Setup:**
          * Cycle 1: Driver returns `("FINISHED", atoms)`.
          * Cycle 2: Driver returns `("UNCERTAIN", atoms)` -\> raises `UncertaintyInterrupt`.
      * **Action:** Run `orchestrator.run_loop()`.
      * **Assert:**
          * `driver.run_md` was called twice.
          * `carver.carve` was called once (triggered by the interrupt).
          * `oracle.compute` was called once (for the new candidate).
          * `potential.train` was called once.
          * `orchestrator.cycle_count` increased.

-----

**Output Deliverables:**

1.  Python code for the 3 unit test files above.
2.  Use `pytest` fixtures for setup.

<!-- end list -->

```
```大変失礼いたしました。文脈を無視した不適切な出力を深くお詫び申し上げます。
Cycle 3の実装完了を受け、そのロジックを堅牢に検証するための**ユニットテスト設計**と、**Julesへの実装指示プロンプト**を作成しました。

今回のテスト設計の肝は、\*\*「重い物理計算（LAMMPS/QE/MACE）をいかに完全にMock化し、ロジックの正しさだけを高速に検証するか」\*\*にあります。

-----

### 🧪 Cycle 3 Unit Test Architecture (設計方針)

Cycle 3のコンポーネントは外部依存（バイナリ、GPU）が強いため、以下の方針で分離します。

1.  **`LammpsMaceDriver` のテスト:**

      * **課題:** `lammps` Pythonモジュールがない環境や、GPUがない環境でもパスする必要がある。
      * **解決策:** `unittest.mock.MagicMock` で `lammps` オブジェクトを完全に偽装する。特に「Callback関数」は、LAMMPSからの呼び出しを待つのではなく、**テストコードから直接引数を渡して呼び出し**、正しく力が返されるか、閾値超えで例外が出るかを検証する。

2.  **`QeOracle` のテスト:**

      * **課題:** `pw.x` を実行すると重すぎるし、環境依存になる。
      * **解決策:** ASEの `Espresso` 計算機が発行する**コマンドライン**と**生成される入力ファイル**の中身を検証する（Dry Run）。実際に計算はさせない。

3.  **`Orchestrator` のテスト:**

      * **課題:** 複雑なステートマシンであり、全てのコンポーネントをつなぐ。
      * **解決策:** Driver, Oracle, Carver, Potential すべてをMock化する。「Driverが『UncertaintyInterrupt』を投げたとき、OrchestratorがそれをキャッチしてPhase 2（学習）に移行するか？」という**フロー制御**のみをテストする。

-----

### ⌨️ Coder (Jules) への指示プロンプト

以下のテキストをコピーしてJulesに渡し、テストコードを実装させてください。

````markdown
# Cycle 3: Unit Test Implementation Specification

**Role:**
You are the Lead QA Engineer.
Now that the Cycle 3 implementation is complete, your mission is to implement the **Unit Test Suite**.

**Core Philosophy:**
**"Test the Logic, Mock the Physics."**
Unit tests must run **instantly (< 1s)**, without GPU, and without external binaries (`pw.x`, `lmp`).
We assume the external tools work; we verify only that your Python code orchestrates them correctly.

**Directory Structure:**
```text
tests/
├── unit/
│   ├── engines/
│   │   └── test_lammps_mace.py   # Target: src/engines/lammps_mace.py
│   ├── oracles/
│   │   └── test_qe_oracle.py     # Target: src/oracles/qe_oracle.py
│   └── core/
│       └── test_orchestrator.py  # Target: src/core/orchestrator.py
└── integration/
    └── test_real_loop.py         # (Keep existing Integration Test)
````

**Requirements by Module:**

-----

### 1\. `tests/unit/engines/test_lammps_mace.py`

**Challenge:** Test the driver without `lammps` installed and without a GPU.

  * **Mocking:**

      * Use `unittest.mock.MagicMock` to replace `lammps.lammps`.
      * Mock `driver.potential` (MacePotential) to return deterministic forces/uncertainty.

  * **Test Case 1: Callback - Normal Step**

      * **Setup:** Mock potential returns `uncertainty = 0.0` (safe).
      * **Action:** Manually call `driver._callback(caller=None, ntimestep=1, nlocal=2, tag=[1,2], x=[...], f=[0,0...])`.
      * **Assert:**
          * The `f` array passed in is populated with forces from the potential.
          * `potential.predict_forces` was called.
          * No exception is raised.

  * **Test Case 2: Callback - Uncertainty Interrupt**

      * **Setup:** Mock potential returns `uncertainty = 10.0` (above threshold).
      * **Action:** Call `driver._callback(...)`.
      * **Assert:** `UncertaintyInterrupt` is raised.
      * **Assert:** The exception object contains the `Atoms` and the correct uncertainty value.

  * **Test Case 3: LAMMPS Configuration**

      * **Action:** Call `driver.run_md(atoms, lammps_script="fix 1...")`.
      * **Assert:**
          * `mock_lammps.command` was called with "pair\_style none".
          * `mock_lammps.set_fix_external_callback` was called.
          * `mock_lammps.commands_string` was called with the user script.

-----

### 2\. `tests/unit/oracles/test_qe_oracle.py`

**Challenge:** Test QE generation without running `pw.x`.

  * **Mocking:**

      * Mock `subprocess.Popen` (used by ASE) to prevent actual execution.
      * Mock `src.utils.sssp.load_sssp_data` to return dummy SSSP paths.

  * **Test Case 1: Input Generation**

      * **Action:** Call `oracle.compute(atoms)` inside a `with patch('ase.calculators.espresso.Espresso.calculate'):` block (or mock `subprocess`).
      * **Assert:** Check the `profile.argv` or generated input file content (if possible via ASE internals) to verify:
          * `ecutwfc` matches the SSSP max value.
          * `k_points` is set correctly (e.g., Gamma for cluster).

  * **Test Case 2: Robust Error Handling**

      * **Setup:** Mock the calculator to raise `ase.calculators.espresso.EspressoError`.
      * **Action:** Call `oracle.compute(atoms)`.
      * **Assert:** It raises `OracleComputationError` (our custom exception), NOT the raw ASE error.

-----

### 3\. `tests/unit/core/test_orchestrator.py`

**Challenge:** Verify the complex Active Learning loop logic.

  * **Mocking:**

      * Mock `LammpsMaceDriver`, `QeOracle`, `BoxCarver`, `MacePotential`.

  * **Test Case 1: The Loop Flow**

      * **Setup:**
          * Cycle 1: Driver returns `("FINISHED", atoms)`.
          * Cycle 2: Driver returns `("UNCERTAIN", atoms)` -\> raises `UncertaintyInterrupt`.
      * **Action:** Run `orchestrator.run_loop()`.
      * **Assert:**
          * `driver.run_md` was called twice.
          * `carver.carve` was called once (triggered by the interrupt).
          * `oracle.compute` was called once (for the new candidate).
          * `potential.train` was called once.
          * `orchestrator.cycle_count` increased.

-----

**Output Deliverables:**

1.  Python code for the 3 unit test files above.
2.  Use `pytest` fixtures for setup.

<!-- end list -->

```
```
