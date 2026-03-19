**下記のうち結構な割合で実装済みだ。実装済みなものについては妥当性を検証しつつ、全体最適を続けながら実装をせよ。**

mlip_pipelines 高効率MLIP構築・運用システム 要件定義書 (Comprehensive Edition)1. プロジェクト概要本プロジェクトは、原子間ポテンシャル作成ツール「Pacemaker (ACE: Atomic Cluster Expansion)」を核とし、材料科学の専門知識が浅いユーザーであっても、最小限の工数で「State-of-the-Art（最先端）」な機械学習ポテンシャル (MLIP) を構築・運用できる自動化システムを開発することを目的とする。1.1 背景と課題：原子シミュレーションの民主化に向けて現代の計算材料科学において、第一原理計算（DFT）の精度と古典分子動力学（MD）のスケールを両立させるMLIPは不可欠なツールとなっている。しかし、高品質なMLIPの構築には、データサイエンスと計算物理の両面にわたる深い専門知識が必要であり、多くの実験研究者や企業の研究者にとって参入障壁が高いのが現状である。従来のMLIP構築フローは、専門家による手動の繰り返し作業（構造作成→DFT→学習→検証）に依存しており、以下の構造的な課題を抱えていた。構造サンプリングの偏りと「外挿」の危険性:標準的な平衡状態のMDだけでは、相転移、化学反応、破壊現象などで現れる「レアイベント」や「高エネルギー配置」を網羅できない。未知の領域（Extrapolation region）にシミュレーションが突入した際、ポテンシャルが物理的にあり得ない力（例: 原子核同士が重なっても斥力が働かないなど）を出力し、シミュレーションが破綻するリスクがある。「ゴミ」データの蓄積と計算資源の浪費:物理的に類似した構造（相関の強いスナップショット）を大量にDFT計算しても、ポテンシャルの精度向上には寄与しない。情報量の低い構造に高価な計算リソースを費やすことは、プロジェクトのコスト効率を著しく低下させる。運用開始後のメンテナンスコスト:シミュレーション中にポテンシャルが破綻した場合、原因となる構造を特定し、再学習を行い、再度シミュレーションを流すという手戻り作業が煩雑であり、研究のボトルネックとなる。1.2 目標 (Success Metrics)本システムは、以下の指標を達成することを目標とする。工数の劇的削減 (Zero-Config Workflow):初期設定ファイル (YAML) 1つで、初期構造生成から学習完了までのパイプラインを無人で完走させる。ユーザーはPythonスクリプトを書く必要がない。データ効率の最大化 (Data Efficiency):能動学習 (Active Learning) と物理に基づいた高度なサンプリング手法を組み合わせ、ランダムサンプリングと比較して1/10以下のDFT計算量で同等の精度（RMSE Energy < 1 meV/atom, Force < 0.05 eV/Å）を目指す。物理的堅牢性 (Physics-Informed Robustness):Core-Repulsion（原子核近傍の斥力）における物理的な正しさを、LJポテンシャルからのデルタ学習によって強制する。データが存在しない領域でも、少なくとも原子が重なり合って崩壊しない「物理的な安全性」を担保する。スケーラビリティと拡張性:局所的なActive Learningから、数百万原子規模のMDや、秒単位の時間スケールを扱うkMCシミュレーションへとシームレスに展開できるアーキテクチャを構築する。2. システムアーキテクチャ詳細本システムは、モジュール間の疎結合を保ちつつ、全体を統括する Pythonベースの「Orchestrator」 を中心に設計される。各モジュールは独立したコンテナ（Docker/Singularity）として動作可能とし、ローカルワークステーションからクラウド/HPC環境へのデプロイを容易にする。2.1 モジュール構成Structure Generator (構造探索モジュール):役割: 未知の化学空間・構造空間を探索し、学習候補となる原子配置を提案する「探検家」。特徴: 物理的直感に基づいたバイアス（温度、圧力、欠陥、化学ポテンシャル）を印加し、闇雲なランダム探索ではなく「意味のある」多様性を追求する。Oracle (教師データ生成モジュール):役割: 提案された構造に対し、第一原理計算 (DFT) を実行して正解データ（エネルギー、力、応力）を付与する「賢者」。特徴: 計算の失敗（SCF収束エラー等）を自動検知し、パラメータ（Mixing beta, Smearing等）を動的に調整して再計算する自己修復機能を持つ。Trainer (学習モジュール):役割: Pacemakerエンジンを駆動し、ACEポテンシャルをフィッティングする「学習者」。特徴: 物理ベースライン（LJ/ZBL）との差分学習を管理し、過学習を防ぐための正則化（Regularization）を自動調整する。Dynamics Engine (推論・運用モジュール):役割: 生成されたポテンシャルを用いてMD/kMCを実行し、リアルタイムで信頼性（Uncertainty）を監視する「実行者」。特徴: LAMMPSおよびEON (kMC) との密な連携インターフェース（Python C-API等）を持ち、シミュレーションを中断・再開する制御権を持つ。2.2 データフロー (The Active Learning Cycle)システムは以下のサイクルを自律的に繰り返す。Exploration (探索): Dynamics Engine または Structure Generator が構造空間を探索する。Detection (検知): 不確実性指標 $\gamma$ (extrapolation grade) が閾値を超えた構造、または幾何学的に新規な構造を検出する。Selection & Calculation (選別と計算):冗長な構造（既存データと類似した構造）を除外。Periodic Embedding により計算コストを最小化した小さなセルを作成。Oracle へ DFT 計算リクエストを送信。Refinement (学習):新規データをデータセットに統合。Trainer がポテンシャルを更新（Fine-tuning）。Deployment (配備):更新された potential.yace を Dynamics Engine へホットデプロイ。シミュレーションを再開。3. 機能要件詳細3.1 自動トレーニング構造作成機能 (Structure Generator)学習データの「質」を決定づける最重要モジュールである。本システムでは、従来の固定的なルールベース（暗黙知）による切り替えを廃止し、対象系の特徴量から最適な探索戦略を動的に導出する 「適応的探索ポリシー (Adaptive Exploration Policy)」 を実装する。適応的探索ポリシーエンジン (Adaptive Exploration Policy Engine):概要:「合金だからMC」といった単純な分岐ではなく、組成、予測物性、不確実性分布を入力とし、MD/MC比率や温度スケジュールなどのハイパーパラメータを出力する決定モデル（Policy）を構築する。入力特徴量 (Input Features):初期探索（M3GNet/CHGNet等）の結果およびPymatgen等の静的解析から以下を取得する。Material DNA: 元素種、組成比、平均価電子数、結晶系、空間群。Predicted Properties: 汎用ポテンシャルで推定したバンドギャップ ($E_g$)、融点 ($T_m^{pred}$)、体積弾性率 ($B_0$)。Uncertainty State: 初期構造に対する $\gamma$ 値の分布形状（平均値、分散、最大値）。出力戦略パラメータ (Output Strategy Parameters):以下のパラメータを動的に決定し、LAMMPS/Pythonスクリプトへ渡す。$R_{MD/MC}$ (MD/MC Ratio): 純粋なMDステップに対する原子交換MCの頻度。$T_{schedule}$ (Temperature Schedule): 最高温度 ($T_{max}$)、昇温速度 ($dT/dt$)。$N_{defects}$ (Defect Density): 導入すべき空孔・格子間原子の濃度。$\epsilon_{range}$ (Strain Range): EOS/弾性計算のための歪み印加範囲。ポリシー決定ロジック (Decision Logic Example):| 特徴量条件 | 判定される物理的レジーム | 適用される探索ポリシー (Action) || $E_g \approx 0$ (金属) & 多成分系 | 拡散・配置エントロピー支配 | High-MC Policy: $R_{MD/MC}$を高く設定 (例: 100 step毎)。$T_{max}$は$0.8 T_m$付近を重点的に。 || $E_g > 0$ (絶縁体) & 複雑な単位胞 | 格子歪み・欠陥支配 | Defect-Driven Policy: $N_{defects}$を高く設定 (スーパーセル内に多種類の欠陥導入)。MCはOFFまたは低頻度。 || $B_0$ が高い (硬い材料) | 共有結合・方向性支配 | Strain-Heavy Policy: $\epsilon_{range}$を広め ($\pm 15\%$) に設定し、せん断変形を多めにサンプリング。 || 初期 $\gamma$ 分散が大きい | 知識不足 (High Uncertainty) | Cautious Exploration: $T_{schedule}$を緩やかにし、低温域でのデータ収集を優先して基礎を固める。 |実行可能なアクション空間 (Action Space):ポリシーによって選択・パラメータ調整される具体的なサンプリング手法群。1. Initial Exploration via Universal Potentials (Cold Start):役割: Policyの入力となる初期特徴量 ($T_m^{pred}$ 等) を取得するための事前調査。動作: ユーザー入力組成に対し、M3GNet等を用いて超高速でスクリーニングMDを行い、大まかな安定構造と物性を推定する。2. Variable T-P Ramping (温度・圧力スキャン):制御: Policyが出力した $T_{max}$ と圧力範囲に基づき、LAMMPSの fix npt 制御変数を設定する。詳細: 低温（フォノン）、中温（熱膨張）、高温・高圧（液体・高密度相）を、指定されたスケジュールで走査する。3. Hybrid MD/MC Sampling (化学空間探索):制御: Policyが出力した $R_{MD/MC}$ に基づき、LAMMPSの fix atom/swap の頻度と確率を調整する。詳細: 拡散が遅い系でも、物理時間を無視して化学組成の局所平衡（規則化・偏析）を探索する。4. Defect & Distortion Engineering (構造空間探索):制御: Policyが出力した $N_{defects}$ と $\epsilon_{range}$ に基づき、Pythonスクリプトで構造生成を行う。詳細 (One Defect Strategy): 空孔、格子間原子、アンチサイトを導入。詳細 (EOS & Strain): 等方的な体積変化 ($\pm \epsilon_{range}$) と、ランダムなせん断歪み (Rattling) を加えた構造を生成し、弾性特性の学習データを確保する。5. Normal Mode & Torsion (分子・局所モード探索):適用: Policyが「分子性」または「低次元構造」と判定した場合に発動。詳細: 結合角や二面角を強制的に操作し、MDでは乗り越えられないエネルギー障壁の向こう側を探索する。3.2 教師データ生成・ラベリング機能 (Oracle)DFT計算は計算コストが高く、かつエラーが発生しやすいため、ここでの自動化と効率化がシステム全体の性能を左右する。添付資料「Automating Quantum Espresso Static Calculations」に基づき、堅牢なプロトコルを実装する。DFT自動計算パイプラインと自己修復機能:入力生成: Quantum Espresso (QE) または VASP の入力ファイルを、構造の特性に応じて動的に最適化する。K-space Sampling: 固定のk点グリッド数（例: $4 \times 4 \times 4$）ではなく、kspacing (逆空間密度、例: $0.03 \sim 0.05 \AA^{-1}$) を指標としてグリッドを自動生成する。これにより、構造探索やEmbeddingによってセルサイズが変動しても、計算精度（エネルギーの収束性）を一貫させ、かつ過剰なk点による計算資源の浪費を防ぐ。Pseudopotentials: ユーザーが個別に指定するのではなく、SSSP (Standard Solid State Pseudopotentials) などの検証済み標準ライブラリ（Precision または Efficiency モード）を自動的に参照・適用する。これにより、ポテンシャルの質に起因する計算エラーやゴーストステートを防ぐ。Smearing: 金属/絶縁体の区別が曖昧な場合、安全側に倒して occupation='smearing' (Marzari-Vanderbilt) を採用し、SCF収束性を高める。Spin Polarization: 遷移金属（Fe, Co, Ni, Mn等）が含まれる場合、自動的にスピン分極計算をONにし、初期磁気モーメントを強磁性に設定して計算崩壊を防ぐ。Error Handling (Self-Correction): 計算が収束しなかった場合、自動的に以下の対策を順次試行するロジックを実装する。混合パラメータ (mixing_beta) を下げる（例: 0.7 -> 0.3）。対角化アルゴリズムを変更する（例: david -> cg）。電子温度（Smearing width）を上げる。Static Calculation (一点計算):設定: 構造緩和 (relax) ではなく、原子位置を固定したまま電子状態のみを解く。QE固有の要件: calculation='scf' を指定するが、デフォルトでは力が計算されないため、必ず tprnfor=.true. および tstress=.true. を設定ファイルに追加する。これにより、原子を動かすことなく、ポテンシャル学習に必要なHellmann-Feynman力と応力テンソルを正確に抽出する。（nscf は電荷密度が必要なため、教師データ生成の文脈では通常使用しない）Periodic Embedding (周期的埋め込み) によるデータ切り出し:背景: 大規模MD中に検出された「不確実な局所領域」だけをDFT計算したいが、単純にクラスターとして切り出すと、表面（真空）の効果がノイズとして混入する。また、球状に切り出すと周期境界条件を適用できない。手法:不確実性の高い原子を中心とし、ACEのカットオフ半径 $R_{cut}$ をカバーする領域を特定する。この領域の周囲に、さらに $R_{buffer}$ (約2層分) のバッファ領域を含めた上で、これらを包含する**直方体状のセル（Orthorhombic Box）**を切り出す。球状に切り出すと周期境界条件を満たせないため、必ず空間を隙間なく埋められる形状とする。この直方体セルを真空に浮かべるのではなく、周期境界条件を持つ小さなスーパーセル として再定義する（Periodic Embedding）。DFT計算後、中心領域（半径 $R_{cut}$）の原子の力のみを正解データとして採用し、境界付近（$R_{buffer}$）の原子の力は「表面効果の影響を受けている」として学習ウェイトをゼロにする（Force Masking）。利点: 界面、転位芯、アモルファス構造などの複雑な局所環境を、表面効果というアーティファクトなしに学習可能にする。3.3 学習機能 (Trainer)Pacemakerの機能をフル活用し、「少ないデータで賢く」学習する。過学習を防ぐための物理的な制約を重視する。Pacemakerのサイトを参照の上フル活用すること (https://pacemaker.readthedocs.io/en/feature-docs/)Delta Learning (LJ Baseline) の強制とハイブリッド運用:理論: 全ポテンシャルエネルギー $E_{total}$ を、$E_{total} = E_{baseline} + E_{ACE}$ と分解する。実装 (学習フェーズ):各元素ペアに対し、原子半径に基づいた標準的なLennard-Jones (LJ) パラメータ、あるいはZBLポテンシャル（近距離核反発）を自動設定する。Pacemaker設定ファイルにおいて、この参照ポテンシャルを定義し、ACE部分は「LJでは表現しきれない多体相互作用の残差」のみを学習するよう構成する。実装 (推論・MDフェーズ):合成ポテンシャルの利用: 学習済みモデルをLAMMPSで実行する際も、単独のACEポテンシャルではなく、必ず ベースライン(LJ/ZBL) と ACE を重ね合わせた合成ポテンシャル を使用する。LAMMPS設定: pair_style hybrid/overlay コマンドを用い、lj/cut (または zbl) と pace を重畳させる記述を in.lammps に自動生成する。これにより、学習時と全く等価なエネルギー局面を再現する。効果: 学習データが存在しない極短距離（原子同士が重なる領域）において、ACE多項式の暴走を防ぎ、必ず物理的な強い斥力が働くことを保証する。これにより、MD中の原子衝突による爆発（Segmentation Fault）をほぼ確実に回避できる。Active Set Optimization (D-Optimality):背景: データ数が増えると学習コストは線形～二乗で増加するが、似通ったデータの追加は精度向上に寄与しない。手法: 蓄積された数千～数万の構造データ全てを使って学習するのではなく、線形代数的な情報量（行列式）が最大となるような「代表構造 (Active Set)」を選別する。ツール: pace_activeset コマンドをバックグラウンドで実行し、MaxVolアルゴリズムを用いて基底関数の係数決定に最も寄与する構造のみをフィルタリングする。これにより、数万の候補構造から数百の「真に重要な構造」だけを抽出して学習に用いる。3.4 推論・On-the-Fly (OTF) 学習機能 (Dynamics Engine)能動学習の現場となる実行エンジンであり、本システムの心臓部である。ここでは、古典分子動力学 (MD) とアダプティブ・キネティック・モンテカルロ (aKMC) をシームレスに統合し、自律的な学習と推論のサイクルを実現する詳細仕様を定義する。ハイブリッドポテンシャルの適用 (Hybrid Potential Application):目的と背景: 純粋なACEポテンシャルは多項式展開であるため、学習データが存在しない原子核同士の極端な接近領域（Core region）において、非物理的な引力井戸を形成する場合がある。これは高エネルギー衝突時にMDシミュレーションの即時崩壊（Segmentation Fault）を招くリスクがある。要件: 3.3節で定義した物理ベースライン（LJまたはZBL）を、推論時にも厳格に適用し、安全装置として機能させる。LAMMPS実装:pair_style hybrid/overlay コマンドを使用し、物理ベースライン (lj/cut や zbl) と機械学習部分 (pace) を加算的に重ね合わせる記述を in.lammps に自動生成する。設定例:# ZBLをベースラインとし、ACEを上乗せする設定例
pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace potential.yace Ti O
pair_coeff * * zbl 22 8  # 原子番号を指定

この構成により、万が一ACE部分が外挿領域で異常な値を返しても、物理ベースラインの強い斥力が支配的となり、原子の重なりとシミュレーションの破綻を物理的に阻止する。不確実性監視 (Uncertainty Quantification) と fix halt:メカニズム:PacemakerのACEポテンシャルは、推論時に各原子の局所環境記述子（$\mathbf{c}_i$）を計算し、学習データセットの張る部分空間からの逸脱度合い（マハラノビス距離に類似）を示す指標 $\gamma$ (extrapolation grade) をリアルタイムに出力する機能を持つ。$\gamma \approx 0$ は学習データに近い既知の領域、$\gamma \gg 1$ は未知の危険領域を意味する。LAMMPS実装とトリガー:compute pace コマンド（USER-PACEパッケージ）を用いて、全原子の最大 $\gamma$ 値 (v_max_gamma) を監視する。fix halt コマンドを使用し、v_max_gamma > threshold (推奨初期値: 5.0, 学習進行に伴い緩和可能) となった瞬間にシミュレーションを強制停止し、終了コードを返す。パフォーマンス最適化:毎ステップの $\gamma$ 計算はコストが高いため、run every 10 などの設定により、10〜100ステップごとの間欠的な監視を行う。これにより、OTF監視による計算オーバーヘッドを10%未満に抑えつつ、破綻前に検知する応答性を確保する。自動再学習ループ (The Anatomy of a Halt Event):停止イベントが発生した際、Pythonオーケストレーターは以下のシーケンスを実行し、ポテンシャルを「治療」して再開する。単なる「穴埋め（一点学習）」ではなく、不確実性が高い領域の「局所的な曲率」を学習し、壁を平滑化するための近傍サンプリング戦略を採用する。Halt & Diagnose (停止と診断):LAMMPSが特定の終了コードで停止し、制御がPythonドライバに戻る。ログを解析し、停止の原因となったタイムステップと、最大 $\gamma$ 値を記録した原子ID群を特定し、その構造 $S_0$ (Halt Structure) を抽出する。Generate Local Candidates (近傍候補の生成):抽出した $S_0$ を中心に、以下のいずれかの手法で 20〜30 個程度の近傍構造候補 $\{S_i\}$ を安価に生成する（この段階ではDFTは行わない）。(A) Normal Mode Approximation: $S_0$ におけるHessianをACEあるいは汎用ポテンシャルで近似計算し、最大曲率方向へ $\pm \epsilon$ 変位させる（推奨）。(B) MD Micro-burst: $S_0$ から非常に短い時間（5〜20ステップ）、高温または小刻みな時間刻みでMDを走らせる。(C) Random Displacement: 高 $\gamma$ 原子に対し、微小なランダム変位 ($0.01 \sim 0.05 \AA$) を加える。Local D-Optimality Selection (局所的な選別):生成された候補構造群 $\{S_i\}$ と $S_0$ に対し、pace_activeset をローカルに実行する。情報行列の行列式 (D-Optimality) を最大化するような 5〜10 個の構造セット $\{S_{selected}\}$ を選出する。この際、$S_0$ は必ず含める（Anchor）。設定: 推奨デフォルト数は 5 点だが、系（金属: 3-5, 分子: 6-10）やHaltの頻度に応じて config.yaml で調整可能とする。目的: 単一点だけでなく、その周囲の「勾配の変化（曲率）」を学習させることで、再び同じ穴に落ちるのを防ぐ（再Halt防止）。Embed (埋め込み):選出された $\{S_{selected}\}$ の各構造に対し、3.2節の Periodic Embedding 処理を適用する。単なるクラスター（真空あり）ではなく、周期境界条件を満たす直方体状の小型スーパーセルを作成し、バルク性質を保ったままDFT計算可能な形式に変換する。Compute (正解データの生成):生成されたセル群をOracleへ一括投入し、DFT計算を実行して正確な力とエネルギーを取得する。Update (再学習):新規データを既存の学習セットに追加する。Pacemakerを実行する際、ゼロから学習するのではなく、前回のモデル重みを初期値として読み込み (--initial_potential)、学習率を下げて数エポックだけ回す Fine-tuning を行う。これにより、学習時間を数時間から数分へと短縮する。Resume (再開):更新された potential.yace をLAMMPSの作業ディレクトリに配置する。LAMMPSを read_restart で停止直前の状態から復帰させ、新しいポテンシャルをロードしてシミュレーションを続行する。Scale-up: MD/kMC 連携:背景: MDはナノ秒スケールの現象しか追えないが、実際の材料劣化や拡散現象は秒〜年のスケールで進行する。これを解決するために、Adaptive Kinetic Monte Carlo (aKMC) を導入し、時間スケールの壁を突破する。kMC (Kinetic Monte Carlo) 実装:EON (Eon Client/Server) 等のaKMCソフトウェアと連携する。サドル点探索（Nudged Elastic BandやDimer法）において、エネルギーと力の計算エンジンとしてACEポテンシャルを使用する。連携ロジック:遷移状態（サドル点）は、ポテンシャルエネルギー曲面上で原子配置が歪んだ高エネルギー状態であり、「未知の領域」になりやすい。サドル点探索中に構造の $\gamma$ 値が高くなった場合、即座に上述のOTFループ（Extract -> Embed -> ...）を起動する。これにより、「MDでは到達できないが、熱力学的に重要な遷移パス」を能動的に学習し、拡散係数や反応速度定数の予測精度を劇的に向上させる。役割分担とシナジー:MD: 短時間の熱振動、液体構造、高速な拡散、およびエントロピー的な配置の探索を担当。kMC: 固体内の空孔拡散、表面吸着、相変態の核生成など、MDでは到達不可能な長時間スケール現象の探索を担当。この両輪を共通のポテンシャル・共通の学習ループで回すことで、時間スケールと空間スケールの双方をカバーする「全能型」のポテンシャルを育成する。3.5 Pythonオーケストレーション詳細設計 (Implementation Blueprints)システム全体を統括するPythonフレームワーク（pyacemaker）の実装設計。ファイル管理の厳格化と、各モジュール（LAMMPS, ASE, Pacemaker）の具体的な連携コードを定義する。ディレクトリ構造 (Directory Strategy):混乱を避けるため、以下のような階層構造を自動生成する。
Core Orchestrator Logic (The Brain):状態遷移を管理するメインループの実装イメージ。import time
from pathlib import Path
import shutil

class Orchestrator:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.md_engine = MDInterface(self.config['lammps'])
        self.oracle = DFTManager(self.config['dft'])
        self.trainer = PacemakerWrapper(self.config['training'])
        self.validator = Validator(self.config['validation'])
        self.iteration = 0

    def run_cycle(self):
        # 最新ポテンシャルの特定
        current_pot = self.trainer.get_latest_potential()

        # 作業ディレクトリ作成
        work_dir = Path(f"active_learning/iter_{self.iteration:03d}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. LAMMPS MD実行 (Exploration)
        halt_info = self.md_engine.run_exploration(
            potential=current_pot,
            work_dir=work_dir / "md_run"
        )

        if not halt_info['halted']:
            print("MD completed without high uncertainty. Converged?")
            return "CONVERGED"

        # 2. 構造抽出 (Selection) - NEW: Generate local candidates & D-opt select
        high_gamma_atoms = self.md_engine.extract_high_gamma_structures(
            dump_file=halt_info['dump_file'],
            threshold=self.config['otf_loop']['uncertainty_threshold']
        )

        # Local D-Optimality Selection Logic (Pseudo-code)
        selected_structures = []
        for s0 in high_gamma_atoms:
            candidates = self.structure_generator.generate_local_candidates(s0, n=20)
            selected = self.trainer.select_local_active_set(candidates, anchor=s0, n=5)
            selected_structures.extend(selected)

        # 3. DFT計算 (Oracle)
        new_data = self.oracle.compute_batch(selected_structures, work_dir / "dft_calc")

        if not new_data:
            print("No valid data obtained from DFT.")
            return "ERROR"

        # 4. 学習データの更新 (Data Management)
        dataset_path = self.trainer.update_dataset(new_data)

        # 5. 再学習 (Refinement)
        new_pot_path = self.trainer.train(
            dataset=dataset_path,
            initial_potential=current_pot,
            output_dir=work_dir / "training"
        )

        # 6. 検証 (Validation)
        validation_result = self.validator.validate(new_pot_path)
        if not validation_result['passed']:
            print(f"Validation failed: {validation_result['reason']}")

        # 次のサイクルへ
        self.iteration += 1
        shutil.copy(new_pot_path, f"potentials/generation_{self.iteration:03d}.yace")

Module Interface Details:A. MDInterface (LAMMPS Control):lammps Pythonモジュール（または subprocess）を使用するが、ここではより柔軟なPythonモジュール版を推奨。from lammps import lammps

class MDInterface:
    def run_exploration(self, potential, work_dir):
        lmp = lammps()
        # ... (LAMMPSの初期化コマンド発行) ...

        # Hybrid Potentialの設定
        lmp.command("pair_style hybrid/overlay pace zbl 1.0 2.0")
        lmp.command(f"pair_coeff * * pace {potential} ...")

        # Watchdogの設定 (gamma監視)
        lmp.command("compute pace_gamma all pace ... gamma_mode=1")
        lmp.command("variable max_gamma equal max(c_pace_gamma)")
        lmp.command("fix watchdog all halt 10 v_max_gamma > 5.0 error hard")

        try:
            lmp.command("run 100000")
            return {'halted': False}
        except Exception as e:
            # LAMMPSがerror hardで終了した場合、Python側でキャッチ可能
            print("Halt triggered by uncertainty watchdog!")
            return {'halted': True, 'dump_file': f"{work_dir}/dump.lammps"}

B. DFTManager (ASE Integration):複雑なDFT入力作成を ase.calculators.espellipso (QE) 等に委譲する。from ase.io import read, write
from ase.calculators.espresso import Espresso

class DFTManager:
    def compute_batch(self, structures, calc_dir):
        results = []
        for atoms in structures:
            # Periodic Embedding処理 (直方体セル化 + バッファ)
            embedded_atoms = self._apply_periodic_embedding(atoms)

            # Calculator設定 (自己修復ロジック付き)
            calc = Espresso(
                pseudopotentials=self.pseudos,
                tprnfor=True, tstress=True,  # 必須: 力と応力
                kpts=self.kpts,
                # ...
            )
            embedded_atoms.calc = calc

            try:
                # 計算実行
                embedded_atoms.get_potential_energy()
                results.append(embedded_atoms)
            except Exception:
                # 収束失敗時のリトライロジックをここに記述
                pass
        return results

C. EONWrapper (kMC Integration):EONはクライアント・サーバーモデルで動作するが、ここではローカルで eonclient を実行し、ポテンシャル評価部分のみをフックする構成とする。ディレクトリ構造:active_learning/iter_XXX/
└── kmc_run/
    ├── config.ini          # EON設定ファイル
    ├── reactant.con        # 初期構造
    ├── potential.yace      # 現在のポテンシャル
    └── potentials/         # EON用ポテンシャルドライバ
        └── pace_driver.py  # EONからPacemakerを呼ぶスクリプト

EON設定 (config.ini) 自動生成:未知の遷移状態を探索するため、NEBではなくDimer法 (process_search) を指定する。[Main]
job = process_search    # Dimer法による探索
temperature = 300.0

[Potential]
potential = script
script_path = ./potentials/pace_driver.py

[Process Search]
min_mode_method = dimer

EON実行とOTF検知ロジック:pace_driver.py 内部でエネルギー計算と同時に $\gamma$ 値をチェックし、閾値を超えたら特定の終了コード（例: 100）でEONを強制終了させる。# pace_driver.py (EONから呼び出されるスクリプト)
import sys
from pyacemaker.calculator import PaceCalculator

# EONの仕様に合わせて座標読み込み -> 計算 -> 出力を行う
# (標準入力から座標を読み込み、標準出力にエネルギーと力を返す)

calc = PaceCalculator("potential.yace")
atoms = read_coordinates_from_stdin() # 擬似コード

gamma = calc.get_extrapolation_grade(atoms)

if gamma > THRESHOLD:
    # 高い不確実性を検知 -> 構造を出力して異常終了
    write_bad_structure("bad_structure.cfg", atoms)
    sys.exit(100)  # 特別な終了コード

# エネルギーと力を標準出力へ
print(calc.get_potential_energy(atoms))
print_forces(calc.get_forces(atoms))

Orchestrator側の制御:class EONWrapper:
    def run_kmc(self, work_dir):
        # EONクライアントを実行
        proc = subprocess.run(["eonclient"], cwd=work_dir)

        if proc.returncode == 100:
            # OTFイベント発生
            return {'halted': True, 'bad_structure': work_dir / "bad_structure.cfg"}
        return {'halted': False}

D. PacemakerWrapper (Dataset & Training):PacemakerはPython APIが公開されていない機能も多いため、CLIをラップするのが確実。import subprocess

class PacemakerWrapper:
    def update_dataset(self, new_atoms_list):
        # ASE atoms -> Pacemaker pckl.gzip 形式への変換
        # pace_collect コマンドなどを利用するか、内部フォーマットに合わせて保存
        pass

    def train(self, dataset, initial_potential, output_dir):
        cmd = [
            "pace_train",
            "--dataset", str(dataset),
            "--initial_potential", str(initial_potential),
            "--max_num_epochs", "50",  # Fine-tuningなので少なく
            "--output_dir", str(output_dir)
        ]
        subprocess.run(cmd, check=True)
        return output_dir / "output_potential.yace"

3.6 品質保証・検証機能 (Validator)作成されたポテンシャルが、学習データ以外の未知のデータや、マクロな物理特性に対しても妥当な予測を行うかを検証する、システム品質の「最後の砦（Quality Assurance Gate）」である。単に数値的な誤差が小さいだけでなく、物理的に正しい振る舞いをするかを多角的に診断する。検証プロトコル (Validation Suite):学習サイクル (Refinement) が完了するたびに、以下のテストスイートを自動実行し、総合的なスコアを算出する。Test Set Evaluation (汎化性能の数値評価):分割戦略: 全データセットを Training:Validation:Test = 80:10:10 にランダム分割、あるいは時系列分割する。特に時系列分割は、類似した構造（相関の強いデータ）がテストセットに含まれるのを防ぎ、真の汎化性能を評価するために推奨される。指標: 以下の目標値をクリアしているか確認する。Energy RMSE < 1-2 meV/atomForce RMSE < 0.03-0.05 eV/ÅStress RMSE < 0.1 GPa可視化: Parity Plot（正解 vs 予測）だけでなく、誤差の累積分布関数 (CDF) をプロットし、95パーセンタイル誤差が許容範囲内かを確認する。Dynamic Stability (Phonon Dispersion):目的: ポテンシャルが「見かけ上の安定点」ではなく、真に動力学的に安定な結晶構造を再現しているか確認する。手法: 外部ツール Phonopy と連携し、有限変位法を用いてフォノン分散関係（バンド構造）を計算する。スーパーセルは、長距離相互作用を捉えるのに十分な大きさ（例: $4 \times 4 \times 4$ 以上）を自動設定する。判定: ブリルアンゾーン全域において、虚数の振動数（Imaginary Frequency, $\omega^2 < 0$）が存在しないことを必須条件とする（ただし、$\Gamma$点近傍の微小な虚数モードは並進対称性の破れによる数値誤差として許容する）。Mechanical Stability & Elasticity (弾性・機械的特性):手法: 単位胞に独立な歪み（$\pm 1\%, \pm 2\%$）を印加し、エネルギー-歪み曲線を多項式フィッティングして弾性定数テンソル ($C_{ij}$) を算出する。安定性チェック: 結晶系に応じた Bornの安定条件（例: 立方晶なら $C_{11} - C_{12} > 0, C_{44} > 0, C_{11} + 2C_{12} > 0$）を満たしているか確認する。精度評価: Voigt-Reuss-Hill平均を用いて体積弾性率 ($B$) とせん断弾性率 ($G$) を算出し、DFT参照値との誤差が許容範囲内（通常 $\pm 10-15\%$ 以内）であることを検証する。Physical Stress Tests (極限環境テスト):EOS Curve (Birch-Murnaghan Fit): 平衡体積から $\pm 20\%$ の範囲で圧縮・膨張させ、得られたエネルギー曲線をBirch-Murnaghan状態方程式にフィッティングする。滑らかさ（微分の連続性）と、$V_0, B_0, B'_0$ の物理的妥当性をチェックする。Melting Behavior: 融点付近（$T_m \pm 100K$）でのNPT-MDを行い、二相共存法またはZ-methodを用いて融点を推定し、実験値またはDFT予測値と大きく乖離していないかを確認する。Automated Gatekeeper Logic (判定とアクション):各テストの結果に基づき、システムは以下のいずれかの決定を下す。PASS (Green): 全ての基準（RMSE, Phonon stability, Born criteria）をクリア。ポテンシャルを production フォルダへデプロイし、バージョン番号を更新する。CONDITIONAL (Yellow): 数値精度は良いが、一部の物理量（例: 弾性率）の誤差がわずかに大きい。警告付きでデプロイしつつ、次回の学習で重み付け調整 (Physics-Informed Loss) を予約する。FAIL (Red): 致命的な欠陥（虚数フォノン、Born条件違反、許容を超えたRMSE）が検出された。デプロイを中止し、失敗の原因となった構造領域（例: 特定の歪み方向）を特定して、Structure Generatorに追加のサンプリング指示を出す。Reporting:検証結果、Parity Plot、フォノンバンド図、EOS曲線、弾性率比較表をまとめた validation_report.html を自動生成し、ユーザーがブラウザで視覚的に品質を確認できるようにする。PYACEMAKER 次世代アーキテクチャ要求定義書 (PRD)Version: 2.1.0 (NextGen Hierarchical Distillation Architecture with FLARE Best Practices)Date: 2026-02-28Status: DRAFT1. プロジェクト背景と目的1.1. 現行システム（Phase 01）の限界と直面した課題Phase 01で実装されたPYACEMAKERの基本オーケストレーションは、Active Learningループの基礎を確立した。しかし、実用的なHPC規模（数万〜数百万原子の長時間分子動力学シミュレーション）への展開を想定した場合、以下の致命的な物理的・システム的制約が判明した。MD連続性の欠落 (Time-Continuity Break):不確実性（$\gamma$）が閾値を超えてHaltした際、再学習後にMDが初期構造から再スタートする設計となっており、相変態や拡散といった長時間の物理現象に到達できない。熱揺らぎによる過敏な停止 (Thermal Noise False Positives):単一の不確実性閾値でHaltを判定しているため、MD中の熱振動による瞬間的なスパイク（物理的には安全なノイズ）に過敏に反応し、不要な計算と無限ループを引き起こす。局所切り出しによる物理的破綻 (Dangling Bond / Dipole Divergence):巨大なシステムから不確実な領域（クラスター）を真空中へ単純に切り出してDFTに渡すと、切断面に大量のダングリングボンドが発生し、電荷の不均衡や双極子モーメントの発散を引き起こす。結果としてDFTのSCFループが収束しない、あるいは「ゴミ（非物理的な電子状態）」を学習してしまう。鈍重なバッチ再学習と破滅的忘却 (Batch Retraining & Catastrophic Forgetting):毎ステップ全データを用いてバッチ学習をやり直す設計は計算量（$O(N^3)$）の爆発を招き、かつエラー構造ばかりを追加することで、安定なバルク構造の記述精度が劣化する。LAMMPS連携の脆弱性 (System Fragility):C++レイヤーでのLAMMPSのクラッシュ（Lost atoms等）がPythonのメインプロセスを道連れにし、状態（State）の保存すら行われずにオーケストレーター全体が停止する。1.2. 先行研究（FLARE）からのベストプラクティス抽出ハーバード大等で開発されたFLAREアーキテクチャのコードベース解析から、以下の4つのパラダイムシフトを本システムに導入する。Master-Slave逆転 (Inversion of Control): PythonがLAMMPSを操作するのではなく、LAMMPSのC++ループ（fix python/invokeや定期的なコールバック）の内部にPythonを従属させ、MDの時間を巻き戻さずに「一時停止→ポテンシャル更新→再開」をシームレスに行う。二段階の不確実性閾値 (Two-Tier Thresholds): 「DFTを呼び出すための閾値（threshold_call_dft）」と「学習データに追加するための閾値（threshold_add_train）」を分離し、ノイズへの耐性を持たせる。計算は全体、学習は局所 (Global Calculation, Local Learning): 物理的に安定した環境で電子状態を計算し、学習モデルの更新には不確実性が高かった「中心原子の力（Force）」のみを用いる。逐次更新 (Incremental Update): バッチ再学習ではなく、過去の重みを初期値とした差分学習（Delta Learning/インクリメンタル更新）により、計算コストをO(1)に保つ。1.3. 次世代アーキテクチャの基本思想上記の課題とFLAREの教訓を統合し、「階層的蒸留（Hierarchical Distillation）」、「インテリジェント・クラスター抽出（Site-specific Cutout & Passivation）」、および**「非同期マスター・スレーブMD（Seamless Resume）」**を中核とする次世代ワークフローを定義する。FLAREが諦めた「数百万原子のMD」を可能にするため、「切り出さない」のではなく「物理的に完璧に修復して切り出す」アプローチを採用する。さらに、基盤モデル（MACE）からの蒸留とファインチューニングの活用をシステム全体で最大化することにより、最大のボトルネックである高コストなDFT計算の試行回数を極限まで最小化しつつ、ターゲット領域においてDFTに迫る高精度な計算を実現することを目指す。2. 4段階・階層的蒸留ワークフロー (Core Workflow)本システムは、対象となる系（例：Fe-Pt-Mg-Oの4元系）に対して、以下の4つのフェーズを順次実行する。Phase 1: ゼロショット蒸留と初期ポテンシャル構築（Zero-Shot Distillation & Baseline Construction）目的: DFTを一切呼び出さず、MACE-MP-0等の基盤モデルが持つ「宇宙の常識（広範な汎化性能）」を抽出し、ACEに焼き付ける。空間分解とコンビナトリアル探索:入力された元素群から、全単体（$N$種）および全二元系（${N}C{2}$種）のサブシステムを自動定義する。各サブシステムに対し、ランダム構造、歪み（Strain）、Rattle、高温スナップショットに加え、**多様な組成比の変動（Stoichiometry variation）**や、欠陥（空孔、格子間原子、アンチサイト等のディフェクト）の導入を網羅した構造プールを生成する。DIRECTサンプリングによる情報量最大化 (Active Set Selection):既存資産である ActiveSetSelector（DIRECTサンプリング / D-Optimality）を活用し、生成された膨大な構造プールから特徴空間において最も情報密度が高く多様性に富んだ構造（数百〜数千個）を抽出する。これにより、冗長なデータを排除し、後段の推論・学習コストを極限まで抑える。確信度フィルタリング:抽出された構造を MACEManager（基盤モデルOracle）に渡し、エネルギー・力・不確実性を推論する。この際、MACEの不確実性が閾値以下の「MACEが自信を持っている安全な構造」のみを正解データとして採用する。ベースラインACE学習 (LJ Delta Learning):2, 3を通じてchemical spaceを十分に広い範囲で、なおかつ確信度フィルタリングを通過した高品質なデータを用いて PacemakerTrainer を起動し、基礎的な多体ポテンシャル（base.yace）を学習させる。この際、超近接領域での原子のすり抜けを防ぐ（物理的破綻を避ける）ための既存機能として、Lennard-Jones (LJ) ポテンシャルからの Delta Learning をデフォルト構成として適用し、短距離反発のベースラインを担保する。Delta Learningは既存のクラスなどを活用し、パラメータを元素によって最適化する機能などを適宜活用することPhase 2: 限界テストと物理バリデーション（Validation & Stress Test）目的: Phase 1で構築された基礎ポテンシャルが、本番環境で最低限の物理的安定性を担保できるか検証する。母物質の物理特性検査:Validator を起動し、各サブシステムの安定相（例: bcc-Fe, fcc-Pt, NaCl-MgO）に対する弾性定数（Bornの安定性基準）、フォノン分散（虚数振動の不在）、および状態方程式（EOS）を計算する。合格基準に達しない場合はPhase 1のサンプリング密度を自動で上げて再学習する。ミニチュアMDによるストレステスト:本番環境の縮小版（数千原子程度のスラブモデル等）を作成し、構築したポテンシャルでMDを走らせる。早期にHaltが発生するか、どの温度帯で不確実性が高まるか（Uncertainty Map）をプロファイリングする。Phase 3: インテリジェント・クラスター抽出（Intelligent Cutout & Passivation）目的: 大規模MD中に未知の局所構造（界面、欠陥、衝突）に遭遇しHaltした際、DFTで計算可能な物理的に妥当なクリーン・クラスターを自動生成する。二段階閾値に基づく震源地特定 (Two-Tier Evaluation):MDのシステム最大不確実性が threshold_call_dft を数ステップ連続で超えた場合にのみHaltを発動させる（熱ノイズの排除）。その後、既存の _get_max_gamma_atom_index などの機能を拡張・活用して個別の原子の不確実性（Site-uncertainty）を評価し、threshold_add_train を超えている原子群を「震源地」として特定する。球状切り出しと重み付け (Local Learning):既存資産である utils.extraction.extract_local_region をそのまま活用し、震源地から半径 $R_{core}$ 内の原子に force_weight = 1.0 を、半径 $R_{buffer}$ 内の原子に force_weight = 0.0 を付与する。さらに既存の utils.embedding.embed_cluster 等を呼び出し、切り出したクラスターを真空層付きの周期境界セル（PBC）に安全に再配置する。これにより学習対象をコアのみに絞り込む。MACEによる境界事前緩和（Pre-relaxation）:既存のMLIPラッパー機構（現行の m3gnet_wrapper.py の枠組み）を基盤モデル（MACE）用に拡張し適用する。**コア原子の座標は固定（Freeze）**したまま、MACEを用いてバッファ領域の原子の座標のみをエネルギー極小化（Relax）する。これにより切り出し時の不自然な結合歪みを解消する。自動終端処理（Auto-Passivation）:バッファ領域外縁の切断された結合（特に酸化物のOやMgなど）に対し、Fractional Hydrogen等のダミー原子を自動配置し、クラスター全体の電荷とダイポールモーメントを中性化する（utils.structure等に新規統合）。クリーンDFT計算:物理的・電気的に安定化されたクラスターを、既存資産である interfaces.qe_driver.QEDriver および core.oracle.DFTManager に渡す。既存の自己修復（Self-Healing）機能（Smearingの拡張やMixing Betaの自動調整）をフル活用し、SCFを確実に収束させてコア原子に対する真の力（Ground Truth Force）を取得する。Phase 4: 階層的ファインチューニング（Hierarchical Delta Learning）目的: 取得した貴重な少数のDFTデータを用いて、MACEとACEを連鎖的にアップデートし、MDを再開する。MACEの覚醒（Finetune MACE）:取得したDFTデータを用いて、MACE自体をファインチューニングする。これにより基盤モデルが「対象系の特異な界面物理」を完全に理解した状態（覚醒MACE）となる。サロゲートデータの爆発的生成:覚醒MACEをOracleとして用い、Haltした周辺のフェーズ空間（ランダム変位や微小MD）で数千個のサロゲートデータを一瞬で生成・推論する。ACEのデルタ学習 (Incremental Update):大量のサロゲートデータと、アンカーとなるDFT真値データを PacemakerTrainer に入力し、ACEポテンシャルを更新する。この際、計算量爆発を防ぐためゼロからの学習は行わず、直前のポテンシャルからの差分学習とし、さらにリプレイバッファを混入させる。シームレス・レジューム (Master-Slave Resume):更新されたポテンシャルを読み込み、Haltした直後のステップ（時間、座標、速度）からMDを安全に再開する。3. モジュール別要求仕様3.1. pyacemaker.utils.extraction (大幅拡張)クラスター抽出と終端処理を担う、本アーキテクチャの要となるモジュール。extract_intelligent_cluster(structure: Atoms, target_atoms: List[int], config: ExtractionConfig) -> Atoms入力: 巨大なASE Atomsオブジェクト、threshold_add_trainを超えた対象原子のインデックスリスト。処理:neighbor_listを用いた $R_{core}$ と $R_{buffer}$ の球状抽出。force_weight 配列の付与（Core=1.0, Buffer=0.0）。_pre_relax_buffer(cluster, mace_calc): コアを ase.constraints.FixAtoms で固定し、MACEを用いてバッファをLBFGSで緩和する。_passivate_surface(cluster): 電気陰性度と結合半径から未結合手を検出し、適宜Hまたは擬似原子を追加する（追加原子の force_weight は当然0.0）。出力: 周期境界を持ち、真空層が挿入され、終端処理が施された計算可能な Atoms オブジェクト。3.2. pyacemaker.core.oracle (多段化)Oracleを抽象化し、基盤モデル（MACE）と第一原理計算（DFT）を透過的に扱えるようにする。class MACEManager(BaseOracle)MACE-MP-0等の推論を実行するラッパー。GPU対応。エネルギー、力に加え、アンサンブル分散や潜在特徴空間の距離に基づく不確実性（Uncertainty）を出力する機能が必須。class TieredOracle(BaseOracle)クエリ戦略を管理する。構造を受け取った際、まずは MACEManager で推論し、不確実性が特定の閾値を超えた場合のみ QEDriver (DFT) にフォールバックするルーティングロジックを持つ。3.3. pyacemaker.core.engine (LAMMPS連携とシームレス再開)LAMMPSのクラッシュに耐え、Halt後の時間を連続させる堅牢なエンジン。FLAREのMaster-Slaveパラダイムを適用。fix python/invoke の活用 (推奨アプローチ):LAMMPSのC++実行ループから直接Pythonの検証スクリプトを毎Nステップ呼び出す。不確実性が閾値を超えた場合はMDをポーズ状態にし、バックグラウンドでOrchestrator（学習パイプライン）を走らせる。完了後、pair_coeffを動的にリロードしてMDを継続する。Process Isolation と read_restart (フォールバック・アプローチ):C++連携が技術的課題となる場合は別プロセス化する。LAMMPSがクラッシュしてもメインループは生き残り、定期保存された.restartファイルから速度分布とアンサンブル状態を完全に引き継いで再開する。ソフトスタート（温度急上昇防止）:ポテンシャルが切り替わった直後のエネルギー不連続による系の破綻を防ぐため、再開直後の $N$ ステップは強いLangevin熱浴（fix langevin）をかけて系を熱化（Thermalize）させるロジックを自動挿入する。3.4. pyacemaker.core.trainer (Pacemaker & MACE Finetune)FinetuneManager:DFTから取得したクリーンなデータセットを用いて、MACEのPyTorchモデルの最終層付近（Readout layer）を短時間学習させるラッパー。PacemakerTrainer のインクリメンタル更新・デルタ学習強化:バッチ学習の計算量爆発を防ぐため、前回のポテンシャル状態を引継ぎ、過去の学習データ（training_history.extxyz）からランダムにサンプリングした固定サイズのリプレイバッファを現在の学習セットに混合する。LJポテンシャルからのDelta Learningを実行するための設定を input.yaml に自動生成する機能を担保する。4. データモデル要件 (domain_models/config.py)新たなワークフローを制御するためのPydanticモデルの拡張。class DistillationConfig(BaseModel):
    """Phase 1: Zero-Shot Distillation設定"""
    enable: bool = True
    mace_model_path: str = "mace-mp-0-medium"
    uncertainty_threshold: float = Field(0.05, description="MACEが自信を持つ閾値")
    sampling_structures_per_system: int = 1000

class ActiveLearningThresholds(BaseModel):
    """FLAREにインスパイアされた二段階閾値"""
    threshold_call_dft: float = Field(0.05, description="MDをHaltしてDFTを呼び出す基準")
    threshold_add_train: float = Field(0.02, description="学習セットに追加する原子を選ぶ基準")
    smooth_steps: int = Field(3, description="熱ノイズ排除のため、閾値超過が連続するべきステップ数")

class CutoutConfig(BaseModel):
    """Phase 3: インテリジェント切り出し設定"""
    core_radius: float = Field(4.0, description="Force Weight 1.0の半径")
    buffer_radius: float = Field(3.0, description="追加の緩和バッファ層の厚さ")
    enable_pre_relaxation: bool = True
    enable_passivation: bool = True
    passivation_element: str = "H"

class LoopStrategyConfig(BaseModel):
    """Active Learning ループの戦略設定"""
    use_tiered_oracle: bool = True
    incremental_update: bool = True
    replay_buffer_size: int = Field(500, description="破滅的忘却を防ぐための過去データ保持数")
    baseline_potential_type: str = Field("LJ", description="ベースラインとなる物理ポテンシャル (LJなど)")
    thresholds: ActiveLearningThresholds = Field(default_factory=ActiveLearningThresholds)
5. 非機能要件・HPC運用要件5.1. ステート管理とトランザクション (Robust Checkpointing)Task-level Checkpointing:イテレーション単位の粗い保存ではなく、DFTの1計算、サロゲートの1生成ごとにJSONまたはSQLiteベースのローカルDBに状態をコミットする。HPCジョブがWall-time（実行時間制限）で強制キルされても、再投入時に秒単位でレジューム可能とする。Artifact Cleanup:数百万ステップのMDから生成されるダンプファイルやQEの巨大な波動関数ファイル（.wfc）は、学習・推論に成功した直後に自動的に圧縮（gzip）または削除するデーモンプロセスを並行稼働させる。5.2. スケジューラ連携と並列化 (HPC Dispatch)Oracle（DFT計算）は直列実行ではなく、concurrent.futures や Dask 等を用いて、利用可能なノード/GPUへ非同期にディスパッチする。PacemakerTrainer のサブプロセス呼び出し時、HPC環境（Slurmの srun, PBSの mpiexec）のプレフィックスを環境変数から動的に組み立てるジョブテンプレート機能（JobDispatcher）を実装する。6. 実装フェーズ（マイルストーン）提案Sprint 1: Core Extraction & Two-Tier Evaluatorextraction.py の再設計。MACEによるPre-relaxationとH終端のアルゴリズム実装・テスト。二段階閾値判定（Call DFT vs Add Train）のロジック構築。Sprint 2: Master-Slave Inversion & Seamless Resumefix python/invokeを用いたLAMMPSとの密結合、または .restart ファイルを用いた堅牢なMD継続機構の実装。LJベースのDelta Learning設定の統合。Sprint 3: Hierarchical Distillation LoopOrchestratorの書き換え。Phase 1〜4のフローを統合し、MACEManager と TieredOracle を接続する。差分学習（Incremental Update）の導入。Sprint 4: Scale & RobustnessSQLiteベースの細かいチェックポイント機能と、マルチノード実行向けの非同期ディスパッチ機構の導入。End of Document
