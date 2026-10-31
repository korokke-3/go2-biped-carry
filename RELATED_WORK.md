# 関連研究・リポジトリ:2026-10-16 調査

## 後ろ足で立つ・二足歩行
| 名前 | ロボット / 環境 | 要点 | コード / ライセンス |
|---|---|---|---|
| Bipedalism for Quadrupedal Robots, arXiv 2507.20382, Humanoids 2025 | Go2 / Isaac Gym | リスク適応型の分布型 PPO。後ろ足歩行で、台車押し・障害物の探索・荷物の運搬を実機で実現。重心を後ろ足の上に保つ報酬。前腕への 10 N の外力で評価 | なし |
| Simulator Adaptation, arXiv 2604.11090 | Go2 / Isaac Lab | 後ろ足歩行。実機では横方向のずれが大きかったが、残差アクチュエータモデルで改善。X 誤差 1.2 m → 0.2 m | なし |
| H-Infinity Locomotion, arXiv 2404.14405, CoRL 2024 | A1 | 後ろ足歩行。外力を出す「妨害役」を方策と敵対的に学習 | HIMLoco 内、CC BY-NC-SA |
| CyberDog2, arXiv 2311.05818, ICRA 2024 | CyberDog2 | 四本足から立ち上がる。遅延 5〜30 ms のランダム化 | — |
| arclab-hku/bipedal_locomotion_for_quadrupedal_robots | Go1 / Isaac Gym | 模倣ベースの二足歩行 | ライセンス不明 |
| M1KELONG/My_go2_lab | Go2 / Isaac Lab | 逆立ち・片足支持・起き上がり。sim2sim まで | ライセンス不明 |
| Sang-SC/go2_quad2hand | Go2 / Isaac Gym | **前足**で立つ逆立ち歩行。後ろ足の接地に減点。MuJoCo と C++ の実機用制御プログラム。状態推定ネットワークで、速度と、重心と圧力中心のずれを推定 | BSD-3 |

## 未知の荷物・外力への強さ
| 名前 | 要点 | コード |
|---|---|---|
| FALCON, arXiv 2505.06776 | 人型。手先にかける外力の上限を、関節トルクの限界から計算。方向を Dirichlet 分布で配分し、全体の強さをカリキュラムで 0→1 に上げる。作用点もランダム化。方策は5ステップ履歴、価値関数は外力を見る | LeCAR-Lab/FALCON、MIT |
| MULE, arXiv 2505.00488 | Go1。基本の方策に補正行動を足す適応方策。関節トルクから足裏の力を計算して観測に追加。荷物 0〜2.5 kg を4秒ごとに入れ替え | なし |
| HIMLoco / DreamWaQ | 履歴から状態を推定する定番の手法。履歴は5ステップ | HIMLoco、CC BY-NC-SA |

## 載せて運ぶ:つかまない運搬
| 名前 | 要点 | コード |
|---|---|---|
| SteadyTray / ReST-RL, arXiv 2603.10306 | 人型 G1 がトレイを運ぶ。学習済みの歩行方策を固定し、残差の補正モジュールを足す構成。物体 0.05〜1.0 kg | AllenHuangGit/steadytray、Apache-2.0、Isaac Lab |
| LocoTouch, CoRL 2025 | 四足の背中に円柱を載せて運ぶ。Isaac Lab で物体を剛体としてランダムに生成 | linchangyi1/LocoTouch、MIT |

## 本プロジェクトへの反映
- **荷物を前腕への外力として扱う方式**:2507.20382 / FALCON。`Unitree-Go2-Biped-Load` で実装済みです。
- **重心を後ろ足の上に保つ報酬**:2507.20382。`loaded_com_over_hind_feet` で実装済みです。
- **方策は固有受容の履歴、価値関数は実際の荷重**:FALCON / MULE。固有受容は関節と IMU です。実装済みで、履歴は3ステップです。
- 今後の候補は次のとおりです。
  - 外力の方向と作用点のランダム化:FALCON。
  - 関節トルクから計算した前腕の荷重を観測に追加:MULE。
  - 残差の補正モジュール:SteadyTray。
  - 実際の箱を載せる学習:LocoTouch / SteadyTray。
  - リスク適応型の PPO:2507.20382。
