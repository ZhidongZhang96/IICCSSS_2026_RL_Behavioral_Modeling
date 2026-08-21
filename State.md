# State — 机制主线研究状态

> 更新于 2026-08-21。本文档对应 `main_mechanism.ipynb`（问题驱动的机制分析），
> 全部"当前结果"来自**缩减配置冒烟运行**（hidden 16、3 epochs、train 60 sessions、
> per-condition cap 30–40、bootstrap 100–200），只验证管道，**不能下结论**。

## 1. 研究问题

**核心问题（不预设结论）**：不同任务之间理应存在共享的学习机制——那具体的差异体现在哪里？

- **Q1（差异定位，共享模型内）**：条件差异落在哪一层——表示（`h_t`）、更新动态
  （`z`/`r` gates）、还是读出（策略/NLL）？
- **Q2（共享 vs 分别训练）**：若每个条件单独训练一个 GRU，拟合代价、gate 动态、
  表示对齐会发生什么？差异是条件固有需求，还是共享容量的假象？
- **Q3（可拓展性）**：同样的分析切换到其他条件特征（`n_actions` / `points_type` /
  `probs_type` / `block_change_type`）时，结论是否一致？

评估统一为 **teacher-forced 固定历史（人类轨迹）**，只回答"条件策略与内部机制的差异"，
不回答实际绩效（closed-loop / ground-truth 在 `main_ground_truth.ipynb`）。

## 2. 分析方法

| 环节 | 方法 | 指标 |
|---|---|---|
| 共享模型 | 设计级切分（task_number），人类数据训练 GRU，held-out 上 teacher-forced replay | NLL/pseudo-R²、P(best) 曲线 |
| 表示层 | 线性 probe（session 级切分）+ LDA 判别投影 vs PCA + per-unit selectivity + top-k 可解码性 | val acc vs chance、between/within 比、selectivity ∈ [0,1] |
| 动态层 | 每条件 z/r 均值曲线；两两 gate 距离（bootstrap 95% CI，session 重采样） | mean\|Δz(t)\|、mean\|Δr(t)\| |
| 读出层 | 按条件分组的 held-out NLL/pseudo-R² 与 P(best) 曲线 | 每条件指标表 |
| 分条件训练 | `train_gru_per_condition`（同超参、cap 平衡、设计级切分） | 每条件模型 |
| 拟合代价 | 同 session 配对 ΔNLL = shared − per-condition | 均值 + bootstrap CI |
| 动态对比 | per-condition 模型之间、以及与共享模型（同 session）的 gate 距离 | 距离 + CI |
| 表示对齐 | shared-probe 迁移（同一读出套到各模型 h）+ CCA（同 session/trial 对齐） | 迁移 acc、CCA top1/mean top8 |
| 合并可视化 | 共享 vs 分条件模型的 h 各做一次 PCA/UMAP/t-SNE，按条件着色；分条件模型的 gate 曲线 | 两两并排图 |

端到端封装：`utils.per_condition_report(train_df, val_df, test_df, by, shared_model, ...)`
一次输出训练、latents、配对 NLL、gate 距离、probe 迁移、CCA 与合并嵌入；
`CONDITION_FEATURE` / `scan_features` 控制按哪个特征分析。

## 3. 预期结论（判读表）

**Direction 1（差异定位）**

| 观察 | 结论 |
|---|---|
| 条件可从 h 解码（probe/LDA 显著高于 chance），且不随结构控制特征消失 | 表示层编码条件 |
| 可解码但 ≈ n_actions/n_states 控制 | 差异由任务结构驱动，而非条件特征本身 |
| 仅 gates 显著不同 | 差异在更新动态（参数性） |
| 仅 NLL/P(best) 显著不同 | 差异在读出或数据难度 |
| 三层皆无差异 | 共享机制主导，条件对模型影响小 |

**Direction 2（共享 vs 分别训练）**

| 观察 | 结论 |
|---|---|
| per-condition NLL 更优（CI>0）且动态/表示不同 | 条件确实需要不同机制，共享模型付真实混合代价 |
| per-condition ≈ 共享且 CCA 高 / probe 可迁移 | 共享机制足够 |
| per-condition 更差 | 条件内样本不足/过拟合，解释受限 |
| 分条件模型彼此 gate 不同 | 条件的最优动态固有不同 |
| CCA 高但 gate 值不同 | 参数性差异（同一机制、不同节奏） |
| CCA 低 | 结构性差异（不同子空间/专用机制） |

## 4. 当前结果（缩减配置冒烟）

### 4.1 visibility — Direction 1（共享模型）

| 指标 | 数值 |
|---|---|
| 解码 visibility 的 probe val acc（full hidden） | **0.821**（chance 0.25） |
| top-8 选择性单位 probe val acc | **0.777**（≈full，条件信息较局部化） |
| LDA between/within 比 | 0.227 |
| 2-D 投影线性 val acc | PCA **0.388** vs LDA **0.475** |
| gate z 两两距离 | 0.0006–0.012（actions vs 其余 0.011–0.012，其余对 ≤0.0012） |
| gate r 两两距离 | 0.25–0.83（actions 与其余差距最大 0.375–0.825） |

要点：表示层明显编码条件（0.82）；2-D 线性投影分离一般（"有模式但不清楚"的量化）；
update gate z 几乎不区分条件，**reset gate r 差异大且以 actions 为主**——候选机制偏向
"共享表示 + 条件化的 reset 动态"。

### 4.2 visibility — Direction 2（分条件训练）

| 指标 | 数值 |
|---|---|
| per-condition 训练 NLL/trial（train） | actions 1.072 / states 1.184 / sta 1.064 / sta_actions 1.120 |
| per-condition vs 共享模型 gate 距离（同 session） | z 0.0004–0.0015，r 0.0001–0.0022（极小） |
| 分条件模型两两 gate 距离 | z 0.0009–0.013（actions vs 其余大）；r 0.25–0.83（模式与共享模型内一致） |
| probe 迁移（shared→per） | actions 0.749→**0.650**（掉 0.10）；其余几乎不掉（0.584/0.597/0.592 → 0.584/0.595/0.597） |
| CCA（shared vs per，同 session） | top1 0.984–0.995；mean top8 0.803–0.937 |

要点：单独训练后 gate 模式与共享模型几乎相同 → 条件的动态差异更可能是**固有需求**而非
共享容量假象；CCA 高 → 共享线性子空间；**actions 的 probe 迁移掉点**是需要盯着的反例
（可能与其 n_states=1 的退化结构有关）。

### 4.3 probs_type — Direction 1（`CONDITION_FEATURE` 切换验证）

| 指标 | 数值 |
|---|---|
| 解码 probs_type 的 probe val acc | **0.922**（注：子样本 chance 显示 0.333，4 类里最小类可能未被抽到，见注意事项） |
| top-8 单位 probe val acc | 0.818 |
| LDA between/within 比 | 0.268 |
| 2-D 投影线性 val acc | PCA **0.431** vs LDA **0.501** |
| gate z 两两距离 | binary_and_stable vs 其余 0.014–0.016，其余对 0.0008–0.0018 |
| gate r 两两距离 | 0.34–1.76（all_1 vs continuous_and_drifting 最大 1.76） |

### 4.4 probs_type — Direction 2（分条件训练）

| 条件 | 配对 ΔNLL（shared−per） | 95% CI | CCA mean top8 | probe 迁移（shared→per） |
|---|---|---|---|---|
| all_1 | **+1.31** | 0.11–2.46 | 0.943 | 0.673→0.656 |
| binary_and_stable | −0.13 | −0.48–0.20 | 0.914 | 0.720→0.713 |
| continuous_and_drifting | −0.29 | −1.33–0.62 | 0.922 | 0.558→0.553 |
| continuous_and_stable | +0.01 | −0.11–0.11 | 0.914 | 0.626→0.625 |

另：per-condition vs 共享模型 gate 距离极小（z 0.0002–0.0045，r 0.0003–0.0027）；
分条件模型两两 gate 距离模式与共享模型内一致（z 0.0018–0.0202，r 0.34–1.76）。

要点：**除 all_1 外没有显著混合代价**；all_1（确定性奖励）单独训练拟合明显更好，
与 Direction 1 中 all_1 的 gate r 距离最大一致——确定性条件可能是最"特殊"的一类。

### 4.5 per-feature scan

`main_mechanism.ipynb` 末尾"Per-Feature Scan"节（循环 `scan_features`，每个特征
调用 `per_condition_report` 并输出配对表、合并嵌入、gate 曲线）—— 部分feature的hidden space确实有cluster pattern，但需要考虑模型训练的variability，需多次训练以得到stable的solution。

## 5. 注意事项与下一步

- **以上均为缩减配置冒烟**：hidden 16、3 epochs、train 60 sessions、cap 30–40、
  bootstrap 100–200。正式结论需全量配置（hidden 32、epochs 8+、cap 150、n_boot 2000、多 seed）。
- probs_type 的 probe 子样本疑似丢了最小类（chance 显示 0.333 而非 0.25）：需调大
  `probe_sessions` 或保证每个类都被抽到。
- 小样本条件的 bootstrap CI 偏宽（如 all_1 仅 36 个 test sessions）。
- actions 条件恒 n_states=1，visibility 与任务结构混杂；归因时需匹配子集或显式标注。
- 待办（主线不 work 再启用）：时间分辨/事件条件化分析、完整混杂匹配、hidden 维度扫描、
  closed-loop / ground-truth 方向。
