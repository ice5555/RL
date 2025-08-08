太好了！现在**训练—评估—画图**链路都通了，接下来就是把实验做“像论文”。我给你一份**两阶段路线图**（本地→Colab/GPU），每阶段要跑哪些实验、怎么记录、预期产出是什么。你照着清单执行就行。

# 阶段 A｜本地（MPS/CPU）——把质量提起来

目标：从“EM/F1≈0”到至少有可见提升；定稿日志/图表格式。

## A1. 提升可答对率（最重要）

* 模型切换：`google/flan-t5-small`（mps），`max_new_tokens: 32`
* Prompt 收紧：结尾用

  > “Answer ONLY with the correct answer, as a short phrase, based strictly on the provided CONTEXTS.”
* 训练量：

  * `data.sample: 200`
  * `train.total_timesteps: 5_000`
  * `ppo.n_steps: 32`，`batch_size: 32`
* 评估频率：`eval.freq: 200`，`eval.episodes: min(100, len(dev))`

## A2. 奖励塑形（让 RL 真有信号）

* 文本归一：小写、去标点/冠词、压缩空格（EM/F1 计算前做）。
* 奖励：`reward = 1.0*EM + 0.5*F1 - 0.05*latency`
  观察 reward 分布：如果还是全 0，就把 F1 系数暂时拉到 1.0。
* 记录：在 `train_log.csv` 多写一列 `em_raw/f1_raw`（可选），方便排查。

## A3. 检索质量与动作空间

* 先测 Retrieval Recall（你已有）：确保 @5 ≥ 0.9。
* 把 `rerank_on=1` 接上 `CrossEncoder`（`cross-encoder/ms-marco-MiniLM-L-6-v2`）

  * 动作空间继续包含 `rerank_on ∈ {0,1}`
  * `info["latency_sec"]` 会变大，看看 RL 会不会学会只在必要时打开 rerank。
* 如果生成太慢：把 `temperature` 候选减少为 `{0.0, 0.7}` 先。

## A4. 固定日志 & 图表

* 你现在已有：

  * 训练日志：`train_log.csv`（per-episode）
  * 训练期评估：`eval_log.csv`（per N steps）
  * baseline：`eval_baseline_*.txt`
  * 画图：`plot_rl_with_baseline.py`
* 再加两张表（论文用）：

  1. **表 1：Baseline vs PPO（dev 集）**
     列：EM、F1、Avg Reward、Latency（取 `eval(full)`）
  2. **表 2：消融**（有没有 rerank、不同 top\_k 动作集合）

> 阶段 A 结束标志：
>
> * eval 曲线能看到**reward 上升**（最好 > fixed baseline）
> * EM/F1 有**非零**点（即使很小也行）
> * 日志/图表格式固定

# 阶段 B｜Colab/GPU——放大样本 & 跑对比

目标：拿到可写论文的主结果图与对比表。

## 迁移到 Colab 的时机

* 当本地跑 **FLAN + n\_steps≥32 + total\_timesteps≥5k** 能看到趋势，就上 Colab。
* 你需要的就是更大样本和更久训练。

## Colab 计划

* 运行环境：T4/L4/RTX（T4 也够）
* 代码准备：

  ```bash
  !git clone <你的仓库>
  %cd RL
  !pip install -q -r requirements.txt
  ```
* 模型：先 `flan-t5-small`，如顺利再试 `flan-t5-base`（小心显存）
* 规模：

  * `data.sample: 2,000`（或直接用 dev\_fullwiki 的子集）
  * `train.total_timesteps: 50_000 ~ 100_000`
  * `ppo.n_steps: 64`，`batch_size: 64`
  * `eval.freq: 1_000`，`eval.episodes: 200`
* 产物：

  * 完整的 `train_log.csv / eval_log.csv`
  * 主图：**Eval Reward/EM/F1 vs Timesteps**（叠 baseline）
  * 次图：**Action 分布**、**Latency 曲线**
  * 主表：Baseline vs PPO（dev/test）

## 对比实验矩阵（建议最少跑这些）

* **Prompt**：基本 vs 强约束（两行）
* **Rerank**：off/on（两行）
* **动作空间**：`top_k` 集合 `{5,10,20,50}` vs `{10,20}`（两行）
* **奖励系数**：`(EM=1,F1=0.5)` vs `(EM=1,F1=1)`（两行）

> 结果写法：曲线 + 汇总表（均值 ± 标准差，跑 3 个不同 seed）

# 可选扩展（有时间再做）

* **SFT/DPO** 作为初始化（warm start）：

  * 用 5k-20k 热身样本对 `flan-t5-small` 做 SFT
  * DPO（对比采样）需要偏好的成对数据，成本高；放弃也行
  * 把 “SFT 初始化的 PPO” vs “直接 PPO” 做一组对比
* **GRPO**：如果你想展示“RL for LLM”多算法对比，可以把 GRPO 做一次短跑（只需一条曲线 + 一行表），方法上和 PPO 近似，但实现工作量比 SFT/DPO 大，谨慎评估时间。

# 实验记录与复现

* 统一命名：`run_name = <model>-<nsteps>-<topkset>-<rerank>-<seed>`
* 固定种子：`seed in {42, 43, 44}`
* 每次实验保存：

  * `config.yaml`（当次运行配置）
  * `train_log.csv / eval_log.csv`
  * `plots/*.png`
* 结果收敛后，把**论文图**和**最终表**导出到 `paper_figs/`，避免被后续实验覆盖。

# 你现在要做的三件事

1. 把 config 切回 **FLAN + mps**，跑一个中等规模（`sample=200`, `steps=5000`），确认 EM/F1 > 0。
2. 打开 **reranker**（让 `rerank_on=1` 真正生效），再跑一遍，比较两条曲线。
3. 把这两次的 `eval_log.csv` 画成主图（timesteps vs EM/F1/Reward），看看提升是否稳定。

如果你愿意，我可以直接给你**一键切换到 FLAN 的配置**和**把 reranker 接入 QASystem 的补丁**，然后你在本地先小跑验证一下再迁 Colab。





python -m cli.train_rl --config config/hotpotqa_rl.yaml --baseline_only
python -m cli.train_rl --config config/hotpotqa_rl.yaml --baseline_only --baseline_mode random





python -m cli.train_rl --config config/hotpotqa_rl.yaml --baseline_only
python -m cli.train_rl --config config/hotpotqa_rl.yaml --baseline_only --baseline_mode random



python scripts/plot_rl_with_baseline.py