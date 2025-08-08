绝了—we’ve got the whole pipeline running locally ✅
下一步就可以把它搬到 Colab 跑正式实验了。我给你一份可执行的实验计划 + 跑法清单，从准备环境 → 基线 → PPO → 消融 → 画图出表，一条龙。

实验目标（论文结论想证明啥）
	1.	RL 能提升 RAG 问答质量：PPO 相比 fixed / random baseline 至少在 Reward 上超过，最好 EM/F1 也抬头。
	2.	哪些因素最关键：动作空间、奖励塑形、重排序、Prompt、模型规模等。
	3.	可复现：不同随机种子结果稳定；有清晰日志和图表。

计算设置
	•	Colab：T4/A100/GPU（优先选择 A100 Colab Pro，如果有）
	•	数据：HotpotQA（已在仓库 data/）
	•	训练/评估样本：
	•	快速跑：sample=1k
	•	正式跑：sample=5k（时间允许再更大）
	•	模型：google/flan-t5-small（起步）→ 如有预算再尝试 flan-t5-base

Colab 运行手册（最短路径）

# 0) GPU
!nvidia-smi
!python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 1) 拉代码（你自己仓库）
!git clone <your_repo_url> RL
%cd RL

# 2) 安装
!pip install -q transformers datasets sentence-transformers rank-bm25 evaluate faiss-cpu stable-baselines3

# 3) 关掉 tokenizer 并行警告（脚本里也有兜底）
%env TOKENIZERS_PARALLELISM=false

# 4) 快速 Sanity：只跑 baseline
!python -m cli.train_rl --config config/hotpotqa_rl.yaml --baseline_only

# 5) 正式训练（先 5k timesteps，确认日志在增）
!python -m cli.train_rl --config config/hotpotqa_rl.yaml

# 6) 画图（含 baseline + eval 曲线）
!python scripts/plot_rl_with_baseline.py

建议把 config/hotpotqa_rl.yaml 切成Colab 版（示例）：

paths:
  dataset_json: data/hotpot_train_v1.1.json

data:
  sample: 1000
  cache_dir: .cache/prep

model:
  name: google/flan-t5-small
  device: cuda
  max_new_tokens: 32

train:
  total_timesteps: 20000

ppo:
  n_steps: 64
  batch_size: 64
  gamma: 0.99
  learning_rate: 3e-4

baseline:
  mode: fixed
  episodes: 200
  fixed: {top_k: 20, temperature: 0.7, rerank_on: 0}

eval:
  freq: 500
  episodes: 200

实验矩阵（主线 + 消融）

最少版你可以做 6 组，时间足够再扩展到 10–12 组：

主线
	1.	Baseline-fixed：固定 (top_k=20, temp=0.7, rerank=0)，作为强基线
	2.	Baseline-random：随机动作
	3.	PPO（默认动作空间）：top_k ∈ {5,10,20,50}；temp ∈ {0.0,0.3,0.7,1.0}；rerank_on ∈ {0,1}

消融
	4.	无重排序：动作空间去掉 rerank_on（或强制 0）
	5.	有重排序：rerank_on=1 时启用 Cross-Encoder（ms-marco-MiniLM-L-6-v2）
	6.	奖励塑形对比：
	•	A：reward = EM + 0.5*F1 - 0.05*latency（当前）
	•	B：reward = 0.5*EM + F1 - 0.05*latency（更看重 F1）
	•	C：reward = F1（去掉 latency，看趋势）
（奖励切换可以用 config/flag 控）。

扩展项（看时间）：
	•	Prompt 变体（强约束 vs 弱约束）
	•	模型规模（flan-t5-small → base）
	•	动作空间加 multi-query（例如两路 query 合并检索）

评估与记录
	•	训练期：
	•	.cache/rl_logs/train_log.csv（每 episode：reward / em / f1 / latency / action）
	•	.cache/rl_logs/eval_log.csv（每 eval.freq steps：avg_reward / avg_em / avg_f1 / avg_latency）
	•	训练后：
	•	Eval(3eps) 快速 sanity
	•	Eval(full) = min(200, eval_df)
	•	多次运行：每次实验写到独立目录，如 .cache/rl_runs/run_YYYYMMDD_HHMM/（可以在 CLI 增加 --run_name）

图表与表格（论文用）
	•	训练曲线（Episode 轴）：Reward（含 baseline 线）、Latency
	•	评估曲线（Timesteps 轴）：avg_EM、avg_F1、avg_Reward（含 baseline 线）
	•	动作分布：top_k、temperature 柱状
	•	汇总表：每组实验的 final avg_EM/F1/Reward/Latency（来自 eval_log.csv 最后一行或训练后 Eval(full)）

复现实验的命名与脚本

建议做个最简壳脚本 scripts/launch.sh（或 Colab 单元）：

# 1) baseline
python -m cli.train_rl --config config/hotpotqa_rl.yaml --baseline_only
python -m cli.train_rl --config config/hotpotqa_rl.yaml --baseline_only --baseline_mode random

# 2) ppo 默认
python -m cli.train_rl --config config/hotpotqa_rl.yaml

# 3) 无重排序
python -m cli.train_rl --config config/hotpotqa_rl_norerank.yaml

# 4) 有重排序
python -m cli.train_rl --config config/hotpotqa_rl_rerank.yaml

# 5/6) 奖励塑形 A/B/C（写三份 config 或加 --reward_mode 参数）
python -m cli.train_rl --config config/hotpotqa_rl_rewardA.yaml
python -m cli.train_rl --config config/hotpotqa_rl_rewardB.yaml
python -m cli.train_rl --config config/hotpotqa_rl_rewardC.yaml

# 画图
python scripts/plot_rl_with_baseline.py

预期现象 & 判定标准
	•	tiny 模型：EM/F1 常为 0 → 看 reward 超过 baseline 即可
	•	flan-t5-small（GPU）：随着步数↑、样本↑，avg_F1 应该上升，至少在 eval 曲线上出现比 baseline 稍高的段落（哪怕不大）。
	•	reranker 打开时，reward 与 F1应该更稳/更高（因为相关上下文更准）。
	•	奖励 B 通常让 F1更敏感（但也可能引入更多高温动作和更长延迟）。

稳定性与复现
	•	设 seed（numpy/python/torch）→ 在 env 初始化里确保写死
	•	每组 3 次独立运行，报告均值±方差（至少在最终表格）
	•	Checkpoint：ppo_rag_steps.zip（SB3 自动存），保留最后一次

⸻

你要的话，我可以：
	•	给你三份额外 config（无重排序 / 有重排序 / 奖励B/C）
	•	给 CLI 加一个 --run_name 参数，把日志输出到 .cache/rl_runs/<run_name>/，便于多组对比
	•	给画图脚本加“对比多个 run 目录”的功能，一张图上叠多条曲线

你决定下想跑哪几组，我把对应配置直接写好。