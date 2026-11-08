# lora-finetune-demo

LoRA（Low-Rank Adaptation）参数高效微调的最小对照实验：
**零依赖、纯 Python 手写 LoRA 数学**，在玩具任务上把「全量微调」与「LoRA 微调」
做参数量与效果对照；另附工业标准栈（HuggingFace PEFT）的真实落地模板。

> 数据与模型均为合成/公开资源，不含任何真实业务数据。

## 两条路径

| | `lora_demo.py`（主演示） | `train_peft.py`（生产模板） |
|---|---|---|
| 依赖 | **零**（Python 标准库） | torch / transformers / peft / datasets |
| 环境 | 任何机器，`python lora_demo.py` 即跑 | GPU / Colab |
| 作用 | 手写 ΔW=B·A 的前向与反向，讲清 LoRA 改了什么 | 在真实预训练 LM 上挂 adapter 并只保存 adapter |
| 状态 | **已端到端实跑**，测试通过 | 未在受限沙箱实跑（无 GPU），仅保证 API 用法正确 |

## 快速开始（零依赖）

```bash
python lora_demo.py
python tests/test_lora.py
```

实跑输出（Python 3.13，D/H/C/R = 10/20/4/2，4 类玩具分类，60 epoch）：

```
方法                可训练参数      参数占比       训练准确率       测试准确率
----------------------------------------------------------------
全量微调               280      100%     100.0%     100.0%
LoRA(r=2)          108     38.6%     100.0%     100.0%
----------------------------------------------------------------
结论：LoRA 用 38.6% 的可训练参数，达到与全量微调可比的适配效果。
```

## LoRA 数学（这就是 `lora_demo.py` 手写的部分）

对一个冻结的权重矩阵 `W_base ∈ ℝ^{out×in}`，不直接更新它，而是学习一个低秩增量：

```
ΔW = B · A      B ∈ ℝ^{out×r},  A ∈ ℝ^{r×in},  r ≪ min(in,out)
y  = W_base · x + scaling · (B · A) · x
```

- **初始化**：`A` 随机小值，`B` 置零 → 训练起点 `ΔW = 0`，不破坏预训练底座。
- **训练时**：冻结 `W_base`，只优化 `A`、`B`。本 demo 手写链式法则把梯度
  拆回 `A`/`B`（见 `train_lora` 中 `G1/G2 → dA/dB` 的推导）。
- **参数量**：全量 `in·out`；LoRA `r·(in+out)`。两层 MLP 实测 **280 → 108（38.6%）**。
- **工程价值**：只保存几十 KB～几 MB 的 adapter，而不是整个底座；
  显存、存储、多任务切换成本、过拟合风险都显著下降。

## 真实 PEFT 路径（可选）

```bash
pip install -r requirements-peft.txt
python train_peft.py
```

`train_peft.py` 用 `LoraConfig(r=8, lora_alpha=32, target_modules=["c_attn"])`
在中文 GPT-2 小模型上做指令微调，并调用 `model.save_pretrained()` **只保存 adapter**。
脚本内有 `print_trainable()` 打印可训练参数占比，可直接与上面的理论值对照。

## 目录结构

```
lora-finetune-demo/
├── lora_demo.py              # 零依赖手写 LoRA 主演示（已实跑）
├── train_peft.py             # HuggingFace PEFT 生产模板（需 GPU）
├── tests/test_lora.py        # 参数效率 + 适配效果回归测试
├── requirements.txt          # 核心：零依赖
├── requirements-peft.txt     # 可选：真实 PEFT 依赖
└── README.md
```

## 边界与说明

- 主演示是**玩具任务**（4 类合成数据 + 2 层 MLP），用于精确验证
  「低秩适配器能以更少参数达到可比效果」这一机制，**不是**真实 LLM 评测指标；
  100% 准确率只说明该玩具任务对两种方法都很简单，不代表生产效果。
- `train_peft.py` 需要 GPU 与第三方栈，**未在受限沙箱内实跑**，
  仅保证 HuggingFace PEFT API 用法正确；请在 Colab / GPU 机器上自行验证。
- 结论只声称「参数效率 + 机制正确」，不虚标任何生产吞吐或业务指标。

## 相关项目

同一套工程方法下的检索侧对照与落地实现：

- [rag-vector-vs-keyword](https://github.com/Babymrbbbb/rag-vector-vs-keyword)：两种检索方式在中文语料上的同口径对照评测。
- [langchain-kb-qa](https://github.com/Babymrbbbb/langchain-kb-qa)：把对照结论落到可部署的检索增强问答服务。
