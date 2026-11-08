"""真实 PEFT/LoRA 微调路径（生产模板，需 GPU / Colab，本沙箱未实跑）。

与 lora_demo.py 的关系：
- lora_demo.py：零依赖、纯 Python 手写 LoRA 数学，本机可跑、已实跑验证；
  用于讲清「LoRA 到底改了什么」以及参数效率，是本仓库的主演示。
- 本文件：把同一套思想落到工业标准栈（transformers + peft），
  在真实预训练语言模型上做 LoRA 指令微调，产出可加载的 adapter。

诚实声明：本脚本需要 torch / transformers / peft / datasets 与 GPU 环境，
作者未在受限沙箱内实跑，仅保证 HuggingFace PEFT API 用法正确。
请在 Colab / 有 GPU 的机器上运行。

运行：
    pip install -r requirements-peft.txt
    python train_peft.py            # 默认小样本 + 小模型，几分钟内可跑完
"""
import os

# 避免在无 GPU 环境误触大下载
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# —— 小模型中文本地/在线均可；Colab 首次会下载约 100MB ——
MODEL_NAME = os.getenv("LORA_BASE_MODEL", "uer/gpt2-chinese-cluecorpussmall")
OUTPUT_DIR = os.getenv("LORA_OUTPUT_DIR", "./lora_adapter")

# —— 虚构的小样本指令数据（演示用，不含任何真实业务数据）——
SAMPLES = [
    {"instruction": "查询门店营业时间", "output": "门店营业时间为每日 08:00 至 22:00。"},
    {"instruction": "如何申请发票", "output": "在订单详情页点击「申请发票」，填写抬头与税号后提交。"},
    {"instruction": "会员卡怎么办理", "output": "在前台或小程序「会员中心」填写手机号即可免费办理。"},
    {"instruction": "蛋糕需要提前预订吗", "output": "定制蛋糕建议提前 48 小时预订，现货蛋糕可到店购买。"},
    {"instruction": "支持配送吗", "output": "市区范围内支持同城配送，满 88 元免配送费。"},
]


def build_dataset(tokenizer):
    texts = [
        f"指令：{s['instruction']}\n回答：{s['output']}" for s in SAMPLES
    ]
    enc = tokenizer(texts, truncation=True, max_length=128, padding="max_length")
    return Dataset.from_dict(
        {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    )


def print_trainable(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"可训练参数 {trainable:,} / 总参数 {total:,} = {100 * trainable / total:.2f}%")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # 核心：只给注意力投影层挂 LoRA 适配器（r=8, alpha=32）
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn"],  # GPT2 系；LLaMA 系用 q_proj/v_proj
        bias="none",
    )
    model = get_peft_model(base, config)
    print_trainable(model)

    dataset = build_dataset(tokenizer)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=20,
        per_device_train_batch_size=2,
        learning_rate=2e-3,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        fp16=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    # 只保存 adapter（几十 KB～几 MB），底座不动——这正是 LoRA 的工程价值
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"adapter 已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit(
            "未安装 PEFT 依赖。请执行：pip install -r requirements-peft.txt\n"
            "（本脚本需要 GPU/Colab；纯本地零依赖演示请直接运行 python lora_demo.py）"
        )
    main()
