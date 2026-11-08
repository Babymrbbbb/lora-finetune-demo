"""核心 demo 回归测试：参数效率 + 适配效果，纯标准库。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lora_demo import run_comparison


def test_param_efficiency():
    r = run_comparison()
    assert r["lora_params"] < r["full_params"], "LoRA 可训练参数必须少于全量微调"
    assert 0.3 < r["lora_ratio"] < 0.6, f"参数占比异常：{r['lora_ratio']}"
    print("PASS param efficiency:", r["full_params"], "->", r["lora_params"])


def test_comparable_accuracy():
    r = run_comparison()
    assert r["lora_test"] >= 0.9, f"LoRA 测试准确率过低：{r['lora_test']}"
    assert r["full_test"] - r["lora_test"] <= 0.1, "LoRA 不应显著劣于全量微调"
    print(
        "PASS accuracy: full=%.1f%% lora=%.1f%%"
        % (r["full_test"] * 100, r["lora_test"] * 100)
    )


if __name__ == "__main__":
    test_param_efficiency()
    test_comparable_accuracy()
