from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_comparison_results(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_accuracy(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "-"


def _format_int(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value))
    return "-"


def render_markdown(rows: list[dict[str, object]]) -> str:
    ordered = sorted(rows, key=lambda row: float(row.get("test_accuracy", 0.0)), reverse=True)
    best = ordered[0] if ordered else None

    lines: list[str] = ["# 结果分析摘要", "", "## 实验对比表", ""]
    lines.append("| 实验 | 训练方式 | 预训练 | 学习率 | 最佳 epoch | 验证准确率 | 测试准确率 | 可训练参数 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | ---: |")

    for row in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("experiment", "-")),
                    str(row.get("train_mode", "-")),
                    "yes" if row.get("pretrained") else "no",
                    _format_accuracy(row.get("lr")),
                    _format_int(row.get("best_epoch")),
                    _format_accuracy(row.get("best_val_accuracy")),
                    _format_accuracy(row.get("test_accuracy")),
                    _format_int(row.get("trainable_parameters")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 结论自动摘要",
            "",
        ]
    )
    if best is not None:
        lines.append(
            f"- 当前测试集表现最好的实验是 {best.get('experiment', '-')}，测试准确率为 {_format_accuracy(best.get('test_accuracy'))}。"
        )
        lines.append(
            f"- 它对应的训练方式是 {best.get('train_mode', '-')}, 学习率为 {_format_accuracy(best.get('lr'))}。"
        )

    lines.extend(
        [
            "- 如果预训练线性探测优于从头训练，说明 ImageNet 特征对该任务有明显迁移价值。",
            "- 如果微调优于线性探测，说明目标数据集与预训练分布存在一定差异，解冻骨干可以进一步适配。",
            "- 如果较小学习率带来更高的验证准确率和更平稳的曲线，说明微调阶段应采用更保守的更新幅度。",
            "",
            "## 报告写作框架",
            "",
            "1. 数据集与划分：说明 dhole 和 fox 的数据来源、样本数量、训练/验证/测试比例。",
            "2. 模型与实验设置：说明 ResNet18、三组实验、学习率和训练轮数。",
            "3. 结果展示：给出准确率、混淆矩阵、训练曲线、正确与错误样例。",
            "4. 结果分析：分析预训练收益、微调收益以及学习率影响。",
            "5. 总结：给出最优方案和后续改进方向。",
        ]
    )

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a markdown summary for experiment comparison results")
    parser.add_argument("--comparison-json", type=Path, default=Path("runs/comparison.json"), help="Input comparison JSON")
    parser.add_argument("--output", type=Path, default=Path("runs/report_summary.md"), help="Output markdown file")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rows = load_comparison_results(args.comparison_json)
    report = render_markdown(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "experiments": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()