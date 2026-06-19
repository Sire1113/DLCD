# DLCD
南京邮电大学 深度学习概论大作业

## 题目

基于 ResNet18 的 dhole vs fox 二分类迁移学习对比实验。

数据集默认从 iNaturalist 公开物种观察图像检索下载，适合作为一个陌生域二分类任务。

## 项目结构

- `src/dlcd/data.py`：数据扫描、分层划分、数据增强与 DataLoader 构建
- `src/dlcd/models.py`：ResNet18 模型构建，以及从头训练、线性探测、微调模式控制
- `src/dlcd/engine.py`：训练、验证、测试与检查点保存
- `src/dlcd/visualize.py`：训练曲线、混淆矩阵、正确/错误样例可视化
- `src/dlcd/cli.py`：命令行入口，统一运行三组对比实验
- `src/dlcd/experiments.py`：实验配置，便于扩展学习率扫描

## 实验设计

本项目固定使用 ResNet18，做三组对比：

1. 从头训练 ResNet18，不使用预训练权重，作为基线。
2. 使用 ImageNet 预训练 ResNet18，只训练最后分类层，观察迁移学习收益。
3. 在预训练基础上解冻部分骨干网络进行微调，比较进一步提升。

如果需要，还可以在第三组基础上增加学习率扫描，比较 `1e-3`、`1e-4`、`1e-5` 的效果。

## 数据集目录

将数据整理成下面的文件结构：

```text
data/
	dhole/
		xxx.jpg
	fox/
		yyy.jpg
```

每个类别一个文件夹，程序会自动完成训练集、验证集和测试集划分。

## 运行方式

先安装依赖：

```bash
uv sync
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu131
```

运行三组对比实验：

```bash
uv run dlcd compare --data-root data --output-dir runs
```

如果你想让所有实验跑更久，可以直接加 `--epochs`，比如：

```bash
uv run dlcd compare --data-root data --output-dir runs --epochs 40
```

先下载数据集：

```bash
uv run dlcd-download --output-dir data --per-class 120
```

默认会分别用 `dhole` 和 `red fox` 作为物种查询词，也可以用 `--class-spec` 自定义类别和查询词。
如果网络较慢，可以加大超时并增加重试次数，例如 `--timeout 180 --retries 5`。

如果要附加学习率对比：

```bash
uv run dlcd compare --data-root data --output-dir runs --lr-sweep
```

单独运行某个实验：

```bash
uv run dlcd train --data-root data --output-dir runs --experiment scratch_resnet18
```

同样可以加 `--epochs 40` 覆盖默认轮数。

生成结果分析摘要：

```bash
uv run dlcd-report --comparison-json runs/comparison.json --output runs/report_summary.md
```

## 输出结果

每个实验会生成一个独立目录，包含：

- `best.pt`：验证集最佳模型权重
- `history.png`：训练曲线
- `confusion_matrix.png`：混淆矩阵
- `correct_examples.png`：正确分类样例
- `wrong_examples.png`：错误分类样例
- `summary.json`：实验结果汇总

## 代码入口

命令行主入口是 `src/dlcd/cli.py`，核心训练流程在 `src/dlcd/engine.py`，模型结构在 `src/dlcd/models.py`。

## 结果分析框架


1. 数据集说明：写明 dhole 和 fox 的来源、数量、划分比例，以及是否存在样本不平衡。
2. 方法说明：比较从头训练、预训练线性探测、预训练微调三种设置。
3. 定量结果：报告验证集和测试集准确率、混淆矩阵和最佳 epoch。
4. 定性结果：展示正确样例和错误样例，分析误判原因。
5. 学习率分析：比较不同学习率下的收敛速度、稳定性和最终性能。
6. 结论：总结预训练和微调是否带来收益，以及在该数据集上的最优策略。

自动生成的摘要文件是 `runs/report_summary.md`，可直接作为结果分析初稿。
