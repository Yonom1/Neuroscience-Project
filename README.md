# 项目脚本说明文档

本项目包含多个 Python 脚本，用于数据集整理、变体生成、模型训练与评估、结果可视化以及人类实验。以下是各脚本的详细说明。

## 1. 数据准备脚本

### `data_organizer.py`
*   **用途**: 将原始的 ImageNet 数据集（按 WNID 命名的文件夹）整理成 PyTorch `ImageFolder` 格式的训练集和验证集。
*   **输入**: 
    *   `./my_experiment_data/` (默认源目录，包含 `n02123394` 等文件夹)
*   **输出**: 
    *   `./dataset/train/`
    *   `./dataset/val/`
*   **配置**: 脚本内可修改 `SOURCE_MAPPING` (类别映射) 和 `TRAIN_RATIO` (训练集比例)。

### `build_wild_test.py`
*   **用途**: 使用爬虫从 Bing 图片搜索下载额外的测试图片（Wild Test Set），用于构建测试集。
*   **输入**: 无（使用网络爬虫）。
*   **输出**: 
    *   `./dataset/test/persian_cat/`
    *   `./dataset/test/siamese_cat/`
*   **配置**: 脚本内可修改 `categories` (搜索关键词) 和 `num_images` (下载数量)。

### `generate_variants.py`
*   **用途**: 基于原始数据集，生成不同程度的**对比度降低**和**高斯噪声**变体，用于测试模型的鲁棒性。
*   **输入**: 
    *   `./dataset/` (原始数据集)
*   **输出**: 
    *   `./dataset_variants/` (包含多个子文件夹，如 `contrast_level_1`...`contrast_level_5`, `noise_level_1`...`noise_level_5`)
*   **配置**: 脚本内可定义 `CONTRAST_LEVELS` (对比度档位) 和 `NOISE_LEVELS` (噪声档位)。

---

## 2. 实验与评估脚本

### `run.py`
*   **用途**: 核心实验脚本。依次在原始数据集及其所有变体上微调并评估三个模型（AlexNet, VGG16, ResNet18）。
*   **输入**: 
    *   `./dataset/`
    *   `./dataset_variants/`
*   **输出**: 
    *   `experiment_results.csv`: 包含所有实验的详细指标（Dataset, Model, TN, FP, FN, TP）。
*   **逻辑**: 自动遍历所有定义的数据集路径，加载数据，微调模型最后一层，并在测试集上计算混淆矩阵。

### `plot_results.py`
*   **用途**: 读取实验结果日志，绘制模型准确率随干扰强度变化的曲线图。
*   **输入**: 
    *   `experiment_results.csv`
*   **输出**: 
    *   `accuracy_plot.png`: 包含两个子图（准确率 vs 对比度，准确率 vs 噪声）。

---

## 3. 人类实验脚本

### `human_experiment.py`
*   **用途**: 交互式 GUI 程序，用于测试人类在相同数据集上的分类准确率，作为基准（Human Baseline）。
*   **输入**: 
    *   `./dataset/` 和 `./dataset_variants/` 中的测试集图片。
*   **输出**: 
    *   `./human_results/human_experiment_results_X.csv`: 记录每张图片的判断结果、反应时间等。
*   **操作**: 
    *   运行后输入用户 ID。
    *   **左方向键 (←)**: 波斯猫
    *   **右方向键 (→)**: 暹罗猫
    *   每次随机抽取 80 张图片进行测试。

---

## 4. 依赖库
*   `torch`, `torchvision`
*   `numpy`, `pandas`, `matplotlib`, `seaborn`
*   `scikit-image`, `Pillow`
*   `tkinter` (用于 GUI)
*   `icrawler` (用于爬虫)
