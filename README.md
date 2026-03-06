# 北京中考体育能力预测模型

## 项目概述

本项目基于学生体质测试数据，构建了一个用于预测北京中考体育成绩的机器学习模型。模型可以根据学生的各项体育测试成绩，预测其中考体育分数（满分60分）。

## 项目结构

```
中考体育能力预测模型/
├── data_preprocessing.py          # 数据预处理脚本
├── model_training_fixed.py        # 模型训练脚本（修复版）
├── predict_cli.py                 # 命令行预测界面
├── user_interface.py              # Streamlit Web界面
├── test_model.py                  # 模型测试脚本
├── 高中学生体质测试信息202508010326.json  # 原始数据
├── cleaned_sports_data.csv        # 清洗后的数据
├── preprocessed_data.npz          # 预处理后的数据
├── sports_score_predictor.*.pkl   # 模型文件
├── model_metrics.csv              # 模型性能指标
├── model_comprehensive_report.txt # 模型综合报告
├── feature_importance.png         # 特征重要性图
├── model_performance.png          # 模型性能图
├── score_distribution.png         # 分数分布图
└── model_test_performance.png     # 测试性能图
```

## 模型性能

### 主要指标
- **测试集 R²分数**: 0.9378
- **测试集 MAE**: 1.6780
- **测试集 RMSE**: 2.1529
- **等级预测准确率**: 85.0%

### 误差分析
- 误差 ≤ 1分的准确率: 39.0%
- 误差 ≤ 2分的准确率: 65.4%
- 误差 ≤ 3分的准确率: 84.1%
- 误差 ≤ 5分的准确率: 98.0%

## 特征重要性

根据随机森林模型分析，最重要的特征包括：

1. **lm (1000米跑)** - 16.6%
2. **mmm (800米跑)** - 15.3%
3. **zwtqq (坐位体前屈)** - 15.1%
4. **mm (50米跑)** - 15.0%
5. **ldty (立定跳远)** - 9.8%

## 快速开始

### 1. 环境要求

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

### 2. 使用命令行界面

```bash
# 运行预测系统
python predict_cli.py

# 选择操作：
# 1. 进行预测 - 输入学生信息进行预测
# 2. 查看示例 - 查看示例预测结果
# 3. 退出
```

### 3. 使用Web界面（需要Streamlit）

```bash
# 安装Streamlit
pip install streamlit

# 运行Web界面
streamlit run user_interface.py
```

### 4. 直接使用模型进行预测

```python
import joblib
import numpy as np

# 加载模型
pipeline = joblib.load('sports_score_predictor.pipeline.pkl')

# 准备输入数据（男生示例）
input_data = {
    'gender': 0,      # 男生
    'sg': 170,        # 身高170cm
    'tz': 60,         # 体重60kg
    'bmi': 60 / ((170/100)**2),  # BMI
    'mm': 7.5,        # 50米7.5秒
    'lm': 3.5,        # 1000米3.5分钟
    'mmm': 0,         # 800米（男生为0）
    'ytxs': 10,       # 引体向上10个
    'ywqz': 0,        # 仰卧起坐（男生为0）
    'ldty': 230,      # 立定跳远230cm
    'zwtqq': 15,      # 坐位体前屈15cm
    'fhl': 4000       # 肺活量4000ml
}

# 转换为数组并预测
X = np.array([list(input_data.values())])
score = pipeline.predict(X)[0]
print(f"预测中考体育分数: {score:.1f}/60")
```

## 输入数据说明

### 男生需要输入的项目：
1. 性别：男生 (gender=0)
2. 身高 (sg)：单位cm
3. 体重 (tz)：单位kg
4. 50米跑 (mm)：单位秒
5. 1000米跑 (lm)：单位分钟
6. 引体向上 (ytxs)：单位个
7. 立定跳远 (ldty)：单位cm
8. 坐位体前屈 (zwtqq)：单位cm
9. 肺活量 (fhl)：单位ml

### 女生需要输入的项目：
1. 性别：女生 (gender=1)
2. 身高 (sg)：单位cm
3. 体重 (tz)：单位kg
4. 50米跑 (mm)：单位秒
5. 800米跑 (mmm)：单位分钟
6. 仰卧起坐 (ywqz)：单位个
7. 立定跳远 (ldty)：单位cm
8. 坐位体前屈 (zwtqq)：单位cm
9. 肺活量 (fhl)：单位ml

## 评分等级标准

| 等级 | 分数范围 | 说明 |
|------|----------|------|
| 优秀 | ≥54分 | 体育成绩非常出色 |
| 良好 | 48-53分 | 体育成绩良好 |
| 及格 | 36-47分 | 达到及格标准 |
| 待提高 | <36分 | 需要加强锻炼 |

## 模型训练过程

### 1. 数据预处理
- 加载原始JSON数据（1228条记录）
- 处理缺失值（使用中位数填充）
- 处理异常值（基于合理范围裁剪）
- 创建衍生特征（BMI）
- 编码分类变量（性别）
- 创建目标变量（模拟中考体育分数）

### 2. 特征工程
- 选择12个特征进行建模
- 标准化所有数值特征
- 分割数据集（训练集80%，测试集20%）

### 3. 模型训练
- 使用随机森林回归算法
- 参数设置：n_estimators=200, max_depth=20
- 交叉验证评估模型性能

### 4. 模型评估
- 计算R²分数、MAE、RMSE等指标
- 分析特征重要性
- 可视化预测结果

## 改进建议

基于预测结果，系统会提供针对性的改进建议，包括：

- 🏃 速度训练建议
- 🏃‍♂️/🏃‍♀️ 耐力训练建议
- 💪 力量训练建议
- 🦘 爆发力训练建议
- 🧘 柔韧性训练建议
- 🫁 心肺功能训练建议

## 注意事项

1. **数据来源**：模型基于高中学生体质测试数据训练
2. **评分标准**：使用模拟的中考体育评分标准
3. **预测范围**：分数范围为15-58分（满分60分）
4. **使用建议**：预测结果仅供参考，应结合专业评估
5. **局限性**：模型对极端值的预测可能不够准确

## 文件说明

### 核心文件
- `predict_cli.py` - 主要使用界面
- `sports_score_predictor.pipeline.pkl` - 预测管道
- `cleaned_sports_data.csv` - 清洗后的数据

### 辅助文件
- `model_comprehensive_report.txt` - 详细模型报告
- `feature_importance.png` - 特征重要性可视化
- `model_performance.png` - 模型性能可视化

## 后续改进方向

1. **数据增强**：收集更多中考体育实际数据
2. **模型优化**：尝试其他算法如XGBoost、LightGBM
3. **特征工程**：添加更多衍生特征
4. **实时更新**：建立模型定期更新机制
5. **个性化建议**：提供更详细的训练计划

## 技术支持

如有问题或建议，请参考模型报告中的详细信息，或检查相关可视化图表了解模型性能。

---

**最后更新**: 2026-03-06
**模型版本**: 1.0
**作者**: zhouzdx
**联系方式**: zhouzdx@outlook.com
