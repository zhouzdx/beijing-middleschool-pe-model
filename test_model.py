import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_model_performance():
    """测试模型性能"""
    print("="*50)
    print("中考体育能力预测模型 - 性能测试")
    print("="*50)

    # 加载数据
    print("\n1. 加载测试数据...")
    data = np.load('preprocessed_data.npz', allow_pickle=True)
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']

    # 加载模型
    print("2. 加载模型...")
    pipeline = joblib.load('sports_score_predictor.pipeline.pkl')

    # 预测
    print("3. 进行预测...")
    y_pred = pipeline.predict(X_test)

    # 计算指标
    print("4. 计算性能指标...")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 显示结果
    print("\n" + "="*50)
    print("📊 模型性能测试结果")
    print("="*50)

    print(f"\n测试集统计:")
    print(f"  样本数量: {len(y_test)}")
    print(f"  实际分数范围: {y_test.min():.1f} - {y_test.max():.1f}")
    print(f"  实际分数平均值: {y_test.mean():.1f}")
    print(f"  实际分数标准差: {y_test.std():.1f}")

    print(f"\n预测性能指标:")
    print(f"  R²分数: {r2:.4f}")
    print(f"  平均绝对误差 (MAE): {mae:.4f}")
    print(f"  均方根误差 (RMSE): {rmse:.4f}")

    # 误差分析
    print(f"\n误差分析:")
    errors = y_test - y_pred
    abs_errors = np.abs(errors)

    print(f"  最大正误差: {errors.max():.2f}")
    print(f"  最大负误差: {errors.min():.2f}")
    print(f"  平均绝对误差: {abs_errors.mean():.2f}")
    print(f"  误差标准差: {errors.std():.2f}")

    # 准确率分析
    print(f"\n准确率分析:")

    # 不同误差阈值下的准确率
    thresholds = [1, 2, 3, 5]
    for threshold in thresholds:
        accuracy = np.mean(abs_errors <= threshold) * 100
        print(f"  误差 ≤ {threshold} 分的准确率: {accuracy:.1f}%")

    # 等级准确率
    def get_level(score):
        if score >= 54:
            return "优秀"
        elif score >= 48:
            return "良好"
        elif score >= 36:
            return "及格"
        else:
            return "待提高"

    actual_levels = [get_level(score) for score in y_test]
    predicted_levels = [get_level(score) for score in y_pred]

    level_accuracy = np.mean(np.array(actual_levels) == np.array(predicted_levels)) * 100
    print(f"  等级预测准确率: {level_accuracy:.1f}%")

    # 创建性能可视化
    create_performance_visualization(y_test, y_pred, errors)

    # 测试示例预测
    print("\n" + "="*50)
    print("🧪 示例预测测试")
    print("="*50)

    test_examples()

def create_performance_visualization(y_test, y_pred, errors):
    """创建性能可视化"""
    print("\n5. 创建性能可视化图表...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 实际值 vs 预测值散点图
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('实际分数')
    axes[0, 0].set_ylabel('预测分数')
    axes[0, 0].set_title('实际值 vs 预测值')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 残差图
    axes[0, 1].scatter(y_pred, errors, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('预测分数')
    axes[0, 1].set_ylabel('残差')
    axes[0, 1].set_title('残差图')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 误差分布直方图
    axes[0, 2].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('误差')
    axes[0, 2].set_ylabel('频数')
    axes[0, 2].set_title('误差分布')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 分数分布对比
    axes[1, 0].hist(y_test, bins=30, alpha=0.5, label='实际分数', edgecolor='black')
    axes[1, 0].hist(y_pred, bins=30, alpha=0.5, label='预测分数', edgecolor='black')
    axes[1, 0].set_xlabel('分数')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('分数分布对比')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 累积误差分布
    sorted_abs_errors = np.sort(np.abs(errors))
    cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)
    axes[1, 1].plot(sorted_abs_errors, cumulative, marker='.', linestyle='none')
    axes[1, 1].set_xlabel('绝对误差')
    axes[1, 1].set_ylabel('累积比例')
    axes[1, 1].set_title('累积误差分布')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. 误差箱线图
    axes[1, 2].boxplot(errors)
    axes[1, 2].set_ylabel('误差')
    axes[1, 2].set_title('误差分布箱线图')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_test_performance.png', dpi=300, bbox_inches='tight')
    print(f"  性能测试图已保存到: [model_test_performance.png](file:///D:/agent/模型/model_test_performance.png)")

def test_examples():
    """测试示例预测"""
    # 加载模型
    pipeline = joblib.load('sports_score_predictor.pipeline.pkl')

    # 定义测试用例
    test_cases = [
        {
            'name': '优秀男生',
            'data': {
                'gender': 0, 'sg': 175, 'tz': 65, 'bmi': 65/((175/100)**2),
                'mm': 6.5, 'lm': 3.2, 'mmm': 0, 'ytxs': 15,
                'ywqz': 0, 'ldty': 250, 'zwtqq': 20, 'fhl': 5000
            },
            'expected_level': '优秀'
        },
        {
            'name': '良好女生',
            'data': {
                'gender': 1, 'sg': 165, 'tz': 55, 'bmi': 55/((165/100)**2),
                'mm': 8.0, 'lm': 0, 'mmm': 3.3, 'ytxs': 0,
                'ywqz': 40, 'ldty': 190, 'zwtqq': 18, 'fhl': 3500
            },
            'expected_level': '良好'
        },
        {
            'name': '及格男生',
            'data': {
                'gender': 0, 'sg': 170, 'tz': 70, 'bmi': 70/((170/100)**2),
                'mm': 8.5, 'lm': 4.0, 'mmm': 0, 'ytxs': 5,
                'ywqz': 0, 'ldty': 210, 'zwtqq': 10, 'fhl': 3000
            },
            'expected_level': '及格'
        },
        {
            'name': '待提高女生',
            'data': {
                'gender': 1, 'sg': 155, 'tz': 45, 'bmi': 45/((155/100)**2),
                'mm': 10.0, 'lm': 0, 'mmm': 4.5, 'ytxs': 0,
                'ywqz': 20, 'ldty': 150, 'zwtqq': 5, 'fhl': 2000
            },
            'expected_level': '待提高'
        }
    ]

    print("\n测试用例结果:")
    print("-" * 80)

    for i, test_case in enumerate(test_cases, 1):
        # 预测
        X = np.array([list(test_case['data'].values())])
        score = pipeline.predict(X)[0]

        # 确定等级
        def get_level(score):
            if score >= 54:
                return "优秀"
            elif score >= 48:
                return "良好"
            elif score >= 36:
                return "及格"
            else:
                return "待提高"

        predicted_level = get_level(score)
        is_correct = predicted_level == test_case['expected_level']

        print(f"\n{i}. {test_case['name']}:")
        print(f"   预测分数: {score:.1f}/60")
        print(f"   预测等级: {predicted_level}")
        print(f"   期望等级: {test_case['expected_level']}")
        print(f"   结果: {'✅ 正确' if is_correct else '❌ 错误'}")

        # 显示关键指标
        if test_case['name'] == '优秀男生':
            print(f"   关键指标: 50米{test_case['data']['mm']}秒, 1000米{test_case['data']['lm']}分, 引体{test_case['data']['ytxs']}个")
        elif test_case['name'] == '良好女生':
            print(f"   关键指标: 50米{test_case['data']['mm']}秒, 800米{test_case['data']['mmm']}分, 仰卧起坐{test_case['data']['ywqz']}个")

    print("\n" + "="*50)
    print("测试完成！")

def create_model_report():
    """创建模型报告"""
    print("\n" + "="*50)
    print("📋 模型综合报告")
    print("="*50)

    # 加载数据
    data = np.load('preprocessed_data.npz', allow_pickle=True)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']

    # 加载模型
    pipeline = joblib.load('sports_score_predictor.pipeline.pkl')

    # 训练集和测试集预测
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # 计算指标
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # 创建报告
    report = f"""中考体育能力预测模型 - 综合报告
{'='*50}

一、模型基本信息
  模型类型: 随机森林回归
  特征数量: {len(feature_names)}
  训练样本数: {len(X_train)}
  测试样本数: {len(X_test)}

二、特征列表
{chr(10).join(f'  {i+1:2d}. {feature}' for i, feature in enumerate(feature_names))}

三、性能指标
  训练集 R²: {train_r2:.4f}
  测试集 R²: {test_r2:.4f}
  训练集 MAE: {train_mae:.4f}
  测试集 MAE: {test_mae:.4f}

四、数据统计
  目标变量范围: {y_train.min():.1f} - {y_train.max():.1f}
  目标变量平均值: {y_train.mean():.1f}
  目标变量标准差: {y_train.std():.1f}

五、模型文件
  1. 预测管道: sports_score_predictor.pipeline.pkl
  2. 标准化器: sports_score_predictor.scaler.pkl
  3. 特征名称: sports_score_predictor.features.npy
  4. 清洗数据: cleaned_sports_data.csv

六、使用说明
  1. 使用 predict_cli.py 进行交互式预测
  2. 使用 user_interface.py 启动Web界面(需要Streamlit)
  3. 直接调用模型进行批量预测

七、注意事项
  1. 模型基于模拟的中考体育评分标准
  2. 预测结果仅供参考
  3. 实际中考评分标准可能有所不同
  4. 建议结合专业体育教师评估

{'='*50}
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # 保存报告
    with open('model_comprehensive_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n完整报告已保存到: [model_comprehensive_report.txt](file:///D:/agent/模型/model_comprehensive_report.txt)")

def main():
    """主函数"""
    print("开始测试模型性能...")

    # 测试模型性能
    test_model_performance()

    # 创建模型报告
    create_model_report()

    print("\n" + "="*50)
    print("✅ 所有测试完成！")
    print("="*50)

if __name__ == "__main__":
    main()