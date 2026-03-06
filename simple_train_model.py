import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """加载和准备数据"""
    print("="*60)
    print("加载和准备真实中考分数数据")
    print("="*60)

    # 加载处理后的数据
    df = pd.read_csv('processed_score_data.csv', encoding='utf-8-sig')

    print(f"原始数据形状: {df.shape}")
    print(f"总记录数: {len(df)}")

    # 目标变量
    y = df['总分数'].values
    print(f"\n目标变量 - 总分数:")
    print(f"  范围: {y.min():.2f} - {y.max():.2f}")
    print(f"  平均值: {y.mean():.2f}")
    print(f"  标准差: {y.std():.2f}")

    # 选择特征 - 使用最相关的特征
    features = [
        # 基础测试成绩
        '身高_cm',
        '体重_kg',
        '肺活量_ml',
        '50米跑_秒',
        '坐位体前屈_cm',
        '跳绳_个',

        # 各项评分
        'BMI分数',
        '体重分数',
        '肺活量分数',
        '50米跑分数',
        '坐位体前屈分数',
        '跳绳分数',

        # 标准分
        '标准分数'
    ]

    # 检查哪些特征可用
    available_features = [f for f in features if f in df.columns]
    print(f"\n可用特征数量: {len(available_features)}")
    print("可用特征:")
    for i, feature in enumerate(available_features, 1):
        print(f"  {i:2d}. {feature}")

    # 准备特征数据
    X = df[available_features].copy()

    # 处理缺失值 - 使用中位数填充
    print(f"\n处理缺失值...")
    missing_before = X.isnull().sum().sum()
    print(f"  缺失值总数: {missing_before}")

    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"  {col}: 用中位数 {median_val:.2f} 填充")

    missing_after = X.isnull().sum().sum()
    print(f"  处理后缺失值: {missing_after}")

    return X.values, y, available_features

def train_random_forest(X, y, feature_names):
    """训练随机森林模型"""
    print("\n" + "="*60)
    print("训练随机森林模型")
    print("="*60)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"特征数量: {X_train.shape[1]}")

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建随机森林模型
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    print("\n训练模型...")
    rf_model.fit(X_train_scaled, y_train)

    # 预测
    y_pred_train = rf_model.predict(X_train_scaled)
    y_pred_test = rf_model.predict(X_test_scaled)

    # 计算指标
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # 交叉验证
    print("进行交叉验证...")
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"\n模型性能:")
    print(f"  训练集 R²: {train_r2:.4f}")
    print(f"  测试集 R²: {test_r2:.4f}")
    print(f"  测试集 MAE: {test_mae:.4f}")
    print(f"  测试集 RMSE: {test_rmse:.4f}")
    print(f"  交叉验证 R²: {cv_mean:.4f} (±{cv_std:.4f})")

    # 误差分析
    errors = y_test - y_pred_test
    abs_errors = np.abs(errors)

    print(f"\n误差分析:")
    print(f"  平均绝对误差: {abs_errors.mean():.2f}")
    print(f"  最大正误差: {errors.max():.2f}")
    print(f"  最大负误差: {errors.min():.2f}")

    # 准确率分析
    print(f"\n准确率分析:")
    thresholds = [1, 2, 3, 5]
    for threshold in thresholds:
        accuracy = np.mean(abs_errors <= threshold) * 100
        print(f"  误差 ≤ {threshold} 分的准确率: {accuracy:.1f}%")

    return rf_model, scaler, X_test_scaled, y_test, y_pred_test, {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_mean': cv_mean,
        'cv_std': cv_std
    }

def analyze_feature_importance(model, feature_names):
    """分析特征重要性"""
    print("\n" + "="*60)
    print("特征重要性分析")
    print("="*60)

    importances = model.feature_importances_

    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': importances
    })

    # 按重要性排序
    feature_importance_df = feature_importance_df.sort_values('重要性', ascending=False)

    print("\n特征重要性排名:")
    print(feature_importance_df.to_string(index=False))

    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(feature_importance_df)), feature_importance_df['重要性'])
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['特征'])
    plt.xlabel('特征重要性')
    plt.title('中考体育分数预测 - 特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importance_real.png', dpi=300, bbox_inches='tight')
    print(f"\n特征重要性图已保存到: feature_importance_real.png")

    return feature_importance_df

def visualize_results(model, X_test, y_test, y_pred, feature_names):
    """可视化结果"""
    print("\n" + "="*60)
    print("结果可视化")
    print("="*60)

    # 1. 实际值 vs 预测值散点图
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际分数')
    plt.ylabel('预测分数')
    plt.title('实际值 vs 预测值')
    plt.grid(True, alpha=0.3)

    # 2. 残差图
    residuals = y_test - y_pred

    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测分数')
    plt.ylabel('残差')
    plt.title('残差图')
    plt.grid(True, alpha=0.3)

    # 3. 误差分布
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('误差')
    plt.ylabel('频数')
    plt.title('误差分布')
    plt.grid(True, alpha=0.3)

    # 4. 分数分布对比
    plt.subplot(2, 2, 4)
    plt.hist(y_test, bins=30, alpha=0.5, label='实际分数', edgecolor='black')
    plt.hist(y_pred, bins=30, alpha=0.5, label='预测分数', edgecolor='black')
    plt.xlabel('分数')
    plt.ylabel('频数')
    plt.title('实际分数与预测分数分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_performance_real.png', dpi=300, bbox_inches='tight')
    print(f"模型性能图已保存到: model_performance_real.png")

def save_model_and_results(model, scaler, feature_names, metrics):
    """保存模型和结果"""
    print("\n" + "="*60)
    print("保存模型和结果")
    print("="*60)

    # 保存模型
    joblib.dump(model, '中考体育分数预测模型.model.pkl')
    print(f"模型已保存到: 中考体育分数预测模型.model.pkl")

    # 保存scaler
    joblib.dump(scaler, '中考体育分数预测模型.scaler.pkl')
    print(f"标准化器已保存到: 中考体育分数预测模型.scaler.pkl")

    # 保存特征名称
    np.save('中考体育分数预测模型.features.npy', feature_names)
    print(f"特征名称已保存到: 中考体育分数预测模型.features.npy")

    # 保存指标
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('模型性能指标.csv', index=False, encoding='utf-8-sig')
    print(f"模型性能指标已保存到: 模型性能指标.csv")

    # 保存详细报告
    report = f"""
中考体育分数预测模型 - 训练报告
{'='*60}

训练时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

模型信息:
  模型类型: 随机森林回归
  特征数量: {len(feature_names)}
  训练样本数: 4000
  测试样本数: 1000

性能指标:
  测试集 R²: {metrics['test_r2']:.4f}
  测试集 MAE: {metrics['test_mae']:.4f}
  测试集 RMSE: {metrics['test_rmse']:.4f}
  交叉验证 R²: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})

特征列表:
{chr(10).join(f'  {i+1:2d}. {feature}' for i, feature in enumerate(feature_names))}

使用说明:
  1. 加载模型: joblib.load('中考体育分数预测模型.model.pkl')
  2. 加载标准化器: joblib.load('中考体育分数预测模型.scaler.pkl')
  3. 加载特征: np.load('中考体育分数预测模型.features.npy', allow_pickle=True)
  4. 进行预测:
     - 准备特征数据 (与训练时相同的特征顺序)
     - 标准化: X_scaled = scaler.transform(X)
     - 预测: y_pred = model.predict(X_scaled)

注意事项:
  1. 模型基于真实中考体育分数数据训练
  2. 预测结果仅供参考
  3. 确保输入数据的特征顺序与训练时一致

{'='*60}
"""

    with open('模型训练报告.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"模型训练报告已保存到: 模型训练报告.txt")

def create_prediction_example(model, scaler, feature_names):
    """创建预测示例"""
    print("\n" + "="*60)
    print("创建预测示例")
    print("="*60)

    # 从训练数据中获取特征范围
    df = pd.read_csv('processed_score_data.csv', encoding='utf-8-sig')

    # 创建示例数据
    example_data = {}

    for feature in feature_names:
        if feature in df.columns:
            # 使用该特征的中位数作为示例值
            median_val = df[feature].median()
            example_data[feature] = median_val

    print("示例输入数据 (使用各特征中位数):")
    for i, (feature, value) in enumerate(example_data.items(), 1):
        print(f"  {i:2d}. {feature}: {value:.2f}")

    # 准备输入数据
    X_example = np.array([list(example_data.values())])

    # 标准化
    X_example_scaled = scaler.transform(X_example)

    # 预测
    score = model.predict(X_example_scaled)[0]

    print(f"\n预测结果:")
    print(f"  预测中考体育分数: {score:.2f}")

    # 根据分数给出评价
    if score >= 90:
        level = "优秀"
        comment = "🎉 体育成绩非常优秀！"
    elif score >= 80:
        level = "良好"
        comment = "👍 体育成绩良好！"
    elif score >= 60:
        level = "及格"
        comment = "✅ 达到及格标准。"
    else:
        level = "待提高"
        comment = "💪 需要加强锻炼。"

    print(f"  成绩等级: {level}")
    print(f"  评价: {comment}")

def main():
    """主函数"""
    print("开始基于真实中考分数数据训练模型...")

    # 1. 加载和准备数据
    X, y, feature_names = load_and_prepare_data()

    # 2. 训练随机森林模型
    model, scaler, X_test, y_test, y_pred, metrics = train_random_forest(X, y, feature_names)

    # 3. 分析特征重要性
    feature_importance_df = analyze_feature_importance(model, feature_names)

    # 4. 可视化结果
    visualize_results(model, X_test, y_test, y_pred, feature_names)

    # 5. 保存模型和结果
    save_model_and_results(model, scaler, feature_names, metrics)

    # 6. 创建预测示例
    create_prediction_example(model, scaler, feature_names)

    print("\n" + "="*60)
    print("🎉 模型训练完成！")
    print("="*60)

    print(f"\n📊 关键成果:")
    print(f"1. 使用真实中考分数数据训练 (5000条记录)")
    print(f"2. 测试集 R²: {metrics['test_r2']:.4f}")
    print(f"3. 测试集 MAE: {metrics['test_mae']:.4f}")
    print(f"4. 误差 ≤ 2分的准确率: {np.mean(np.abs(y_test - y_pred) <= 2) * 100:.1f}%")
    print(f"5. 模型文件已保存到当前目录")

    print(f"\n🚀 下一步:")
    print(f"1. 使用模型进行预测")
    print(f"2. 查看特征重要性了解关键因素")
    print(f"3. 根据模型结果制定训练计划")

    return model, scaler, feature_names

if __name__ == "__main__":
    model, scaler, feature_names = main()