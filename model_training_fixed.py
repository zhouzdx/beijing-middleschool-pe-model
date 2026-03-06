import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_clean_data():
    """加载并清理数据"""
    print("加载和清理数据...")

    # 加载数据
    data = np.load('preprocessed_data.npz', allow_pickle=True)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']

    # 加载scaler
    scaler = joblib.load('scaler.pkl')

    print(f"原始训练集形状: {X_train.shape}")
    print(f"原始测试集形状: {X_test.shape}")

    # 检查NaN值
    print(f"\n训练集NaN数量: {np.isnan(X_train).sum()}")
    print(f"测试集NaN数量: {np.isnan(X_test).sum()}")

    # 使用中位数填充NaN值
    imputer = SimpleImputer(strategy='median')
    X_train_clean = imputer.fit_transform(X_train)
    X_test_clean = imputer.transform(X_test)

    print(f"\n清理后训练集形状: {X_train_clean.shape}")
    print(f"清理后测试集形状: {X_test_clean.shape}")
    print(f"清理后训练集NaN数量: {np.isnan(X_train_clean).sum()}")
    print(f"清理后测试集NaN数量: {np.isnan(X_test_clean).sum()}")

    return X_train_clean, X_test_clean, y_train, y_test, scaler, feature_names, imputer

def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """训练随机森林模型"""
    print("\n" + "="*50)
    print("训练随机森林模型")
    print("="*50)

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

    print("训练模型...")
    rf_model.fit(X_train, y_train)

    # 预测
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)

    # 计算指标
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # 交叉验证
    print("进行交叉验证...")
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"\n模型性能:")
    print(f"  训练集 R²: {train_r2:.4f}")
    print(f"  测试集 R²: {test_r2:.4f}")
    print(f"  测试集 MAE: {test_mae:.4f}")
    print(f"  测试集 RMSE: {test_rmse:.4f}")
    print(f"  交叉验证 R²: {cv_mean:.4f} (±{cv_std:.4f})")

    return rf_model, {
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
    print("\n" + "="*50)
    print("特征重要性分析")
    print("="*50)

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
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance_df)), feature_importance_df['重要性'])
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['特征'])
    plt.xlabel('特征重要性')
    plt.title('中考体育能力预测 - 特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n特征重要性图已保存到: [feature_importance.png](file:///D:/agent/模型/feature_importance.png)")

    return feature_importance_df

def visualize_results(model, X_test, y_test, feature_names):
    """可视化结果"""
    print("\n" + "="*50)
    print("结果可视化")
    print("="*50)

    # 预测
    y_pred = model.predict(X_test)

    # 1. 实际值 vs 预测值散点图
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际分数')
    plt.ylabel('预测分数')
    plt.title('实际值 vs 预测值')
    plt.grid(True, alpha=0.3)

    # 2. 残差图
    residuals = y_test - y_pred

    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
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

    # 4. 预测误差箱线图
    plt.subplot(2, 2, 4)
    error_percentage = np.abs(residuals / y_test) * 100
    plt.boxplot(error_percentage)
    plt.ylabel('相对误差百分比 (%)')
    plt.title('预测相对误差分布')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    print(f"模型性能图已保存到: [model_performance.png](file:///D:/agent/模型/model_performance.png)")

    # 5. 分数分布对比
    plt.figure(figsize=(10, 6))
    plt.hist(y_test, bins=30, alpha=0.5, label='实际分数', edgecolor='black')
    plt.hist(y_pred, bins=30, alpha=0.5, label='预测分数', edgecolor='black')
    plt.xlabel('分数')
    plt.ylabel('频数')
    plt.title('实际分数与预测分数分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('score_distribution.png', dpi=300, bbox_inches='tight')
    print(f"分数分布图已保存到: [score_distribution.png](file:///D:/agent/模型/score_distribution.png)")

    # 6. 特征重要性可视化（单独图）
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title('特征重要性排序')
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('特征')
    plt.ylabel('重要性')
    plt.tight_layout()
    plt.savefig('feature_importance_sorted.png', dpi=300, bbox_inches='tight')
    print(f"特征重要性排序图已保存到: [feature_importance_sorted.png](file:///D:/agent/模型/feature_importance_sorted.png)")

def save_model_and_results(model, imputer, scaler, feature_names, metrics):
    """保存模型和结果"""
    print("\n" + "="*50)
    print("保存模型和结果")
    print("="*50)

    # 创建处理管道
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('imputer', imputer),
        ('model', model)
    ])

    # 保存管道
    joblib.dump(pipeline, 'sports_score_predictor.pipeline.pkl')
    print(f"预测管道已保存到: [sports_score_predictor.pipeline.pkl](file:///D:/agent/模型/sports_score_predictor.pipeline.pkl)")

    # 单独保存模型
    joblib.dump(model, 'sports_score_predictor.model.pkl')
    print(f"模型已保存到: [sports_score_predictor.model.pkl](file:///D:/agent/模型/sports_score_predictor.model.pkl)")

    # 保存scaler
    joblib.dump(scaler, 'sports_score_predictor.scaler.pkl')
    print(f"标准化器已保存到: [sports_score_predictor.scaler.pkl](file:///D:/agent/模型/sports_score_predictor.scaler.pkl)")

    # 保存特征名称
    np.save('sports_score_predictor.features.npy', feature_names)
    print(f"特征名称已保存到: [sports_score_predictor.features.npy](file:///D:/agent/模型/sports_score_predictor.features.npy)")

    # 保存指标
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('model_metrics.csv', index=False, encoding='utf-8-sig')
    print(f"模型指标已保存到: [model_metrics.csv](file:///D:/agent/模型/model_metrics.csv)")

    # 保存详细报告
    with open('model_report.txt', 'w', encoding='utf-8') as f:
        f.write("中考体育能力预测模型 - 训练报告\n")
        f.write("="*50 + "\n\n")
        f.write(f"模型类型: 随机森林回归\n")
        f.write(f"特征数量: {len(feature_names)}\n")
        f.write(f"训练样本数: {X_train.shape[0]}\n")
        f.write(f"测试样本数: {X_test.shape[0]}\n\n")
        f.write("性能指标:\n")
        f.write(f"  测试集 R²: {metrics['test_r2']:.4f}\n")
        f.write(f"  测试集 MAE: {metrics['test_mae']:.4f}\n")
        f.write(f"  测试集 RMSE: {metrics['test_rmse']:.4f}\n")
        f.write(f"  交叉验证 R²: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})\n\n")
        f.write("特征列表:\n")
        for i, feature in enumerate(feature_names, 1):
            f.write(f"  {i:2d}. {feature}\n")

    print(f"模型报告已保存到: [model_report.txt](file:///D:/agent/模型/model_report.txt)")

def create_prediction_example():
    """创建预测示例"""
    print("\n" + "="*50)
    print("创建预测示例")
    print("="*50)

    # 加载模型和工具
    pipeline = joblib.load('sports_score_predictor.pipeline.pkl')
    scaler = joblib.load('sports_score_predictor.scaler.pkl')
    feature_names = np.load('sports_score_predictor.features.npy', allow_pickle=True)

    # 创建示例数据（男生）
    example_male = {
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

    # 创建示例数据（女生）
    example_female = {
        'gender': 1,      # 女生
        'sg': 160,        # 身高160cm
        'tz': 50,         # 体重50kg
        'bmi': 50 / ((160/100)**2),  # BMI
        'mm': 8.5,        # 50米8.5秒
        'lm': 0,          # 1000米（女生为0）
        'mmm': 3.5,       # 800米3.5分钟
        'ytxs': 0,        # 引体向上（女生为0）
        'ywqz': 35,       # 仰卧起坐35个
        'ldty': 180,      # 立定跳远180cm
        'zwtqq': 18,      # 坐位体前屈18cm
        'fhl': 3000       # 肺活量3000ml
    }

    # 转换为数组
    X_male = np.array([list(example_male.values())])
    X_female = np.array([list(example_female.values())])

    # 预测
    score_male = pipeline.predict(X_male)[0]
    score_female = pipeline.predict(X_female)[0]

    print("预测示例:")
    print(f"\n男生示例:")
    for key, value in example_male.items():
        print(f"  {key}: {value}")
    print(f"  预测中考体育分数: {score_male:.1f}/60")

    print(f"\n女生示例:")
    for key, value in example_female.items():
        print(f"  {key}: {value}")
    print(f"  预测中考体育分数: {score_female:.1f}/60")

    # 保存示例
    examples = {
        '男生示例': example_male,
        '女生示例': example_female,
        '预测分数': {
            '男生': float(score_male),
            '女生': float(score_female)
        }
    }

    import json
    with open('prediction_examples.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"\n预测示例已保存到: [prediction_examples.json](file:///D:/agent/模型/prediction_examples.json)")

def main():
    """主函数"""
    print("="*50)
    print("中考体育能力预测模型 - 模型训练（修复版）")
    print("="*50)

    # 1. 加载并清理数据
    X_train, X_test, y_train, y_test, scaler, feature_names, imputer = load_and_clean_data()

    # 2. 训练随机森林模型
    rf_model, metrics = train_random_forest(X_train, X_test, y_train, y_test, feature_names)

    # 3. 分析特征重要性
    feature_importance_df = analyze_feature_importance(rf_model, feature_names)

    # 4. 可视化结果
    visualize_results(rf_model, X_test, y_test, feature_names)

    # 5. 保存模型和结果
    save_model_and_results(rf_model, imputer, scaler, feature_names, metrics)

    # 6. 创建预测示例
    create_prediction_example()

    print("\n" + "="*50)
    print("模型训练完成！")
    print("="*50)

    return rf_model, imputer, scaler, feature_names

if __name__ == "__main__":
    # 全局变量用于示例函数
    global X_train, X_test
    rf_model, imputer, scaler, feature_names = main()