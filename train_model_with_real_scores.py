import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("="*60)
    print("加载和预处理真实中考分数数据")
    print("="*60)

    # 加载处理后的数据
    df = pd.read_csv('processed_score_data.csv', encoding='utf-8-sig')

    print(f"原始数据形状: {df.shape}")
    print(f"列数: {len(df.columns)}")

    # 显示数据信息
    print("\n数据列:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    # 检查目标变量
    print("\n目标变量分析:")
    if '总分数' in df.columns:
        target = df['总分数']
        print(f"  目标变量: 总分数")
        print(f"  非空数量: {target.count()}")
        print(f"  范围: {target.min():.2f} - {target.max():.2f}")
        print(f"  平均值: {target.mean():.2f}")
        print(f"  标准差: {target.std():.2f}")
    else:
        print("  错误: 未找到总分数列")
        return None, None, None

    return df

def prepare_features_and_target(df):
    """准备特征和目标变量"""
    print("\n" + "="*60)
    print("准备特征和目标变量")
    print("="*60)

    # 目标变量
    y = df['总分数'].values

    # 选择特征
    # 基础特征（原始测试成绩）
    base_features = [
        '身高_cm',           # 身高
        '体重_kg',           # 体重
        '肺活量_ml',         # 肺活量
        '50米跑_秒',         # 50米跑成绩
        '坐位体前屈_cm',     # 坐位体前屈
        '跳绳_个'           # 跳绳
    ]

    # 评分特征
    score_features = [
        'BMI分数',           # BMI评分
        '体重分数',          # 体重评分
        '肺活量分数',        # 肺活量评分
        '50米跑分数',        # 50米跑评分
        '坐位体前屈分数',    # 坐位体前屈评分
        '跳绳分数'          # 跳绳评分
    ]

    # 等级特征（需要编码）
    level_features = [
        '体重等级',          # 体重等级
        '肺活量等级',        # 肺活量等级
        '50米跑等级',        # 50米跑等级
        '坐位体前屈等级',    # 坐位体前屈等级
        '跳绳等级'          # 跳绳等级
    ]

    # 其他特征
    other_features = [
        '性别',              # 性别（需要编码）
        '标准分数',          # 标准分
        '附加分'            # 附加分
    ]

    # 检查哪些特征可用
    available_features = []
    all_possible_features = base_features + score_features + level_features + other_features

    for feature in all_possible_features:
        if feature in df.columns:
            available_features.append(feature)

    print(f"可用特征数量: {len(available_features)}")
    print(f"可用特征: {available_features}")

    # 准备特征数据
    X = df[available_features].copy()

    # 处理分类特征
    print("\n处理分类特征...")

    # 编码性别（如果存在）
    if '性别' in X.columns:
        # 性别编码：1=男，2=女 -> 0=男，1=女
        X['性别'] = X['性别'].map({1: 0, 2: 1})
        print(f"  性别编码完成: 男=0, 女=1")

    # 编码等级特征
    level_mapping = {
        '不及格': 0,
        '及格': 1,
        '良好': 2,
        '优秀': 3
    }

    for level_feature in level_features:
        if level_feature in X.columns:
            X[level_feature] = X[level_feature].map(level_mapping)
            print(f"  {level_feature} 编码完成")

    # 处理缺失值
    print("\n处理缺失值...")
    missing_before = X.isnull().sum().sum()
    print(f"  缺失值总数: {missing_before}")

    # 使用中位数填充数值特征
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy='median')

    # 确保所有数值列都是数值类型
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X_imputed = imputer.fit_transform(X[numeric_cols])
    X_imputed_df = pd.DataFrame(X_imputed, columns=numeric_cols, index=X.index)

    # 更新数值列
    for col in numeric_cols:
        X[col] = X_imputed_df[col]

    # 对于非数值列，使用众数填充
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if X[col].isnull().any():
            mode_val = X[col].mode()[0]
            X[col].fillna(mode_val, inplace=True)

    missing_after = X.isnull().sum().sum()
    print(f"  处理后缺失值: {missing_after}")

    # 确保所有数据都是数值类型
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col].fillna(X[col].median(), inplace=True)
            except:
                # 如果无法转换为数值，使用标签编码
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

    return X.values, y, available_features, imputer

def train_and_evaluate_models(X, y, feature_names):
    """训练和评估多个模型"""
    print("\n" + "="*60)
    print("训练和评估多个模型")
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

    # 定义模型
    models = {
        '随机森林': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        '梯度提升': GradientBoostingRegressor(n_estimators=100, random_state=42),
        '线性回归': LinearRegression(),
        '岭回归': Ridge(alpha=1.0)
    }

    results = {}

    for name, model in models.items():
        print(f"\n训练 {name}...")

        try:
            # 训练模型
            model.fit(X_train_scaled, y_train)

            # 预测
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # 计算指标
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            # 交叉验证
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            results[name] = {
                'model': model,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'y_pred_test': y_pred_test
            }

            print(f"  训练集 R²: {train_r2:.4f}")
            print(f"  测试集 R²: {test_r2:.4f}")
            print(f"  测试集 MAE: {test_mae:.4f}")
            print(f"  测试集 RMSE: {test_rmse:.4f}")
            print(f"  交叉验证 R²: {cv_mean:.4f} (±{cv_std:.4f})")

        except Exception as e:
            print(f"  训练失败: {e}")
            results[name] = None

    return results, X_test_scaled, y_test, scaler

def select_best_model(results):
    """选择最佳模型"""
    print("\n" + "="*60)
    print("模型性能比较")
    print("="*60)

    # 创建比较表格
    comparison = []
    for name, result in results.items():
        if result is not None:
            comparison.append({
                '模型': name,
                '测试集R²': result['test_r2'],
                '测试集MAE': result['test_mae'],
                '测试集RMSE': result['test_rmse'],
                '交叉验证R²': result['cv_mean'],
                '交叉验证标准差': result['cv_std']
            })

    df_comparison = pd.DataFrame(comparison)
    df_comparison = df_comparison.sort_values('测试集R²', ascending=False)

    print("\n模型性能排名:")
    print(df_comparison.to_string(index=False))

    # 选择最佳模型（基于测试集R²）
    best_model_name = df_comparison.iloc[0]['模型']
    best_result = results[best_model_name]

    print(f"\n🎯 最佳模型: {best_model_name}")
    print(f"   测试集 R²: {best_result['test_r2']:.4f}")
    print(f"   测试集 MAE: {best_result['test_mae']:.4f}")
    print(f"   测试集 RMSE: {best_result['test_rmse']:.4f}")
    print(f"   交叉验证 R²: {best_result['cv_mean']:.4f} (±{best_result['cv_std']:.4f})")

    return best_model_name, best_result

def analyze_feature_importance(model, feature_names, scaler):
    """分析特征重要性"""
    print("\n" + "="*60)
    print("特征重要性分析")
    print("="*60)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            '特征': feature_names,
            '重要性': importances
        })

        # 按重要性排序
        feature_importance_df = feature_importance_df.sort_values('重要性', ascending=False)

        print("\n特征重要性排名 (前15项):")
        print(feature_importance_df.head(15).to_string(index=False))

        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['重要性'])
        plt.yticks(range(len(top_features)), top_features['特征'])
        plt.xlabel('特征重要性')
        plt.title('中考体育分数预测 - 特征重要性 (前15项)')
        plt.tight_layout()
        plt.savefig('feature_importance_real_scores.png', dpi=300, bbox_inches='tight')
        print(f"\n特征重要性图已保存到: feature_importance_real_scores.png")

        return feature_importance_df

    elif hasattr(model, 'coef_'):
        # 线性模型系数
        coefficients = model.coef_

        # 创建系数DataFrame
        coef_df = pd.DataFrame({
            '特征': feature_names,
            '系数': coefficients
        })

        # 按绝对值排序
        coef_df['绝对值'] = np.abs(coef_df['系数'])
        coef_df = coef_df.sort_values('绝对值', ascending=False)

        print("\n特征系数排名 (前15项):")
        print(coef_df.head(15)[['特征', '系数']].to_string(index=False))

        return coef_df

    else:
        print("该模型不支持特征重要性分析")
        return None

def visualize_results(results, X_test, y_test, feature_names):
    """可视化结果"""
    print("\n" + "="*60)
    print("结果可视化")
    print("="*60)

    # 获取最佳模型的预测结果
    best_model_name = None
    best_result = None

    for name, result in results.items():
        if result is not None:
            if best_result is None or result['test_r2'] > best_result['test_r2']:
                best_model_name = name
                best_result = result

    if best_result is None:
        print("没有可用的模型结果进行可视化")
        return

    y_pred = best_result['y_pred_test']

    # 1. 实际值 vs 预测值散点图
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际分数')
    plt.ylabel('预测分数')
    plt.title(f'实际值 vs 预测值 ({best_model_name})')
    plt.grid(True, alpha=0.3)

    # 2. 残差图
    residuals = y_test - y_pred

    plt.subplot(2, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测分数')
    plt.ylabel('残差')
    plt.title('残差图')
    plt.grid(True, alpha=0.3)

    # 3. 误差分布
    plt.subplot(2, 3, 3)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('误差')
    plt.ylabel('频数')
    plt.title('误差分布')
    plt.grid(True, alpha=0.3)

    # 4. 分数分布对比
    plt.subplot(2, 3, 4)
    plt.hist(y_test, bins=30, alpha=0.5, label='实际分数', edgecolor='black')
    plt.hist(y_pred, bins=30, alpha=0.5, label='预测分数', edgecolor='black')
    plt.xlabel('分数')
    plt.ylabel('频数')
    plt.title('实际分数与预测分数分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. 累积误差分布
    sorted_abs_errors = np.sort(np.abs(residuals))
    cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)

    plt.subplot(2, 3, 5)
    plt.plot(sorted_abs_errors, cumulative, marker='.', linestyle='none')
    plt.xlabel('绝对误差')
    plt.ylabel('累积比例')
    plt.title('累积误差分布')
    plt.grid(True, alpha=0.3)

    # 6. 模型性能对比
    plt.subplot(2, 3, 6)
    model_names = []
    test_r2_scores = []

    for name, result in results.items():
        if result is not None:
            model_names.append(name)
            test_r2_scores.append(result['test_r2'])

    plt.bar(range(len(model_names)), test_r2_scores)
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.ylabel('测试集 R²')
    plt.title('模型性能对比')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_performance_real_scores.png', dpi=300, bbox_inches='tight')
    print(f"模型性能图已保存到: model_performance_real_scores.png")

def save_model_and_results(best_model, scaler, imputer, feature_names, results):
    """保存模型和结果"""
    print("\n" + "="*60)
    print("保存模型和结果")
    print("="*60)

    # 创建处理管道
    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('model', best_model)
    ])

    # 保存管道
    joblib.dump(pipeline, '中考体育分数预测模型.pipeline.pkl')
    print(f"预测管道已保存到: 中考体育分数预测模型.pipeline.pkl")

    # 保存scaler
    joblib.dump(scaler, '中考体育分数预测模型.scaler.pkl')
    print(f"标准化器已保存到: 中考体育分数预测模型.scaler.pkl")

    # 保存特征名称
    np.save('中考体育分数预测模型.features.npy', feature_names)
    print(f"特征名称已保存到: 中考体育分数预测模型.features.npy")

    # 保存模型比较结果
    results_df = pd.DataFrame([
        {
            '模型': name,
            '训练集R²': result['train_r2'],
            '测试集R²': result['test_r2'],
            '测试集MAE': result['test_mae'],
            '测试集RMSE': result['test_rmse'],
            '交叉验证R²': result['cv_mean'],
            '交叉验证标准差': result['cv_std']
        }
        for name, result in results.items() if result is not None
    ])

    results_df = results_df.sort_values('测试集R²', ascending=False)
    results_df.to_csv('模型比较结果.csv', index=False, encoding='utf-8-sig')
    print(f"模型比较结果已保存到: 模型比较结果.csv")

    # 保存最佳模型详细信息
    best_model_name = None
    best_result = None

    for name, result in results.items():
        if result is not None:
            if best_result is None or result['test_r2'] > best_result['test_r2']:
                best_model_name = name
                best_result = result

    if best_result:
        with open('最佳模型信息.txt', 'w', encoding='utf-8') as f:
            f.write(f"最佳模型: {best_model_name}\n")
            f.write(f"测试集 R²: {best_result['test_r2']:.4f}\n")
            f.write(f"测试集 MAE: {best_result['test_mae']:.4f}\n")
            f.write(f"测试集 RMSE: {best_result['test_rmse']:.4f}\n")
            f.write(f"交叉验证 R²: {best_result['cv_mean']:.4f} (±{best_result['cv_std']:.4f})\n")
            f.write(f"\n特征数量: {len(feature_names)}\n")
            f.write(f"训练样本数: 4000\n")
            f.write(f"测试样本数: 1000\n")

        print(f"最佳模型信息已保存到: 最佳模型信息.txt")

def main():
    """主函数"""
    print("开始基于真实中考分数数据训练模型...")

    # 1. 加载和预处理数据
    df = load_and_preprocess_data()
    if df is None:
        return

    # 2. 准备特征和目标变量
    X, y, feature_names, imputer = prepare_features_and_target(df)

    # 3. 训练和评估模型
    results, X_test, y_test, scaler = train_and_evaluate_models(X, y, feature_names)

    # 4. 选择最佳模型
    best_model_name, best_result = select_best_model(results)
    best_model = best_result['model']

    # 5. 分析特征重要性
    feature_importance_df = analyze_feature_importance(best_model, feature_names, scaler)

    # 6. 可视化结果
    visualize_results(results, X_test, y_test, feature_names)

    # 7. 保存模型和结果
    save_model_and_results(best_model, scaler, imputer, feature_names, results)

    print("\n" + "="*60)
    print("🎉 模型训练完成！")
    print("="*60)

    print(f"\n📊 关键成果:")
    print(f"1. 使用真实中考分数数据训练")
    print(f"2. 最佳模型: {best_model_name}")
    print(f"3. 测试集 R²: {best_result['test_r2']:.4f}")
    print(f"4. 测试集 MAE: {best_result['test_mae']:.4f}")
    print(f"5. 模型已保存到当前目录")

    print(f"\n🚀 使用说明:")
    print(f"1. 加载模型: joblib.load('中考体育分数预测模型.pipeline.pkl')")
    print(f"2. 进行预测: model.predict(X_new)")
    print(f"3. 查看特征重要性: 特征重要性排名")

    return best_model, scaler, imputer, feature_names

if __name__ == "__main__":
    best_model, scaler, imputer, feature_names = main()