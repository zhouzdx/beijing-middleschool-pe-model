import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载预处理后的数据"""
    print("加载预处理数据...")

    # 加载数据
    data = np.load('preprocessed_data.npz', allow_pickle=True)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']

    # 加载scaler
    scaler = joblib.load('scaler.pkl')

    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"特征数量: {len(feature_names)}")
    print(f"特征名称: {feature_names}")

    return X_train, X_test, y_train, y_test, scaler, feature_names

def train_models(X_train, X_test, y_train, y_test, feature_names):
    """训练多个模型并比较性能"""
    print("\n" + "="*50)
    print("训练和比较多个模型")
    print("="*50)

    models = {
        '线性回归': LinearRegression(),
        '岭回归': Ridge(alpha=1.0),
        'Lasso回归': Lasso(alpha=0.1),
        '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
        '梯度提升': GradientBoostingRegressor(n_estimators=100, random_state=42),
        '支持向量机': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
        '神经网络': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n训练 {name}...")

        try:
            # 训练模型
            model.fit(X_train, y_train)

            # 预测
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # 计算指标
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
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
                'cv_std': cv_std
            }

            print(f"  训练集 R²: {train_r2:.4f}")
            print(f"  测试集 R²: {test_r2:.4f}")
            print(f"  测试集 MAE: {test_mae:.4f}")
            print(f"  交叉验证 R²: {cv_mean:.4f} (±{cv_std:.4f})")

        except Exception as e:
            print(f"  训练失败: {e}")
            results[name] = None

    return results

def select_best_model(results):
    """选择最佳模型"""
    print("\n" + "="*50)
    print("模型性能比较")
    print("="*50)

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

    print(f"\n最佳模型: {best_model_name}")
    print(f"测试集 R²: {best_result['test_r2']:.4f}")
    print(f"测试集 MAE: {best_result['test_mae']:.4f}")
    print(f"交叉验证 R²: {best_result['cv_mean']:.4f} (±{best_result['cv_std']:.4f})")

    return best_model_name, best_result

def optimize_random_forest(X_train, y_train):
    """优化随机森林模型"""
    print("\n" + "="*50)
    print("优化随机森林模型")
    print("="*50)

    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }

    # 创建基础模型
    rf = RandomForestRegressor(random_state=42)

    # 网格搜索
    print("进行网格搜索...")
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='r2',
        n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def analyze_feature_importance(model, feature_names):
    """分析特征重要性"""
    print("\n" + "="*50)
    print("特征重要性分析")
    print("="*50)

    if hasattr(model, 'feature_importances_'):
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
        plt.title('随机森林特征重要性')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\n特征重要性图已保存到: feature_importance.png")

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

        print("\n特征系数排名:")
        print(coef_df[['特征', '系数']].to_string(index=False))

    else:
        print("该模型不支持特征重要性分析")

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
    print(f"模型性能图已保存到: model_performance.png")

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
    print(f"分数分布图已保存到: score_distribution.png")

def save_model(model, model_name, results):
    """保存模型和结果"""
    print("\n" + "="*50)
    print("保存模型和结果")
    print("="*50)

    # 保存模型
    model_filename = f'{model_name.lower().replace(" ", "_")}_model.pkl'
    joblib.dump(model, model_filename)
    print(f"模型已保存到: {model_filename}")

    # 保存结果
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
    results_df.to_csv('model_comparison_results.csv', index=False, encoding='utf-8-sig')
    print(f"模型比较结果已保存到: model_comparison_results.csv")

    # 保存最佳模型详细信息
    best_result = results[model_name]
    with open('best_model_info.txt', 'w', encoding='utf-8') as f:
        f.write(f"最佳模型: {model_name}\n")
        f.write(f"测试集 R²: {best_result['test_r2']:.4f}\n")
        f.write(f"测试集 MAE: {best_result['test_mae']:.4f}\n")
        f.write(f"测试集 RMSE: {best_result['test_rmse']:.4f}\n")
        f.write(f"交叉验证 R²: {best_result['cv_mean']:.4f} (±{best_result['cv_std']:.4f})\n")

    print(f"最佳模型信息已保存到: best_model_info.txt")

def main():
    """主函数"""
    print("="*50)
    print("中考体育能力预测模型 - 模型训练")
    print("="*50)

    # 1. 加载数据
    X_train, X_test, y_train, y_test, scaler, feature_names = load_data()

    # 2. 训练多个模型
    results = train_models(X_train, X_test, y_train, y_test, feature_names)

    # 3. 选择最佳模型
    best_model_name, best_result = select_best_model(results)
    best_model = best_result['model']

    # 4. 优化随机森林（如果是最佳模型）
    if best_model_name == '随机森林':
        print("\n对最佳模型（随机森林）进行优化...")
        best_model = optimize_random_forest(X_train, y_train)

        # 重新评估优化后的模型
        y_pred_test = best_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        print(f"优化后测试集 R²: {test_r2:.4f}")
        print(f"优化后测试集 MAE: {test_mae:.4f}")

    # 5. 分析特征重要性
    analyze_feature_importance(best_model, feature_names)

    # 6. 可视化结果
    visualize_results(best_model, X_test, y_test, feature_names)

    # 7. 保存模型和结果
    save_model(best_model, best_model_name, results)

    print("\n" + "="*50)
    print("模型训练完成！")
    print("="*50)

    return best_model, scaler, feature_names

if __name__ == "__main__":
    best_model, scaler, feature_names = main()