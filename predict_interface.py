import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import StringIO
import sys

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_model():
    """加载模型和相关工具"""
    print("🔧 加载中考体育分数预测模型...")
    try:
        model = joblib.load('中考体育分数预测模型.model.pkl')
        scaler = joblib.load('中考体育分数预测模型.scaler.pkl')
        feature_names = np.load('中考体育分数预测模型.features.npy', allow_pickle=True)

        print("✅ 模型加载成功！")
        print(f"   模型类型: {type(model).__name__}")
        print(f"   特征数量: {len(feature_names)}")

        return model, scaler, feature_names
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None, None, None

def print_header():
    """打印标题"""
    print("="*70)
    print("          中考体育分数预测系统 (基于真实数据训练)")
    print("="*70)
    print()

def get_user_input(feature_names):
    """获取用户输入"""
    print("\n📝 请输入学生体育测试成绩:")
    print("-" * 50)

    input_data = {}

    # 特征说明
    feature_descriptions = {
        '身高_cm': '身高 (厘米)',
        '体重_kg': '体重 (公斤)',
        '肺活量_ml': '肺活量 (毫升)',
        '50米跑_秒': '50米跑成绩 (秒)',
        '坐位体前屈_cm': '坐位体前屈成绩 (厘米)',
        '跳绳_个': '跳绳个数 (个)',
        'BMI分数': 'BMI评分 (0-100分)',
        '体重分数': '体重评分 (0-100分)',
        '肺活量分数': '肺活量评分 (0-100分)',
        '50米跑分数': '50米跑评分 (0-100分)',
        '坐位体前屈分数': '坐位体前屈评分 (0-100分)',
        '跳绳分数': '跳绳评分 (0-100分)',
        '标准分数': '标准分 (0-100分)'
    }

    # 获取每个特征的输入
    for feature in feature_names:
        if feature in feature_descriptions:
            description = feature_descriptions[feature]

            while True:
                try:
                    # 显示建议范围
                    if feature == '身高_cm':
                        suggestion = "(建议: 小学生 120-160cm, 初中生 140-180cm)"
                    elif feature == '体重_kg':
                        suggestion = "(建议: 小学生 20-50kg, 初中生 30-70kg)"
                    elif feature == '肺活量_ml':
                        suggestion = "(建议: 小学生 800-2500ml, 初中生 1500-4000ml)"
                    elif feature == '50米跑_秒':
                        suggestion = "(建议: 小学生 9-13秒, 初中生 7-11秒)"
                    elif feature == '坐位体前屈_cm':
                        suggestion = "(建议: 0-25cm)"
                    elif feature == '跳绳_个':
                        suggestion = "(建议: 30-150个)"
                    elif '分数' in feature:
                        suggestion = "(0-100分)"
                    else:
                        suggestion = ""

                    value = input(f"{description} {suggestion}: ").strip()

                    if value == "":
                        # 使用默认值
                        if '分数' in feature:
                            default = 75.0
                        elif feature == '身高_cm':
                            default = 150.0
                        elif feature == '体重_kg':
                            default = 45.0
                        elif feature == '肺活量_ml':
                            default = 2000.0
                        elif feature == '50米跑_秒':
                            default = 10.0
                        elif feature == '坐位体前屈_cm':
                            default = 12.0
                        elif feature == '跳绳_个':
                            default = 80.0
                        else:
                            default = 0.0

                        input_data[feature] = default
                        print(f"  使用默认值: {default}")
                        break
                    else:
                        input_data[feature] = float(value)
                        break

                except ValueError:
                    print("  请输入有效的数字！")
                except Exception as e:
                    print(f"  输入错误: {e}")

    return input_data

def predict_score(model, scaler, input_data, feature_names):
    """预测分数"""
    # 确保特征顺序正确
    X = np.array([[input_data[feature] for feature in feature_names]])

    # 标准化
    X_scaled = scaler.transform(X)

    # 预测
    score = model.predict(X_scaled)[0]

    return score

def get_score_level(score):
    """获取分数等级"""
    if score >= 90:
        return "优秀", "🎉 体育成绩非常优秀！继续保持！"
    elif score >= 80:
        return "良好", "👍 体育成绩良好，还有提升空间！"
    elif score >= 60:
        return "及格", "✅ 达到及格标准，需要加强锻炼。"
    else:
        return "待提高", "💪 需要加强锻炼，提升体育成绩。"

def analyze_weaknesses(input_data, feature_names):
    """分析薄弱环节"""
    weaknesses = []

    # 各项测试的参考标准
    standards = {
        '50米跑_秒': {'good': 8.5, 'fair': 10.0, 'comment': '速度能力'},
        '坐位体前屈_cm': {'good': 15.0, 'fair': 10.0, 'comment': '柔韧性'},
        '跳绳_个': {'good': 100, 'fair': 70, 'comment': '协调性'},
        '肺活量_ml': {'good': 2500, 'fair': 1800, 'comment': '心肺功能'}
    }

    for feature, standard in standards.items():
        if feature in input_data:
            value = input_data[feature]

            if feature == '50米跑_秒':  # 时间越短越好
                if value > standard['fair']:
                    weaknesses.append(f"🏃 {standard['comment']}: {value}{'秒' if feature == '50米跑_秒' else ''} (需要提高)")
            else:  # 数值越大越好
                if value < standard['fair']:
                    weaknesses.append(f"💪 {standard['comment']}: {value}{'个' if feature == '跳绳_个' else 'ml' if feature == '肺活量_ml' else 'cm'} (需要提高)")

    # 检查各项分数
    score_features = [f for f in feature_names if '分数' in f and f != '标准分数']
    for feature in score_features:
        if feature in input_data:
            score = input_data[feature]
            if score < 70:
                feature_name = feature.replace('分数', '')
                weaknesses.append(f"📊 {feature_name}评分: {score}分 (需要提高)")

    return weaknesses

def display_results(input_data, score, feature_names):
    """显示预测结果"""
    print("\n" + "="*70)
    print("📊 预测结果")
    print("="*70)

    # 基本信息
    print(f"\n📋 输入数据:")
    for i, feature in enumerate(feature_names, 1):
        if feature in input_data:
            # 格式化显示
            if '身高' in feature:
                unit = 'cm'
            elif '体重' in feature:
                unit = 'kg'
            elif '肺活量' in feature:
                unit = 'ml'
            elif '秒' in feature:
                unit = '秒'
            elif 'cm' in feature:
                unit = 'cm'
            elif '个' in feature:
                unit = '个'
            elif '分数' in feature:
                unit = '分'
            else:
                unit = ''

            print(f"  {i:2d}. {feature}: {input_data[feature]:.1f}{unit}")

    # 预测分数
    print(f"\n🎯 预测结果:")
    print(f"  中考体育预测分数: {score:.1f}")

    # 等级评价
    level, message = get_score_level(score)
    print(f"  成绩等级: {level}")
    print(f"  评价: {message}")

    # 分数解释
    print(f"\n📈 分数解释:")
    if score >= 90:
        print("  • 体育各项能力均衡发展")
        print("  • 具备良好的运动基础")
        print("  • 中考体育有望获得高分")
    elif score >= 80:
        print("  • 体育能力总体良好")
        print("  • 部分项目有提升空间")
        print("  • 通过训练可以进一步提高")
    elif score >= 60:
        print("  • 达到基本体育要求")
        print("  • 需要系统性训练")
        print("  • 重点关注薄弱项目")
    else:
        print("  • 需要加强体育锻炼")
        print("  • 建议制定训练计划")
        print("  • 从基础项目开始提高")

    # 薄弱环节分析
    weaknesses = analyze_weaknesses(input_data, feature_names)
    if weaknesses:
        print(f"\n💡 薄弱环节分析:")
        for i, weakness in enumerate(weaknesses, 1):
            print(f"  {i}. {weakness}")

        print(f"\n🏋️ 训练建议:")
        if any('速度能力' in w for w in weaknesses):
            print("  • 进行短跑训练，提高爆发力")
        if any('柔韧性' in w for w in weaknesses):
            print("  • 每天进行拉伸练习")
        if any('协调性' in w for w in weaknesses):
            print("  • 练习跳绳，提高协调性")
        if any('心肺功能' in w for w in weaknesses):
            print("  • 进行有氧运动，如跑步、游泳")
    else:
        print(f"\n🎯 各项能力均衡，继续保持！")

    # 创建可视化
    create_score_visualization(score, level)

def create_score_visualization(score, level):
    """创建分数可视化"""
    try:
        plt.figure(figsize=(10, 6))

        # 创建仪表盘
        ax = plt.subplot(1, 1, 1, polar=True)

        # 设置角度
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)

        # 绘制背景
        ax.plot(x, y, color='gray', linewidth=20, alpha=0.2)

        # 根据分数计算角度
        score_angle = (score / 120) * 2 * np.pi  # 假设满分120

        # 绘制分数弧
        score_x = np.cos(np.linspace(0, score_angle, 100))
        score_y = np.sin(np.linspace(0, score_angle, 100))

        # 根据等级选择颜色
        if level == "优秀":
            color = 'green'
        elif level == "良好":
            color = 'yellow'
        elif level == "及格":
            color = 'orange'
        else:
            color = 'red'

        ax.plot(score_x, score_y, color=color, linewidth=20, alpha=0.8)

        # 添加指针
        pointer_x = np.cos(score_angle) * 0.9
        pointer_y = np.sin(score_angle) * 0.9
        ax.plot([0, pointer_x], [0, pointer_y], color='black', linewidth=3)

        # 添加分数文本
        ax.text(0, 0.2, f'{score:.1f}', ha='center', va='center',
                fontsize=40, fontweight='bold')
        ax.text(0, -0.1, level, ha='center', va='center',
                fontsize=20, color=color)

        # 设置坐标轴
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        ax.set_title('中考体育分数预测', size=16, y=1.1)

        plt.tight_layout()
        plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 预测结果图已保存到: prediction_result.png")

    except Exception as e:
        print(f"  创建可视化失败: {e}")

def save_prediction_result(input_data, score, level):
    """保存预测结果"""
    import json
    from datetime import datetime

    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prediction': {
            'score': float(score),
            'level': level,
            'max_score': 120  # 假设满分120
        },
        'input_data': input_data
    }

    # 生成文件名
    filename = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"💾 预测结果已保存到: {filename}")
    return filename

def show_example_prediction(model, scaler, feature_names):
    """显示示例预测"""
    print("\n" + "="*70)
    print("📋 示例预测")
    print("="*70)

    # 示例数据（基于数据的中位数）
    example_data = {}

    # 加载数据获取中位数
    try:
        df = pd.read_csv('processed_score_data.csv', encoding='utf-8-sig', nrows=1)
        for feature in feature_names:
            if feature in df.columns:
                # 使用该特征的中位数作为示例值
                median_val = pd.read_csv('processed_score_data.csv', encoding='utf-8-sig')[feature].median()
                example_data[feature] = median_val
    except:
        # 如果无法读取数据，使用默认值
        defaults = {
            '身高_cm': 150.0,
            '体重_kg': 45.0,
            '肺活量_ml': 2000.0,
            '50米跑_秒': 10.0,
            '坐位体前屈_cm': 12.0,
            '跳绳_个': 80.0,
            'BMI分数': 75.0,
            '体重分数': 85.0,
            '肺活量分数': 75.0,
            '50米跑分数': 75.0,
            '坐位体前屈分数': 75.0,
            '跳绳分数': 75.0,
            '标准分数': 75.0
        }

        for feature in feature_names:
            example_data[feature] = defaults.get(feature, 0.0)

    print("\n示例输入数据 (基于平均学生水平):")
    for i, (feature, value) in enumerate(example_data.items(), 1):
        print(f"  {i:2d}. {feature}: {value:.1f}")

    # 预测
    score = predict_score(model, scaler, example_data, feature_names)

    print(f"\n🎯 示例预测结果:")
    print(f"  预测中考体育分数: {score:.1f}")

    level, message = get_score_level(score)
    print(f"  成绩等级: {level}")
    print(f"  评价: {message}")

def main():
    """主函数"""
    print_header()

    # 加载模型
    model, scaler, feature_names = load_model()
    if model is None:
        return

    while True:
        print("\n请选择操作:")
        print("1. 进行中考体育分数预测")
        print("2. 查看示例预测")
        print("3. 查看模型信息")
        print("4. 退出")

        choice = input("\n请输入选择 (1-4): ").strip()

        if choice == '1':
            # 获取用户输入
            input_data = get_user_input(feature_names)

            # 预测分数
            score = predict_score(model, scaler, input_data, feature_names)

            # 显示结果
            display_results(input_data, score, feature_names)

            # 保存结果
            level, _ = get_score_level(score)
            save_file = save_prediction_result(input_data, score, level)

            print(f"\n📁 相关文件:")
            print(f"  • 预测结果: {save_file}")
            print(f"  • 可视化图: prediction_result.png")

        elif choice == '2':
            # 显示示例
            show_example_prediction(model, scaler, feature_names)

        elif choice == '3':
            # 显示模型信息
            print("\n" + "="*70)
            print("🔧 模型信息")
            print("="*70)

            print(f"\n模型类型: 随机森林回归")
            print(f"训练数据: 5000条真实中考体育分数记录")
            print(f"特征数量: {len(feature_names)}")
            print(f"模型性能: R² = 0.9711, MAE = 0.72分")

            print(f"\n特征列表:")
            for i, feature in enumerate(feature_names, 1):
                print(f"  {i:2d}. {feature}")

            print(f"\n💡 使用说明:")
            print(f"  1. 输入各项体育测试成绩")
            print(f"  2. 系统预测中考体育分数")
            print(f"  3. 提供薄弱环节分析和训练建议")

        elif choice == '4':
            print("\n感谢使用中考体育分数预测系统！再见！👋")
            break

        else:
            print("❌ 无效的选择，请重新输入。")

        # 询问是否继续
        if choice in ['1', '2', '3']:
            continue_choice = input("\n是否继续使用？ (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\n感谢使用中考体育分数预测系统！再见！👋")
                break

if __name__ == "__main__":
    main()