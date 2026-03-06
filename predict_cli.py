import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_model():
    """加载模型和相关工具"""
    try:
        pipeline = joblib.load('sports_score_predictor.pipeline.pkl')
        scaler = joblib.load('sports_score_predictor.scaler.pkl')
        feature_names = np.load('sports_score_predictor.features.npy', allow_pickle=True)
        print("✅ 模型加载成功！")
        return pipeline, scaler, feature_names
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None, None, None

def get_user_input():
    """获取用户输入"""
    print("\n" + "="*50)
    print("中考体育能力预测系统")
    print("="*50)

    # 性别选择
    print("\n👤 学生基本信息:")
    gender = input("请输入性别 (1-男生, 2-女生): ").strip()
    while gender not in ['1', '2']:
        print("请输入有效的选择 (1-男生, 2-女生)")
        gender = input("请输入性别 (1-男生, 2-女生): ").strip()

    gender_code = 0 if gender == '1' else 1
    gender_text = "男生" if gender == '1' else "女生"

    print(f"\n📏 身体测量数据 ({gender_text}):")
    height = float(input("身高 (cm, 例如: 170): "))
    weight = float(input("体重 (kg, 例如: 60): "))
    bmi = weight / ((height / 100) ** 2)

    print(f"\n🏃 运动测试成绩 ({gender_text}):")

    if gender == '1':  # 男生
        run_50m = float(input("50米跑成绩 (秒, 例如: 7.5): "))
        run_1000m = float(input("1000米跑成绩 (分钟, 例如: 3.5): "))
        pull_ups = int(input("引体向上个数 (个, 例如: 10): "))
        sit_ups = 0
        run_800m = 0
    else:  # 女生
        run_50m = float(input("50米跑成绩 (秒, 例如: 8.5): "))
        run_800m = float(input("800米跑成绩 (分钟, 例如: 3.5): "))
        sit_ups = int(input("仰卧起坐个数 (个, 例如: 35): "))
        pull_ups = 0
        run_1000m = 0

    # 共同项目
    long_jump = float(input("立定跳远成绩 (cm, 例如: 200): "))
    sit_reach = float(input("坐位体前屈成绩 (cm, 例如: 15): "))
    vital_capacity = int(input("肺活量 (ml, 例如: 3500): "))

    # 创建输入字典
    input_data = {
        'gender': gender_code,
        'sg': height,
        'tz': weight,
        'bmi': bmi,
        'mm': run_50m,
        'lm': run_1000m,
        'mmm': run_800m,
        'ytxs': pull_ups,
        'ywqz': sit_ups,
        'ldty': long_jump,
        'zwtqq': sit_reach,
        'fhl': vital_capacity
    }

    return input_data, gender_text

def predict_score(pipeline, input_data):
    """预测分数"""
    # 转换为数组
    X = np.array([list(input_data.values())])

    # 预测
    score = pipeline.predict(X)[0]

    return score

def get_score_level(score):
    """获取分数等级"""
    if score >= 54:
        return "优秀", "🎉 非常出色！你的体育成绩很优秀。"
    elif score >= 48:
        return "良好", "👍 成绩良好，继续保持！"
    elif score >= 36:
        return "及格", "✅ 达到及格标准，还有提升空间。"
    else:
        return "待提高", "💪 需要加强锻炼，提升体育成绩。"

def get_bmi_category(bmi):
    """获取BMI分类"""
    if bmi < 18.5:
        return "偏瘦"
    elif bmi < 24:
        return "正常"
    elif bmi < 28:
        return "超重"
    else:
        return "肥胖"

def create_improvement_suggestions(input_data, gender):
    """创建改进建议"""
    suggestions = []

    # 速度建议
    if gender == "男生" and input_data['mm'] > 8:
        suggestions.append("🏃 速度训练: 50米跑成绩有待提高，建议进行短跑训练和爆发力练习。")
    elif gender == "女生" and input_data['mm'] > 9:
        suggestions.append("🏃 速度训练: 50米跑成绩有待提高，建议进行短跑训练和爆发力练习。")

    # 耐力建议
    if gender == "男生" and input_data['lm'] > 4:
        suggestions.append("🏃‍♂️ 耐力训练: 1000米跑需要加强，建议进行有氧跑步训练。")
    elif gender == "女生" and input_data['mmm'] > 4:
        suggestions.append("🏃‍♀️ 耐力训练: 800米跑需要加强，建议进行有氧跑步训练。")

    # 力量建议
    if gender == "男生" and input_data['ytxs'] < 5:
        suggestions.append("💪 上肢力量: 引体向上数量较少，建议进行引体向上辅助训练和背部力量练习。")
    elif gender == "女生" and input_data['ywqz'] < 25:
        suggestions.append("💪 核心力量: 仰卧起坐数量有待提高，建议进行核心肌群训练。")

    # 爆发力建议
    if gender == "男生" and input_data['ldty'] < 220:
        suggestions.append("🦘 爆发力训练: 立定跳远距离较短，建议进行腿部爆发力训练。")
    elif gender == "女生" and input_data['ldty'] < 170:
        suggestions.append("🦘 爆发力训练: 立定跳远距离较短，建议进行腿部爆发力训练。")

    # 柔韧性建议
    if input_data['zwtqq'] < 10:
        suggestions.append("🧘 柔韧性训练: 坐位体前屈成绩一般，建议进行拉伸练习提高柔韧性。")

    # 心肺功能建议
    if gender == "男生" and input_data['fhl'] < 3500:
        suggestions.append("🫁 心肺功能: 肺活量有待提高，建议进行有氧运动和呼吸训练。")
    elif gender == "女生" and input_data['fhl'] < 2500:
        suggestions.append("🫁 心肺功能: 肺活量有待提高，建议进行有氧运动和呼吸训练。")

    return suggestions

def display_results(input_data, gender, score):
    """显示预测结果"""
    print("\n" + "="*50)
    print("📊 预测结果")
    print("="*50)

    # 基本信息
    print(f"\n👤 学生信息:")
    print(f"  性别: {gender}")
    print(f"  身高: {input_data['sg']} cm")
    print(f"  体重: {input_data['tz']} kg")
    print(f"  BMI: {input_data['bmi']:.1f} ({get_bmi_category(input_data['bmi'])})")

    # 运动成绩
    print(f"\n🏃 运动成绩:")
    if gender == "男生":
        print(f"  50米跑: {input_data['mm']} 秒")
        print(f"  1000米跑: {input_data['lm']} 分钟")
        print(f"  引体向上: {input_data['ytxs']} 个")
    else:
        print(f"  50米跑: {input_data['mm']} 秒")
        print(f"  800米跑: {input_data['mmm']} 分钟")
        print(f"  仰卧起坐: {input_data['ywqz']} 个")

    print(f"  立定跳远: {input_data['ldty']} cm")
    print(f"  坐位体前屈: {input_data['zwtqq']} cm")
    print(f"  肺活量: {input_data['fhl']} ml")

    # 预测分数
    print(f"\n🎯 预测结果:")
    print(f"  中考体育预测分数: {score:.1f}/60")

    # 等级评价
    level, message = get_score_level(score)
    print(f"  成绩等级: {level}")
    print(f"  评价: {message}")

    # 改进建议
    suggestions = create_improvement_suggestions(input_data, gender)
    if suggestions:
        print(f"\n💡 改进建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    else:
        print(f"\n🎯 各项能力均衡，继续保持！")

    # 保存结果
    save_results = input("\n💾 是否保存预测结果到文件？ (y/n): ").strip().lower()
    if save_results == 'y':
        save_to_file(input_data, gender, score, level, suggestions)

def save_to_file(input_data, gender, score, level, suggestions):
    """保存结果到文件"""
    import json
    from datetime import datetime

    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'gender': gender,
        'input_data': input_data,
        'prediction': {
            'score': float(score),
            'level': level,
            'max_score': 60
        },
        'suggestions': suggestions
    }

    # 生成文件名
    filename = f"prediction_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 预测结果已保存到: {filename}")

def create_example_prediction():
    """创建示例预测"""
    print("\n" + "="*50)
    print("📋 示例预测")
    print("="*50)

    # 加载模型
    pipeline, scaler, feature_names = load_model()
    if pipeline is None:
        return

    # 男生示例
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

    # 女生示例
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

    print("\n1. 男生示例:")
    score_male = predict_score(pipeline, example_male)
    display_results(example_male, "男生", score_male)

    print("\n2. 女生示例:")
    score_female = predict_score(pipeline, example_female)
    display_results(example_female, "女生", score_female)

def main():
    """主函数"""
    print("🏃 北京中考体育能力预测系统")
    print("="*50)

    # 加载模型
    pipeline, scaler, feature_names = load_model()
    if pipeline is None:
        return

    while True:
        print("\n请选择操作:")
        print("1. 进行预测")
        print("2. 查看示例")
        print("3. 退出")

        choice = input("\n请输入选择 (1-3): ").strip()

        if choice == '1':
            # 获取用户输入
            input_data, gender = get_user_input()

            # 预测分数
            score = predict_score(pipeline, input_data)

            # 显示结果
            display_results(input_data, gender, score)

        elif choice == '2':
            # 显示示例
            create_example_prediction()

        elif choice == '3':
            print("\n感谢使用中考体育能力预测系统！再见！👋")
            break

        else:
            print("❌ 无效的选择，请重新输入。")

        # 询问是否继续
        if choice in ['1', '2']:
            continue_choice = input("\n是否继续使用？ (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\n感谢使用中考体育能力预测系统！再见！👋")
                break

if __name__ == "__main__":
    main()