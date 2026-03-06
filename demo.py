#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中考体育能力预测系统 - 演示脚本
"""

import numpy as np
import joblib
import pandas as pd

def print_header():
    """打印标题"""
    print("="*60)
    print("          中考体育能力预测系统 - 演示")
    print("="*60)
    print()

def load_model():
    """加载模型"""
    print("🔧 加载预测模型...")
    try:
        pipeline = joblib.load('sports_score_predictor.pipeline.pkl')
        print("✅ 模型加载成功！")
        return pipeline
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def demo_male_example(pipeline):
    """演示男生示例"""
    print("\n" + "-"*60)
    print("👦 男生示例预测")
    print("-"*60)

    # 男生示例数据
    example_male = {
        'gender': 0,      # 男生
        'sg': 170,        # 身高170cm
        'tz': 60,         # 体重60kg
        'bmi': 60 / ((170/100)**2),  # BMI: 20.8
        'mm': 7.5,        # 50米7.5秒
        'lm': 3.5,        # 1000米3.5分钟
        'mmm': 0,         # 800米（男生为0）
        'ytxs': 10,       # 引体向上10个
        'ywqz': 0,        # 仰卧起坐（男生为0）
        'ldty': 230,      # 立定跳远230cm
        'zwtqq': 15,      # 坐位体前屈15cm
        'fhl': 4000       # 肺活量4000ml
    }

    print("📋 输入数据:")
    print(f"  性别: 男生")
    print(f"  身高: {example_male['sg']} cm")
    print(f"  体重: {example_male['tz']} kg")
    print(f"  BMI: {example_male['bmi']:.1f}")
    print(f"  50米跑: {example_male['mm']} 秒")
    print(f"  1000米跑: {example_male['lm']} 分钟")
    print(f"  引体向上: {example_male['ytxs']} 个")
    print(f"  立定跳远: {example_male['ldty']} cm")
    print(f"  坐位体前屈: {example_male['zwtqq']} cm")
    print(f"  肺活量: {example_male['fhl']} ml")

    # 预测
    X = np.array([list(example_male.values())])
    score = pipeline.predict(X)[0]

    print(f"\n🎯 预测结果:")
    print(f"  中考体育预测分数: {score:.1f}/60")

    # 等级评价
    if score >= 54:
        level = "优秀"
        comment = "🎉 非常出色！"
    elif score >= 48:
        level = "良好"
        comment = "👍 成绩良好！"
    elif score >= 36:
        level = "及格"
        comment = "✅ 达到及格标准。"
    else:
        level = "待提高"
        comment = "💪 需要加强锻炼。"

    print(f"  成绩等级: {level}")
    print(f"  评价: {comment}")

    return score, level

def demo_female_example(pipeline):
    """演示女生示例"""
    print("\n" + "-"*60)
    print("👧 女生示例预测")
    print("-"*60)

    # 女生示例数据
    example_female = {
        'gender': 1,      # 女生
        'sg': 160,        # 身高160cm
        'tz': 50,         # 体重50kg
        'bmi': 50 / ((160/100)**2),  # BMI: 19.5
        'mm': 8.5,        # 50米8.5秒
        'lm': 0,          # 1000米（女生为0）
        'mmm': 3.5,       # 800米3.5分钟
        'ytxs': 0,        # 引体向上（女生为0）
        'ywqz': 35,       # 仰卧起坐35个
        'ldty': 180,      # 立定跳远180cm
        'zwtqq': 18,      # 坐位体前屈18cm
        'fhl': 3000       # 肺活量3000ml
    }

    print("📋 输入数据:")
    print(f"  性别: 女生")
    print(f"  身高: {example_female['sg']} cm")
    print(f"  体重: {example_female['tz']} kg")
    print(f"  BMI: {example_female['bmi']:.1f}")
    print(f"  50米跑: {example_female['mm']} 秒")
    print(f"  800米跑: {example_female['mmm']} 分钟")
    print(f"  仰卧起坐: {example_female['ywqz']} 个")
    print(f"  立定跳远: {example_female['ldty']} cm")
    print(f"  坐位体前屈: {example_female['zwtqq']} cm")
    print(f"  肺活量: {example_female['fhl']} ml")

    # 预测
    X = np.array([list(example_female.values())])
    score = pipeline.predict(X)[0]

    print(f"\n🎯 预测结果:")
    print(f"  中考体育预测分数: {score:.1f}/60")

    # 等级评价
    if score >= 54:
        level = "优秀"
        comment = "🎉 非常出色！"
    elif score >= 48:
        level = "良好"
        comment = "👍 成绩良好！"
    elif score >= 36:
        level = "及格"
        comment = "✅ 达到及格标准。"
    else:
        level = "待提高"
        comment = "💪 需要加强锻炼。"

    print(f"  成绩等级: {level}")
    print(f"  评价: {comment}")

    return score, level

def demo_extreme_cases(pipeline):
    """演示极端情况"""
    print("\n" + "-"*60)
    print("⚡ 极端情况演示")
    print("-"*60)

    cases = [
        {
            'name': '体育特长生',
            'data': {
                'gender': 0, 'sg': 175, 'tz': 65, 'bmi': 65/((175/100)**2),
                'mm': 6.0, 'lm': 3.0, 'mmm': 0, 'ytxs': 20,
                'ywqz': 0, 'ldty': 260, 'zwtqq': 25, 'fhl': 5000
            }
        },
        {
            'name': '需要加强锻炼',
            'data': {
                'gender': 1, 'sg': 155, 'tz': 70, 'bmi': 70/((155/100)**2),
                'mm': 11.0, 'lm': 0, 'mmm': 5.0, 'ytxs': 0,
                'ywqz': 15, 'ldty': 140, 'zwtqq': 5, 'fhl': 2000
            }
        }
    ]

    for case in cases:
        print(f"\n📌 {case['name']}:")

        # 预测
        X = np.array([list(case['data'].values())])
        score = pipeline.predict(X)[0]

        print(f"  预测分数: {score:.1f}/60")

        # 简要分析
        if case['name'] == '体育特长生':
            print(f"  分析: 各项指标优秀，体育成绩突出")
        else:
            print(f"  分析: 多项指标需要提高，建议制定锻炼计划")

def show_model_info():
    """显示模型信息"""
    print("\n" + "-"*60)
    print("📊 模型信息")
    print("-"*60)

    try:
        # 加载特征名称
        feature_names = np.load('sports_score_predictor.features.npy', allow_pickle=True)

        print(f"特征数量: {len(feature_names)}")
        print(f"特征列表: {', '.join(feature_names[:6])}...")

        # 加载指标
        metrics_df = pd.read_csv('model_metrics.csv')
        print(f"\n模型性能:")
        print(f"  测试集 R²: {metrics_df['test_r2'].values[0]:.4f}")
        print(f"  测试集 MAE: {metrics_df['test_mae'].values[0]:.4f}")

    except Exception as e:
        print(f"加载模型信息失败: {e}")

def main():
    """主函数"""
    print_header()

    # 加载模型
    pipeline = load_model()
    if pipeline is None:
        return

    # 演示男生示例
    demo_male_example(pipeline)

    # 演示女生示例
    demo_female_example(pipeline)

    # 演示极端情况
    demo_extreme_cases(pipeline)

    # 显示模型信息
    show_model_info()

    print("\n" + "="*60)
    print("🎉 演示完成！")
    print("="*60)
    print("\n💡 使用建议:")
    print("  1. 运行 'python predict_cli.py' 进行交互式预测")
    print("  2. 查看 'README.md' 获取详细使用说明")
    print("  3. 查看 'model_comprehensive_report.txt' 获取模型详细信息")
    print()

if __name__ == "__main__":
    main()