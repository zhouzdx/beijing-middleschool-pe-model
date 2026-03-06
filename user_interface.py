import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# 设置页面
st.set_page_config(
    page_title="中考体育能力预测系统",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载模型和工具
@st.cache_resource
def load_model():
    """加载模型和相关工具"""
    try:
        pipeline = joblib.load('sports_score_predictor.pipeline.pkl')
        scaler = joblib.load('sports_score_predictor.scaler.pkl')
        feature_names = np.load('sports_score_predictor.features.npy', allow_pickle=True)
        return pipeline, scaler, feature_names
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None, None, None

def create_input_form():
    """创建输入表单"""
    st.sidebar.header("📝 学生基本信息")

    # 性别选择
    gender = st.sidebar.radio("性别", ["男生", "女生"], index=0)
    gender_code = 0 if gender == "男生" else 1

    st.sidebar.header("🏋️ 身体测量数据")

    # 身高体重
    col1, col2 = st.sidebar.columns(2)
    with col1:
        height = st.number_input("身高 (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
    with col2:
        weight = st.number_input("体重 (kg)", min_value=30.0, max_value=150.0, value=60.0, step=0.1)

    # 计算BMI
    bmi = weight / ((height / 100) ** 2)
    st.sidebar.info(f"BMI: {bmi:.1f} ({get_bmi_category(bmi)})")

    st.sidebar.header("🏃 运动测试成绩")

    # 根据性别显示不同的项目
    if gender == "男生":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            run_50m = st.number_input("50米跑 (秒)", min_value=5.0, max_value=15.0, value=7.5, step=0.1)
        with col2:
            run_1000m = st.number_input("1000米跑 (分钟)", min_value=2.0, max_value=10.0, value=3.5, step=0.1)

        pull_ups = st.sidebar.number_input("引体向上 (个)", min_value=0, max_value=50, value=10)
        sit_ups = 0  # 男生不做仰卧起坐
        run_800m = 0  # 男生不做800米
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            run_50m = st.number_input("50米跑 (秒)", min_value=5.0, max_value=15.0, value=8.5, step=0.1)
        with col2:
            run_800m = st.number_input("800米跑 (分钟)", min_value=2.0, max_value=10.0, value=3.5, step=0.1)

        sit_ups = st.sidebar.number_input("仰卧起坐 (个)", min_value=0, max_value=100, value=35)
        pull_ups = 0  # 女生不做引体向上
        run_1000m = 0  # 女生不做1000米

    # 共同项目
    col1, col2 = st.sidebar.columns(2)
    with col1:
        long_jump = st.number_input("立定跳远 (cm)", min_value=100.0, max_value=300.0, value=200.0, step=1.0)
    with col2:
        sit_reach = st.number_input("坐位体前屈 (cm)", min_value=-20.0, max_value=40.0, value=15.0, step=0.1)

    vital_capacity = st.sidebar.number_input("肺活量 (ml)", min_value=1000, max_value=10000, value=3500, step=100)

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

    return input_data, gender

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

def predict_score(pipeline, input_data, feature_names):
    """预测分数"""
    # 转换为数组
    X = np.array([list(input_data.values())])

    # 预测
    score = pipeline.predict(X)[0]

    return score

def create_radar_chart(input_data, score, gender):
    """创建雷达图显示各项能力"""
    # 各项能力的标准化值（0-10分）
    abilities = {
        '速度能力': max(0, 10 - (input_data['mm'] - 6) * 2) if gender == "男生" else max(0, 10 - (input_data['mm'] - 7) * 2),
        '耐力能力': max(0, 10 - (input_data['lm'] - 3) * 3) if gender == "男生" else max(0, 10 - (input_data['mmm'] - 3) * 3),
        '力量能力': min(10, input_data['ytxs'] / 1.5) if gender == "男生" else min(10, input_data['ywqz'] / 5),
        '爆发力': min(10, (input_data['ldty'] - 150) / 10) if gender == "男生" else min(10, (input_data['ldty'] - 120) / 6),
        '柔韧性': min(10, (input_data['zwtqq'] + 10) / 3),
        '心肺功能': min(10, input_data['fhl'] / 500) if gender == "男生" else min(10, input_data['fhl'] / 400)
    }

    # 创建雷达图
    categories = list(abilities.keys())
    values = list(abilities.values())

    # 闭合图形
    values += values[:1]
    categories += categories[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])
    ax.set_ylim(0, 10)
    ax.set_title('运动能力雷达图', size=20, y=1.1)
    ax.grid(True)

    return fig

def create_score_gauge(score):
    """创建分数仪表盘"""
    fig, ax = plt.subplots(figsize=(8, 4))

    # 创建半圆
    angles = np.linspace(0, np.pi, 100)
    x = np.cos(angles)
    y = np.sin(angles)

    # 绘制背景弧
    ax.plot(x, y, color='gray', linewidth=20, alpha=0.2)

    # 根据分数计算角度
    score_angle = (score / 60) * np.pi

    # 绘制分数弧
    score_x = np.cos(np.linspace(0, score_angle, 100))
    score_y = np.sin(np.linspace(0, score_angle, 100))

    # 根据分数选择颜色
    if score >= 48:
        color = 'green'
    elif score >= 36:
        color = 'yellow'
    else:
        color = 'red'

    ax.plot(score_x, score_y, color=color, linewidth=20, alpha=0.8)

    # 添加指针
    pointer_x = np.cos(score_angle) * 0.9
    pointer_y = np.sin(score_angle) * 0.9
    ax.plot([0, pointer_x], [0, pointer_y], color='black', linewidth=3)

    # 添加分数文本
    ax.text(0, 0.2, f'{score:.1f}', ha='center', va='center', fontsize=40, fontweight='bold')
    ax.text(0, -0.1, '/60', ha='center', va='center', fontsize=20)

    # 添加等级标签
    ax.text(-0.9, -0.2, '0', ha='center', va='center', fontsize=12)
    ax.text(0, -0.2, '30', ha='center', va='center', fontsize=12)
    ax.text(0.9, -0.2, '60', ha='center', va='center', fontsize=12)

    # 设置坐标轴
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')

    return fig

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

def create_improvement_suggestions(input_data, gender):
    """创建改进建议"""
    suggestions = []

    # 速度建议
    if input_data['mm'] > (8 if gender == "男生" else 9):
        suggestions.append("🏃 **速度训练**: 50米跑成绩有待提高，建议进行短跑训练和爆发力练习。")

    # 耐力建议
    if gender == "男生" and input_data['lm'] > 4:
        suggestions.append("🏃‍♂️ **耐力训练**: 1000米跑需要加强，建议进行有氧跑步训练。")
    elif gender == "女生" and input_data['mmm'] > 4:
        suggestions.append("🏃‍♀️ **耐力训练**: 800米跑需要加强，建议进行有氧跑步训练。")

    # 力量建议
    if gender == "男生" and input_data['ytxs'] < 5:
        suggestions.append("💪 **上肢力量**: 引体向上数量较少，建议进行引体向上辅助训练和背部力量练习。")
    elif gender == "女生" and input_data['ywqz'] < 25:
        suggestions.append("💪 **核心力量**: 仰卧起坐数量有待提高，建议进行核心肌群训练。")

    # 爆发力建议
    if input_data['ldty'] < (220 if gender == "男生" else 170):
        suggestions.append("🦘 **爆发力训练**: 立定跳远距离较短，建议进行腿部爆发力训练。")

    # 柔韧性建议
    if input_data['zwtqq'] < 10:
        suggestions.append("🧘 **柔韧性训练**: 坐位体前屈成绩一般，建议进行拉伸练习提高柔韧性。")

    # 心肺功能建议
    if input_data['fhl'] < (3500 if gender == "男生" else 2500):
        suggestions.append("🫁 **心肺功能**: 肺活量有待提高，建议进行有氧运动和呼吸训练。")

    return suggestions

def main():
    """主函数"""
    # 标题
    st.title("🏃 北京中考体育能力预测系统")
    st.markdown("---")

    # 加载模型
    pipeline, scaler, feature_names = load_model()

    if pipeline is None:
        st.error("无法加载模型，请确保模型文件存在。")
        return

    # 创建两列布局
    col1, col2 = st.columns([1, 2])

    with col1:
        # 输入表单
        input_data, gender = create_input_form()

        # 预测按钮
        if st.button("🔮 预测中考体育分数", type="primary", use_container_width=True):
            with st.spinner("正在计算预测分数..."):
                # 预测分数
                score = predict_score(pipeline, input_data, feature_names)

                # 保存预测结果到session state
                st.session_state['prediction_score'] = score
                st.session_state['input_data'] = input_data
                st.session_state['gender'] = gender

                st.success("预测完成！")

    with col2:
        # 显示预测结果
        if 'prediction_score' in st.session_state:
            score = st.session_state['prediction_score']
            input_data = st.session_state['input_data']
            gender = st.session_state['gender']

            # 分数显示
            st.header("📊 预测结果")

            # 创建两列显示分数和等级
            col_a, col_b = st.columns(2)

            with col_a:
                # 分数仪表盘
                gauge_fig = create_score_gauge(score)
                st.pyplot(gauge_fig)

            with col_b:
                # 等级和建议
                level, message = get_score_level(score)
                st.metric("预测分数", f"{score:.1f}/60")
                st.metric("成绩等级", level)
                st.info(message)

            st.markdown("---")

            # 能力分析
            st.subheader("📈 运动能力分析")

            # 雷达图
            radar_fig = create_radar_chart(input_data, score, gender)
            st.pyplot(radar_fig)

            # 改进建议
            st.subheader("💡 改进建议")
            suggestions = create_improvement_suggestions(input_data, gender)

            if suggestions:
                for suggestion in suggestions:
                    st.markdown(suggestion)
            else:
                st.success("🎯 各项能力均衡，继续保持！")

            # 详细数据
            with st.expander("📋 查看详细输入数据"):
                df_input = pd.DataFrame([input_data])
                df_input.columns = [
                    '性别', '身高(cm)', '体重(kg)', 'BMI',
                    '50米(秒)', '1000米(分)', '800米(分)',
                    '引体向上(个)', '仰卧起坐(个)',
                    '立定跳远(cm)', '坐位体前屈(cm)', '肺活量(ml)'
                ]
                st.dataframe(df_input, use_container_width=True)

    # 侧边栏底部信息
    st.sidebar.markdown("---")
    st.sidebar.info("""
    ### 📌 使用说明
    1. 在左侧填写学生基本信息
    2. 输入各项体育测试成绩
    3. 点击"预测中考体育分数"按钮
    4. 查看预测结果和改进建议

    ### 🎯 评分标准
    - 优秀: ≥54分
    - 良好: 48-53分
    - 及格: 36-47分
    - 待提高: <36分
    """)

    # 页脚
    st.markdown("---")
    st.caption("© 2026 中考体育能力预测系统 | 基于机器学习模型预测")

if __name__ == "__main__":
    main()