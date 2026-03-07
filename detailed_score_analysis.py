import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_score_data():
    """详细分析包含分数的数据"""
    print("="*60)
    print("详细分析包含中考分数的数据")
    print("="*60)

    # 读取包含分数的CSV文件
    filepath = '中小学生体育测试成绩信息_0 (1).csv'

    print(f"\n分析文件: {filepath}")
    print("-" * 50)

    # 读取数据
    df = pd.read_csv(filepath, encoding='gbk')

    print(f"数据形状: {df.shape}")
    print(f"总记录数: {len(df)}")

    # 显示列信息
    print("\n列信息:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col} ({df[col].dtype})")

    # 分析分数列
    print("\n" + "="*60)
    print("分数数据分析")
    print("="*60)

    # 总分分析
    if '总分' in df.columns:
        total_scores = df['总分'].dropna()
        print(f"\n总分统计:")
        print(f"  非空数量: {len(total_scores)}")
        print(f"  平均值: {total_scores.mean():.2f}")
        print(f"  标准差: {total_scores.std():.2f}")
        print(f"  最小值: {total_scores.min():.2f}")
        print(f"  最大值: {total_scores.max():.2f}")
        print(f"  中位数: {total_scores.median():.2f}")

        # 分数分布
        print(f"\n总分分布:")
        bins = [0, 60, 70, 80, 90, 100, 110, 120]
        labels = ['<60', '60-69', '70-79', '80-89', '90-99', '100-109', '110+']
        score_dist = pd.cut(total_scores, bins=bins, labels=labels, right=False)
        dist_counts = score_dist.value_counts().sort_index()
        for label, count in dist_counts.items():
            percentage = count / len(total_scores) * 100
            print(f"  {label}: {count}人 ({percentage:.1f}%)")

    # 标准分分析
    if '标准分' in df.columns:
        standard_scores = df['标准分'].dropna()
        print(f"\n标准分统计:")
        print(f"  非空数量: {len(standard_scores)}")
        print(f"  平均值: {standard_scores.mean():.2f}")
        print(f"  范围: {standard_scores.min():.2f}-{standard_scores.max():.2f}")

    # 各项分数分析
    print("\n" + "="*60)
    print("各项测试分数分析")
    print("="*60)

    score_columns = {
        'BMI评分': 'BMI分数',
        '体重评分': '体重分数',
        '肺活量评分': '肺活量分数',
        '50米跑评分': '50米跑分数',
        '坐位体前屈评分': '坐位体前屈分数',
        '跳绳评分': '跳绳分数'
    }

    for col, name in score_columns.items():
        if col in df.columns:
            scores = df[col].dropna()
            if len(scores) > 0:
                print(f"\n{name}:")
                print(f"  非空: {len(scores)}")
                print(f"  平均: {scores.mean():.2f}")
                print(f"  范围: {scores.min():.2f}-{scores.max():.2f}")

    # 等级分析
    print("\n" + "="*60)
    print("等级分布分析")
    print("="*60)

    level_columns = {
        '体重等级': '体重等级',
        '肺活量等级': '肺活量等级',
        '50米跑等级': '50米跑等级',
        '坐位体前屈等级': '坐位体前屈等级',
        '跳绳等级': '跳绳等级',
        '总分等级': '总分等级'
    }

    for col, name in level_columns.items():
        if col in df.columns:
            levels = df[col].dropna()
            if len(levels) > 0:
                print(f"\n{name}分布:")
                level_counts = levels.value_counts()
                for level, count in level_counts.items():
                    percentage = count / len(levels) * 100
                    print(f"  {level}: {count}人 ({percentage:.1f}%)")

    # 基本特征分析
    print("\n" + "="*60)
    print("基本特征分析")
    print("="*60)

    # 性别分析
    if '性别' in df.columns:
        gender_mapping = {1: '男', 2: '女'}
        df['性别_文字'] = df['性别'].map(gender_mapping)
        gender_counts = df['性别_文字'].value_counts()
        print(f"\n性别分布:")
        for gender, count in gender_counts.items():
            percentage = count / len(df) * 100
            print(f"  {gender}: {count}人 ({percentage:.1f}%)")

    # 身高体重分析
    if '身高（CM）' in df.columns and '体重（KG）' in df.columns:
        heights = df['身高（CM）'].dropna()
        weights = df['体重（KG）'].dropna()
        print(f"\n身体测量统计:")
        print(f"  身高: {heights.mean():.1f}cm ({heights.min():.1f}-{heights.max():.1f})")
        print(f"  体重: {weights.mean():.1f}kg ({weights.min():.1f}-{weights.max():.1f})")

        # 计算BMI
        df['BMI_计算'] = df['体重（KG）'] / ((df['身高（CM）'] / 100) ** 2)
        bmi_values = df['BMI_计算'].dropna()
        print(f"  BMI: {bmi_values.mean():.1f} ({bmi_values.min():.1f}-{bmi_values.max():.1f})")

    # 测试项目分析
    print("\n" + "="*60)
    print("测试项目成绩分析")
    print("="*60)

    test_columns = {
        '肺活量': '肺活量(ml)',
        '50米跑': '50米跑(秒)',
        '坐位体前屈': '坐位体前屈(cm)',
        '跳绳': '跳绳(个)'
    }

    for col, name in test_columns.items():
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"\n{name}:")
                print(f"  非空: {len(values)}")
                print(f"  平均: {values.mean():.2f}")
                print(f"  范围: {values.min():.2f}-{values.max():.2f}")

    # 相关性分析
    print("\n" + "="*60)
    print("相关性分析")
    print("="*60)

    if '总分' in df.columns:
        # 选择数值列进行相关性分析
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 计算与总分的相关性
        correlations = {}
        for col in numeric_cols:
            if col != '总分' and col != '序号' and col != '班级编号':
                corr = df['总分'].corr(df[col])
                if not pd.isna(corr):
                    correlations[col] = corr

        # 按相关性绝对值排序
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        print(f"\n与总分的相关性 (前10项):")
        for i, (col, corr) in enumerate(sorted_correlations[:10], 1):
            print(f"  {i:2d}. {col}: {corr:.3f}")

    # 创建可视化
    create_visualizations(df)

    # 保存处理后的数据
    save_processed_data(df)

    return df

def create_visualizations(df):
    """创建数据可视化"""
    print("\n" + "="*60)
    print("创建数据可视化")
    print("="*60)

    # 1. 总分分布直方图
    if '总分' in df.columns:
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.hist(df['总分'].dropna(), bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('总分')
        plt.ylabel('频数')
        plt.title('总分分布')
        plt.grid(True, alpha=0.3)

    # 2. 各项分数箱线图
    score_cols = ['BMI评分', '肺活量评分', '50米跑评分', '坐位体前屈评分', '跳绳评分']
    available_scores = [col for col in score_cols if col in df.columns]

    if available_scores:
        plt.subplot(2, 2, 2)
        score_data = [df[col].dropna() for col in available_scores]
        plt.boxplot(score_data, labels=available_scores)
        plt.xticks(rotation=45)
        plt.ylabel('分数')
        plt.title('各项测试分数分布')
        plt.grid(True, alpha=0.3)

    # 3. 等级分布饼图
    if '总分等级' in df.columns:
        plt.subplot(2, 2, 3)
        level_counts = df['总分等级'].value_counts()
        plt.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%')
        plt.title('总分等级分布')

    # 4. 性别与总分关系
    if '性别_文字' in df.columns and '总分' in df.columns:
        plt.subplot(2, 2, 4)
        gender_groups = df.groupby('性别_文字')['总分']
        genders = []
        scores_data = []

        for gender, group in gender_groups:
            genders.append(gender)
            scores_data.append(group.dropna().values)

        plt.boxplot(scores_data, labels=genders)
        plt.xlabel('性别')
        plt.ylabel('总分')
        plt.title('性别与总分关系')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('score_data_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  可视化图表已保存到: score_data_analysis.png")

    # 5. 相关性热力图
    if '总分' in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 选择与总分相关性较高的列
        relevant_cols = ['总分']
        for col in numeric_cols:
            if col != '总分' and col not in ['序号', '班级编号', '附加分']:
                if col in df.columns:
                    relevant_cols.append(col)

        if len(relevant_cols) > 1:
            corr_matrix = df[relevant_cols].corr()

            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, linewidths=0.5)
            plt.title('特征相关性热力图')
            plt.tight_layout()
            plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"  相关性热力图已保存到: correlation_heatmap.png")

def save_processed_data(df):
    """保存处理后的数据"""
    print("\n" + "="*60)
    print("保存处理后的数据")
    print("="*60)

    # 创建处理后的数据副本
    processed_df = df.copy()

    # 重命名列，使其更易理解
    column_mapping = {
        '身高（CM）': '身高_cm',
        '体重（KG）': '体重_kg',
        '肺活量': '肺活量_ml',
        '50米跑': '50米跑_秒',
        '坐位体前屈': '坐位体前屈_cm',
        '跳绳': '跳绳_个',
        'BMI评分': 'BMI分数',
        '体重评分': '体重分数',
        '肺活量评分': '肺活量分数',
        '50米跑评分': '50米跑分数',
        '坐位体前屈评分': '坐位体前屈分数',
        '跳绳评分': '跳绳分数',
        '标准分': '标准分数',
        '总分': '总分数'
    }

    # 应用重命名
    for old_col, new_col in column_mapping.items():
        if old_col in processed_df.columns:
            processed_df.rename(columns={old_col: new_col}, inplace=True)

    # 添加计算字段
    if '身高_cm' in processed_df.columns and '体重_kg' in processed_df.columns:
        processed_df['BMI_计算'] = processed_df['体重_kg'] / ((processed_df['身高_cm'] / 100) ** 2)

    # 保存为CSV
    processed_df.to_csv('processed_score_data.csv', index=False, encoding='utf-8-sig')
    print(f"处理后的数据已保存到: processed_score_data.csv")
    print(f"记录数: {len(processed_df)}")
    print(f"字段数: {len(processed_df.columns)}")

    # 创建数据字典
    data_dict = {}
    for col in processed_df.columns:
        data_dict[col] = {
            '数据类型': str(processed_df[col].dtype),
            '非空数量': processed_df[col].count(),
            '缺失数量': processed_df[col].isnull().sum(),
            '缺失比例': f"{processed_df[col].isnull().sum() / len(processed_df) * 100:.1f}%"
        }

        if processed_df[col].dtype in [np.int64, np.float64]:
            non_null = processed_df[col].dropna()
            if len(non_null) > 0:
                data_dict[col].update({
                    '最小值': non_null.min(),
                    '最大值': non_null.max(),
                    '平均值': non_null.mean(),
                    '中位数': non_null.median()
                })

    # 保存数据字典
    import json
    with open('data_dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    print(f"数据字典已保存到: data_dictionary.json")

def main():
    """主函数"""
    print("开始详细分析包含中考分数的数据...")

    df = analyze_score_data()

    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)

    print(f"\n关键发现:")
    print(f"1. 数据包含 {len(df)} 条学生记录")
    print(f"2. 包含真实的中考体育分数数据（总分列）")
    print(f"3. 包含多项测试的原始成绩和评分")
    print(f"4. 数据质量较好，适合训练预测模型")

    print(f"\n下一步:")
    print(f"1. 使用 processed_score_data.csv 训练模型")
    print(f"2. 目标变量: 总分数")
    print(f"3. 特征变量: 各项测试成绩和评分")

    return df

if __name__ == "__main__":
    df = main()