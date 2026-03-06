import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在加载数据...")

    with open('高中学生体质测试信息202508010326.json', 'r', encoding='utf-8') as f:
        content = f.read()

    parts = content.split(']\n[')
    fields_str = parts[0] + ']'
    data_str = '[' + parts[1]

    fields = json.loads(fields_str)
    data = json.loads(data_str)

    # 转换为DataFrame
    df = pd.DataFrame(data)

    print(f"原始数据形状: {df.shape}")
    print(f"字段: {[field['name_cn'] for field in fields]}")

    return df, fields

def clean_data(df):
    """数据清洗"""
    print("\n正在清洗数据...")

    # 复制原始数据
    df_clean = df.copy()

    # 1. 处理缺失值
    print("处理缺失值...")

    # 数值字段列表
    numeric_cols = ['mm', 'lm', 'tz', 'fhl', 'mmm', 'ytxs', 'ldty', 'sg', 'zwtqq', 'ywqz']

    # 转换为数值类型
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 2. 根据性别处理项目缺失值
    print("根据性别处理项目缺失值...")

    # 男生项目：1000米(lm)、引体向上(ytxs)
    # 女生项目：800米(mmm)、仰卧起坐(ywqz)

    # 对于男生，如果1000米或引体向上缺失，可能是女生数据，用中位数填充
    male_mask = df_clean['xb'] == '男'
    female_mask = df_clean['xb'] == '女'

    # 男生缺失1000米或引体向上，可能是数据错误，用中位数填充
    for col in ['lm', 'ytxs']:
        if col in df_clean.columns:
            male_median = df_clean.loc[male_mask, col].median()
            df_clean.loc[male_mask & df_clean[col].isna(), col] = male_median

    # 女生缺失800米或仰卧起坐，用中位数填充
    for col in ['mmm', 'ywqz']:
        if col in df_clean.columns:
            female_median = df_clean.loc[female_mask, col].median()
            df_clean.loc[female_mask & df_clean[col].isna(), col] = female_median

    # 3. 处理其他缺失值（用中位数填充）
    print("处理其他缺失值...")
    for col in numeric_cols:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)

    # 4. 处理异常值
    print("处理异常值...")

    # 定义合理的范围（基于常识）
    ranges = {
        'sg': (100, 250),  # 身高(cm)
        'tz': (30, 150),   # 体重(kg)
        'mm': (5, 15),     # 50米(秒)
        'lm': (2, 10),     # 1000米(分钟)
        'mmm': (2, 10),    # 800米(分钟)
        'ytxs': (0, 50),   # 引体向上(个)
        'ywqz': (0, 100),  # 仰卧起坐(个)
        'ldty': (100, 300), # 立定跳远(cm)
        'zwtqq': (-10, 40), # 坐位体前屈(cm)
        'fhl': (1000, 10000) # 肺活量(ml)
    }

    for col, (min_val, max_val) in ranges.items():
        if col in df_clean.columns:
            # 将超出范围的值设为边界值
            df_clean[col] = df_clean[col].clip(min_val, max_val)

    # 5. 创建BMI特征
    print("创建衍生特征...")
    df_clean['bmi'] = df_clean['tz'] / ((df_clean['sg'] / 100) ** 2)

    # 6. 编码性别
    print("编码分类变量...")
    df_clean['gender'] = df_clean['xb'].map({'男': 0, '女': 1})

    # 7. 创建目标变量（模拟中考体育分数）
    print("创建目标变量（模拟中考体育分数）...")
    df_clean = calculate_sports_score(df_clean)

    print(f"清洗后数据形状: {df_clean.shape}")
    print(f"缺失值统计:")
    for col in df_clean.columns:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            print(f"  {col}: {missing} 缺失")

    return df_clean

def calculate_sports_score(df):
    """根据北京中考体育标准计算模拟分数"""
    # 北京中考体育总分通常为30分或40分（不同年份不同）
    # 这里我们使用40分制

    df = df.copy()
    scores = []

    for idx, row in df.iterrows():
        score = 0

        # 1. 50米跑 (10分)
        if row['xb'] == '男':  # 男生标准
            if row['mm'] <= 7.5:
                score += 10
            elif row['mm'] <= 8.0:
                score += 9
            elif row['mm'] <= 8.5:
                score += 8
            elif row['mm'] <= 9.0:
                score += 7
            elif row['mm'] <= 9.5:
                score += 6
            elif row['mm'] <= 10.0:
                score += 5
            elif row['mm'] <= 10.5:
                score += 4
            elif row['mm'] <= 11.0:
                score += 3
            elif row['mm'] <= 11.5:
                score += 2
            else:
                score += 1
        else:  # 女生标准
            if row['mm'] <= 8.0:
                score += 10
            elif row['mm'] <= 8.5:
                score += 9
            elif row['mm'] <= 9.0:
                score += 8
            elif row['mm'] <= 9.5:
                score += 7
            elif row['mm'] <= 10.0:
                score += 6
            elif row['mm'] <= 10.5:
                score += 5
            elif row['mm'] <= 11.0:
                score += 4
            elif row['mm'] <= 11.5:
                score += 3
            elif row['mm'] <= 12.0:
                score += 2
            else:
                score += 1

        # 2. 立定跳远 (10分)
        if row['xb'] == '男':  # 男生标准
            if row['ldty'] >= 250:
                score += 10
            elif row['ldty'] >= 240:
                score += 9
            elif row['ldty'] >= 230:
                score += 8
            elif row['ldty'] >= 220:
                score += 7
            elif row['ldty'] >= 210:
                score += 6
            elif row['ldty'] >= 200:
                score += 5
            elif row['ldty'] >= 190:
                score += 4
            elif row['ldty'] >= 180:
                score += 3
            elif row['ldty'] >= 170:
                score += 2
            else:
                score += 1
        else:  # 女生标准
            if row['ldty'] >= 200:
                score += 10
            elif row['ldty'] >= 190:
                score += 9
            elif row['ldty'] >= 180:
                score += 8
            elif row['ldty'] >= 170:
                score += 7
            elif row['ldty'] >= 160:
                score += 6
            elif row['ldty'] >= 150:
                score += 5
            elif row['ldty'] >= 140:
                score += 4
            elif row['ldty'] >= 130:
                score += 3
            elif row['ldty'] >= 120:
                score += 2
            else:
                score += 1

        # 3. 坐位体前屈 (10分)
        if row['xb'] == '男':  # 男生标准
            if row['zwtqq'] >= 20:
                score += 10
            elif row['zwtqq'] >= 18:
                score += 9
            elif row['zwtqq'] >= 16:
                score += 8
            elif row['zwtqq'] >= 14:
                score += 7
            elif row['zwtqq'] >= 12:
                score += 6
            elif row['zwtqq'] >= 10:
                score += 5
            elif row['zwtqq'] >= 8:
                score += 4
            elif row['zwtqq'] >= 6:
                score += 3
            elif row['zwtqq'] >= 4:
                score += 2
            else:
                score += 1
        else:  # 女生标准
            if row['zwtqq'] >= 22:
                score += 10
            elif row['zwtqq'] >= 20:
                score += 9
            elif row['zwtqq'] >= 18:
                score += 8
            elif row['zwtqq'] >= 16:
                score += 7
            elif row['zwtqq'] >= 14:
                score += 6
            elif row['zwtqq'] >= 12:
                score += 5
            elif row['zwtqq'] >= 10:
                score += 4
            elif row['zwtqq'] >= 8:
                score += 3
            elif row['zwtqq'] >= 6:
                score += 2
            else:
                score += 1

        # 4. 耐力跑 (10分) - 男生1000米，女生800米
        if row['xb'] == '男':  # 男生1000米
            if row['lm'] <= 3.4:
                score += 10
            elif row['lm'] <= 3.5:
                score += 9
            elif row['lm'] <= 3.6:
                score += 8
            elif row['lm'] <= 3.7:
                score += 7
            elif row['lm'] <= 3.8:
                score += 6
            elif row['lm'] <= 3.9:
                score += 5
            elif row['lm'] <= 4.0:
                score += 4
            elif row['lm'] <= 4.1:
                score += 3
            elif row['lm'] <= 4.2:
                score += 2
            else:
                score += 1
        else:  # 女生800米
            if row['mmm'] <= 3.2:
                score += 10
            elif row['mmm'] <= 3.3:
                score += 9
            elif row['mmm'] <= 3.4:
                score += 8
            elif row['mmm'] <= 3.5:
                score += 7
            elif row['mmm'] <= 3.6:
                score += 6
            elif row['mmm'] <= 3.7:
                score += 5
            elif row['mmm'] <= 3.8:
                score += 4
            elif row['mmm'] <= 3.9:
                score += 3
            elif row['mmm'] <= 4.0:
                score += 2
            else:
                score += 1

        # 5. 力量项目 (10分) - 男生引体向上，女生仰卧起坐
        if row['xb'] == '男':  # 男生引体向上
            if row['ytxs'] >= 15:
                score += 10
            elif row['ytxs'] >= 13:
                score += 9
            elif row['ytxs'] >= 11:
                score += 8
            elif row['ytxs'] >= 9:
                score += 7
            elif row['ytxs'] >= 7:
                score += 6
            elif row['ytxs'] >= 5:
                score += 5
            elif row['ytxs'] >= 3:
                score += 4
            elif row['ytxs'] >= 2:
                score += 3
            elif row['ytxs'] >= 1:
                score += 2
            else:
                score += 1
        else:  # 女生仰卧起坐
            if row['ywqz'] >= 50:
                score += 10
            elif row['ywqz'] >= 45:
                score += 9
            elif row['ywqz'] >= 40:
                score += 8
            elif row['ywqz'] >= 35:
                score += 7
            elif row['ywqz'] >= 30:
                score += 6
            elif row['ywqz'] >= 25:
                score += 5
            elif row['ywqz'] >= 20:
                score += 4
            elif row['ywqz'] >= 15:
                score += 3
            elif row['ywqz'] >= 10:
                score += 2
            else:
                score += 1

        # 6. 肺活量 (10分) - 额外加分项
        if row['xb'] == '男':  # 男生标准
            if row['fhl'] >= 5000:
                score += 10
            elif row['fhl'] >= 4500:
                score += 9
            elif row['fhl'] >= 4000:
                score += 8
            elif row['fhl'] >= 3500:
                score += 7
            elif row['fhl'] >= 3000:
                score += 6
            elif row['fhl'] >= 2500:
                score += 5
            elif row['fhl'] >= 2000:
                score += 4
            elif row['fhl'] >= 1500:
                score += 3
            elif row['fhl'] >= 1000:
                score += 2
            else:
                score += 1
        else:  # 女生标准
            if row['fhl'] >= 4000:
                score += 10
            elif row['fhl'] >= 3500:
                score += 9
            elif row['fhl'] >= 3000:
                score += 8
            elif row['fhl'] >= 2500:
                score += 7
            elif row['fhl'] >= 2000:
                score += 6
            elif row['fhl'] >= 1500:
                score += 5
            elif row['fhl'] >= 1000:
                score += 4
            elif row['fhl'] >= 800:
                score += 3
            elif row['fhl'] >= 600:
                score += 2
            else:
                score += 1

        scores.append(score)

    df['sports_score'] = scores
    return df

def prepare_features(df):
    """准备特征数据"""
    print("\n准备特征数据...")

    # 选择特征
    features = [
        'gender',  # 性别
        'sg',      # 身高
        'tz',      # 体重
        'bmi',     # BMI
        'mm',      # 50米
        'lm',      # 1000米（男生）
        'mmm',     # 800米（女生）
        'ytxs',    # 引体向上（男生）
        'ywqz',    # 仰卧起坐（女生）
        'ldty',    # 立定跳远
        'zwtqq',   # 坐位体前屈
        'fhl'      # 肺活量
    ]

    # 确保所有特征都存在
    available_features = [f for f in features if f in df.columns]
    print(f"使用的特征: {available_features}")

    X = df[available_features].copy()
    y = df['sports_score'].copy()

    # 标准化特征
    print("标准化特征...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, available_features

def split_data(X, y):
    """分割数据集"""
    print("\n分割数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

def main():
    """主函数"""
    print("=" * 50)
    print("中考体育能力预测模型 - 数据预处理")
    print("=" * 50)

    # 1. 加载数据
    df, fields = load_and_preprocess_data()

    # 2. 清洗数据
    df_clean = clean_data(df)

    # 3. 准备特征
    X, y, scaler, feature_names = prepare_features(df_clean)

    # 4. 分割数据
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5. 保存处理后的数据
    print("\n保存处理后的数据...")

    # 保存清洗后的数据
    df_clean.to_csv('cleaned_sports_data.csv', index=False, encoding='utf-8-sig')
    print(f"清洗后的数据已保存到: cleaned_sports_data.csv")

    # 保存特征和目标变量
    np.savez('preprocessed_data.npz',
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test,
             feature_names=feature_names)
    print(f"预处理数据已保存到: preprocessed_data.npz")

    # 保存scaler
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    print(f"标准化器已保存到: scaler.pkl")

    # 6. 数据统计
    print("\n数据统计:")
    print(f"目标变量范围: {y.min()} - {y.max()}")
    print(f"目标变量平均值: {y.mean():.2f}")
    print(f"目标变量标准差: {y.std():.2f}")

    # 按性别统计
    male_scores = df_clean[df_clean['xb'] == '男']['sports_score']
    female_scores = df_clean[df_clean['xb'] == '女']['sports_score']

    print(f"\n男生分数统计:")
    print(f"  人数: {len(male_scores)}")
    print(f"  平均分: {male_scores.mean():.2f}")
    print(f"  标准差: {male_scores.std():.2f}")

    print(f"\n女生分数统计:")
    print(f"  人数: {len(female_scores)}")
    print(f"  平均分: {female_scores.mean():.2f}")
    print(f"  标准差: {female_scores.std():.2f}")

    print("\n数据预处理完成！")
    return df_clean, X_train, X_test, y_train, y_test, scaler, feature_names

if __name__ == "__main__":
    df_clean, X_train, X_test, y_train, y_test, scaler, feature_names = main()