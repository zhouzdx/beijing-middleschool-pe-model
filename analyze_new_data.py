import pandas as pd
import numpy as np
import json
import os
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

def analyze_csv_file(filepath):
    """分析CSV文件"""
    print(f"\n分析CSV文件: {os.path.basename(filepath)}")
    print("-" * 50)

    try:
        # 尝试不同的编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, nrows=1000)
                print(f"  使用编码: {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            # 尝试自动检测编码
            with open(filepath, 'rb') as f:
                raw_data = f.read(10000)
                import chardet
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                print(f"  自动检测编码: {encoding}")

            df = pd.read_csv(filepath, encoding=encoding, nrows=1000)

        print(f"  数据形状: {df.shape}")
        print(f"  列数: {len(df.columns)}")

        print("\n  列名:")
        for i, col in enumerate(df.columns, 1):
            print(f"    {i:2d}. {col}")

        print("\n  前3行数据:")
        print(df.head(3).to_string())

        print("\n  数据类型:")
        print(df.dtypes)

        print("\n  缺失值统计:")
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                print(f"    {col}: {count} 缺失 ({count/len(df)*100:.1f}%)")

        # 检查是否有中考分数列
        score_columns = [col for col in df.columns if any(keyword in str(col).lower() for keyword in ['分数', 'score', '总分', '成绩'])]
        if score_columns:
            print(f"\n  ⭐ 发现分数列: {score_columns}")
            for col in score_columns:
                if col in df.columns:
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        print(f"    {col}: {len(non_null)}非空, 范围: {non_null.min():.1f}-{non_null.max():.1f}")

        return df

    except Exception as e:
        print(f"  读取失败: {e}")
        return None

def analyze_json_file(filepath):
    """分析JSON文件"""
    print(f"\n分析JSON文件: {os.path.basename(filepath)}")
    print("-" * 50)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # JSON文件包含两个数组
        parts = content.split(']\n[')
        if len(parts) == 2:
            fields_str = parts[0] + ']'
            data_str = '[' + parts[1]

            fields = json.loads(fields_str)
            data = json.loads(data_str)

            print(f"  记录数: {len(data)}")
            print(f"  字段数: {len(fields)}")

            # 转换为DataFrame
            df = pd.DataFrame(data)

            print(f"  数据形状: {df.shape}")

            # 检查是否有中考分数信息
            print("\n  字段列表:")
            for i, field in enumerate(fields, 1):
                print(f"    {i:2d}. {field['name_cn']} ({field['name_en']})")

            return df, fields

    except Exception as e:
        print(f"  读取失败: {e}")
        return None, None

def analyze_xls_file(filepath):
    """分析XLS文件"""
    print(f"\n分析XLS文件: {os.path.basename(filepath)}")
    print("-" * 50)

    try:
        df = pd.read_excel(filepath, nrows=1000)

        print(f"  数据形状: {df.shape}")
        print(f"  列数: {len(df.columns)}")

        print("\n  列名:")
        for i, col in enumerate(df.columns, 1):
            print(f"    {i:2d}. {col}")

        # 检查是否有中考分数列
        score_columns = [col for col in df.columns if any(keyword in str(col).lower() for keyword in ['分数', 'score', '总分', '成绩'])]
        if score_columns:
            print(f"\n  ⭐ 发现分数列: {score_columns}")

        return df

    except Exception as e:
        print(f"  读取失败: {e}")
        return None

def main():
    """主函数"""
    print("="*60)
    print("2.0文件夹 - 新数据全面分析")
    print("="*60)

    data_folder = "D:\\agent\\模型\\2.0"
    os.chdir(data_folder)

    all_dataframes = {}

    # 1. 分析CSV文件
    csv_files = [
        '中小学生体育测试成绩信息_0 (1).csv',
        '高中学生体质测试信息202508010326.csv'
    ]

    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = analyze_csv_file(csv_file)
            if df is not None:
                all_dataframes[csv_file] = df

    # 2. 分析JSON文件
    json_file = '高中学生体质测试信息202508010326.json'
    if os.path.exists(json_file):
        json_df, json_fields = analyze_json_file(json_file)
        if json_df is not None:
            all_dataframes[json_file] = json_df

    # 3. 分析XLS文件
    xls_files = [
        '高中学生体质测试信息202411141903.xls',
        '高中学生体质测试信息202411141903 (1).xls'
    ]

    for xls_file in xls_files:
        if os.path.exists(xls_file):
            df = analyze_xls_file(xls_file)
            if df is not None:
                all_dataframes[xls_file] = df

    # 4. 数据对比和合并分析
    print("\n" + "="*60)
    print("数据对比和合并分析")
    print("="*60)

    if all_dataframes:
        print(f"\n成功读取 {len(all_dataframes)} 个数据文件:")
        for name, df in all_dataframes.items():
            print(f"  {name}: {df.shape[0]} 行 × {df.shape[1]} 列")

        # 寻找包含中考分数的文件
        print("\n寻找包含中考分数的数据:")
        score_data_found = False

        for name, df in all_dataframes.items():
            # 检查是否有分数列
            score_columns = [col for col in df.columns if any(keyword in str(col).lower() for keyword in ['分数', 'score', '总分', '成绩'])]
            if score_columns:
                print(f"\n  ✅ {name} 包含分数数据:")
                for col in score_columns:
                    non_null = df[col].dropna()
                    if len(non_null) > 0:
                        print(f"    列: {col}, 非空: {len(non_null)}, 范围: {non_null.min():.1f}-{non_null.max():.1f}")
                score_data_found = True

        if not score_data_found:
            print("  ⚠️ 未找到包含中考分数的数据文件")

        # 寻找共同字段
        print("\n寻找共同字段（用于数据合并）:")
        common_fields = None
        for name, df in all_dataframes.items():
            if common_fields is None:
                common_fields = set(df.columns)
            else:
                common_fields = common_fields.intersection(set(df.columns))

        if common_fields:
            print(f"  找到 {len(common_fields)} 个共同字段:")
            for field in sorted(common_fields):
                print(f"    - {field}")
        else:
            print("  未找到共同字段")

    # 5. 保存分析结果
    print("\n" + "="*60)
    print("保存分析结果")
    print("="*60)

    # 创建分析报告
    report = f"""
2.0文件夹数据文件分析报告
{'='*60}

分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
分析文件夹: {data_folder}

文件分析结果:
"""

    for name, df in all_dataframes.items():
        report += f"\n{os.path.basename(name)}:"
        report += f"\n  记录数: {df.shape[0]}"
        report += f"\n  字段数: {df.shape[1]}"

        # 检查分数列
        score_columns = [col for col in df.columns if any(keyword in str(col).lower() for keyword in ['分数', 'score', '总分', '成绩'])]
        if score_columns:
            report += f"\n  包含分数列: {score_columns}"

    report += f"""

数据质量评估:
- CSV文件: 需要检查编码和数据结构
- JSON文件: 结构清晰但可能缺少分数数据
- XLS文件: 格式统一但数据量较小

建议:
1. 优先使用包含中考分数的CSV文件
2. 检查数据一致性
3. 合并多个数据源以增加数据量

{'='*60}
"""

    with open('new_data_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"详细报告已保存到: new_data_analysis_report.txt")

    return all_dataframes

if __name__ == "__main__":
    all_dataframes = main()