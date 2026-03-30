import pandas as pd
import numpy as np
import pickle
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter

# ========================
# 常量定义区
# ========================
MARKERS_KEYWORDS = ['标记']  # 完整列表需补充
COLUMNS_TO_REMOVE = ['体检危害因素24', '所属部门20']  # 完整列表需补充
PRIVACY_COLUMNS = ['姓名0', '体检编号1', '监测类型18', '在岗状态19', '所属部门20', '体检危害因素24', '接触危害因素25'
                   '起止日期26', '部门车间28', '诊断日期31', '起止日期35', '主检建议1528',
                   '职业禁忌证名称1529', '疑似职业病名称1530', '体检机构1531', '体检日期1532', '报告出具日期1533',
                   '报告日期1534']
UNIT_KEYWORDS = ['单位']  # 标识单位字段的关键词

# 定义需要规范化的特征值映射
NORMALIZATION_MAPPING = {
    # 通用规范化值（适用于所有特征）
    'common': {
        '未见异常': '无',
        '无特殊情况': '无',
        '未见明显异常': '无',
        '目前无不适症状': '无',
        '正常': '无',
        '无': '无'  # 保持不变，但包含在映射中确保一致性
    },
    # 特定特征的额外规范化
    'specific': {
        '肌力结果350': {
            '正常（Ⅴ级）': '无',
            '5级': '无'
        },
        '肌张力结果356': {
            '正常肌张力': '无'
        }
    }
}


# ========================
# 数据加载模块
# ========================
def load_data(filepath: str) -> pd.DataFrame:
    """从pickle或csv文件加载数据"""
    if filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return pd.read_csv(filepath)


# ========================
# 预处理模块
# ========================
def normalize_special_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    规范化特定特征值：
    1. 将常见正常描述(未见异常、无特殊情况等)统一替换为'无'
    2. 对特定特征(肌力结果350、肌张力结果356)进行额外处理
    """
    # 首先应用通用规范化
    common_mapping = NORMALIZATION_MAPPING['common']
    for col in data.columns:
        # 只处理字符串类型的列
        if data[col].dtype == 'object':
            # 创建映射字典，保留原始值大小写
            col_mapping = {value: '无' for value in common_mapping.keys()}

            # 添加特定列的额外映射
            if col in NORMALIZATION_MAPPING['specific']:
                col_mapping.update(NORMALIZATION_MAPPING['specific'][col])

            # 应用映射
            data[col] = data[col].map(col_mapping).fillna(data[col])

    return data


def create_hazard_dummies(data: pd.DataFrame) -> pd.DataFrame:
    """
    对'接触危害因素25'进行哑变量拆分，只保留出现比例最高的前20个危害因素
    1. 提取所有危害因素并计算频率
    2. 选择频率最高的前20个危害因素
    3. 为这些危害因素创建哑变量
    4. 删除原始'接触危害因素25'列
    """
    if '接触危害因素25' not in data.columns:
        print("警告: '接触危害因素25'列不存在，跳过哑变量创建")
        return data

    # 填充缺失值为空字符串
    data['接触危害因素25'] = data['接触危害因素25'].fillna('')

    # 统计所有危害因素的出现频率
    hazard_counter = Counter()
    for hazards in data['接触危害因素25']:
        if hazards:  # 非空值
            # 按'、'拆分并添加到计数器中
            for hazard in hazards.split(','):
                cleaned_hazard = hazard.strip()
                if cleaned_hazard:  # 非空字符串
                    hazard_counter[cleaned_hazard] += 1

    # 获取总记录数
    total_records = len(data)
    print(f"总记录数: {total_records}")

    # 选择频率最高的前20个危害因素
    top_hazards = [hazard for hazard, _ in hazard_counter.most_common(20)]

    # 打印前20个危害因素及其比例
    print("\n前20个危害因素及其出现比例:")
    for i, hazard in enumerate(top_hazards, 1):
        count = hazard_counter[hazard]
        proportion = count / total_records
        print(f"{i}. {hazard}: {count}次 ({proportion:.2%})")

    # 创建哑变量列
    for hazard in top_hazards:
        col_name = f'危害因素_{hazard}'
        # 检查该危害因素是否存在于每个记录的危害因素列表中
        data[col_name] = data['接触危害因素25'].apply(
            lambda x: 1 if hazard in [h.strip() for h in x.split('、')] else 0
        )

    # 删除原始列
    data = data.drop(columns=['接触危害因素25'])
    print(f"已创建 {len(top_hazards)} 个哑变量列（前20个最高频危害因素）")

    return data


def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    数据预处理流程：
    1. 规范化特定特征值
    2. 填充特定字段缺失值
    3. 转换二值特征
    4. 工龄格式标准化
    5. 对接触危害因素进行哑变量拆分
    """
    # 1. 规范化特定特征值
    data = normalize_special_values(raw_data)

    # 2. 缺失值填充
    fill_rules = {
        '家族史42': '无',
        '其他症状46': '无',
        '个人史43': '无',
        '放射线种类41': '无',
        '治疗经过33': '已成'
    }
    data = data.fillna(fill_rules)

    # 3. 二值特征转换
    binary_features = ['家族史42', '个人史43', '其他44', '自觉症状45', '其他症状46', '既往病史疾病名称30', '防护措施29']
    for col in binary_features:
        data[col] = (~data[col].astype(str).str.contains('无|未|不', na=False)).astype(int)

    # 4. 工龄转换（年/月→总月数）
    for col in ['总工龄22', '接害工龄23']:
        data[col] = data[col].astype(str).apply(convert_duration_to_months)

    # 5. 对接触危害因素进行哑变量拆分
    data = create_hazard_dummies(data)

    return data


def convert_duration_to_months(duration: str) -> int:
    """将'X年Y月'格式转换为月数"""
    match = re.match(r"(\d+)年(\d+)月", str(duration))
    return int(match.group(1)) * 12 + int(match.group(2)) if match else 0


# ========================
# 数据清洗模块（新增单位字段删除功能）
# ========================
def clean_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    数据清洗主流程：
    1. 删除全缺失字段
    2. 删除低方差字段
    3. 移除隐私字段
    4. 删除包含"单位"的字段（新增）
    返回清洗后数据和日志
    """
    log = {}
    original_columns = set(data.columns)

    # 1. 删除全缺失字段
    missing_ratio = data.isnull().mean()
    full_missing_cols = missing_ratio[missing_ratio == 1].index.tolist()
    data = data.drop(columns=full_missing_cols)
    log['full_missing_removed'] = full_missing_cols

    # 2. 删除单值字段
    single_value_cols = [
        col for col in data.select_dtypes(include=['object']).columns
        if data[col].nunique() == 1
    ]
    data = data.drop(columns=single_value_cols)
    log['single_value_removed'] = single_value_cols

    # 3. 移除隐私字段
    privacy_cols = [col for col in PRIVACY_COLUMNS if col in data.columns]
    data = data.drop(columns=privacy_cols)
    log['privacy_removed'] = privacy_cols

    # 4. 删除包含单位的字段（新增功能）
    unit_cols = [
        col for col in data.columns
        if any(keyword in col for keyword in UNIT_KEYWORDS)
    ]
    data = data.drop(columns=unit_cols)
    log['unit_columns_removed'] = unit_cols

    # 5. 移除其他指定无关字段
    removed_cols = [col for col in COLUMNS_TO_REMOVE if col in data.columns]
    data = data.drop(columns=removed_cols)
    log['other_removed'] = removed_cols

    # 记录最终列变化
    log['columns_removed'] = list(original_columns - set(data.columns))
    log['remaining_columns'] = list(data.columns)

    return data, log


# ========================
# 标记处理模块
# ========================
def process_markers(data: pd.DataFrame) -> pd.DataFrame:
    """处理合格标记字段（填充缺失值）"""
    MARKERS_COLUMNS = [
        col for col in data.columns
        if any(keyword in col for keyword in MARKERS_KEYWORDS)
    ]
    marker_cols = [col for col in MARKERS_COLUMNS if col in data.columns]
    data[marker_cols] = data[marker_cols].fillna('合格')
    return data


# ========================
# 数据保存模块
# ========================
def save_data(data: pd.DataFrame, filepath: str, format: str = None):
    """
    保存数据到文件
    支持格式：csv/pkl/xlsx（根据文件后缀自动判断或显式指定）
    """
    # 自动判断格式
    if format is None:
        if filepath.endswith('.csv'):
            format = 'csv'
        elif filepath.endswith('.pkl'):
            format = 'pkl'
        elif filepath.endswith('.xlsx'):
            format = 'xlsx'
        else:
            raise ValueError("无法从文件后缀推断格式，请显式指定format参数")

    # 执行保存
    if format == 'csv':
        data.to_csv(filepath, index=False, encoding='utf-8-sig')
    elif format == 'pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'xlsx':
        # 处理Excel列宽自动调整
        writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
        data.to_excel(writer, index=False, sheet_name='Sheet1')

        # 获取工作表对象
        worksheet = writer.sheets['Sheet1']

        # 自动调整列宽
        for idx, col in enumerate(data.columns):
            max_len = max((
                data[col].astype(str).map(len).max(),  # 数据最大长度
                len(str(col))  # 列名长度
            )) + 2  # 添加缓冲
            worksheet.set_column(idx, idx, max_len)

        writer.close()
    else:
        raise ValueError(f"不支持的格式: {format}")


# ========================
# 主控制流程
# ========================
def main(input_path: str, output_path: str):
    """完整数据处理流程"""
    print("=== 开始数据处理 ===")

    # 1. 加载数据
    raw_data = load_data(input_path)
    print(f"原始数据形状: {raw_data.shape}")

    # 2. 预处理
    preprocessed = preprocess_data(raw_data)

    # 3. 数据清洗
    cleaned, log = clean_data(preprocessed)
    print(f"清洗后数据形状: {cleaned.shape}")

    # 4. 标记处理
    final_data = process_markers(cleaned)

    # 5. 保存结果
    save_format = 'pkl' if output_path.endswith('.pkl') else 'csv'
    save_data(final_data, output_path, format=save_format)

    # 打印清洗摘要
    print("\n=== 清洗摘要 ===")
    print(f"删除全缺失字段: {len(log.get('full_missing_removed', []))}个")
    print(f"删除单值字段: {len(log.get('single_value_removed', []))}个")
    print(f"删除隐私字段: {len(log.get('privacy_removed', []))}个")
    print(f"删除单位字段: {len(log.get('unit_columns_removed', []))}个")
    print(f"其他删除字段: {len(log.get('other_removed', []))}个")
    print(f"最终保留字段: {final_data.shape[1]}个")

    # 返回处理报告
    return {
        'original_shape': raw_data.shape,
        'final_shape': final_data.shape,
        'processing_log': log
    }


if __name__ == "__main__":
    # 配置输入输出路径
    input_file = "data.pkl"  # 支持.csv/.pkl
    output_file = "final_data.xlsx"

    # 执行处理流程
    processing_report = main(input_file, output_file)

    # 保存处理报告
    with open("processing_report.pkl", 'wb') as f:
        pickle.dump(processing_report, f)