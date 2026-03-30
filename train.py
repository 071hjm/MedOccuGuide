# ========================
# 导入必要的库
# ========================
import os
import re
import time
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, brier_score_loss,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, LabelEncoder,
    LabelBinarizer, MinMaxScaler
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


from sklearn.linear_model import LogisticRegression

# 遗传算法相关库 - 使用维护版本 sklearn-genetic-opt
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous

# ========================
# 全局配置
# ========================
SEED = 42
DATA_DIR = "processed_data"
MODEL_DIR = "saved_models"
FIG_DIR = "figures"
SHAP_DIR = "shap_values"
CM_DIR = os.path.join(FIG_DIR, "confusion_matrices")  # 混淆矩阵保存目录

# 创建目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)  # 创建混淆矩阵目录

# 设置字体支持中文
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
shap.initjs()

# 标签映射
LABEL_MAPPING = {
    0: "其他疾病或异常",
    1: "目前未见异常",
    2: "复查",
    3: "职业禁忌证",
    4: "疑似职业病"
}

LABEL_MAPPING_REVERSE = {
    "其他疾病或异常": 0,
    "目前未见异常": 1,
    "复查": 2,
    "职业禁忌证": 3,
    "疑似职业病": 4
}


# ========================
# 数据处理函数
# ========================
def load_and_preprocess_data(logger=None):
    """
    加载和预处理数据集

    返回:
    X (numpy.ndarray): 特征数据
    y (numpy.ndarray): 标签数据
    feature_names (list): 特征名称列表
    """
    # 加载特征数据
    with open('final_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # 获取标签列
    fi = data['主检结论1527']
    data = data.drop(columns=['主检结论1527'])

    # 检查NaN值
    nan_count = fi.isna().sum()
    if nan_count > 0:
        print(f"警告: 发现 {nan_count} 个NaN标签，将删除这些样本")

        # 获取非NaN的索引
        not_nan_idx = fi.notna()

        # 删除NaN样本
        fi = fi[not_nan_idx].copy()
        data = data[not_nan_idx].copy()

    # 转换中文标签为数字
    if isinstance(fi.iloc[0], str):  # 使用iloc[0]更安全
        try:
            fi = np.array([LABEL_MAPPING_REVERSE[label] for label in fi])
        except KeyError as e:
            # 找出无效标签
            invalid_labels = set(label for label in fi if label not in LABEL_MAPPING_REVERSE)
            print(f"错误: 发现无效标签 - {invalid_labels}")
            raise ValueError(f"无效标签值: {invalid_labels}。请检查数据或更新LABEL_MAPPING_REVERSE")

    # 确保为整数类型
    fi = fi.astype(int)

    # 验证处理结果
    print(f"处理后样本数量: {len(data)}")
    print("标签分布:", dict(zip(*np.unique(fi, return_counts=True))))

    # 显示原始标签分布
    if logger:
        logger.info("=" * 80)
        logger.info("原始标签分布:")
    else:
        print("=" * 80)
        print("原始标签分布:")

    if hasattr(fi, 'value_counts'):
        counts = fi.value_counts()
        total = len(fi)
        for label, count in counts.items():
            msg = f"标签 {label}: {count} 个样本 ({count / total * 100:.2f}%)"
            if logger:
                logger.info(msg)
            else:
                print(msg)
    else:
        unique, counts = np.unique(fi, return_counts=True)
        total = len(fi)
        for u, c in zip(unique, counts):
            msg = f"标签 {u}: {c} 个样本 ({c / total * 100:.2f}%)"
            if logger:
                logger.info(msg)
            else:
                print(msg)

    # 找出需要删除的样本索引（标签为3或4的样本）
    delete_indices = [i for i, label in enumerate(fi) if label in {3, 4}]
    msg = f"将删除 {len(delete_indices)} 个样本 (标签值为3或4)"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    # 删除对应样本
    if isinstance(data, np.ndarray):
        data_filtered = np.delete(data, delete_indices, axis=0)
        feature_names = [f"feature_{i}" for i in range(data_filtered.shape[1])]
    else:  # 假设为pandas DataFrame
        data_filtered = data.drop(delete_indices).reset_index(drop=True)
        feature_names = data_filtered.columns.tolist()

    if isinstance(fi, np.ndarray):
        fi_filtered = np.delete(fi, delete_indices)
    else:  # 假设为pandas Series
        fi_filtered = fi.drop(delete_indices).reset_index(drop=True)

    # 显示过滤后的标签分布
    if logger:
        logger.info("\n过滤后的标签分布:")
    else:
        print("\n过滤后的标签分布:")

    if hasattr(fi_filtered, 'value_counts'):
        counts = fi_filtered.value_counts()
        total = len(fi_filtered)
        for label, count in counts.items():
            msg = f"标签 {label}: {count} 个样本 ({count / total * 100:.2f}%)"
            if logger:
                logger.info(msg)
            else:
                print(msg)
    else:
        unique, counts = np.unique(fi_filtered, return_counts=True)
        total = len(fi_filtered)
        for u, c in zip(unique, counts):
            msg = f"标签 {u}: {c} 个样本 ({c / total * 100:.2f}%)"
            if logger:
                logger.info(msg)
            else:
                print(msg)

    X = data_filtered.values if hasattr(data_filtered, 'values') else data_filtered
    y = np.array(fi_filtered)

    msg = f"最终数据集大小: {X.shape}"
    if logger:
        logger.info(msg)
        logger.info("=" * 80)
    else:
        print(msg)
        print("=" * 80)

    return X, y, feature_names


def process_feature_names(feature_names):
    """
    处理特征名称：去掉数字后缀

    输入:
    - feature_names: 原始特征名称列表

    输出:
    - processed_names: 处理后的特征名称列表
    """
    processed_names = []
    for name in feature_names:
        # 去掉数字后缀（如"特征_123" -> "特征"）
        base_name = name.split('_')[0]
        processed_names.append(base_name)
    return processed_names


def smart_type_conversion(df):
    """
    智能类型转换：将混合类型的列转换为统一的数值类型

    输入:
    - df: 原始数据框

    输出:
    - df: 转换后的数据框
    """
    for col in df.columns:
        # 尝试转换为数值类型
        converted = pd.to_numeric(df[col], errors='coerce')
        # 计算转换成功率
        conversion_rate = converted.notna().mean()
        if conversion_rate > 0.8:  # 如果80%以上的值可以转换为数字
            # 将非数值值设为0
            df[col] = converted.fillna(0).astype(float)
            print(f"将特征 '{col}' 转换为浮点型 (转换率: {conversion_rate:.2%})")
        else:
            # 保留为字符串类型
            df[col] = df[col].astype(str)
            print(f"特征 '{col}' 保留为字符串类型 (转换率: {conversion_rate:.2%})")
    return df


# ========================
# 模型评估函数
# ========================
def multiclass_brier_score(y_true, y_prob):
    """多分类Brier分数计算"""
    lb = LabelBinarizer()
    Y = lb.fit_transform(y_true)
    return brier_score_loss(Y.ravel(), y_prob.ravel())


def multiclass_specificity(y_true, y_pred):
    """计算多分类宏平均特异度"""
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(cm.shape[0]):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)
    return np.mean(specificities)


def multiclass_npv(y_true, y_pred):
    """计算多分类宏平均阴性预测值"""
    cm = confusion_matrix(y_true, y_pred)
    npvs = []
    for i in range(cm.shape[0]):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fn = np.sum(cm[i, :]) - cm[i, i]
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        npvs.append(npv)
    return np.mean(npvs)


def calculate_metric_with_ci(y_true, y_pred, y_prob, metric_func, n_bootstraps=500, alpha=0.05):
    """
    计算指标及其95%置信区间

    输入:
    - y_true: 真实标签
    - y_pred: 预测标签
    - y_prob: 预测概率
    - metric_func: 指标计算函数
    - n_bootstraps: 重采样次数
    - alpha: 显著性水平

    输出:
    - mean_score: 指标平均值
    - (lower_bound, upper_bound): 置信区间
    """
    scores = []
    n_samples = len(y_true)
    rng = np.random.RandomState(SEED)
    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, n_samples, replace=True)
        try:
            # 注意：这里根据指标函数需要的参数传递
            if metric_func.__name__ in ['multiclass_brier_score', 'roc_auc_score']:
                # 这些指标需要y_prob
                score = metric_func(y_true[indices], y_prob[indices])
            elif metric_func.__name__ in ['multiclass_specificity', 'multiclass_npv']:
                # 这些指标需要y_pred
                score = metric_func(y_true[indices], y_pred[indices])
            else:
                # 其他指标需要y_pred和y_prob
                score = metric_func(y_true[indices], y_pred[indices], y_prob[indices])
            scores.append(score)
        except Exception as e:
            # 如果计算失败，跳过
            # print(f"计算失败: {str(e)}")
            continue
    if len(scores) > 0:
        mean_score = np.mean(scores)
        lower_bound = np.percentile(scores, 100 * alpha / 2)
        upper_bound = np.percentile(scores, 100 * (1 - alpha / 2))
    else:
        mean_score = np.nan
        lower_bound = np.nan
        upper_bound = np.nan
    return mean_score, (lower_bound, upper_bound)


def calculate_all_metrics(y_true, y_pred, y_prob):
    """
    计算所有需要的指标

    输入:
    - y_true: 真实标签
    - y_pred: 预测标签
    - y_prob: 预测概率

    输出:
    - scores: 包含所有指标的字典
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')  # PPV
    recall_macro = recall_score(y_true, y_pred, average='macro')  # Sensitivity
    f1_macro = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    brier = multiclass_brier_score(y_true, y_prob)

    try:
        auc_ovo = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
    except Exception as e:
        print(f"计算AUC-ROC失败: {str(e)}")
        auc_ovo = np.nan

    specificity_macro = multiclass_specificity(y_true, y_pred)  # Specificity
    npv_macro = multiclass_npv(y_true, y_pred)  # NPV

    return {
        'Accuracy': accuracy,
        'Precision_macro': precision_macro,  # PPV
        'Recall_macro': recall_macro,  # Sensitivity
        'F1_macro': f1_macro,
        'AUC_ovo_macro': auc_ovo,
        'Specificity_macro': specificity_macro,
        'NPV_macro': npv_macro,
        'MCC': mcc,
        'Kappa': kappa,
        'BrierScore': brier
    }


def save_confusion_matrix(y_true, y_pred, model_name, label_mapping):
    """
    计算、保存混淆矩阵为CSV和图片

    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - model_name: 模型名称
    - label_mapping: 标签映射字典
    """

    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        # 如果 y_pred 是二维的，将其转换为一维
        y_pred = y_pred.flatten()

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 获取实际存在的标签
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    label_names = [label_mapping[label] for label in unique_labels]

    # 保存为CSV文件
    cm_df = pd.DataFrame(
        cm,
        index=label_names,
        columns=label_names
    )
    cm_df.to_csv(f"{CM_DIR}/confusion_matrix_{model_name}.csv")

    # 绘制混淆矩阵热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar=True
    )
    plt.title(f'{model_name} 混淆矩阵', fontsize=16)
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 保存图片
    plt.savefig(f"{CM_DIR}/confusion_matrix_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  混淆矩阵已保存: {CM_DIR}/confusion_matrix_{model_name}.csv 和 .png")


# 在数据处理函数区域添加清理文件名的函数
def clean_filename(filename):
    """清理文件名中的无效字符（适用于Windows系统）"""
    invalid_chars = '/\\:*?"<>| \t\n\r'  # 包含制表符\t等无效字符
    for char in invalid_chars:
        filename = filename.replace(char, '_')  # 用下划线替换
    return filename



# ========================
# 主程序逻辑
# ========================
def main():
    # 确保目录存在
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("shap_values", exist_ok=True)
    os.makedirs("processed_data", exist_ok=True)
    os.makedirs(CM_DIR, exist_ok=True)  # 确保混淆矩阵目录存在

    # ========================
    # 数据加载与预处理
    # ========================
    print("加载数据...")
    X, y, feature_names = load_and_preprocess_data()
    y = y.astype(int)

    # 将标签5映射为2（复查）
    y[y == 5] = 2

    # 获取过滤后的标签映射（只包含实际存在的标签）
    present_labels = np.unique(y)
    filtered_label_mapping = {k: v for k, v in LABEL_MAPPING.items() if k in present_labels}

    # 转换为DataFrame并智能类型转换
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)
    X = smart_type_conversion(X)

    # 分离数值型和类别型特征
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()

    # 保存特征类型信息
    feature_info = {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'all_features': X.columns.tolist()
    }
    joblib.dump(feature_info, "processed_data/feature_info.pkl")
    print(f"数值型特征数量: {len(numeric_cols)}")
    print(f"类别型特征数量: {len(categorical_cols)}")
    print(f"总特征数量: {len(feature_info['all_features'])}")

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=SEED
    )

    if categorical_cols:
        print("处理类别型特征（标签编码）...")

        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()

        for col in categorical_cols:
            le = LabelEncoder()
            # 合并训练测试集统一编码
            combined = pd.concat([X_train[col], X_test[col]], axis=0)
            le.fit(combined)

            # 转换数据
            X_train_processed[col] = le.transform(X_train[col])
            X_test_processed[col] = le.transform(X_test[col])

            # 清理特征名中的无效字符
            cleaned_col = clean_filename(col)
            encoder_path = os.path.join("saved_models", f"label_encoder_{cleaned_col}.pkl")

            # 保存编码器
            try:
                joblib.dump(le, encoder_path)
                print(f"已保存编码器: {encoder_path} (原始特征名: {col})")
            except Exception as e:
                print(f"警告: 无法保存编码器 {col} (清理后: {cleaned_col}): {str(e)}")
                continue

    # 保存处理后的数据
    joblib.dump(X_train_processed, f"{DATA_DIR}/X_train_processed.pkl")
    joblib.dump(X_test_processed, f"{DATA_DIR}/X_test_processed.pkl")
    joblib.dump(y_train, f"{DATA_DIR}/y_train.pkl")
    joblib.dump(y_test, f"{DATA_DIR}/y_test.pkl")

    # 转换为NumPy数组
    X_train = X_train_processed.values.astype(np.float32)
    X_test = X_test_processed.values.astype(np.float32)
    feature_names = X_train_processed.columns.tolist()

    # 创建全局scaler
    global_scaler = StandardScaler().fit(X_train)
    joblib.dump(global_scaler, f"{MODEL_DIR}/scaler.pkl")

    # ========================
    # 模型定义与遗传算法参数空间
    # ========================
    param_spaces = {
        'LR': {
            'clf__C': Continuous(0.01, 10),
            'clf__penalty': Categorical(['l1', 'l2']),
            'clf__solver': Categorical(['saga'])
        },
        'DT': {
            'clf__max_depth': Integer(10, 5000),
            'clf__min_samples_split': Integer(2, 50)
        },
        'RF': {
            'clf__n_estimators': Integer(100, 5000),
            'clf__max_features': Categorical(['sqrt', 'log2']),
            'clf__max_depth': Integer(10, 50)
        },
        'kNN': {
            'clf__n_neighbors': Integer(3, 20),
            'clf__weights': Categorical(['uniform', 'distance'])
        },
        'NB': {
            'clf__var_smoothing': Continuous(1e-10, 1e-5)
        },
        'AdaBoost': {
            'clf__n_estimators': Integer(50, 5000),
            'clf__learning_rate': Continuous(0.01, 1)
        },
        'XGBoost': {
            'clf__n_estimators': Integer(100, 5000),
            'clf__learning_rate': Continuous(0.01, 1),
            'clf__max_depth': Integer(3, 20),
            'clf__subsample': Continuous(0.5, 1)
        },
        'LightGBM': {
            'clf__n_estimators': Integer(100, 5000),
            'clf__learning_rate': Continuous(0.01, 1),
            'clf__num_leaves': Integer(20, 150),
            'clf__subsample': Continuous(0.5, 1)
        },
        'CatBoost': {
            'clf__iterations': Integer(100, 1000),
            'clf__depth': Integer(4, 12),
            'clf__learning_rate': Continuous(0.01, 1),
            'clf__l2_leaf_reg': Continuous(1, 10)
        }
    }

    model_space = {
        'LR': {'estimator': LogisticRegression(solver='saga', max_iter=10000, n_jobs=-1)},
        'DT': {'estimator': DecisionTreeClassifier()},
        'RF': {'estimator': RandomForestClassifier(n_jobs=-1)},
        'kNN': {'estimator': KNeighborsClassifier(n_jobs=-1)},
        'NB': {'estimator': GaussianNB()},
        'AdaBoost': {'estimator': AdaBoostClassifier()},
        'XGBoost': {
            'estimator': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)},
        'LightGBM': {'estimator': LGBMClassifier(n_jobs=-1)},
        'CatBoost': {'estimator': CatBoostClassifier(verbose=0, thread_count=-1)}
    }

    # ========================
    # 结果存储初始化
    # ========================
    results_file = "model_final_results.csv"
    params_file = "best_models_summary.csv"
    shap_status_file = "shap_status.csv"
    ci_results_file = "model_results_with_ci.csv"

    # 初始化结果数据框
    if os.path.exists(results_file):
        df_results = pd.read_csv(results_file, index_col='Model')
    else:
        df_results = pd.DataFrame(columns=[
            'Model', 'Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro',
            'AUC_ovo_macro', 'Specificity_macro', 'NPV_macro', 'MCC', 'Kappa', 'BrierScore', 'TrainTime(s)'
        ]).set_index('Model')

    if os.path.exists(params_file):
        df_params = pd.read_csv(params_file, index_col=0)
    else:
        df_params = pd.DataFrame(columns=['cv_f1_macro', 'tune_time_s'])

    # 初始化置信区间结果
    if os.path.exists(ci_results_file):
        df_ci = pd.read_csv(ci_results_file, index_col='Model')
    else:
        metrics = [
            'Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro',
            'AUC_ovo_macro', 'Specificity_macro', 'NPV_macro', 'MCC', 'Kappa', 'BrierScore'
        ]
        columns = []
        for metric in metrics:
            columns.extend([f"{metric}", f"{metric}_ci_lower", f"{metric}_ci_upper"])
        df_ci = pd.DataFrame(columns=['Model', 'TrainTime(s)'] + columns).set_index('Model')

    # 初始化SHAP状态记录
    if os.path.exists(shap_status_file):
        df_shap_status = pd.read_csv(shap_status_file, index_col='Model')
    else:
        df_shap_status = pd.DataFrame(columns=['Model', 'SHAP_success']).set_index('Model')

    # ========================
    # 模型训练与评估循环
    # ========================
    for name, cfg in model_space.items():
        model_path = f"saved_models/{name}.pkl"
        shap_path = f"shap_values/shap_values_{name}.npz"

        # 检查模型是否已训练
        model_trained = os.path.exists(model_path)
        shap_completed = name in df_shap_status.index and df_shap_status.loc[name, 'SHAP_success']

        if model_trained:
            print(f">>> {name} 已训练，加载模型及超参数")
            # 加载模型数据（可能是单纯模型对象或新格式字典）
            loaded_obj = joblib.load(model_path)

            # 区分旧格式（单纯模型对象）和新格式（包含超参数的字典）
            if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
                # 新格式：从字典中提取模型和超参数
                pipe = loaded_obj['model']
                best_params = loaded_obj.get('hyperparameters', {})
                print(f"  加载新格式模型，超参数: {best_params}")
            else:
                # 旧格式：直接是模型对象（单纯的Pipeline等）
                pipe = loaded_obj
                # 尝试从df_params中获取超参数（如果存在）
                if name in df_params.index:
                    best_params = df_params.loc[name].to_dict()
                    # 移除非超参数列
                    for key in ['cv_f1_macro', 'tune_time_s', 'seed']:
                        best_params.pop(key, None)
                    print(f"  加载旧格式模型，从参数文件获取超参数: {best_params}")
                else:
                    best_params = {}
                    print(f"  加载旧格式模型，未找到超参数信息")

            # 处理训练时间
            if name in df_params.index:
                elapsed = df_params.loc[name, 'tune_time_s']
            else:
                # 从新格式字典中获取训练时间
                if isinstance(loaded_obj, dict) and 'train_time' in loaded_obj:
                    elapsed = loaded_obj['train_time']
                    print(f"  加载训练时间: {elapsed:.1f}秒")
                else:
                    elapsed = np.nan
                    print(f"  未找到训练时间记录")
        else:
            # 训练新模型，按新格式保存（包含模型+超参数）
            print(f">>> 训练 & 调参 {name} (使用遗传算法)")
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=SEED)),
                ('clf', cfg['estimator'])
            ])
            np.random.seed(SEED)
            search = GASearchCV(
                estimator=pipe,
                param_grid=param_spaces[name],
                scoring='f1_macro',
                population_size=5,
                generations=10,
                tournament_size=3,
                elitism=True,
                crossover_probability=0.7,
                mutation_probability=0.3,
                cv=5,
                verbose=True,
                n_jobs=-1,
            )

            t0 = time.time()
            search.fit(X_train, y_train)
            elapsed = time.time() - t0
            pipe = search.best_estimator_
            best_params = search.best_params_

            # 保存超参数到CSV
            df_params.loc[name] = {
                **best_params,
                'cv_f1_macro': round(search.best_score_, 4),
                'tune_time_s': round(elapsed, 1)
            }
            df_params.to_csv(params_file)
            best_score = search.best_score_
            del search
            gc.collect()

            # 新格式：保存为包含模型和超参数的字典
            model_data = {
                'model': pipe,
                'hyperparameters': best_params,
                'feature_names': feature_names,
                'train_time': elapsed,
                'cv_score': best_score,
                'seed': SEED
            }
            joblib.dump(model_data, model_path)
            print(f"  新格式模型已保存（含超参数）到 {model_path}")
            print(f"  最佳参数: {best_params}")
            print(f"  最佳f1分数: {df_params.loc[name, 'cv_f1_macro']:.4f}")
            shap_completed = False

        # ========================
        # 模型评估
        # ========================
        if not model_trained or name not in df_results.index or name=="CatBoost":
            if model_trained and os.path.exists(f"saved_models/{name}.pkl"):
                model_data = joblib.load(f"saved_models/{name}.pkl")
                # 从字典中提取实际的模型管道
                pipe = model_data['model']
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)

            # 计算所有指标
            scores = calculate_all_metrics(y_test, y_pred, y_prob)
            scores['TrainTime(s)'] = elapsed

            # 保存点估计结果
            df_results.loc[name] = scores
            df_results.to_csv(results_file)

            # 计算并保存混淆矩阵
            save_confusion_matrix(y_test, y_pred, name, filtered_label_mapping)

            # 计算置信区间
            ci_results = {'Model': name, 'TrainTime(s)': elapsed}

            # 为每个指标计算置信区间
            metrics_functions = {
                'Accuracy': lambda y_t, y_p, y_pr: accuracy_score(y_t, y_p),
                'Precision_macro': lambda y_t, y_p, y_pr: precision_score(y_t, y_p, average='macro'),
                'Recall_macro': lambda y_t, y_p, y_pr: recall_score(y_t, y_p, average='macro'),
                'F1_macro': lambda y_t, y_p, y_pr: f1_score(y_t, y_p, average='macro'),
                'AUC_ovo_macro': lambda y_t, y_p, y_pr: roc_auc_score(y_t, y_pr, multi_class='ovo', average='macro'),
                'Specificity_macro': lambda y_t, y_p, y_pr: multiclass_specificity(y_t, y_p),
                'NPV_macro': lambda y_t, y_p, y_pr: multiclass_npv(y_t, y_p),
                'MCC': lambda y_t, y_p, y_pr: matthews_corrcoef(y_t, y_p),
                'Kappa': lambda y_t, y_p, y_pr: cohen_kappa_score(y_t, y_p),
                'BrierScore': lambda y_t, y_p, y_pr: multiclass_brier_score(y_t, y_pr)
            }

            for metric_name, metric_func in metrics_functions.items():
                try:
                    mean_score, (lower, upper) = calculate_metric_with_ci(
                        y_test, y_pred, y_prob, metric_func
                    )
                    ci_results[metric_name] = mean_score
                    ci_results[f"{metric_name}_ci_lower"] = lower
                    ci_results[f"{metric_name}_ci_upper"] = upper
                except Exception as e:
                    print(f"  计算 {metric_name} 置信区间失败: {str(e)}")
                    ci_results[metric_name] = np.nan
                    ci_results[f"{metric_name}_ci_lower"] = np.nan
                    ci_results[f"{metric_name}_ci_upper"] = np.nan

            # 保存置信区间结果
            df_ci.loc[name] = ci_results
            df_ci.to_csv(ci_results_file)
            print(f"  {name} 模型评估完成")
        else:
            print(f"  {name} 模型评估结果已存在，跳过评估")

        # ========================
        # SHAP解释性分析
        # ========================
        if not shap_completed:
            print(f"  计算 {name} 的 SHAP 值...")
            shap_values = None
            try:
                # 获取模型和缩放器
                clf = pipe.named_steps['clf']
                scaler = pipe.named_steps['scaler']
                X_test_scaled = scaler.transform(X_test)

                # 对于大型数据集进行采样
                sample_size = min(2000, X_test_scaled.shape[0])
                sample_idx = np.random.choice(X_test_scaled.shape[0], sample_size, replace=False)
                X_test_sampled = X_test_scaled[sample_idx]

                # 根据模型类型选择合适的解释器
                tree_models = ['DT', 'RF', 'XGBoost', 'LightGBM', 'CatBoost']
                if name in tree_models:
                    # 移除check_additivity参数，该参数在新版本SHAP中已被移除
                    explainer = shap.TreeExplainer(clf)
                elif name in ['LR', 'AdaBoost']:
                    explainer = shap.LinearExplainer(clf, X_test_sampled)
                elif name in ['kNN', 'NB']:
                    background = shap.kmeans(X_test_sampled, min(10, X_test_sampled.shape[0]))
                    explainer = shap.KernelExplainer(clf.predict_proba, background)
                else:
                    explainer = shap.Explainer(clf, X_test_sampled)

                # 计算SHAP值
                shap_values = explainer(X_test_sampled)

                # 保存完整的SHAP数据
                shap_data = {
                    'values': shap_values.values,
                    'base_values': shap_values.base_values,
                    'data': shap_values.data,
                    'feature_names': feature_names,
                    'processed_feature_names': process_feature_names(feature_names),
                    'sample_indices': sample_idx
                }
                joblib.dump(shap_data, f"shap_values/shap_data_{name}.pkl")
                print(f"  完整的SHAP数据已保存到 shap_values/shap_data_{name}.pkl")

                # 1. 绘制每个类别的SHAP汇总图
                n_classes = shap_values.shape[2] if len(shap_values.shape) == 3 else 1
                processed_feature_names = process_feature_names(feature_names)

                # 为每个类别创建SHAP汇总图
                for class_idx in range(n_classes):
                    plt.figure(figsize=(12, 10))

                    if n_classes > 1:
                        class_shap = shap_values[:, :, class_idx]
                        class_title = f"{name} SHAP摘要图 - 类别: {filtered_label_mapping.get(class_idx, f'类别{class_idx}')}"
                    else:
                        class_shap = shap_values
                        class_title = f"{name} SHAP摘要图"

                    shap.summary_plot(
                        class_shap,
                        X_test_sampled,
                        feature_names=processed_feature_names,
                        show=False,
                        plot_size=None
                    )
                    plt.title(class_title, fontsize=16)
                    plt.tight_layout()
                    plt.savefig(f"figures/shap_summary_{name}_class{class_idx}.png", dpi=150, bbox_inches='tight')
                    plt.close()

                # 2. 绘制所有类别平均的SHAP汇总图
                if n_classes > 1:
                    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=2)
                    plt.figure(figsize=(12, 10))
                    shap.summary_plot(
                        mean_abs_shap,
                        X_test_sampled,
                        feature_names=processed_feature_names,
                        show=False,
                        plot_type="bar",
                        plot_size=None
                    )
                    plt.title(f"{name} SHAP摘要图 - 所有类别平均", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(f"figures/shap_summary_{name}_global_avg.png", dpi=150, bbox_inches='tight')
                    plt.close()

                    # 3. 绘制多类别SHAP条形图
                    plt.figure(figsize=(14, 10))
                    shap.summary_plot(
                        shap_values,
                        X_test_sampled,
                        feature_names=processed_feature_names,
                        show=False,
                        plot_type="bar",
                        class_names=[filtered_label_mapping.get(i, f"类别{i}") for i in range(n_classes)]
                    )
                    plt.title(f"{name} 多类别SHAP重要性", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(f"figures/shap_bar_{name}_multi_class.png", dpi=150, bbox_inches='tight')
                    plt.close()
                else:
                    # 单类别模型的条形图
                    plt.figure(figsize=(12, 10))
                    shap.summary_plot(
                        shap_values,
                        X_test_sampled,
                        feature_names=processed_feature_names,
                        show=False,
                        plot_type="bar"
                    )
                    plt.title(f"{name} SHAP特征重要性", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(f"figures/shap_bar_{name}.png", dpi=150, bbox_inches='tight')
                    plt.close()

                # 记录SHAP分析成功
                df_shap_status.loc[name] = {'SHAP_success': True}
                df_shap_status.to_csv(shap_status_file)
                print(f"  SHAP 分析完成，所有图表已保存。")
            except Exception as e:
                import traceback
                print(f"  计算SHAP失败: {str(e)}")
                print(traceback.format_exc())
                df_shap_status.loc[name] = {'SHAP_success': False}
                df_shap_status.to_csv(shap_status_file)
            finally:
                # 安全清理资源
                if 'clf' in locals():
                    del clf
                if 'scaler' in locals():
                    del scaler
                if shap_values is not None:
                    del shap_values
                gc.collect()
        else:
            print(f"  SHAP分析已完成，跳过")

    # ========================
    # 结果可视化
    # ========================
    # 确保figures目录存在
    os.makedirs("figures", exist_ok=True)

    # 指标对比柱状图（带置信区间）
    if len(df_ci) > 0:
        metrics = [
            'Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro',
            'AUC_ovo_macro', 'Specificity_macro', 'NPV_macro', 'MCC', 'Kappa', 'BrierScore'
        ]

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 8))

            # 确保有置信区间数据
            if metric not in df_ci.columns or f"{metric}_ci_lower" not in df_ci.columns:
                print(f"警告: {metric} 缺少置信区间数据")
                continue

            # 提取点估计和置信区间
            point_estimates = df_ci[metric]
            lower_bounds = df_ci[f"{metric}_ci_lower"]
            upper_bounds = df_ci[f"{metric}_ci_upper"]

            # 计算误差条位置
            lower_errors = point_estimates - lower_bounds
            upper_errors = upper_bounds - point_estimates
            yerr = [lower_errors, upper_errors]

            # 创建柱状图
            bars = ax.bar(
                x=range(len(df_ci.index)),
                height=point_estimates,
                color=sns.color_palette("Set2"),
                yerr=yerr,
                capsize=5,
                error_kw={'elinewidth': 2, 'capthick': 2}
            )

            ax.set_title(f"{metric} 模型对比 (95% CI)", fontsize=16)
            ax.set_xticks(range(len(df_ci.index)))
            ax.set_xticklabels(df_ci.index, rotation=45, ha='right', fontsize=12)
            ax.set_ylabel(metric, fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)

            # 添加数据标签
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.01,
                    f'{height:.3f}\n({lower_bounds[i]:.3f}-{upper_bounds[i]:.3f})',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )

            plt.tight_layout()
            fig.savefig(f"figures/{metric}_bar_CI.png", dpi=150)
            plt.close()

        # 置信区间比较图
        fig, ax = plt.subplots(figsize=(14, 10))
        ci_data = []
        for model in df_ci.index:
            for metric in ['Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro', 'AUC_ovo_macro']:
                if f"{metric}_ci_lower" in df_ci.columns:
                    ci_data.append({
                        'Model': model,
                        'Metric': metric,
                        'Value': df_ci.loc[model, metric],
                        'CI_lower': df_ci.loc[model, f"{metric}_ci_lower"],
                        'CI_upper': df_ci.loc[model, f"{metric}_ci_upper"]
                    })
        ci_df = pd.DataFrame(ci_data)
        sns.pointplot(
            x='Metric',
            y='Value',
            hue='Model',
            data=ci_df,
            join=False,
            capsize=0.1,
            markers='o',
            scale=0.8,
            errwidth=1.5,
            ax=ax
        )
        ax.set_title("模型性能比较 (95%置信区间)", fontsize=18)
        ax.set_ylabel("得分", fontsize=14)
        ax.set_xlabel("指标", fontsize=14)
        ax.legend(title='模型', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        fig.savefig("figures/metrics_CI_comparison.png", dpi=150)
        plt.close()

    # 训练时间对比图
    if len(df_ci) > 0:
        # 垂直条形图
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=df_ci.index,
            y=df_ci['TrainTime(s)'] / 60,
            hue=df_ci.index,
            palette="rocket",
            ax=ax,
            legend=False
        )
        ax.set_title("训练时间对比 (分钟)")
        ax.set_ylabel("分钟")
        ax.set_xticks(range(len(df_ci.index)))
        ax.set_xticklabels(df_ci.index, rotation=45, ha='right')
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        fig.savefig("figures/TrainingTime_bar.png", dpi=150)
        plt.close()

        # 水平条形图
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(
            y=df_ci.index,
            x=df_ci['TrainTime(s)'] / 60,
            hue=df_ci.index,
            palette="rocket",
            ax=ax,
            legend=False,
            orient='h'
        )
        ax.set_title("训练时间对比 (分钟)")
        ax.set_xlabel("分钟")
        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 0.1, p.get_y() + p.get_height() / 2,
                    f'{width:.1f}',
                    ha='left', va='center')
        plt.tight_layout()
        fig.savefig("figures/TrainingTime_horizontal_bar.png", dpi=150)
        plt.close()

        # 饼图
        fig, ax = plt.subplots(figsize=(10, 8))
        time_percent = (df_ci['TrainTime(s)'] / df_ci['TrainTime(s)'].sum()) * 100
        explode = [0.1 if t == max(time_percent) else 0 for t in time_percent]
        ax.pie(
            time_percent,
            labels=df_ci.index,
            autopct='%1.1f%%',
            startangle=90,
            explode=explode,
            shadow=True,
            colors=sns.color_palette("rocket", len(df_ci))
        )
        ax.set_title("训练时间占比")
        plt.tight_layout()
        fig.savefig("figures/TrainingTime_pie.png", dpi=150)
        plt.close()

    # 雷达图展示性能
    if len(df_ci) > 0:
        radar_metrics = ['Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro', 'AUC_ovo_macro', 'MCC']
        has_ci = all(f"{metric}_ci_lower" in df_ci.columns for metric in radar_metrics)
        if has_ci:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10), subplot_kw=dict(polar=True))
            ax1, ax2 = axes

            # 归一化处理
            scaler = MinMaxScaler()
            norm_point = pd.DataFrame(
                scaler.fit_transform(df_ci[radar_metrics]),
                columns=radar_metrics,
                index=df_ci.index
            )

            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
            angles += angles[:1]

            # 点估计雷达图
            for m in norm_point.index:
                vals = norm_point.loc[m].tolist()
                vals += vals[:1]
                ax1.plot(angles, vals, label=m, linewidth=2, marker='o', markersize=6)
                ax1.fill(angles, vals, alpha=0.25)
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(radar_metrics, fontsize=12)
            ax1.set_title("模型性能雷达图 (点估计)", fontsize=16, y=1.1)
            ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
            ax1.set_rlabel_position(30)
            ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax1.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
            ax1.set_ylim(0, 1)

            # 置信区间雷达图
            for m in norm_point.index:
                lower_vals = [df_ci.loc[m, f"{metric}_ci_lower"] for metric in radar_metrics]
                upper_vals = [df_ci.loc[m, f"{metric}_ci_upper"] for metric in radar_metrics]
                # 归一化置信区间
                lower_vals = scaler.transform(np.array(lower_vals).reshape(1, -1)).flatten()
                upper_vals = scaler.transform(np.array(upper_vals).reshape(1, -1)).flatten()
                lower_vals = np.append(lower_vals, lower_vals[0])
                upper_vals = np.append(upper_vals, upper_vals[0])
                ax2.fill_between(angles, lower_vals, upper_vals, alpha=0.3, label=m)
                point_vals = norm_point.loc[m].tolist()
                point_vals += point_vals[:1]
                ax2.plot(angles, point_vals, linewidth=1.5)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(radar_metrics, fontsize=12)
            ax2.set_title("模型性能置信区间雷达图", fontsize=16, y=1.1)
            ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
            ax2.set_rlabel_position(30)
            ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax2.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
            ax2.set_ylim(0, 1)
            plt.tight_layout()
            fig.savefig("figures/performance_radar_CI.png", dpi=150, bbox_inches='tight')
            plt.close()

    # 模型综合得分图
    if len(df_ci) > 0:
        score_metrics = ['Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro', 'AUC_ovo_macro', 'MCC']
        composite_scores = df_ci[score_metrics].mean(axis=1)
        sorted_indices = composite_scores.sort_values(ascending=False).index
        sorted_scores = composite_scores.loc[sorted_indices]

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(
            x=range(len(sorted_scores)),
            height=sorted_scores,
            color=sns.color_palette("viridis", len(sorted_scores))
        )
        ax.set_title("模型综合性能得分", fontsize=16)
        ax.set_ylabel("综合得分", fontsize=14)
        ax.set_xticks(range(len(sorted_scores)))
        ax.set_xticklabels(sorted_scores.index, rotation=45, ha='right', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        fig.savefig("figures/model_composite_score.png", dpi=150)
        plt.close()

    print("全部模型训练、评估 及SHAP解释性分析完成！")
    print(f"混淆矩阵已保存至: {CM_DIR}")


# ========================
# 程序入口
# ========================
if __name__ == "__main__":
    main()