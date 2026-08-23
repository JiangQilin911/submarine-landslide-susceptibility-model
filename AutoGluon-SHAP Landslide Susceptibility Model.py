#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
滑坡易发性评价 - AutoGluon 集成与 SHAP 解释
========================================================

"""

import os
import sys
import re
import random
import shutil
import time
import argparse
import logging
import warnings
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, precision_score, recall_score
# AutoGluon
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    raise ImportError("请安装 AutoGluon: pip install autogluon")

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ==================== 全局配置 ====================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# AutoGluon 训练配置
AUTOGLUON_CONFIG = {
    "time_limit": 1800,
    "presets": "best_quality",
    "auto_stack": True,
    "num_stack_levels": 2,
    "fit_weighted_ensemble": True,
    "fit_full_last_level_weighted_ensemble": True,
    "excluded_model_types": ['NN_TORCH'],
    "verbosity": 2,
    "hyperparameters": {
        'GBM': {'num_boost_round': 150, 'num_leaves': 31, 'max_depth': 6,
                'learning_rate': 0.05, 'min_child_samples': 20, 'subsample': 0.8,
                'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1},
        'XGB': {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.05,
                'reg_alpha': 0.1, 'reg_lambda': 0.5, 'min_child_weight': 3,
                'subsample': 0.8, 'colsample_bytree': 0.8},
        'CAT': {'iterations': 150, 'depth': 6, 'learning_rate': 0.05,
                'l2_leaf_reg': 3, 'random_strength': 1, 'bagging_temperature': 1},
        'RF': {'n_estimators': 150, 'max_depth': 7, 'min_samples_leaf': 8, 'max_features': 0.6},
        'XT': {'n_estimators': 150, 'max_depth': 7, 'min_samples_leaf': 8, 'max_features': 0.6},
        'KNN': {'n_neighbors': 10, 'weights': 'distance', 'p': 2},
    }
}

# SHAP 配置
SHAP_CONFIG = {
    "background_samples": 100,
    "nsamples": 50,
    "max_samples": 200,
}

# 特征名称映射（中文->英文）
FEATURE_NAMES_MAP = {
    '坡度': 'Slope gradient', '水深': 'Water depth', '坡向': 'Aspect',
    '曲率': 'Curvature', '地形起伏度': 'Topographic relief',
    '地面粗糙度': 'Terrain roughness', '距水道距离': 'Distance to channels',
    '距断层距离': 'Distance to faults', '地震影响强度': 'Seismic‑influence intensity',
    '相干性': 'Seismic coherence', '浅层气风险': 'Shallow‑gas risk',
    '天然含水量': 'Natural water content', '塑性指数': 'Plasticity index',
    '湿容重': 'Wet bulk unit weight', '细粒含量': 'Fine‑grain content'
}

# ==================== 辅助函数 ====================
def ensure_2d_shap(shap_values, n_features: int) -> Optional[np.ndarray]:
    if shap_values is None:
        return None
    if isinstance(shap_values, list):
        if len(shap_values) == 2 and isinstance(shap_values[0], np.ndarray):
            shap_values = shap_values[1]
        else:
            try:
                shap_values = np.array(shap_values)
            except Exception:
                return None
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            if shap_values.shape[2] == 2:
                shap_values = shap_values[:, :, 1]
            else:
                shap_values = shap_values[:, :, 0]
        elif shap_values.ndim == 3 and shap_values.shape[2] == 1:
            shap_values = shap_values[:, :, 0]
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        if shap_values.shape[1] != n_features:
            if shap_values.shape[1] > n_features:
                shap_values = shap_values[:, :n_features]
            else:
                pad = np.zeros((shap_values.shape[0], n_features - shap_values.shape[1]))
                shap_values = np.hstack([shap_values, pad])
        return shap_values
    return None

# ==================== 两步特征选择器 ====================
class TwoStepFeatureSelector:
    def __init__(self, correlation_threshold: float = 0.8, iv_threshold: float = 0.02):
        self.correlation_threshold = correlation_threshold
        self.iv_threshold = iv_threshold
        self.selected_features = None
        self.feature_names = None
        self.removed_by_iv = []
        self.removed_by_corr = []
        self.iv_scores = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        if feature_names is None:
            feature_names = [f'feat_{i}' for i in range(X.shape[1])]
        self.feature_names = feature_names

        logger.info("=" * 70)
        logger.info(f"步骤1: 移除 IV < {self.iv_threshold}")
        logger.info(f"步骤2: 移除 |相关| >= {self.correlation_threshold} (四舍五入)")
        logger.info("=" * 70)

        # 步骤1：IV 过滤
        iv_scores = self._calculate_iv(X, y, feature_names)
        self.iv_scores = dict(zip(feature_names, iv_scores))
        iv_remove = [i for i, iv in enumerate(iv_scores) if iv < self.iv_threshold]
        self.removed_by_iv = [feature_names[i] for i in iv_remove]
        keep_idx = [i for i, iv in enumerate(iv_scores) if iv >= self.iv_threshold]
        logger.info(f"IV < {self.iv_threshold} 移除 {len(self.removed_by_iv)} 个特征")
        current_features = [feature_names[i] for i in keep_idx]
        X_current = X[:, keep_idx]

        # 步骤2：相关性过滤
        logger.info(f"步骤2: 相关性过滤 (阈值 >= {self.correlation_threshold})")
        if len(current_features) > 1:
            corr = np.corrcoef(X_current.T)
            to_remove = set()
            for i in range(len(current_features)):
                for j in range(i+1, len(current_features)):
                    r = abs(corr[i, j])
                    if round(r, 2) >= self.correlation_threshold:
                        # 移除 IV 较低的特征
                        iv_i = iv_scores[keep_idx[i]]
                        iv_j = iv_scores[keep_idx[j]]
                        if iv_i < iv_j:
                            to_remove.add(current_features[i])
                        else:
                            to_remove.add(current_features[j])
            self.removed_by_corr = list(to_remove)
            current_features = [f for f in current_features if f not in to_remove]
            logger.info(f"移除 {len(self.removed_by_corr)} 个高相关特征")
        else:
            self.removed_by_corr = []

        self.selected_features = current_features
        logger.info(f"最终保留特征数: {len(self.selected_features)}")
        logger.info(f"特征列表: {self.selected_features}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_features is None:
            raise ValueError("选择器尚未拟合")
        idx = [self.feature_names.index(f) for f in self.selected_features]
        return X[:, idx]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> np.ndarray:
        self.fit(X, y, feature_names)
        return self.transform(X)

    def _calculate_iv(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], n_bins: int = 10) -> np.ndarray:
        ivs = []
        total_bad = np.sum(y == 1)
        total_good = len(y) - total_bad
        for i in range(X.shape[1]):
            col = X[:, i]
            bins = np.percentile(col, np.linspace(0, 100, n_bins + 1))
            bins[0] = -np.inf
            bins[-1] = np.inf
            digitized = np.digitize(col, bins[1:-1])
            iv = 0
            for b in range(n_bins):
                mask = (digitized == b)
                if not np.any(mask):
                    continue
                bad = np.sum(y[mask] == 1)
                good = np.sum(y[mask] == 0)
                if bad == 0 or good == 0:
                    continue
                bad_pct = bad / total_bad
                good_pct = good / total_good
                woe = np.log(bad_pct / good_pct)
                iv += (bad_pct - good_pct) * woe
            ivs.append(iv)
        return np.array(ivs)

# ==================== AutoGluon 训练 ====================
def train_autogluon(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    X_total: np.ndarray, feature_names: List[str],
                    save_dir: str) -> Tuple[TabularPredictor, np.ndarray, Dict[str, float]]:

    logger.info("=" * 70)
    logger.info("AutoGluon 堆叠集成训练")
    logger.info("=" * 70)

    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test

    model_path = os.path.join(save_dir, 'autogluon_model')
    os.makedirs(model_path, exist_ok=True)

    predictor = TabularPredictor(
        label='target',
        path=model_path,
        problem_type='binary',
        eval_metric='roc_auc',
        verbosity=AUTOGLUON_CONFIG.get("verbosity", 2)
    )

    logger.info("开始训练 AutoGluon（时间限制: {} 秒）...".format(AUTOGLUON_CONFIG["time_limit"]))
    predictor.fit(
        train_df,
        time_limit=AUTOGLUON_CONFIG["time_limit"],
        presets=AUTOGLUON_CONFIG["presets"],
        auto_stack=AUTOGLUON_CONFIG["auto_stack"],
        num_stack_levels=AUTOGLUON_CONFIG["num_stack_levels"],
        fit_weighted_ensemble=AUTOGLUON_CONFIG["fit_weighted_ensemble"],
        fit_full_last_level_weighted_ensemble=AUTOGLUON_CONFIG["fit_full_last_level_weighted_ensemble"],
        excluded_model_types=AUTOGLUON_CONFIG["excluded_model_types"],
        hyperparameters=AUTOGLUON_CONFIG["hyperparameters"],
    )

    # 预测概率
    total_df = pd.DataFrame(X_total, columns=feature_names)
    total_proba = predictor.predict_proba(total_df)
    if isinstance(total_proba, pd.DataFrame):
        total_proba = total_proba.iloc[:, 1].values
    elif isinstance(total_proba, np.ndarray) and total_proba.ndim > 1:
        total_proba = total_proba[:, 1]

    test_proba = predictor.predict_proba(test_df)
    if isinstance(test_proba, pd.DataFrame):
        test_proba = test_proba.iloc[:, 1].values
    elif isinstance(test_proba, np.ndarray) and test_proba.ndim > 1:
        test_proba = test_proba[:, 1]

    # 计算测试集 AUC
    auc = roc_auc_score(y_test, test_proba)
    logger.info(f"AutoGluon 测试集 AUC: {auc:.4f}")

    # 获取最优阈值（基于测试集）
    fpr, tpr, thresholds = roc_curve(y_test, test_proba)
    opt_idx = np.argmax(tpr - fpr)
    opt_threshold = thresholds[opt_idx] if len(thresholds) > 0 else 0.5
    logger.info(f"最优阈值: {opt_threshold:.4f}")

    test_pred = (test_proba >= opt_threshold).astype(int)
    scores = {
        'AUC': auc,
        'Accuracy': accuracy_score(y_test, test_pred),
        'F1': f1_score(y_test, test_pred),
        'Precision': precision_score(y_test, test_pred),
        'Recall': recall_score(y_test, test_pred)
    }

    logger.info(f"测试集性能: {scores}")
    return predictor, total_proba, scores

# ==================== SHAP 分析（KernelExplainer + 交互近似） ====================
def analyze_shap(predictor: TabularPredictor, X_train: np.ndarray,
                 feature_names: List[str], save_dir: str) -> Dict:
    logger.info("=" * 70)
    logger.info("SHAP 分析（KernelExplainer）")
    logger.info("=" * 70)

    # 采样用于 SHAP 计算的样本
    n_samples = min(len(X_train), SHAP_CONFIG["max_samples"])
    idx = np.random.choice(len(X_train), n_samples, replace=False)
    X_sample = X_train[idx]

    # 准备背景数据
    background = shap.sample(pd.DataFrame(X_train, columns=feature_names),
                             SHAP_CONFIG["background_samples"], random_state=RANDOM_STATE)

    # 定义预测函数
    def predict_fn(x):
        if isinstance(x, pd.DataFrame):
            proba = predictor.predict_proba(x)
        else:
            proba = predictor.predict_proba(pd.DataFrame(x, columns=feature_names))
        if isinstance(proba, pd.DataFrame):
            return proba.iloc[:, 1].values
        elif isinstance(proba, np.ndarray) and proba.ndim > 1:
            return proba[:, 1]
        return np.array(proba, dtype=float)

    # 创建 KernelExplainer
    explainer = shap.KernelExplainer(predict_fn, background)

    # 计算 SHAP 值
    logger.info(f"计算 SHAP 值（采样数: {SHAP_CONFIG['nsamples']}）...")
    shap_vals = explainer.shap_values(pd.DataFrame(X_sample, columns=feature_names),
                                      nsamples=SHAP_CONFIG["nsamples"])
    shap_vals = ensure_2d_shap(shap_vals, len(feature_names))
    if shap_vals is None:
        raise RuntimeError("SHAP 值计算失败")

    # 特征重要性（平均绝对值）
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_abs_SHAP': mean_abs_shap
    }).sort_values('Mean_abs_SHAP', ascending=False)
    logger.info("特征重要性（按 Mean_abs_SHAP 排序）:")
    logger.info(importance_df.to_string(index=False))

    # 保存重要性到 CSV
    importance_path = os.path.join(save_dir, 'SHAP_Feature_Importance.csv')
    importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
    logger.info(f"特征重要性已保存至: {importance_path}")

    # 近似交互效应（通过特征置换，仅对 Top-5 重要特征计算）
    n_features = len(feature_names)
    top_k = min(5, n_features)
    top_indices = importance_df.head(top_k)['Feature'].index.tolist()
    # 将特征名转换为列索引
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    top_indices = [feat_to_idx[f] for f in top_indices]

    logger.info(f"计算近似交互矩阵（Top-{top_k} 特征）...")
    n_samples_shap = shap_vals.shape[0]
    interaction_matrix = np.zeros((n_samples_shap, n_features, n_features))

    # 对特征对进行置换
    for i in range(len(top_indices)):
        for j in range(i+1, len(top_indices)):
            idx_i, idx_j = top_indices[i], top_indices[j]
            # 置换两个特征
            X_perm = X_sample.copy()
            perm_idx = np.random.permutation(n_samples_shap)
            X_perm[:, idx_i] = X_sample[perm_idx, idx_i]
            X_perm[:, idx_j] = X_sample[perm_idx, idx_j]
            # 计算置换后的 SHAP 值
            shap_perm = explainer.shap_values(pd.DataFrame(X_perm, columns=feature_names),
                                              nsamples=min(30, SHAP_CONFIG["nsamples"]))
            shap_perm = ensure_2d_shap(shap_perm, n_features)
            if shap_perm is not None:
                # 交互效应近似为置换前后该特征 SHAP 的差异
                interaction = shap_perm[:, idx_i] - shap_vals[:, idx_i]
                interaction_matrix[:, idx_i, idx_j] = interaction
                interaction_matrix[:, idx_j, idx_i] = interaction

    # 计算交互强度（平均绝对交互值）
    inter_intensity = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                inter_intensity[i, j] = np.abs(interaction_matrix[:, i, j]).mean()
            else:
                inter_intensity[i, j] = 0.0

    # 保存交互矩阵（仅保存特征对强度）
    pair_list = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            pair_list.append({
                'Feature1': feature_names[i],
                'Feature2': feature_names[j],
                'Interaction_Intensity': inter_intensity[i, j]
            })
    inter_df = pd.DataFrame(pair_list).sort_values('Interaction_Intensity', ascending=False)
    inter_path = os.path.join(save_dir, 'SHAP_Interaction_Intensities.csv')
    inter_df.to_csv(inter_path, index=False, encoding='utf-8-sig')
    logger.info(f"交互强度已保存至: {inter_path}")

    # 返回结果
    result = {
        'shap_values': shap_vals,
        'mean_abs_shap': mean_abs_shap,
        'importance_df': importance_df,
        'interaction_matrix': interaction_matrix,
        'interaction_intensity': inter_intensity,
        'interaction_df': inter_df
    }
    return result

# ==================== 结果保存 ====================
def save_predictions(predictions: np.ndarray, first_three_cols: pd.DataFrame,
                     best_name: str, save_dir: str, output_csv: str) -> None:
    logger.info("保存易发性预测结果...")
    pred_clipped = np.clip(predictions, 0, 1)
    df_result = pd.DataFrame({f'{best_name}_Probability': pred_clipped})
    output = pd.concat([first_three_cols.reset_index(drop=True), df_result], axis=1)
    output.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logger.info(f"结果已保存至: {output_csv}")
    logger.info(f"预测值范围: [{pred_clipped.min():.4f}, {pred_clipped.max():.4f}]")

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description='AutoGluon 易发性评价 + SHAP 分析')
    parser.add_argument('--input', required=True, help='输入 CSV/Excel 文件路径')
    parser.add_argument('--output', required=True, help='输出目录或 CSV 文件路径')
    parser.add_argument('--target', default='label', help='目标列名')
    parser.add_argument('--iv_threshold', type=float, default=0.02, help='IV 阈值')
    parser.add_argument('--corr_threshold', type=float, default=0.8, help='相关性阈值')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    global RANDOM_STATE
    RANDOM_STATE = args.seed
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    input_path = args.input
    output_path = args.output
    target_col = args.target

    if os.path.isdir(output_path):
        output_csv = os.path.join(output_path, 'Susceptibility_Results.csv')
        save_dir = output_path
    else:
        output_csv = output_path
        save_dir = os.path.dirname(output_path)
    os.makedirs(save_dir, exist_ok=True)

    # 中文特征列表
    original_feature_names_cn = ['水深', '坡度', '坡向', '曲率', '地形起伏度', '地面粗糙度',
                                 '距水道距离', '距断层距离', '地震影响强度', '相干性',
                                 '浅层气风险', '天然含水量', '塑性指数', '湿容重', '细粒含量']

    logger.info("=" * 70)
    logger.info("AutoGluon 易发性评价 + SHAP 分析")
    logger.info("=" * 70)

    # ---------- 数据加载 ----------
    logger.info("加载数据...")
    try:
        if input_path.endswith('.xls'):
            df = pd.read_excel(input_path, engine='xlrd')
        elif input_path.endswith('.xlsx'):
            df = pd.read_excel(input_path, engine='openpyxl')
        else:
            df = pd.read_csv(input_path, encoding='gbk')
    except Exception as e:
        logger.error(f"文件读取错误: {e}")
        try:
            df = pd.read_excel(input_path)
        except:
            df = pd.read_csv(input_path, encoding='utf-8')

    first_three_cols = df.iloc[:, :3].copy().reset_index(drop=True)
    available_cn = [f for f in original_feature_names_cn if f in df.columns]
    available_en = [FEATURE_NAMES_MAP.get(f, f) for f in available_cn]
    missing = set(original_feature_names_cn) - set(available_cn)
    if missing:
        logger.warning(f"缺失特征: {missing}")

    X_raw = df[available_cn].values
    y_raw = df[target_col].values
    logger.info(f"样本数: {X_raw.shape[0]}, 特征数: {X_raw.shape[1]}")
    logger.info(f"标签分布: {dict(zip(*np.unique(y_raw, return_counts=True)))}")

    # ---------- 数据分割 ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.3, random_state=RANDOM_STATE, stratify=y_raw
    )
    X_total = X_raw.copy()

    # ---------- 特征选择 ----------
    logger.info("=" * 70)
    logger.info("特征选择")
    logger.info("=" * 70)
    selector = TwoStepFeatureSelector(
        correlation_threshold=args.corr_threshold,
        iv_threshold=args.iv_threshold
    )
    X_train_sel = selector.fit_transform(X_train, y_train, available_en)
    X_test_sel = selector.transform(X_test)
    X_total_sel = selector.transform(X_total)
    feature_names = selector.selected_features
    logger.info(f"最终特征: {feature_names}")

    # ---------- 训练 AutoGluon ----------
    predictor, total_proba, test_scores = train_autogluon(
        X_train_sel, y_train, X_test_sel, y_test, X_total_sel, feature_names, save_dir
    )

    # ---------- SHAP 分析 ----------
    shap_results = analyze_shap(predictor, X_train_sel, feature_names, save_dir)

    # ---------- 保存预测结果 ----------
    save_predictions(total_proba, first_three_cols, 'AutoGluon', save_dir, output_csv)

    # 输出摘要
    logger.info("=" * 70)
    logger.info("分析完成！")
    logger.info(f"测试集 AUC: {test_scores['AUC']:.4f}")
    logger.info(f"Top-5 重要特征: {shap_results['importance_df'].head(5)['Feature'].tolist()}")
    logger.info(f"所有结果已保存至: {save_dir}")
    logger.info("=" * 70)

if __name__ == '__main__':
    main()