"""
中文图表生成模块 (Chinese Chart Generation for PrNet Experiments)
================================================================
生成所有中文版本的实验图表，包括伪距数据、定位数据、
消融实验和计算开销对比。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC',
                                    'SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# 方法颜色映射
METHOD_COLORS = {
    'WLS': '#1f77b4',
    'WLS+EKF': '#ff7f0e',
    'WLS+Corr': '#2ca02c',
    'WLS+EKF+RTS': '#d62728',
    'WLS+MHE': '#9467bd',
    'WLS+PrNet': '#e377c2',
    'WLS+PrNet+RTS': '#7f7f7f',
    'WLS+EKF+PrNet': '#bcbd22',
    'WLS+MHE+PrNet': '#17becf',
    '真实轨迹': '#000000',
    'Ground Truth': '#000000',
}

# 方法中文名称映射
METHOD_CN = {
    'WLS': 'WLS（加权最小二乘）',
    'WLS+EKF': 'WLS+EKF（扩展卡尔曼滤波）',
    'WLS+EKF+RTS': 'WLS+EKF+RTS（RTS平滑）',
    'WLS+MHE': 'WLS+MHE（移动窗口估计）',
    'WLS+PrNet': 'WLS+PrNet（神经网络校正）',
    'WLS+EKF+PrNet': 'WLS+EKF+PrNet',
    'WLS+PrNet+RTS': 'WLS+PrNet+RTS',
    'WLS+MHE+PrNet': 'WLS+MHE+PrNet',
    'Ground Truth': '真实轨迹',
    'Full PrNet': '完整PrNet',
}

SCENARIO_CN = {
    'RouteR': '郊区路线（RouteR）',
    'RouteU': '城区路线（RouteU）',
}


def _get_cn_method(method):
    """获取方法的中文名称"""
    return METHOD_CN.get(method, method)


def _get_cn_scenario(scenario):
    """获取场景的中文名称"""
    for key, val in SCENARIO_CN.items():
        if key in scenario:
            return val
    return scenario


def _setup_plot_style():
    """设置统一的绘图风格"""
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


# ============================================================
# 1. 各卫星原始伪距时序图
# ============================================================

def plot_raw_pseudorange_timeseries_cn(epoch_data_dict, sorted_epochs, scenario_name, output_dir):
    """绘制各卫星原始伪距随时间变化的时序图。"""
    _setup_plot_style()

    prn_data = {}
    for ep in sorted_epochs:
        ed = epoch_data_dict[ep]
        for i, prn in enumerate(ed['prns']):
            if prn not in prn_data:
                prn_data[prn] = {'epochs': [], 'pr': []}
            prn_data[prn]['epochs'].append(ep)
            prn_data[prn]['pr'].append(ed['raw_pr'][i])

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.tab20
    prn_list = sorted(prn_data.keys())

    for idx, prn in enumerate(prn_list):
        color = cmap(idx / max(len(prn_list), 1))
        ax.plot(prn_data[prn]['epochs'], prn_data[prn]['pr'],
                '.', markersize=2, color=color, alpha=0.7, label=f'PRN {prn}')

    ax.set_xlabel('历元编号')
    ax.set_ylabel('原始伪距（米）')
    ax.set_title(f'{_get_cn_scenario(scenario_name)} — 各卫星原始伪距时序图')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, f'{scenario_name}_raw_pseudorange_timeseries_cn.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 2. 伪距误差分布直方图（多方法对比）
# ============================================================

def plot_pr_error_distribution_comparison_cn(method_pr_errors, scenario_name, output_dir):
    """绘制多种方法的伪距误差分布直方图。"""
    _setup_plot_style()

    n_methods = len(method_pr_errors)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for idx, (method, errors) in enumerate(method_pr_errors.items()):
        ax = axes[idx]
        if len(errors) == 0:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(_get_cn_method(method))
            continue

        color = METHOD_COLORS.get(method, '#333333')
        ax.hist(errors, bins=60, color=color, edgecolor='white', alpha=0.7)
        ax.axvline(np.mean(errors), color='red', ls='--', lw=1.5,
                   label=f'均值: {np.mean(errors):.2f} 米')
        ax.axvline(np.median(errors), color='green', ls='--', lw=1.5,
                   label=f'中位数: {np.median(errors):.2f} 米')
        ax.set_xlabel('伪距误差（米）')
        ax.set_ylabel('频数')
        ax.set_title(_get_cn_method(method))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{_get_cn_scenario(scenario_name)} — 伪距误差分布', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_pr_error_distribution_cn.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 3. 伪距误差CDF曲线
# ============================================================

def plot_pr_error_cdf_cn(method_pr_errors, scenario_name, output_dir):
    """绘制多种方法的伪距误差累积分布函数（CDF）曲线。"""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, errors in method_pr_errors.items():
        if len(errors) == 0:
            continue
        sorted_err = np.sort(np.abs(errors))
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        color = METHOD_COLORS.get(method, '#333333')
        ax.plot(sorted_err, cdf, label=_get_cn_method(method), color=color, linewidth=1.5)

    ax.set_xlabel('|伪距误差|（米）')
    ax.set_ylabel('累积分布函数（CDF）')
    ax.set_title(f'{_get_cn_scenario(scenario_name)} — 伪距误差累积分布函数')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_pr_error_cdf_cn.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 4. 伪距误差箱线图
# ============================================================

def plot_pr_error_boxplot_cn(method_pr_errors, scenario_name, output_dir):
    """绘制多种方法的伪距误差箱线图。"""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    data = []
    for method, errors in method_pr_errors.items():
        if len(errors) > 0:
            labels.append(_get_cn_method(method))
            data.append(errors)

    if not data:
        plt.close()
        return None

    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    colors = [METHOD_COLORS.get(l, '#cccccc') for l in method_pr_errors.keys()]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('伪距误差（米）')
    ax.set_title(f'{_get_cn_scenario(scenario_name)} — 伪距误差对比箱线图')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_pr_error_boxplot_cn.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 5. 轨迹对比图
# ============================================================

def plot_trajectory_comparison_cn(method_positions, gt_positions, scenario_name, output_dir):
    """绘制多种方法的定位轨迹与真实轨迹对比图。"""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 9))

    if gt_positions is not None and len(gt_positions) > 0:
        gt = np.array(gt_positions)
        ax.plot(gt[:, 1], gt[:, 0], 'k-', linewidth=2, label='真实轨迹', zorder=10)

    for method, positions in method_positions.items():
        if len(positions) == 0:
            continue
        pos = np.array(positions)
        color = METHOD_COLORS.get(method, '#333333')
        ax.plot(pos[:, 1], pos[:, 0], '.', markersize=3, color=color,
                alpha=0.5, label=_get_cn_method(method))

    ax.set_xlabel('经度（°）')
    ax.set_ylabel('纬度（°）')
    ax.set_title(f'{_get_cn_scenario(scenario_name)} — 定位轨迹对比')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_trajectory_comparison_cn.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 6. 定位误差时序图
# ============================================================

def plot_position_error_timeseries_cn(method_errors, scenario_name, output_dir):
    """绘制多种方法的水平定位误差随时间变化图。"""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    for method, (epochs, errors) in method_errors.items():
        if len(errors) == 0:
            continue
        color = METHOD_COLORS.get(method, '#333333')
        ax.plot(epochs, errors, '-', linewidth=0.8, color=color,
                alpha=0.7, label=_get_cn_method(method))

    ax.set_xlabel('历元编号')
    ax.set_ylabel('水平定位误差（米）')
    ax.set_title(f'{_get_cn_scenario(scenario_name)} — 定位误差时序图')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_position_error_timeseries_cn.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 7. 定位误差CDF
# ============================================================

def plot_position_error_cdf_cn(method_errors, scenario_name, output_dir):
    """绘制多种方法的定位误差累积分布函数（CDF）。"""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, (epochs, errors) in method_errors.items():
        if len(errors) == 0:
            continue
        sorted_err = np.sort(errors)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        color = METHOD_COLORS.get(method, '#333333')
        ax.plot(sorted_err, cdf, label=_get_cn_method(method), color=color, linewidth=1.5)

    ax.set_xlabel('水平定位误差（米）')
    ax.set_ylabel('累积分布函数（CDF）')
    ax.set_title(f'{_get_cn_scenario(scenario_name)} — 定位误差累积分布函数')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_position_error_cdf_cn.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 8. 定位误差统计表（图表形式）
# ============================================================

def plot_position_error_table_cn(method_errors, scenario_name, output_dir):
    """生成定位误差统计表图片。"""
    _setup_plot_style()

    headers = ['方法', '均值（米）', '中位数（米）', '50分位数', '67分位数', '95分位数', '标准差（米）']
    rows = []

    for method, (epochs, errors) in method_errors.items():
        if len(errors) == 0:
            rows.append([_get_cn_method(method), 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
            continue
        rows.append([
            _get_cn_method(method),
            f'{np.mean(errors):.2f}',
            f'{np.median(errors):.2f}',
            f'{np.percentile(errors, 50):.2f}',
            f'{np.percentile(errors, 67):.2f}',
            f'{np.percentile(errors, 95):.2f}',
            f'{np.std(errors):.2f}',
        ])

    fig, ax = plt.subplots(figsize=(14, max(2, 0.5 * len(rows) + 1.5)))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')

    ax.set_title(f'{_get_cn_scenario(scenario_name)} — 定位误差统计表', fontsize=13, pad=20)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_position_error_table_cn.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 9. 消融实验柱状图
# ============================================================

def plot_ablation_results_cn(ablation_results, baseline_error, scenario_name, output_dir):
    """绘制消融实验结果柱状图。"""
    _setup_plot_style()

    ABLATION_CN = {
        'Without CN0': '移除CN0',
        'Without sinE/cosE': '移除仰角',
        'Without PRN': '移除PRN',
        'Without WLS Position': '移除WLS位置',
        'Without Geometry Vector': '移除几何矢量',
        'Without Heading': '移除航向角',
    }

    names = ['完整PrNet'] + [ABLATION_CN.get(r['name'], r['name']) for r in ablation_results.values()]
    losses = [baseline_error] + [r['avg_loss'] for r in ablation_results.values()]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ca02c'] + ['#ff7f0e'] * len(ablation_results)
    bars = ax.bar(range(len(names)), losses, color=colors, edgecolor='white', alpha=0.8)

    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('平均MSE损失')
    ax.set_title(f'{_get_cn_scenario(scenario_name)} — 特征消融实验')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_ablation_bar_cn.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 10. 消融实验统计表
# ============================================================

def plot_ablation_table_cn(ablation_results, baseline_stats, scenario_name, output_dir):
    """生成消融实验统计表图片。"""
    _setup_plot_style()

    ABLATION_CN = {
        'Without CN0': '移除CN0',
        'Without sinE/cosE': '移除仰角',
        'Without PRN': '移除PRN',
        'Without WLS Position': '移除WLS位置',
        'Without Geometry Vector': '移除几何矢量',
        'Without Heading': '移除航向角',
    }

    headers = ['配置', '输入维度', '参数量', '平均损失', '训练时间（秒）', '推理时间（毫秒）']
    rows = [['完整PrNet（基线）', '16', str(baseline_stats.get('num_params', 'N/A')),
             f"{baseline_stats.get('avg_loss', 0):.6f}",
             f"{baseline_stats.get('train_time', 0):.1f}",
             f"{baseline_stats.get('avg_inference_time', 0) * 1000:.2f}"]]

    for abl_type, res in ablation_results.items():
        rows.append([
            ABLATION_CN.get(res['name'], res['name']),
            str(res['input_size']),
            str(res['num_params']),
            f"{res['avg_loss']:.6f}",
            f"{res['train_time']:.1f}",
            f"{res['avg_inference_time'] * 1000:.2f}",
        ])

    fig, ax = plt.subplots(figsize=(14, max(2, 0.5 * len(rows) + 1.5)))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')

    ax.set_title(f'{_get_cn_scenario(scenario_name)} — 特征消融实验统计', fontsize=13, pad=20)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{scenario_name}_ablation_table_cn.png')
    plt.savefig(path)
    plt.close()
    return path


# ============================================================
# 11. 计算开销对比柱状图
# ============================================================

def plot_computational_overhead_cn(model_stats, output_dir):
    """绘制不同模型计算开销对比柱状图。"""
    _setup_plot_style()

    models = list(model_stats.keys())
    params = [model_stats[m].get('params', 0) for m in models]
    inference_times = [model_stats[m].get('inference_time', 0) * 1000 for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000'][:len(models)]

    # 参数量
    bars1 = ax1.bar(models, params, color=colors, edgecolor='white', alpha=0.8)
    for bar, val in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:,}', ha='center', va='bottom', fontsize=9)
    ax1.set_ylabel('参数数量')
    ax1.set_title('模型复杂度')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha='right')

    # 推理时间
    bars2 = ax2.bar(models, inference_times, color=colors, edgecolor='white', alpha=0.8)
    for bar, val in zip(bars2, inference_times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('推理时间（毫秒/样本）')
    ax2.set_title('推理速度')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20, ha='right')

    fig.suptitle('计算开销对比', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'computational_overhead_comparison_cn.png')
    plt.savefig(path)
    plt.close()
    return path


def plot_computational_overhead_table_cn(model_stats, output_dir):
    """生成计算开销汇总表图片。"""
    _setup_plot_style()

    headers = ['模型', '参数量', '推理时间（毫秒）', '训练时间（秒）', '内存估计（MB）']
    rows = []
    for model, stats in model_stats.items():
        mem_mb = stats.get('params', 0) * 8 / (1024 * 1024)
        rows.append([
            model,
            f"{stats.get('params', 0):,}",
            f"{stats.get('inference_time', 0) * 1000:.3f}",
            f"{stats.get('train_time', 0):.1f}",
            f"{mem_mb:.2f}",
        ])

    fig, ax = plt.subplots(figsize=(12, max(2, 0.5 * len(rows) + 1.5)))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')

    ax.set_title('计算开销汇总', fontsize=13, pad=20)
    plt.tight_layout()
    path = os.path.join(output_dir, 'computational_overhead_table_cn.png')
    plt.savefig(path)
    plt.close()
    return path
