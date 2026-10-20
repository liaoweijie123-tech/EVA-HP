# process_private_data.py
import os
import re
from pathlib import Path
import numpy as np
import scipy.signal
import h5py
from dataclasses import dataclass
from typing import ClassVar
import matplotlib.pyplot as plt
import random # 导入 random 模块用于随机选择
from typing import List, Optional # 导入 List 和 Optional 用于类型提示
from matplotlib.colors import LinearSegmentedColormap
import math

# ======================================================================================
# 1. 集中化配置 (适配您的私有数据集)
# ======================================================================================
CONFIG = {
    # --- 输入路径 (关键修改) ---
    "private_data_root": Path('E:/hand_mesh/Process_EVA-HP_1/split'),  # 已分割好的数据根目录

    # --- 输出路径 (流水线会自动创建这些目录) ---
    "stage1_output_file": Path('E:/hand_mesh/Process_EVA-HP_1/private_S2_Scalers/per_subject_scalers.npz'),
    "stage2_output_dir": Path('E:/hand_mesh/Process_EVA-HP_1/private_S3_Preprocessed/'),


    # # --- 数据集参数 ---
    # "subjects": [1, 2, 3, 4, 5, 6],  # 您的受试者列表
    # "actions": [1, 2, 3, 4, 5, 6, 7, 8], # 您的动作列表

    # --- 数据集参数 ---
    "subjects": [1],  # 您的受试者列表
    "actions": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], # 您的动作列表

    # --- 信号处理参数 (与公开数据集保持一致) ---
    "sampling_rate": 200, # Hz 
    "fps": 35, # Hz (假设与Ninapro一致，如果不同请修改)
    "emg_channels": 8,     # 根据您的私有数据修改
    "glove_channels_to_keep": list(range(24)), # 假设有20个关节，全保留
    "rms_window_ms": 100,
    "rms_increment_ms": 5,
}

CONFIG["glove_final_channels"] = len(CONFIG["glove_channels_to_keep"])

# ======================================================================================
# 2. 辅助类与函数 (从原脚本复用)
# ======================================================================================
###############专利###########################

# def plot_emg_channels_stacked(emg_raw: np.ndarray, 
#                               sampling_rate: int, 
#                               title: str = None,
#                               duration: float = 2.0,
#                               save_path: str = None,
#                               spacing_factor: float = 3.0):  # <--- [核心修改] 调大此数值可增加间距
#     """
#     [专利级] 可视化多通道原始EMG信号 (堆叠视图)，支持自定义间距。

#     Args:
#         emg_raw (np.ndarray): 原始EMG数据 (样本数, 通道数)。
#         sampling_rate (int): 采样率 (Hz)。
#         title (str, optional): 图表标题。
#         duration (float): 仅显示前多少秒的数据。
#         save_path (str, optional): 保存路径 (推荐 .pdf 或 .png)。
#         spacing_factor (float): [关键] 控制通道垂直间距的倍数。
#                                 2.0 = 紧凑 (波形可能重叠)
#                                 4.0 = 标准 (波形分离)
#                                 6.0 = 宽松 (适合展示大幅度动作)
#     """
#     if not isinstance(emg_raw, np.ndarray) or emg_raw.ndim != 2:
#         raise ValueError("emg_raw 必须是一个二维 NumPy 数组。")

#     num_samples, num_channels = emg_raw.shape

#     plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
    
#     # [关键] 解决负号 '-' 显示为方块的问题
#     plt.rcParams['axes.unicode_minus'] = False
#     # --- 1. 设置专业绘图风格 ---
#     plt.rcParams['font.family'] = 'sans-serif'

#     plt.rcParams['font.size'] = 11
#     plt.rcParams['axes.linewidth'] = 1.2 

#     # --- 2. 数据切片 ---
#     points_to_show = int(duration * sampling_rate)
#     if points_to_show > num_samples:
#         points_to_show = num_samples
    
#     data_slice = emg_raw[:points_to_show, :]
#     time_axis = np.arange(points_to_show) / sampling_rate

#     # --- 3. 计算自适应偏移量 ---
#     # 使用 99.5 分位数找到信号的最大幅度 (忽略极个别异常值)
#     global_max_amp = np.percentile(np.abs(data_slice), 99.5)
    
#     # [核心逻辑] 偏移量 = 最大幅度 * 间距系数
#     offset_step = global_max_amp * spacing_factor
    
#     # --- 4. 绘图 ---
#     # 动态计算画布高度：间距越大，图越高
#     fig_height = max(4, 0.6 * num_channels * (spacing_factor / 3.0))
#     fig, ax = plt.subplots(figsize=(8, fig_height), dpi=300)

#     # 隐藏上、右边框
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     y_ticks = []
#     y_tick_labels = []

#     # 倒序遍历：Channel 1 在最上方
#     for i in range(num_channels):
#         ch_idx = i
#         # 计算当前通道的垂直位置
#         display_pos = (num_channels - 1 - i) * offset_step
        
#         signal = data_slice[:, ch_idx]
        
#         # 绘制纯黑线条
#         ax.plot(time_axis, signal + display_pos, color='k', linewidth=0.8)
        
#         y_ticks.append(display_pos)
#         y_tick_labels.append(f"CH {ch_idx + 1}")

#     # --- 5. 坐标轴修饰 ---
#     ax.set_yticks(y_ticks)
#     ax.set_yticklabels(y_tick_labels, fontweight='bold')
#     ax.set_xlabel("时间(s)", fontsize=12, fontweight='bold')
#     ax.set_xlim(0, time_axis[-1])
    
#     # 移除 Y 轴刻度线 (只留文字)
#     ax.tick_params(axis='y', length=0) 

#     if title:
#         ax.set_title(title, fontsize=12, pad=10)

#     # --- 6. 添加垂直比例尺 (Scale Bar) ---
#     # 在右下角画一条竖线，代表幅值大小
#     bar_x = time_axis[-1] * 1.02 # 放在 X 轴最右侧外面
#     bar_bottom = y_ticks[0]      # 最下方通道的基线
#     bar_height = offset_step * 0.5 # 高度设为间距的一半
    
#     ax.plot([bar_x, bar_x], [bar_bottom, bar_bottom + bar_height], 
#             color='k', linewidth=2.0, clip_on=False)
    

#     plt.tight_layout()

#     # --- 7. 保存 ---
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
#         print(f"Figure saved to: {save_path}")
    
#     plt.show()


def plot_joint_angles_subplots(joint_angles: np.ndarray, 
                               sampling_rate: int, 
                               joint_indices_to_display: list = None,
                               title: str = None,
                               save_path: str = None):
    """
    [专利级] 可视化多通道关节角度数据 (网格视图)。
    特点：5行4列紧凑布局，适合展示 20 个关节，纯黑白风格。

    Args:
        joint_angles (np.ndarray): 关节角度数据 (样本数, 关节数)。
        sampling_rate (int): 采样率 (Hz)。
        joint_indices_to_display (list): 要显示的关节索引 (默认为全部)。
        title (str): 图表总标题 (可选)。
        save_path (str): 保存路径。
    """
    # --- 1. 数据准备 ---
    if not isinstance(joint_angles, np.ndarray) or joint_angles.ndim != 2:
        raise ValueError("joint_angles 必须是一个二维 NumPy 数组。")
    
    num_samples, total_joints = joint_angles.shape
    
    if joint_indices_to_display is None:
        # 默认显示前 20 个，如果不足 20 个则显示全部
        limit = min(total_joints, 20)
        joint_indices_to_display = list(range(limit))
    
    num_plots = len(joint_indices_to_display)
    
    # --- 2. 布局计算 (核心) ---
    # 强制使用 4 列布局，行数自动计算
    cols = 4
    rows = math.ceil(num_plots / cols)
    
    # 动态计算图表大小：每个子图约 3x2 英寸
    fig_width = 12 
    fig_height = 2.0 * rows 
    
    # --- 3. 设置绘图风格 ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.0
    
    # 创建画布，使用 constrained_layout 自动调整间距
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=300, constrained_layout=True)
    
    # 展平 axes 数组以便一维遍历
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes] # 只有一个子图的情况

    time_axis = np.arange(num_samples) / sampling_rate
    
    # --- 4. 绘图循环 ---
    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        
        # 如果还有数据要画
        if i < num_plots:
            joint_real_idx = joint_indices_to_display[i]
            data = joint_angles[:, joint_real_idx]
            
            # [核心] 绘制纯黑线条
            ax.plot(time_axis, data, color='k', linewidth=1.2)
            
            # [核心] 在图内部左上角添加标签 (例如 "J1", "J2")
            # transform=ax.transAxes 确保坐标是相对子图的 (0-1)
            ax.text(0.03, 0.92, f"J{joint_real_idx + 1}", 
                    transform=ax.transAxes, 
                    fontsize=12, fontweight='bold', va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
            
            # 简化边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # 添加淡淡的网格辅助读数
            ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
            
            # 坐标轴设置
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            # 智能显示标签：只有最左侧一列显示 Y 轴标签
            if i % cols == 0:
                ax.set_ylabel("角度(°)", fontsize=16)
            
            # 智能显示标签：只有最下面一行显示 X 轴标签
            # 判断是否是最后一行
            if i >= (rows - 1) * cols:
                ax.set_xlabel("时间(s)", fontsize=16)
                
            ax.set_xlim(0, time_axis[-1])
            
        else:
            # 隐藏多余的空子图 (如果有的话)
            ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    # --- 5. 保存 ---
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Joint Angles Figure saved to: {save_path}")
    
    plt.show()




# def plot_joint_angles_subplots(joint_angles: np.ndarray, 
#                                sampling_rate: int, 
#                                joint_indices_to_display: Optional[List[int]] = None,
#                                title: str = "Joint Angles Over Time",
#                                group_size: int = 4):
#     """
#     可视化多通道关节角度数据，垂直排列子图。
#     并根据关节索引进行分组着色（默认每4个关节一组颜色）。

#     Args:
#         joint_angles (np.ndarray): 关节角度数据，形状为 (样本数, 关节数)。
#         sampling_rate (int): 信号的采样率 (Hz)。
#         joint_indices_to_display (Optional[List[int]]): 要显示的关节索引列表。若为None则显示全部。
#         title (str): 图表总标题。
#         group_size (int): 每多少个关节分为一组进行同色显示。默认 4。
#     """
#     # --- 1. 输入校验 ---
#     if not isinstance(joint_angles, np.ndarray) or joint_angles.ndim != 2:
#         raise ValueError("joint_angles 必须是一个二维 NumPy 数组 (样本数, 关节数)。")

#     num_samples, total_joints = joint_angles.shape

#     # --- 2. 确定要显示的关节 ---
#     if joint_indices_to_display is not None:
#         # 过滤无效索引
#         valid_indices = [i for i in joint_indices_to_display if 0 <= i < total_joints]
#         if len(valid_indices) != len(joint_indices_to_display):
#             print(f"警告: 已忽略部分超出范围的索引。")
#         joint_indices_to_plot = sorted(valid_indices)
#     else:
#         joint_indices_to_plot = list(range(total_joints))
    
#     num_joints_to_plot = len(joint_indices_to_plot)
#     if num_joints_to_plot == 0:
#         print("警告: 没有选择任何关节进行显示。")
#         return

#     # --- 3. 准备颜色调色板 ---
#     # 使用 matplotlib 的 tab10 调色板 (10种高对比度颜色)
#     # 如果组数超过10，颜色会循环使用
#     color_palette = plt.get_cmap("tab10").colors 

#     # --- 4. 创建子图布局 ---
#     # 动态调整高度：每个子图高度约为 2 英寸，保证20个图也能看清
#     fig_height = max(3, 3 * num_joints_to_plot)
#     fig, axes = plt.subplots(num_joints_to_plot, 1, 
#                            figsize=(12, fig_height), 
#                            sharex=True,
#                            constrained_layout=False) # 使用 constrained_layout 自动调整间距
    
#     if num_joints_to_plot == 1:
#         axes = [axes]
        
#     fig.suptitle(title, fontsize=16, fontweight='bold')

#     # --- 5. 计算时间轴 ---
#     time_axis = np.arange(num_samples) / sampling_rate
    
#     # --- 6. 绘图循环 ---
#     for plot_idx, joint_real_idx in enumerate(joint_indices_to_plot):
#         ax = axes[plot_idx]
        
#         # === 核心逻辑：计算颜色 ===
#         # 根据“原始关节索引”除以“组大小”来决定使用哪种颜色
#         # 例如：0-3为组0(蓝色), 4-7为组1(橙色), 8-11为组2(绿色)...
#         group_id = joint_real_idx // group_size
#         color_idx = group_id % len(color_palette)
#         current_color = color_palette[color_idx]
        
#         # 绘制信号
#         ax.plot(time_axis, joint_angles[:, joint_real_idx], 
#                 linewidth=1.5, 
#                 color=current_color)
        
#         # # 设置标题 (显示关节索引和所属组)
#         # ax.set_title(f"Joint {joint_real_idx + 1} (Group {group_id + 1})", 
#         #              loc='left', 
#         #              fontsize=10, 
#         #              color=current_color, # 标题颜色也跟随线条颜色，方便识别
#         #              fontweight='bold',
#         #              pad=3)
        
#         # 设置样式
#         ax.set_ylabel("Angle (°)", fontsize=9)
#         ax.grid(True, linestyle=':', alpha=0.6)
        
#         # 优化Y轴刻度显示，防止重叠
#         ax.tick_params(axis='y', labelsize=8)

#     # --- 7. 底部标签 ---
#     axes[-1].set_xlabel("Time (s)", fontsize=12)
    
#     # 如果没有用 constrained_layout，可以使用 tight_layout
#     # plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
#     plt.show()

# def plot_joint_angles_subplots(joint_angles: np.ndarray, 
#                                sampling_rate: int, 
#                                joint_indices_to_display: Optional[List[int]] = None,
#                                title: str = "Joint Angles Over Time"):
#     """
#     可视化多通道关节角度数据，每个关节一个独立的子图，垂直排列。
#     可以指定一个要显示的关节索引列表。

#     Args:
#         joint_angles (np.ndarray): 关节角度数据，形状为 (样本数, 关节数)。
#                                    假设单位为度 (degrees)。
#         sampling_rate (int): 信号的采样率 (Hz)。
#         joint_indices_to_display (Optional[List[int]]): 
#             一个包含要显示的关节列索引的列表 (e.g., [0, 2, 4, 6, 8])。
#             如果为 None (默认)，则显示所有关节。
#         title (str): 图表的总标题。
#     """
#     if not isinstance(joint_angles, np.ndarray) or joint_angles.ndim != 2:
#         raise ValueError("joint_angles 必须是一个二维 NumPy 数组 (样本数, 关节数)。")

#     num_samples, total_joints = joint_angles.shape

#     # --- 核心修改：根据用户输入的列表选择关节 ---
#     if joint_indices_to_display is not None:
#         # 验证输入的索引是否有效
#         invalid_indices = [i for i in joint_indices_to_display if not (0 <= i < total_joints)]
#         if invalid_indices:
#             raise IndexError(f"提供的关节索引 {invalid_indices} 超出范围 (0-{total_joints-1})。")
#         joint_indices_to_plot = sorted(joint_indices_to_display)
#         print(f"将显示指定的 {len(joint_indices_to_plot)} 个关节: {joint_indices_to_plot}")
#     else:
#         # 如果未提供列表，则显示所有关节
#         joint_indices_to_plot = list(range(total_joints))
#         print(f"未指定关节列表，将显示所有 {total_joints} 个关节。")
    
#     num_joints_to_plot = len(joint_indices_to_plot)
#     if num_joints_to_plot == 0:
#         print("警告: 没有选择任何关节进行显示。")
#         return

#     # --- 创建子图布局 ---
#     fig, axes = plt.subplots(num_joints_to_plot, 1, 
#                            figsize=(14, 1.8 * num_joints_to_plot), 
#                            sharex=True)
#     fig.suptitle(title, fontsize=16)

#     # 确保 axes 总是可迭代的
#     if num_joints_to_plot == 1:
#         axes = [axes]

#     # --- 计算时间轴 ---
#     time_axis = np.arange(num_samples) / sampling_rate
    
#     # --- 遍历被选中的关节索引进行绘图 ---
#     for i, joint_idx in enumerate(joint_indices_to_plot):
#         ax = axes[i]
        
#         # 绘制信号
#         ax.plot(time_axis, joint_angles[:, joint_idx], linewidth=1.2)
        
#         # 自动设置子图标题，标明是哪个关节 (索引从0开始，但通常我们称之为第1个关节)
#         ax.set_title(f"Joint {joint_idx + 1}", loc='left', fontsize=10, pad=3)
        
#         # 设置Y轴标签
#         ax.set_ylabel("Angle (°)")
        
#         ax.grid(True, linestyle='--', alpha=0.7)

#     # --- 美化整体图表 ---
#     axes[-1].set_xlabel("Time (s)", fontsize=12)
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.96])
#     plt.show()

###############专利###########################


##############毕业论文#####################



def plot_8ch_colored_time_series_clean(emg_data, sampling_rate, save_path=None):
    """
    [论文专用 - 精致版] 绘制 8 通道彩色肌电信号时序图。
    优化要点：
    1. 更精细的颜色方案（高饱和度但协调）
    2. 通道标签（CH1~CH8），颜色与波形对应
    3. 通道间细虚线分隔，视觉更清晰
    4. 波形振幅按通道能量差异动态缩放
    5. 低振幅通道（CH7/CH8）降低透明度，减少视觉噪声
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    num_samples, num_channels = emg_data.shape
    num_channels = min(num_channels, 8)

    # --- 颜色方案（与 tab10 兼容但更精细）---
    COLORS = [
        '#4C8EDA',  # CH1 蓝
        '#E8874A',  # CH2 橙
        '#52A96E',  # CH3 绿
        '#D44F4F',  # CH4 红
        '#9370C4',  # CH5 紫
        '#8B6854',  # CH6 棕
        '#D480A0',  # CH7 粉
        '#888880',  # CH8 灰
    ]

    # --- 画布 ---
    fig, axes = plt.subplots(
        num_channels, 1,
        figsize=(10, 6),          # 宽高比 5:3，适合论文双栏
        dpi=300,
        gridspec_kw=dict(
            hspace=0.0,           # 子图间无间隙
            top=0.97, bottom=0.03,
            left=0.07, right=0.99
        )
    )

    fig.patch.set_facecolor('white')
    t = np.arange(num_samples)

    for i, ax in enumerate(axes):
        ch_data = emg_data[:, i]

        # 每通道独立归一化（按 99.9 百分位，保留波形形态）
        peak = np.percentile(np.abs(ch_data), 99.9)
        if peak > 0:
            ch_norm = ch_data / peak
        else:
            ch_norm = ch_data

        # 低振幅通道（CH7/CH8）视觉上稍淡
        alpha = 0.72 if i >= 6 else 0.90
        lw    = 0.85 if i >= 6 else 1.1

        ax.plot(t, ch_norm, color=COLORS[i], linewidth=lw,
                alpha=alpha, rasterized=True)

        # --- 通道标签（左侧，与波形同色）---
        ax.set_ylabel(f'CH{i+1}', color=COLORS[i],
                      fontsize=8, fontweight='bold',
                      rotation=0, labelpad=18, va='center')

        # --- 隐藏所有坐标轴元素 ---
        ax.set_axis_off()
        ax.set_xlim(0, num_samples - 1)
        ax.set_ylim(-1.55, 1.55)

        # --- 通道间分隔虚线（除最后一行）---
        if i < num_channels - 1:
            ax.axhline(y=-1.55, color='#cccccc', linewidth=0.5,
                       linestyle=(0, (3, 5)), alpha=0.7)

        # 重新显示 ylabel（axhline 后需要）
        ax.yaxis.set_visible(True)
        ax.yaxis.set_tick_params(length=0)
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.yaxis.label.set_visible(True)

    # --- 外边框（极细灰线，仅画左+右+上+下）---
    fig.add_artist(plt.matplotlib.patches.FancyBboxPatch(
        (0.07, 0.03), 0.92, 0.94,
        boxstyle="square,pad=0",
        linewidth=0.6, edgecolor='#cccccc', facecolor='none',
        transform=fig.transFigure, zorder=10
    ))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02,
                    dpi=300, transparent=False, facecolor='white')
        print(f"Saved → {save_path}")

    return fig




import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy import signal
 
 
# ─────────────────────────────────────────────
#  色板：高对比度，色盲友好
# ─────────────────────────────────────────────
_CHANNEL_COLORS = [
    '#3A6FC4',  # CH1  蓝
    '#E07B2A',  # CH2  橙
    '#3DA060',  # CH3  绿
    '#C03030',  # CH4  红
    '#7B5EA7',  # CH5  紫
    '#7A4F38',  # CH6  棕
    '#C4608A',  # CH7  玫红
    '#7A7A72',  # CH8  灰
]
 
_MEAN_COLOR  = '#111111'
_PEAK_COLOR  = '#C03030'
_NOTCH_COLOR = '#AAAAAA'
 
# 三频段配色（低饱和度，不抢主图）
_BAND_ZH = {
    'low':  {'range': (0,   20),  'color': '#D6EAF8', 'label': '低频段\n0–20 Hz',    'text_color': '#1A5276'},
    'mid':  {'range': (20,  100), 'color': '#D5F5E3', 'label': '中频段\n20–100 Hz',  'text_color': '#1D6A3C'},
    'high': {'range': (100, 120), 'color': '#FDEBD0', 'label': '高频段\n>100 Hz', 'text_color': '#784212'},
}
_BAND_EN = {
    'low':  {'range': (0,   20),  'color': '#D6EAF8', 'label': 'Low\n0–20 Hz',    'text_color': '#1A5276'},
    'mid':  {'range': (20,  100), 'color': '#D5F5E3', 'label': 'Mid\n20–100 Hz',  'text_color': '#1D6A3C'},
    'high': {'range': (100, 120), 'color': '#FDEBD0', 'label': 'High\n100–120 Hz','text_color': '#784212'},
}
 
 
def _setup_fonts(use_chinese: bool) -> bool:
    """自动检测中文字体并设置 rcParams。返回是否成功启用中文。"""
    plt.rcParams['mathtext.fontset']   = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
 
    if not use_chinese:
        plt.rcParams['font.family'] = 'Times New Roman'
        return False
 
    candidates = [
        'SimHei', 'Microsoft YaHei', 'STHeiti', 'PingFang SC',
        'WenQuanYi Zen Hei', 'Arial Unicode MS', 'Noto Sans CJK SC',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            return True
 
    plt.rcParams['font.family'] = 'Times New Roman'
    return False
 
 
def _log_y_formatter(x, _):
    """Y 轴对数刻度：只显示 10 的整数次幂，LaTeX 上标格式。"""
    if x <= 0:
        return ''
    log_val = np.log10(x)
    p = int(np.round(log_val))
    if not np.isclose(log_val, p, atol=0.01):
        return ''
    if p == 0:  return r'$1$'
    if p == 1:  return r'$10$'
    return rf'$10^{{{p}}}$'
 
 
def plot_emg_psd_paper(
    emg_data,
    sampling_rate: int,
    channel_indices=None,
    channel_names=None,
    save_path: str = None,
    figsize: tuple = (10.0, 5.5),       # 加宽留出右侧图例空间
    freq_range: tuple = (0, 120),
    use_chinese: bool = True,
    show_mean: bool = True,
    show_peak: bool = True,
    show_notch_line: bool = True,
    notch_freq: float = 50.0,
    show_freq_bands: bool = True,
    alpha_individual: float = 0.35,
    linewidth_individual: float = 0.9,
    linewidth_mean: float = 2.3,
    nperseg_factor: int = 2,
):
    """
    绘制 sEMG 信号的功率谱密度图（学术论文版 v2）。
 
    Parameters
    ----------
    emg_data : np.ndarray, shape (N_samples, N_channels)
        原始 EMG 数据。
    sampling_rate : int
        采样率 (Hz)。
    channel_indices : list[int] | None
        要绘制的通道索引，None 则全选（最多 8 条）。
    channel_names : list[str] | None
        通道显示名称；None 时自动生成"通道 X"。
    save_path : str | None
        图片保存路径（含扩展名）；None 则不保存。
    figsize : tuple
        画布尺寸（英寸），默认 (10.0, 5.5)。
    freq_range : tuple
        X 轴显示范围，默认 (0, 120) Hz。
    use_chinese : bool
        是否使用中文标签，自动检测字体。
    show_mean : bool
        是否绘制均值 PSD 曲线。
    show_peak : bool
        是否标注均值 PSD 的峰值。
    show_notch_line : bool
        是否绘制陷波频率标注虚线。
    notch_freq : float
        陷波频率 (Hz)，默认 50 Hz。
    show_freq_bands : bool
        是否绘制低 / 中 / 高频段背景色标注。
    alpha_individual : float
        各通道曲线透明度，默认 0.35。
    linewidth_individual : float
        各通道曲线线宽，默认 0.9。
    linewidth_mean : float
        均值曲线线宽，默认 2.3。
    nperseg_factor : int
        Welch 每段样本数 = nperseg_factor × sampling_rate。
 
    Returns
    -------
    fig, ax : matplotlib Figure & Axes
    """
 
    # ── 0. 字体初始化 ──────────────────────────────────────────────────────
    zh = _setup_fonts(use_chinese)
 
    # ── 1. 全局字号（中文核心期刊规范） ───────────────────────────────────
    plt.rcParams.update({
        'font.size':        10,
        'axes.labelsize':   11,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'legend.fontsize':   12,
        'axes.linewidth':    0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
    })
 
    # ── 2. 数据准备 ────────────────────────────────────────────────────────
    if emg_data.ndim == 1:
        emg_data = emg_data.reshape(-1, 1)
    _, num_channels = emg_data.shape
 
    channels_to_plot = (
        list(channel_indices) if channel_indices is not None
        else list(range(min(num_channels, 8)))
    )
 
    def _ch_name(idx):
        if channel_names and idx < len(channel_names):
            return channel_names[idx]
        return f'通道 {idx + 1}' if zh else f'Ch{idx + 1}'
 
    nperseg  = int(nperseg_factor * sampling_rate)
    noverlap = nperseg // 2
 
    # ── 3. 计算 PSD（Welch 法，汉明窗） ───────────────────────────────────
    all_psds, freqs = [], None
    for ch_idx in channels_to_plot:
        f, psd = signal.welch(
            emg_data[:, ch_idx], fs=sampling_rate,
            window='hamming', nperseg=nperseg,
            noverlap=noverlap, scaling='density', detrend='constant',
        )
        if freqs is None:
            freqs = f
        all_psds.append(psd)
 
    mean_psd = np.mean(all_psds, axis=0) if all_psds else None
 
    # ── 4. 创建画布 ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
 
    # ── 5. 三频段背景色（最底层） ──────────────────────────────────────────
    if show_freq_bands:
        bands = _BAND_ZH if zh else _BAND_EN
        for band in bands.values():
            x0 = max(band['range'][0], freq_range[0])
            x1 = min(band['range'][1], freq_range[1])
            if x0 >= x1:
                continue
            # 背景填充
            ax.axvspan(x0, x1, color=band['color'], alpha=0.55,
                       zorder=0, linewidth=0)
            # 频段标签：X 轴数据坐标居中，Y 轴 axes 比例顶部
            ax.text(
                (x0 + x1) / 2, 0.975,
                band['label'],
                transform     = ax.get_xaxis_transform(),
                ha='center', va='top',
                fontsize      = 12,
                color         = band['text_color'],
                linespacing   = 1.3,
                zorder        = 1,
            )
 
        # 频段分界线（极细浅灰）
        for x_div in [20, 100]:
            if freq_range[0] < x_div < freq_range[1]:
                ax.axvline(x_div, color='#CCCCCC', linewidth=0.5,
                           linestyle='-', zorder=1)
 
    # ── 6. 各通道 PSD ──────────────────────────────────────────────────────
    ch_handles = []
    for i, (ch_idx, psd) in enumerate(zip(channels_to_plot, all_psds)):
        line, = ax.semilogy(
            freqs, psd,
            color=_CHANNEL_COLORS[i % len(_CHANNEL_COLORS)],
            alpha=alpha_individual,
            linewidth=linewidth_individual,
            zorder=2,
            label=_ch_name(ch_idx),
        )
        ch_handles.append(line)
 
    # ── 7. 均值 PSD ────────────────────────────────────────────────────────
    mean_handle = None
    if show_mean and mean_psd is not None:
        mean_label = '均值功率谱' if zh else 'Mean PSD'
        mean_line, = ax.semilogy(
            freqs, mean_psd,
            color=_MEAN_COLOR, linewidth=linewidth_mean,
            zorder=10, label=mean_label,
        )
        mean_handle = mean_line
 
    # ── 8. 峰值标注 ────────────────────────────────────────────────────────
    peak_handle = None
    if show_peak and show_mean and mean_psd is not None:
        mask     = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        peak_idx = np.argmax(mean_psd[mask])
        peak_f   = freqs[mask][peak_idx]
        peak_v   = mean_psd[mask][peak_idx]
        peak_label = f'峰值: {peak_f:.1f} Hz' if zh else f'Peak: {peak_f:.1f} Hz'
        peak_pt, = ax.plot(
            peak_f, peak_v, marker='*',
            color=_PEAK_COLOR, markersize=10,
            linestyle='none', zorder=15, label=peak_label,
        )
        peak_handle = peak_pt
 
    # ── 9. 陷波标注线 ──────────────────────────────────────────────────────
    if show_notch_line and freq_range[0] < notch_freq < freq_range[1]:
        ax.axvline(notch_freq, color=_NOTCH_COLOR,
                   linewidth=0.7, linestyle='--', zorder=3)
        notch_text = f'陷波 {notch_freq:.0f} Hz' if zh else f'Notch {notch_freq:.0f} Hz'
        ax.text(
            notch_freq - 0.8, 1.6e-4,
            notch_text, fontsize=12,
            color=_NOTCH_COLOR, ha='right', va='bottom', zorder=4,
        )
 
    # ── 10. 坐标轴范围与刻度 ───────────────────────────────────────────────
    ax.set_xlim(freq_range)
    ax.set_ylim(bottom=1e-4)
 
    # Y 轴格式：LaTeX 上标，仅整数次幂
    ax.yaxis.set_major_formatter(FuncFormatter(_log_y_formatter))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=50)
    )
 
    # X 轴：步距 20 Hz
    ax.set_xticks(np.arange(freq_range[0], freq_range[1] + 1, 20))
 
    # 刻度内置，四边对称（IEEE / 中文核心规范）
    ax.tick_params(axis='both', which='major',
                   direction='in', length=4.5, width=0.6,
                   top=True, right=True)
    ax.tick_params(axis='both', which='minor',
                   direction='in', length=2.0, width=0.4,
                   top=True, right=True)
 
    xlabel = '频率 (Hz)'                          if zh else 'Frequency (Hz)'
    ylabel = r'功率谱密度 ($\mu V^2$/Hz)'         if zh else r'Power Spectral Density ($\mu V^2$/Hz)'
    ax.set_xlabel(xlabel, fontweight='bold', labelpad=6, fontsize=14)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=6, fontsize=14)
 
    ax.grid(True, which='major', linestyle='--', linewidth=0.45,
            color='#CCCCCC', alpha=0.7, zorder=1)
    ax.grid(True, which='minor', linestyle=':',  linewidth=0.3,
            color='#DDDDDD', alpha=0.5, zorder=1)
 
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
 
    # ── 11. 图例：坐标轴右侧外部，单列 ────────────────────────────────────
    legend_handles = ch_handles[:]
    if mean_handle:
        legend_handles.append(mean_handle)
    if peak_handle:
        legend_handles.append(peak_handle)
 
    ax.legend(
        handles        = legend_handles,
        loc            = 'upper left',
        bbox_to_anchor = (1.01, 1.0),   # 紧贴右侧，顶部对齐
        borderaxespad  = 0,
        ncol           = 1,
        frameon        = True,
        fancybox       = False,
        edgecolor      = '#BBBBBB',
        framealpha     = 0.95,
        handlelength   = 1.8,
        handletextpad  = 0.5,
        borderpad      = 0.7,
        labelspacing   = 0.45,
    )
 
    # ── 12. 布局：右侧留白给图例（约占 figure 宽度 16%） ──────────────────
    fig.tight_layout(pad=0.8)
    fig.subplots_adjust(right=0.81)
 
    # ── 13. 保存 ───────────────────────────────────────────────────────────
    if save_path:
        save_dir = os.path.dirname(save_path) or '.'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            save_path,
            dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none',
        )
        print(f'[PSD] 已保存 → {save_path}')
 
    return fig, ax




COLORS = [
    '#9933cc', '#885500', '#22aa44', '#557733',
    '#3388cc', '#cc2233', '#dd55aa', '#ee8811',
    '#1199bb', '#aa3300', '#33aa99', '#8833cc',
    '#bb7700', '#226699', '#aa5522', '#558822',
]
 
 
def _draw(ax, x, data, lw=1.1):
    """所有通道重叠绘制，无任何装饰"""
    for ch in range(data.shape[1]):
        ax.plot(x, data[:, ch],
                color=COLORS[ch % len(COLORS)],
                lw=lw, alpha=0.88,
                solid_capstyle='round', solid_joinstyle='round')
    ax.set_xlim(x[0], x[-1])
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_visible(False)
 
 
# ── 图1：全局视图 ─────────────────────────────────────────────────
 
def plot_rms_waveform_global(rms_data, figsize=(13, 3.0), dpi=300,
                              lw=0.9, n_local=4, window_ratio=0.10,
                              save_path=None):
    """
    全局重叠波形图，高亮局部窗口位置。
 
    参数:
        rms_data     : (T, C) ndarray，已归一化
        figsize      : 图像尺寸
        dpi          : 分辨率
        lw           : 线宽
        n_local      : 局部段数（用于高亮标注）
        window_ratio : 每段窗口占总长比例
        save_path    : 保存路径
    """
    data = rms_data.astype(float)
    T, C = data.shape
    x = np.arange(T)
    win = max(1, int(T * window_ratio))
    seg_starts = np.linspace(0, T - win, n_local, dtype=int)
 
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='white')
    ax.set_facecolor('white')
    _draw(ax, x, data, lw=lw)
 
 
 
    # 顶部标记方块
    ymax = data.max()
    ax.plot(T / 2, ymax * 1.15,
            's', color='#7722aa', markersize=7,
            clip_on=False, zorder=5)
 
    plt.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.05)
 
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"已保存：{save_path}")
    plt.show()
    return fig, ax
 
 
# ── 图2：局部视图 ─────────────────────────────────────────────────
 
def plot_rms_waveform_local(rms_data, n_local=4, window_ratio=0.10,
                             figsize=(14, 3.4), dpi=300,
                             lw=1.05, save_path=None):
    """
    局部重叠波形图，均匀取 n_local 段放大展示。
 
    参数:
        rms_data     : (T, C) ndarray，已归一化
        n_local      : 段数
        window_ratio : 每段占总长比例
        figsize      : 图像尺寸
        dpi          : 分辨率
        lw           : 线宽
        save_path    : 保存路径
    """
    data = rms_data.astype(float)
    T, C = data.shape
    win = max(1, int(T * window_ratio))
    seg_starts = np.linspace(0, T - win, n_local, dtype=int)
 
    box_w = 0.175
    box_h = 0.70
    box_y = 0.17
    gap   = (1.0 - n_local * box_w) / (n_local + 1)
    lefts = [gap + i * (box_w + gap) for i in range(n_local)]
 
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
 
    for idx, (s, left) in enumerate(zip(seg_starts, lefts)):
        seg = data[s: s + win, :]
        x   = np.arange(win)
 
        ax = fig.add_axes([left, box_y, box_w, box_h])
        ax.set_facecolor('white')
        _draw(ax, x, seg, lw=lw)
 
        # 边框
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.9)
            sp.set_color('#333333')
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
 
        # 顶部连接线 + 圆点
        cx = left + box_w / 2
        fig.add_artist(plt.Line2D(
            [cx, cx], [box_y + box_h, box_y + box_h + 0.09],
            transform=fig.transFigure,
            color='#8833bb', lw=1.2, clip_on=False))
        fig.add_artist(plt.Circle(
            (cx, box_y + box_h + 0.105), 0.011,
            color='#7722aa', transform=fig.transFigure, clip_on=False))
 
        # 时间范围 + 段编号
        fig.text(left + box_w / 2, box_y - 0.06,
                 f'Step {s}–{s+win}',
                 ha='center', va='top', fontsize=7.5,
                 color='#666666', transform=fig.transFigure)
        fig.text(left + box_w / 2, box_y - 0.13,
                 f'Segment {idx+1}',
                 ha='center', va='top', fontsize=8,
                 color='#7722aa', fontweight='medium',
                 transform=fig.transFigure)
 
    # 段间省略号
    for i in range(n_local - 1):
        mid_x = lefts[i] + box_w + gap / 2
        mid_y = box_y + box_h / 2
        for dx in [-0.020, 0.0, 0.020]:
            fig.add_artist(plt.Circle(
                (mid_x + dx, mid_y), 0.0065,
                color='#444444', transform=fig.transFigure, clip_on=False))
 
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"已保存：{save_path}")
    plt.show()
    return fig
 


def plot_rms_heatmap(rms_data, title='RMS Feature Heatmap', 
                     figsize=(10, 6), dpi=300, cmap='viridis',
                     xlabel='Time Steps', ylabel='Channel', 
                     colorbar_label='RMS Amplitude',
                     save_path=None, show_values=False,
                     time_labels=None, channel_labels=None):
    """
    绘制专业级 RMS 特征热图，适用于学术论文。

    参数:
        rms_data (np.ndarray): 形状为 (时间步数, 通道数) 的二维数组
        title (str): 图像标题
        figsize (tuple): 图像尺寸 (宽, 高)
        dpi (int): 图像分辨率
        cmap (str): 颜色映射
        xlabel (str): x轴标签
        ylabel (str): y轴标签
        colorbar_label (str): 颜色条标签
        save_path (str): 保存路径，None则不保存
        show_values (bool): 是否在热图上显示数值
        time_labels (list): 时间轴标签
        channel_labels (list): 通道标签
    """
    if rms_data.ndim != 2:
        raise ValueError("输入的 RMS 数据应为二维数组，形状为 (时间步数, 通道数)")
    
    # 设置论文级别的matplotlib样式
    plt.style.use('seaborn-v0_8-paper')
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 使用更专业的颜色映射
    if cmap == 'viridis':
        colors = ['#440154', '#31688e', '#35b779', '#fde725']
        custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', colors, N=256)
    else:
        custom_cmap = plt.cm.get_cmap(cmap)
    
    # 转置数据以正确显示：行为通道，列为时间
    rms_display = rms_data.T  # 转置后形状为 (通道数, 时间步数)
    
    # 绘制热图
    im = ax.imshow(rms_display, aspect='auto', cmap=custom_cmap, 
                   origin='lower', interpolation='bilinear')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # 设置轴标签和刻度
    num_time_steps, num_channels = rms_data.shape
    
    # Y轴（通道）
    if channel_labels is not None:
        ax.set_yticks(range(num_channels))
        ax.set_yticklabels(channel_labels)
    else:
        ax.set_yticks(range(0, num_channels, max(1, num_channels//10)))
        ax.set_yticklabels([f'Ch{i+1}' for i in range(0, num_channels, max(1, num_channels//10))])
    
    # X轴（时间）
    if time_labels is not None:
        ax.set_xticks(range(len(time_labels)))
        ax.set_xticklabels(time_labels, rotation=45)
    else:
        # 智能设置时间轴刻度
        max_ticks = 10
        step = max(1, num_time_steps // max_ticks)
        tick_positions = range(0, num_time_steps, step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f'{i}' for i in tick_positions])
    
    # 设置标签
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # 如果需要显示数值
    if show_values and rms_data.size < 500:  # 只在数据量较小时显示
        for i in range(num_channels):
            for j in range(num_time_steps):
                text = ax.text(j, i, f'{rms_display[i, j]:.2f}',
                             ha="center", va="center", color="white", 
                             fontsize=8, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 设置刻度参数
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"图像已保存至: {save_path}")
    
    plt.show()
    
    return fig, ax




@dataclass
class Emg2PoseSessionData:
    HDF5_GROUP: ClassVar[str] = "neuropose"
    TIMESERIES: ClassVar[str] = "timeseries"
    EMG: ClassVar[str] = "emg"
    JOINT_ANGLES: ClassVar[str] = "joint_angles"
    SKELETON: ClassVar[str] = "skeleton"
    TIMESTAMPS: ClassVar[str] = "time"
    USER_ID: ClassVar[str] = "user_id"
    MOTION_ID: ClassVar[str] = "motion_id"
    SESSION_NAME: ClassVar[str] = "session"
    START_TIME: ClassVar[str] = "start"
    END_TIME: ClassVar[str] = "end"
    NUM_CHANNELS: ClassVar[str] = "num_channels"
    NUM_JOINTS: ClassVar[str] = "num_joints"
    NUM_SKELETONS: ClassVar[str] = "num_skeletons"
    DATASET_NAME: ClassVar[str] = "dataset"
    SAMPLE_RATE: ClassVar[str] = "sample_rate"
    USER: ClassVar[str] = "user"
    MOTION: ClassVar[str] = "motion"

def preprocess_data_segment(emg_raw, glove_raw, config):
    """(与原脚本完全相同，直接复用)"""
    sr = config["sampling_rate"]
    nyquist = 0.5 * sr

    SELECTED_JOINT_INDICES = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] 

    # plot_joint_angles_subplots(
    #     joint_angles=glove_raw,
    #     sampling_rate=CONFIG['fps'],
    #     joint_indices_to_display=SELECTED_JOINT_INDICES, # <--- 在这里传入列表
    #     title=f"Selected Joint Angles"
    # )




    # Filter design
    b_band, a_band = scipy.signal.butter(4, [10/nyquist, 95/nyquist], btype='bandpass')
    b_notch, a_notch = scipy.signal.iirnotch(50.0, 30.0, sr)
    b_low, a_low = scipy.signal.butter(4, 5/nyquist, btype='low')


    # Apply filters
    emg_filtered = scipy.signal.filtfilt(b_band, a_band, emg_raw, axis=0)
    emg_filtered = scipy.signal.filtfilt(b_notch, a_notch, emg_filtered, axis=0)
    

    glove_smooth = scipy.signal.filtfilt(b_low, a_low, glove_raw, axis=0)


    # plot_8ch_colored_time_series_clean(
    #         emg_data=emg_raw, 
    #         sampling_rate=config["sampling_rate"], 
    #         save_path="time.png", 
    #     )

    # plot_emg_psd_paper(
    #     emg_data        = emg_raw,
    #     sampling_rate   = sr,
    #     freq_range      = (0, 120),
    #     use_chinese     = True,
    #     show_mean       = True,
    #     show_peak       = True,
    #     show_notch_line = True,
    #     show_freq_bands = True,          # 开启三频段背景色
    #     save_path       = 'emg_psd_v2.png',
    # )

    # plot_emg_psd_paper(
    #     emg_data=emg_filtered,
    #     sampling_rate=sr,
    #     save_path="E:\hand_mesh\数据可视化",
    #     freq_range=(0, 120),
    # )

    # plot_emg_channels_stacked(
    #         emg_raw=emg_filtered, 
    #         sampling_rate=config["sampling_rate"],
    #         title=None, # 专利图通常不需要标题，在Latex里写caption
    #         duration=100, # 只看前2秒，波形更清楚
    #         save_path="Patent_Fig_RawEMG.png", # 保存成图片
    #     )

    # plot_joint_angles_subplots(
    #     joint_angles=glove_smooth,  # 或者 glove_smooth
    #     sampling_rate=CONFIG['fps'],
    #     save_path="Patent_Joints_20.png"
    # )

    # plot_joint_angles_subplots(
    #     joint_angles=glove_smooth,
    #     sampling_rate=CONFIG['fps'],
    #     joint_indices_to_display=SELECTED_JOINT_INDICES, # <--- 在这里传入列表
    #     title=f"Selected Joint Angles"
    # )


    # RMS feature extraction
    win_len = int(config["rms_window_ms"] * sr / 1000)
    inc = int(config["rms_increment_ms"] * sr / 1000)
    if emg_filtered.shape[0] < win_len:
        return None, None

    num_frames = (emg_filtered.shape[0] - win_len) // inc + 1
    
    rms_features = np.array([
        np.sqrt(np.mean(emg_filtered[j*inc : j*inc+win_len, :]**2, axis=0)) 
        for j in range(num_frames)
    ])
    
    angle_labels = np.array([
        glove_smooth[j*inc + win_len - 1, :]
        for j in range(num_frames)
    ])

    return rms_features, angle_labels, glove_smooth

# ======================================================================================
# 3. 适配后的流水线阶段函数
# ======================================================================================

def run_stage_1_calculate_scalers(config):
    """(关键修改) 从您的 split 目录下的 _train.hdf5 文件计算归一化参数。"""
    print("\n--- STAGE 2: Calculating Normalization Scalers for Private Data ---")
    output_file = config["stage1_output_file"]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_scalers = {}

    for sub_id in config["subjects"]:
        print(f"  Calculating scalers for Subject S{sub_id}...")
        subject_emg_min = np.full(config["emg_channels"], np.inf)
        subject_emg_max = np.full(config["emg_channels"], -np.inf)
        subject_angle_min = np.full(config["glove_final_channels"], np.inf)
        subject_angle_max = np.full(config["glove_final_channels"], -np.inf)

        # 遍历该受试者的所有训练文件
        subject_dir = config["private_data_root"] / f"subject{sub_id}"
        train_files = list(subject_dir.glob("*_train.hdf5"))

        if not train_files:
            print(f"    Warning: No training files found for subject {sub_id}. Skipping.")
            continue

        for train_file in train_files:
            with h5py.File(train_file, 'r') as f:
                timeseries_data = f['neuropose/timeseries']
                emg_raw, glove_raw = timeseries_data['emg'][:], timeseries_data['joint_angles'][:]
            
            rms_features, _, joint_angles_full_smooth = preprocess_data_segment(emg_raw, glove_raw, config)
            if rms_features is None: continue
            
            subject_emg_min = np.minimum(subject_emg_min, np.min(rms_features, axis=0))
            subject_emg_max = np.maximum(subject_emg_max, np.max(rms_features, axis=0))
            subject_angle_min = np.minimum(subject_angle_min, np.min(joint_angles_full_smooth, axis=0))
            subject_angle_max = np.maximum(subject_angle_max, np.max(joint_angles_full_smooth, axis=0))

        all_scalers[f's{sub_id}'] = {
            'emg_min': subject_emg_min, 'emg_max': subject_emg_max,
            'angle_min': subject_angle_min, 'angle_max': subject_angle_max
        }
        
    np.savez(output_file, **all_scalers)
    print(f"  All scalers saved to: {output_file}")
    print("--- STAGE 2: COMPLETED ---")

def run_stage_2_preprocess_data(config):
    """(关键修改) 对 split 目录下的所有 train/test 文件应用预处理和归一化。"""
    print("\n--- STAGE 3: Preprocessing and Normalizing Private Data ---")
    input_dir = config["private_data_root"]
    output_dir = config["stage2_output_dir"]
    scalers = np.load(config["stage1_output_file"], allow_pickle=True)

    # 遍历所有受试者目录下的所有HDF5文件
    for h5_file in input_dir.rglob("subject*/*.hdf5"):
        # 从文件名解析信息
        match = re.search(r'(\d+)_(\w+)', h5_file.stem) # e.g., "1_train"
        if not match: continue
        action_id_str, split = match.groups()
        
        # 从父目录名解析受试者ID
        sub_id_match = re.search(r'subject(\d+)', h5_file.parent.name)
        if not sub_id_match: continue
        sub_id_str = sub_id_match.group(1)

        # 将解析出的字符串ID转换为整数
        subject_id = int(sub_id_str)
        motion_id = int(action_id_str)

        print(f"  Processing S{sub_id_str}_M{action_id_str}_{split}...")
        
        with h5py.File(h5_file, 'r') as f:
            timeseries_data = f['neuropose/timeseries']
            emg_raw, glove_raw = timeseries_data['emg'][:], timeseries_data['joint_angles'][:]
            skeleton_data = timeseries_data['skeleton'][:]


        rms_features, angle_labels, _ = preprocess_data_segment(emg_raw, glove_raw, config)
        if rms_features is None:
            print(f"    Skipping due to insufficient data.")
            continue
        
        # Normalize data
        scaler_key = f's{sub_id_str}'
        if scaler_key not in scalers: 
            print(f"    Warning: Scaler not found for subject {sub_id_str}. Skipping.")
            continue
        
        subject_scalers = scalers[scaler_key].item()
        emg_min, emg_max = subject_scalers['emg_min'], subject_scalers['emg_max']
        angle_min, angle_max = subject_scalers['angle_min'], subject_scalers['angle_max']
        
        emg_range = emg_max - emg_min; emg_range[emg_range == 0] = 1e-8
        emg_norm = (rms_features - emg_min) / emg_range





        # plot_rms_waveform_global(emg_norm, figsize=(13, 3.0), dpi=300,
        #                         lw=0.9, n_local=4, window_ratio=0.10,
        #                         save_path='rms_global.png')

        # plot_rms_waveform_local(emg_norm, n_local=4, window_ratio=0.10,
        #                         figsize=(14, 3.4), dpi=300,
        #                         lw=1.05, save_path='rms_local.png')
        # plot_rms_heatmap(emg_norm, 
        #                 title='EMG RMS Feature Heatmap',
        #                 save_path='emg_norm_heatmap_test.png',
        #                 channel_labels=[f'EMG-{i+1}' for i in range(emg_norm.shape[1])])


        angle_range = angle_max - angle_min; angle_range[angle_range == 0] = 1e-8
        angle_norm = (angle_labels - angle_min) / angle_range
        
        # --- 4. 封装成最终格式并保存 ---
        num_samples = emg_norm.shape[0]
        
        # 创建新的结构化数组
        # 注意：这里的维度需要和处理后的数据匹配
        # 特别是 joint_angles 的维度
        dtype = np.dtype([
            ('emg', np.float32, (emg_norm.shape[1],)),
            ('joint_angles', np.float32, (angle_norm.shape[1],)),
            ('skeleton', np.float32, (skeleton_data.shape[1],)) if skeleton_data is not None else ('skeleton', np.float32, (1,)), # 处理skeleton不存在的情况
            ('time', np.float64),
            ('user_id', np.int32),   # <--- 确保定义
            ('motion_id', np.int32)
        ])
        
        new_ts_data = np.empty(num_samples, dtype=dtype)
        
        # 填充新数据
        new_ts_data['emg'] = emg_norm
        new_ts_data['joint_angles'] = angle_norm
        
        # 填充旧数据或其他数据
        if skeleton_data is not None and len(skeleton_data) > 0:
            # 需要对 skeleton 数据进行降采样以匹配新的长度
            # 这里我们简单地取样，更精确的方法是插值
            inc = max(1, int(config["rms_increment_ms"] * config["sampling_rate"] / 1000))
            win_len = int(config["rms_window_ms"] * config["sampling_rate"] / 1000)
            skeleton_downsampled = np.array([skeleton_data[j*inc + win_len - 1, :] for j in range(num_samples)])
            new_ts_data['skeleton'] = skeleton_downsampled
        else:
            # 如果没有skeleton数据，填充0
            new_ts_data['skeleton'] = np.zeros((num_samples, dtype.fields['skeleton'][0].shape[0]), dtype=np.float32)

        # 重新生成时间戳
        feature_sample_rate = 1000 / config["rms_increment_ms"]
        new_ts_data['time'] = np.arange(num_samples) / feature_sample_rate
        new_ts_data['user_id'] = subject_id
        new_ts_data['motion_id'] = motion_id


        # 保存到新文件
        out_sub_dir = output_dir / f"subject{sub_id_str}"
        out_sub_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_sub_dir / f"S{sub_id_str}_M{action_id_str}_{split}.hdf5"
        with h5py.File(out_path, 'w') as f_out:
            group = f_out.create_group('neuropose')
            group.create_dataset('timeseries', data=new_ts_data)
            group.attrs['sample_rate'] = feature_sample_rate

    print("--- STAGE 2: COMPLETED ---")






# ======================================================================================
# 4. 主执行块
# ======================================================================================
if __name__ == "__main__":
    print("====== STARTING PRIVATE DATA PROCESSING PIPELINE ======")

    # # 执行从第二阶段开始的流水线
    run_stage_1_calculate_scalers(CONFIG)
    run_stage_2_preprocess_data(CONFIG)
