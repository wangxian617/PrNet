# PrNet 实验评估报告（中文版）

**生成日期**: 2026年6月1日

---

## 一、项目概述

PrNet（Pseudorange Correction Network）是一种基于多层感知机（MLP）的神经网络，用于校正Android智能手机的原始GNSS（全球导航卫星系统）伪距测量值，从而提升定位精度。本报告基于PrNet的完整实验流程，对所有图表内容进行详细的中文描述与解读。

### 1.1 术语定义

在本报告中，各处理阶段的伪距名称定义如下：

| 术语 | 英文名称 | 含义说明 |
|------|----------|----------|
| **原始伪距** | Raw Pseudorange | 智能手机GNSS接收机直接测量得到的伪距值，包含各种误差（对流层、电离层、多径、钟差等） |
| **WLS伪距残差** | WLS Pseudorange Residual | 经过加权最小二乘法（WLS）解算后的伪距残差，已消除接收机钟差的影响 |
| **EKF平滑伪距** | EKF-smoothed Pseudorange | 在WLS基础上，经扩展卡尔曼滤波（EKF）进行时域平滑后的伪距 |
| **RTS平滑伪距** | RTS-smoothed Pseudorange | 在EKF基础上，经Rauch-Tung-Striebel（RTS）后向平滑后的伪距 |
| **PrNet校正伪距** | PrNet-corrected Pseudorange | 经过PrNet神经网络预测并校正后的伪距，去除了多径等系统性误差 |
| **MHE估计伪距** | MHE-estimated Pseudorange | 经移动窗口估计（Moving Horizon Estimation）方法处理后的伪距 |

### 1.2 评估方法

各方法之间的定位组合方式如下：

| 方法简称 | 处理流程 |
|----------|----------|
| **WLS** | 原始伪距 → 加权最小二乘解算 → 定位结果 |
| **WLS+EKF** | 原始伪距 → WLS解算 → EKF时域滤波 → 定位结果 |
| **WLS+EKF+RTS** | 原始伪距 → WLS → EKF → RTS后向平滑 → 定位结果 |
| **WLS+MHE** | 原始伪距 → WLS → 移动窗口估计 → 定位结果 |
| **WLS+PrNet** | 原始伪距 → PrNet校正 → WLS解算 → 定位结果 |
| **WLS+EKF+PrNet** | 原始伪距 → PrNet校正 → WLS → EKF → 定位结果 |
| **WLS+PrNet+RTS** | 原始伪距 → PrNet校正 → WLS → EKF → RTS → 定位结果 |
| **WLS+MHE+PrNet** | 原始伪距 → PrNet校正 → WLS → MHE → 定位结果 |

### 1.3 评价指标与公式

#### 1.3.1 伪距误差（Pseudorange Error）

$$\text{PR\_Error}_i = \rho_i^{\text{measured}} - \rho_i^{\text{true}}$$

其中 $\rho_i^{\text{measured}}$ 为第 $i$ 颗卫星的测量伪距，$\rho_i^{\text{true}}$ 为真实伪距（由真实位置和卫星位置计算得到）。

#### 1.3.2 水平定位误差（Horizontal Position Error）

$$\text{HPE} = R_{\text{earth}} \cdot \arccos\left(\sin\phi_1 \sin\phi_2 + \cos\phi_1 \cos\phi_2 \cos(\lambda_2 - \lambda_1)\right)$$

其中 $(\phi_1, \lambda_1)$ 为估计位置的纬度和经度，$(\phi_2, \lambda_2)$ 为真实位置的纬度和经度，$R_{\text{earth}} = 6371000$ 米。

#### 1.3.3 百分位数指标

- **50th 百分位数**（中位数）：表示50%的定位误差小于该值
- **67th 百分位数**：表示67%的定位误差小于该值（常用于GNSS评估）
- **95th 百分位数**：表示95%的定位误差小于该值

#### 1.3.4 改善率计算公式

$$\text{改善率(\%)} = \frac{E_{\text{baseline}} - E_{\text{improved}}}{E_{\text{baseline}}} \times 100\%$$

其中：
- 分子：$E_{\text{baseline}} - E_{\text{improved}}$ = 基线方法的误差 - 改进方法的误差
- 分母：$E_{\text{baseline}}$ = 基线方法的误差
- $E$ 可以是均值、中位数或某个百分位数处的误差值

#### 1.3.5 消融实验变化率

$$\text{损失变化率(\%)} = \frac{L_{\text{ablated}} - L_{\text{full}}}{L_{\text{full}}} \times 100\%$$

其中：
- 分子：$L_{\text{ablated}} - L_{\text{full}}$ = 移除某特征后的损失 - 完整模型的损失
- 分母：$L_{\text{full}}$ = 完整PrNet模型的损失

---

## 二、模型架构

| 参数 | 值 |
|------|-----|
| 模型类型 | PrNet（基于MLP的多层感知机） |
| 输入特征数 | 16（CN0、sinE、cosE、PRN、WLS位置、几何矢量、航向角） |
| 隐藏层神经元数 | 40 |
| MLP层数 | 20 |
| 总可训练参数数 | 31,881 |
| Dropout | 0 |

### 2.1 输入特征详细说明

PrNet的16维输入特征包括：

| 特征编号 | 特征名称 | 说明 |
|----------|----------|------|
| 0 | CN0/50 | 载噪比（归一化） |
| 1 | sinE | 卫星仰角的正弦值 |
| 2 | cosE | 卫星仰角的余弦值 |
| 3 | PRN/32 | 卫星伪随机噪声码编号（归一化） |
| 4–6 | WLS经度 | WLS解算的经度（度、分、秒） |
| 7–9 | WLS纬度 | WLS解算的纬度（度、分、秒） |
| 10–12 | 几何矢量 | 用户到卫星的单位方向矢量（N、E、D） |
| 13–15 | 航向角 | 智能手机航向角（N、E、D分量） |

---

## 三、实验数据集

使用Google Smartphone Decimeter Challenge (GSDC) 2021的公开数据集，设计了两类场景：

### 3.1 郊区路线（RouteR）

| 测试文件 | 历元数 | 样本总数 | 伪距误差均值（米） | 伪距误差标准差（米） | 伪距误差中位数（米） |
|----------|--------|----------|---------------------|----------------------|----------------------|
| 2020-05-14-US-MTV-1 | 1620 | 13331 | -0.0000 | 11.1782 | 0.1235 |
| 2020-09-04-US-SF-2 | 2231 | 20956 | 0.0539 | 12.0922 | -1.2118 |
| 2021-04-28-US-MTV-1 | 1879 | 14134 | -0.0000 | 16.3983 | -1.7692 |

**PrNet预测的伪距校正量：**

| 测试文件 | 校正均值（米） | 校正标准差（米） | 最小值（米） | 最大值（米） |
|----------|----------------|------------------|--------------|--------------|
| 2020-05-14-US-MTV-1 | -6.9934 | 12.0287 | -108.1513 | 9.0709 |
| 2020-09-04-US-SF-2 | -6.4945 | 13.9629 | -108.3805 | 15.4662 |
| 2021-04-28-US-MTV-1 | -26.0245 | 29.2045 | -112.7067 | 12.0256 |

### 3.2 城区路线（RouteU）

| 测试文件 | 历元数 | 样本总数 | 伪距误差均值（米） | 伪距误差标准差（米） | 伪距误差中位数（米） |
|----------|--------|----------|---------------------|----------------------|----------------------|
| 2021-04-28-US-SJC-1 | 136 | 1130 | 0.0000 | 51.0091 | -11.7041 |
| 2021-04-28-US-SJC-1_G | 86 | 642 | -0.0000 | 22.2158 | -6.3497 |

**PrNet预测的伪距校正量：**

| 测试文件 | 校正均值（米） | 校正标准差（米） | 最小值（米） | 最大值（米） |
|----------|----------------|------------------|--------------|--------------|
| 2021-04-28-US-SJC-1 | -94.2307 | 33.0938 | -163.0645 | -0.5713 |
| 2021-04-28-US-SJC-1_G | -96.9289 | 25.9028 | -147.7514 | -2.4825 |

**注意**：城区场景的伪距误差显著大于郊区场景，这是因为城区存在更严重的多径效应和信号遮挡。

---

## 四、图表详细说明

以下对每类图表的含义、坐标轴、以及如何解读进行详细描述。

### 4.1 各卫星原始伪距时序图

**文件名格式**: `{场景名}_raw_pseudorange_timeseries_cn.png`

**图表说明**：
- **横轴（X轴）**：历元编号（Epoch），代表时间步，每个历元对应接收机的一次测量周期（通常为1秒）
- **纵轴（Y轴）**：原始伪距值（米），即接收机直接测量到的从卫星到接收机的伪距
- **数据点**：不同颜色代表不同卫星（以PRN编号区分），每个点表示该卫星在该历元的原始伪距测量值
- **如何解读**：原始伪距值通常在 $2 \times 10^7$ 米量级（约2万公里，GPS卫星轨道高度），各卫星的伪距随时间连续变化，反映卫星相对运动

### 4.2 伪距误差分布直方图

**文件名格式**: `{场景名}_pr_error_distribution_cn.png`

**图表说明**：
- **横轴（X轴）**：伪距误差（米），即测量伪距与真实伪距的差值
- **纵轴（Y轴）**：频数（Count），表示落在该误差区间内的样本数量
- **子图**：每个子图对应一种方法（WLS、WLS+EKF、WLS+MHE、WLS+PrNet）
- **红色虚线**：均值（Mean），表示伪距误差的平均偏差
- **绿色虚线**：中位数（Median），表示伪距误差的中间值
- **柱高**：代表该误差范围内的测量样本数量
- **如何解读**：分布越集中于零附近，说明该方法的伪距校正效果越好。PrNet校正后的分布应比WLS更窄、更集中于零

### 4.3 伪距误差CDF曲线

**文件名格式**: `{场景名}_pr_error_cdf_cn.png`

**图表说明**：
- **横轴（X轴）**：|伪距误差|（米），即伪距误差的绝对值
- **纵轴（Y轴）**：累积分布函数（CDF），取值范围0~1，表示误差小于等于横轴值的样本比例
- **各曲线**：不同颜色代表不同方法
- **如何解读**：曲线越靠左、越快接近1.0，说明该方法的伪距误差越小。在相同CDF值（如0.67或0.95）处，横轴读数越小越好

### 4.4 伪距误差箱线图

**文件名格式**: `{场景名}_pr_error_boxplot_cn.png`

**图表说明**：
- **横轴（X轴）**：不同定位方法名称
- **纵轴（Y轴）**：伪距误差（米）
- **箱体（Box）**：箱体上下边界分别为第75百分位数（Q3）和第25百分位数（Q1），中间横线为中位数
- **须线（Whisker）**：延伸至1.5×IQR范围内的最大/最小值
- **如何解读**：箱体越窄、越接近零轴，说明该方法的伪距误差分布越集中、偏差越小

### 4.5 轨迹对比图

**文件名格式**: `{场景名}_trajectory_comparison_cn.png`

**图表说明**：
- **横轴（X轴）**：经度（°），地理坐标的东西方向
- **纵轴（Y轴）**：纬度（°），地理坐标的南北方向
- **黑色实线**：真实轨迹（Ground Truth）
- **其他彩色散点**：各方法估算的定位轨迹
- **如何解读**：估算轨迹越贴近黑色真实轨迹，说明定位精度越高。散点越聚集（抖动越小），说明定位稳定性越好

### 4.6 定位误差时序图

**文件名格式**: `{场景名}_position_error_timeseries_cn.png`

**图表说明**：
- **横轴（X轴）**：历元编号，代表时间步
- **纵轴（Y轴）**：水平定位误差（米），即估算位置与真实位置之间的水平距离
- **各曲线**：不同颜色代表不同方法的定位误差随时间的变化
- **如何解读**：曲线越低、越平稳，说明该方法的定位精度越好、越稳定。峰值（spike）通常对应多径效应或信号遮挡严重的时段

### 4.7 定位误差CDF

**文件名格式**: `{场景名}_position_error_cdf_cn.png`

**图表说明**：
- **横轴（X轴）**：水平定位误差（米）
- **纵轴（Y轴）**：累积分布函数（CDF），取值0~1
- **各曲线**：不同方法的误差累积分布
- **如何解读**：可从CDF图读取特定百分位数处的误差值，如67th百分位数对应CDF=0.67时的横轴值。曲线越靠左，说明整体定位精度越高

### 4.8 定位误差统计表

**文件名格式**: `{场景名}_position_error_table_cn.png`

**图表说明**：
该表以图片形式展示各方法的定位误差统计信息：

| 列名 | 说明 |
|------|------|
| 方法 | 定位方法名称 |
| 均值（米） | 定位误差的算术平均值 |
| 中位数（米） | 50%的误差小于该值 |
| 50分位数 | 与中位数相同 |
| 67分位数 | 67%的误差小于该值 |
| 95分位数 | 95%的误差小于该值 |
| 标准差（米） | 误差的离散程度 |

### 4.9 消融实验柱状图

**文件名格式**: `{路线名}_ablation_bar_cn.png`

**图表说明**：
- **横轴（X轴）**：不同特征配置（完整PrNet、移除CN0、移除仰角、移除PRN、移除WLS位置、移除几何矢量、移除航向角）
- **纵轴（Y轴）**：平均MSE损失（Mean Squared Error），值越小说明模型性能越好
- **柱高**：代表该配置下的平均MSE损失值
- **绿色柱**：完整PrNet（基线）
- **橙色柱**：移除某一特征后的模型
- **如何解读**：如果移除某特征后损失显著增大，说明该特征对模型性能贡献重要。增大越多，说明该特征越关键

### 4.10 消融实验统计表

**文件名格式**: `{路线名}_ablation_table_cn.png`

**图表说明**：

| 列名 | 说明 |
|------|------|
| 配置 | 特征消融配置名称 |
| 输入维度 | 消融后的输入特征数 |
| 参数量 | 模型可训练参数总数 |
| 平均损失 | 评估集上的平均MSE损失 |
| 训练时间（秒） | 模型训练所需时间 |
| 推理时间（毫秒） | 单次推理所需时间 |

### 4.11 计算开销对比图

**文件名格式**: `computational_overhead_comparison_cn.png`

**图表说明**：
包含两个子图：

**左图 — 模型复杂度**：
- **横轴**：模型名称（PBC-RF、FCNN-LSTM、Set Transformer、PrNet）
- **纵轴**：参数数量（Number of Parameters）
- **柱高**：代表各模型的可训练参数总数

**右图 — 推理速度**：
- **横轴**：模型名称
- **纵轴**：推理时间（毫秒/样本）
- **柱高**：代表单个样本的推理耗时

### 4.12 计算开销汇总表

**文件名格式**: `computational_overhead_table_cn.png`

| 列名 | 说明 |
|------|------|
| 模型 | 模型名称 |
| 参数量 | 可训练参数总数 |
| 推理时间（毫秒） | 单次推理耗时 |
| 训练时间（秒） | 总训练时间 |
| 内存估计（MB） | 按双精度浮点数估计的内存占用 |

---

## 五、各场景实验结果

### 5.1 郊区场景 — 2020-05-14-US-MTV-1（郊区指纹定位）

**中文图表位置**: `results/2020-05-14-US-MTV-1/chinese/`

#### 图表列表

| 图表 | 文件名 | 说明 |
|------|--------|------|
| 原始伪距时序图 | `2020-05-14-US-MTV-1_raw_pseudorange_timeseries_cn.png` | 各卫星原始伪距随时间变化 |
| 伪距误差分布 | `2020-05-14-US-MTV-1_pr_error_distribution_cn.png` | WLS/EKF/MHE/PrNet四种方法的伪距误差直方图 |
| 伪距误差CDF | `2020-05-14-US-MTV-1_pr_error_cdf_cn.png` | 伪距误差绝对值的累积分布函数 |
| 伪距误差箱线图 | `2020-05-14-US-MTV-1_pr_error_boxplot_cn.png` | 各方法伪距误差对比箱线图 |
| 轨迹对比 | `2020-05-14-US-MTV-1_trajectory_comparison_cn.png` | 各方法定位轨迹与真实轨迹的对比 |
| 定位误差时序 | `2020-05-14-US-MTV-1_position_error_timeseries_cn.png` | 水平定位误差随时间变化 |
| 定位误差CDF | `2020-05-14-US-MTV-1_position_error_cdf_cn.png` | 定位误差的累积分布函数 |
| 定位误差统计表 | `2020-05-14-US-MTV-1_position_error_table_cn.png` | 各方法定位误差统计表格 |

### 5.2 郊区场景 — 2020-09-04-US-SF-2（郊区跨迹定位）

**中文图表位置**: `results/2020-09-04-US-SF-2/chinese/`

#### 图表列表

| 图表 | 文件名 | 说明 |
|------|--------|------|
| 原始伪距时序图 | `2020-09-04-US-SF-2_raw_pseudorange_timeseries_cn.png` | 各卫星原始伪距随时间变化 |
| 伪距误差分布 | `2020-09-04-US-SF-2_pr_error_distribution_cn.png` | 各方法伪距误差直方图 |
| 伪距误差CDF | `2020-09-04-US-SF-2_pr_error_cdf_cn.png` | 伪距误差CDF对比 |
| 伪距误差箱线图 | `2020-09-04-US-SF-2_pr_error_boxplot_cn.png` | 伪距误差箱线图对比 |
| 轨迹对比 | `2020-09-04-US-SF-2_trajectory_comparison_cn.png` | 定位轨迹对比 |
| 定位误差时序 | `2020-09-04-US-SF-2_position_error_timeseries_cn.png` | 定位误差时序图 |
| 定位误差CDF | `2020-09-04-US-SF-2_position_error_cdf_cn.png` | 定位误差CDF |
| 定位误差统计表 | `2020-09-04-US-SF-2_position_error_table_cn.png` | 定位误差统计 |

### 5.3 郊区场景 — 2021-04-28-US-MTV-1（郊区跨迹定位）

**中文图表位置**: `results/2021-04-28-US-MTV-1/chinese/`

（图表列表同5.1格式，文件名前缀为 `2021-04-28-US-MTV-1`）

### 5.4 城区场景 — 2021-04-28-US-SJC-1（城区指纹定位）

**中文图表位置**: `results/2021-04-28-US-SJC-1/chinese/`

（图表列表同5.1格式，文件名前缀为 `2021-04-28-US-SJC-1`）

### 5.5 城区场景 — 2021-04-28-US-SJC-1_G（城区跨迹定位）

**中文图表位置**: `results/2021-04-28-US-SJC-1_G/chinese/`

（图表列表同5.1格式，文件名前缀为 `2021-04-28-US-SJC-1_G`）

---

## 六、特征消融实验

### 6.1 实验设计

消融实验通过逐一移除PrNet的某一类输入特征来评估各特征对模型性能的贡献。共有6组消融实验：

| 消融配置 | 移除的特征 | 移除后输入维度 | 说明 |
|----------|-----------|----------------|------|
| 移除CN0 | CN0/50 | 15 | 移除载噪比特征 |
| 移除仰角 | sinE, cosE | 14 | 移除卫星仰角信息 |
| 移除PRN | PRN/32 | 15 | 移除卫星编号信息 |
| 移除WLS位置 | WLS经纬度（6维） | 10 | 移除WLS估计位置 |
| 移除几何矢量 | N, E, D方向矢量 | 13 | 移除用户—卫星几何向量 |
| 移除航向角 | 航向N, E, D | 13 | 移除手机航向信息 |

### 6.2 结果说明

**中文图表位置**:
- `results/ablation_R/chinese/R_ablation_bar_cn.png` — 郊区路线消融实验柱状图
- `results/ablation_R/chinese/R_ablation_table_cn.png` — 郊区路线消融实验统计表
- `results/ablation_U/chinese/U_ablation_bar_cn.png` — 城区路线消融实验柱状图
- `results/ablation_U/chinese/U_ablation_table_cn.png` — 城区路线消融实验统计表

**解读要点**：
- 如果移除某特征后MSE损失显著增加，则该特征对模型贡献大
- 改善率公式：$\Delta\% = \frac{L_{\text{ablated}} - L_{\text{full}}}{L_{\text{full}}} \times 100\%$
  - 分子：移除该特征后的损失与完整模型损失之差
  - 分母：完整模型的MSE损失

---

## 七、对比模型实验

### 7.1 对比模型说明

| 模型 | 类型 | 说明 |
|------|------|------|
| **PBC-RF** | 随机森林 | Point-Based Correction，使用scikit-learn随机森林回归器 |
| **FCNN-LSTM** | 深度学习 | 全连接网络+LSTM，先提取卫星特征再进行时序建模 |
| **Set Transformer** | 深度学习 | 使用自注意力机制处理卫星集合数据 |
| **PrNet** | 深度学习 | 本文提出的MLP方法 |

### 7.2 结果说明

**中文图表位置**:
- `results/comparison_R/chinese/computational_overhead_comparison_cn.png` — 郊区路线计算开销对比
- `results/comparison_R/chinese/computational_overhead_table_cn.png` — 郊区路线计算开销表
- `results/comparison_U/chinese/computational_overhead_comparison_cn.png` — 城区路线计算开销对比（如适用）
- `results/comparison_U/chinese/computational_overhead_table_cn.png` — 城区路线计算开销表（如适用）

---

## 八、使用说明

### 8.1 生成中文图表

运行以下命令从已有数据重新生成所有中文图表：

```bash
cd PrNet
python generate_chinese_charts.py
```

### 8.2 中文图表模块

中文图表的绘图函数位于 `experiments/charts_cn.py`，提供了与英文版 `experiments/charts.py` 完全对应的中文版函数接口：

| 英文函数 | 中文函数 | 说明 |
|----------|----------|------|
| `plot_raw_pseudorange_timeseries` | `plot_raw_pseudorange_timeseries_cn` | 原始伪距时序图 |
| `plot_pr_error_distribution_comparison` | `plot_pr_error_distribution_comparison_cn` | 伪距误差分布图 |
| `plot_pr_error_cdf` | `plot_pr_error_cdf_cn` | 伪距误差CDF |
| `plot_pr_error_boxplot` | `plot_pr_error_boxplot_cn` | 伪距误差箱线图 |
| `plot_trajectory_comparison` | `plot_trajectory_comparison_cn` | 轨迹对比图 |
| `plot_position_error_timeseries` | `plot_position_error_timeseries_cn` | 定位误差时序图 |
| `plot_position_error_cdf` | `plot_position_error_cdf_cn` | 定位误差CDF |
| `plot_position_error_table` | `plot_position_error_table_cn` | 定位误差统计表 |
| `plot_ablation_results` | `plot_ablation_results_cn` | 消融实验柱状图 |
| `plot_ablation_table` | `plot_ablation_table_cn` | 消融实验统计表 |
| `plot_computational_overhead` | `plot_computational_overhead_cn` | 计算开销柱状图 |
| `plot_computational_overhead_table` | `plot_computational_overhead_table_cn` | 计算开销统计表 |

---

## 九、注意事项

1. 郊区数据使用 `input_size=39`，城区数据使用 `input_size=55`
2. 评估时使用 `batch_size=1`（每个批次对应一个历元）
3. 使用预训练权重文件，位于 `Neural_Pseudorange_Correction/Weights/` 目录
4. 所有评估均在CPU上运行
5. 中文图表需要系统安装中文字体（如WenQuanYi Micro Hei或Noto Sans CJK SC）
