# Battery Materials Discovery for Li-ion Cathodes

> 基于 **Materials Project + Machine Learning + Materials Physics + DFT** 的锂离子电池正极材料筛选项目。
> 核心路线：**Data → Leakage-aware ML → Physics → Pareto → Uncertainty → DFT → Automated Parsing**

---

## 1. Project Overview

本项目关注一个更接近真实材料研发的问题：

> **如何从上千个锂离子电池正极候选中，综合 Voltage、Capacity、Volume Change 和 Stability，逐步筛选出值得进一步计算或实验验证的材料？**

项目从 Materials Project 的 Li insertion electrode 数据出发，结合：

- Framework-aware validation
- XGBoost + SHAP
- Reaction-aware features
- Property-specific physics models
- Pareto multi-objective screening
- Local empirical uncertainty
- Quantum ESPRESSO DFT
- pymatgen + ASE automated workflow

最终从：

```text
1858 modeling records
```

筛选至：

```text
4 DFT shortlist candidates
```

并对排名最高的：

```text
CoPO4 → LiCoPO4
```

完成独立 DFT 验证。

---

## 2. Overall Workflow

```text
Materials Project
        ↓
Data Cleaning & Domain Definition
        ↓
Composition Descriptors
        ↓
Framework GroupKFold
        ↓
Reaction-aware Voltage Modeling
        ↓
XGBoost + SHAP
        ↓
Capacity Physics
        ↓
Endpoint Structure / Volume Physics
        ↓
Engineering Constraints
        ↓
Pareto Optimization
        ↓
Uncertainty Filtering
        ↓
DFT Shortlist
        ↓
Quantum ESPRESSO
        ↓
ASE Automated Output Parsing
        ↓
Battery Voltage Validation
```

核心思路：

> **ML where ML is useful, physics where physics is sufficient, DFT where higher-fidelity validation is needed.**

---

## 3. Dataset & Validation

数据来源：

```text
Materials Project
Li insertion electrode dataset
```

初始数据：

```text
2774 battery records
```

清洗后：

```text
1858 modeling records
1019 unique frameworks
```

主要体系：

```text
Transition metals:
Ti / V / Cr / Mn / Fe / Co / Ni

Anions:
O / P / F / S
```

### Why Framework GroupKFold?

材料数据库中，相似 framework 可能同时出现在训练集和验证集。

Random Split 容易高估模型对真正 unseen material family 的泛化能力，因此项目比较：

| Validation | Mean R² | MAE |
|---|---:|---:|
| Random 5-fold CV | 0.5949 | 0.3667 V |
| Framework GroupKFold | 0.4935 | 0.4185 V |

因此正式 Voltage benchmark 使用：

```text
Framework GroupKFold
```

这一部分的核心结论是：

> **对材料数据来说，合理的数据划分方式有时比换一个更复杂的模型更重要。**

---

## 4. Voltage Prediction

使用 `matminer` 生成约：

```text
142 composition descriptors
```

并建立：

```text
XGBoost Regressor
```

Composition baseline：

```text
R²   = 0.4935
MAE  = 0.4185 V
RMSE = 0.5582 V
```

由于 Average Voltage 是 charge → discharge reaction-dependent property，进一步加入：

```text
normalized_delta_li
relative_delta_li
num_steps
```

Feature ablation：

| Features | Mean R² | MAE |
|---|---:|---:|
| Composition only | 0.4935 | 0.4185 V |
| Reaction only | 0.1026 | 0.5990 V |
| Composition + Reaction | **0.5302** | **0.3966 V** |

说明 Reaction Features 不能替代 composition，但能够补充反应信息。

---

## 5. Model Interpretation with SHAP

最终 Voltage model 使用：

```text
XGBoost
+
Composition Descriptors
+
Reaction Features
```

SHAP top features 包括：

```text
MagpieData mean Column
relative_delta_li
MagpieData mean NdUnfilled
MagpieData mean GSvolume_pa
5-norm
```

其中：

```text
relative_delta_li
```

在 mean absolute SHAP 中排名：

```text
#2
```

说明 charge/discharge reaction extent 对模型预测具有重要贡献。

需要强调：

> **SHAP reflects model attribution, not causality.**

---

## 6. Why Not Use ML for Everything?

项目没有对所有 target 强行使用 Machine Learning。

最终策略：

| Property | Method | Reason |
|---|---|---|
| Voltage | XGBoost + Reaction Features | Reaction-dependent |
| Capacity | Physics Equation | Determined mainly by stoichiometry |
| Volume Change | Endpoint Structure Physics | Strong structure dependence |
| Stability | Screening Constraint | Used for filtering |

### Capacity

最初 composition ML：

```text
R² ≈ 0.18
```

但理论容量可由：

```text
Q = nF / (3.6M)
```

直接计算。

采用正确的 discharged-state molar mass 后：

```text
R² ≈ 1.000
```

因此无需用 ML 重复拟合明确的物理关系。

### Volume Change

Composition model：

```text
R² ≈ 0.12
```

进一步使用 charge / discharge endpoint crystal structures：

```text
R²   ≈ 0.566
MAE  ≈ 0.0098
Corr ≈ 0.820
```

因此最终采用结构物理方法，而不是继续堆模型。

---

## 7. Multi-objective Candidate Screening

筛选同时考虑：

```text
Voltage ↑
Capacity ↑
Volume Change ↓
Stability ↓
```

主要约束：

```text
3.0 V ≤ predicted voltage ≤ 5.0 V
capacity ≥ 100 mAh/g
endpoint ΔV/V ≤ 0.10
max_stability ≤ 0.10 eV/atom
single-step reaction
```

Voltage 使用 Framework GroupKFold OOF prediction，避免 full-data model 对训练样本自我评分。

筛选漏斗：

```text
1858  Modeling records
 ↓
1855  Endpoint structures available
 ↓
629   Physics + stability constraints
 ↓
69    Pareto-optimal candidates
 ↓
53    Unique frameworks
 ↓
20    Uncertainty-filtered candidates
 ↓
4     DFT shortlist
 ↓
1     Full DFT validation
```

---

## 8. Uncertainty-aware Screening

为了降低 extrapolation risk，引入：

```text
Local Empirical Uncertainty
```

方法：

```text
standardized feature space
        ↓
k = 20 nearest samples
        ↓
exclude same framework
        ↓
OOF absolute residuals
        ↓
local 90th percentile error
```

定义：

```text
Voltage_LCB
=
Voltage_OOF
-
local_q90
```

这里的不确定性属于：

```text
framework-excluded
OOF residual-based
empirical uncertainty
```

不是 Bayesian uncertainty，也不是 conformal prediction。

---

## 9. DFT Shortlist

最终 shortlist：

| Rank | Candidate |
|---:|---|
| 1 | CoPO₄ |
| 2 | Ni₂(SO₄)₃ |
| 3 | MnCr₃(PO₄)₆ |
| 4 | V₂O₅ |

CoPO₄ 主要指标：

```text
ML OOF Voltage   ≈ 4.514 V
Local q90        ≈ 0.554 V
Voltage LCB      ≈ 3.960 V
Capacity         ≈ 166.6 mAh/g
Endpoint ΔV/V    ≈ 0.00037
```

最终选择：

```text
CoPO4 → LiCoPO4
```

进行第一性原理验证。

---

## 10. Quantum ESPRESSO DFT Validation

计算设置：

```text
Quantum ESPRESSO 7.5
PBE
spin-polarized

ecutwfc = 70 Ry
ecutrho = 540 Ry

CoPO4 / LiCoPO4:
3 × 3 × 3 k-points

Li metal:
14 × 14 × 14 k-points

vc-relax + final SCF
```

最终能量：

```text
E(CoPO4 cell)
= -1856.45435286 Ry

E(LiCoPO4 cell)
= -1915.41470790 Ry

E(Li metal)
= -14.47195990 Ry / atom
```

得到平均电压：

```text
DFT-PBE Voltage
= 3.6481 V
```

对比：

| Method | Voltage |
|---|---:|
| ML OOF | 4.5139 V |
| Materials Project | 4.3018 V |
| QE-PBE DFT | 3.6481 V |

这里不能简单理解成“ML 比 DFT 更准确”。

当前独立 DFT 使用 plain PBE，而 Co-containing transition-metal compounds 涉及 localized 3d electrons。Materials Project 使用更完整的 database-consistent GGA / GGA+U / correction workflow。

因此本项目将：

```text
3.6481 V
```

定位为：

> **Independent first-principles sanity check**

而不是对 Materials Project 的精确复现。

---

## 11. pymatgen + ASE Automation

为了增强 computational workflow 的自动化和可复现性，项目新增两部分。

### Structure Interoperability

`src/31_ase_structure_bridge.py`

实现：

```text
Materials Project
        ↓
pymatgen Structure
        ↓
ASE Atoms
        ↓
pymatgen Structure
```

CoPO₄ 验证结果：

```text
24 atoms
Formula preserved: True
Atom count preserved: True
Periodic cell preserved
```

### Automated QE Output Parsing

`src/32_qe_output_parser.py`

ASE 直接读取：

```text
CoPO4_final_scf.out
LiCoPO4_final_scf.out
Li_k14.out
```

自动提取：

```text
chemical formula
atom count
final total energy
cell volume
JOB DONE status
```

解析结果：

```text
CoPO4
-25258.34601229 eV

LiCoPO4
-26060.54243921 eV

Li metal
-196.90102806 eV
```

自动计算：

```text
Reaction ΔE
= -14.59231467 eV

ΔE / Li
= -3.64807867 eV

DFT Voltage
= 3.6481 V
```

结果与原先人工读取 QE energies 后计算的电压一致。

因此当前 workflow 已从：

```text
QE output
→ manually copy energy
→ Python
```

升级为：

```text
QE output
→ ASE parser
→ total energy
→ reaction energy
→ battery voltage
```

---

## 12. Project Structure

```text
battery-materials-discovery/
│
├── README.md
├── requirements.txt
├── requirements-gnn.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── xgboost_voltage_final.pkl
│
├── dft/
│   ├── co_po4/
│   ├── li_copo4/
│   ├── li_metal/
│   └── pseudo/
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── qe_parser/
│
└── src/
    ├── 01–09   Data + XGBoost
    ├── 10–14   GNN exploration
    ├── 15–17   Reaction features + SHAP
    ├── 18–22   Physics analysis
    ├── 23–24   Pareto + uncertainty
    ├── 25–30   DFT validation + visualization
    ├── 31_ase_structure_bridge.py
    └── 32_qe_output_parser.py
```

---

## Reproducibility

Main environment:

```bash
pip install -r requirements.txt
```

主要依赖：

```text
numpy
pandas
scikit-learn
xgboost
shap
matminer
pymatgen
mp-api
ase==3.29.0
```

Materials Project API：

```bash
export MP_API_KEY="YOUR_MATERIALS_PROJECT_API_KEY"
```

运行 ASE structure bridge：

```bash
python src/31_ase_structure_bridge.py
```

解析 QE results：

```bash
python src/32_qe_output_parser.py
```

---

## Limitations

当前项目仍有明确边界：

- Materials Project 数据并非真实电池实验数据
- Voltage GroupKFold R² ≈ 0.53，仍存在较大 unexplained variance
- 当前 uncertainty 是 empirical residual uncertainty
- GNN 仅作为 exploratory experiment
- 目前只对一个 shortlist candidate 完成完整 DFT validation
- CoPO₄ / LiCoPO₄ 当前使用 plain PBE，没有进一步进行 DFT+U sensitivity analysis
- ASE 当前用于 structure interoperability 和 QE output parsing，并未声称整个 QE 计算提交过程都由 ASE 自动执行

---

## What This Project Demonstrates

```text
Materials Informatics
Domain-aware Validation
XGBoost
SHAP
Reaction Feature Engineering
Crystal Structure Processing
Physics-based Modeling
Pareto Optimization
Uncertainty-aware Screening
pymatgen
ASE
Quantum ESPRESSO
DFT Validation
Automated Scientific Workflow
```

### Interview Summary

> **我从 Materials Project 获取锂电正极材料数据，通过 framework-aware validation 降低材料数据库中的数据泄漏风险，并利用 XGBoost、reaction features 和 SHAP 建立电压模型。对于容量和体积变化，则根据物理机制分别采用理论公式和 endpoint crystal structures，而不是强行使用机器学习。随后通过 Pareto 和局部经验不确定性筛选 DFT 候选，并使用 Quantum ESPRESSO 对 CoPO₄ → LiCoPO₄ 完成第一性原理验证。最后通过 pymatgen 和 ASE 实现晶体结构互操作和 QE 输出自动解析，将 ML、材料物理与 DFT 串成完整的材料筛选 workflow。**

---

## Core Takeaway

> **Materials discovery is not simply a machine-learning problem.**

更实用的路线是：

```text
Data
+
Domain Knowledge
+
Machine Learning
+
Physics
+
Uncertainty
+
First-principles Calculation
```

最终目标不是预测所有材料，而是：

> **Reduce the search space and prioritize the next high-value calculation or experiment.**