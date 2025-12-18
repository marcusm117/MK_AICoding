# AI Coding 经济影响预测模型 V3（README）

> **时间口径**：默认所有金额为 **年化（USD/year）**。  
> **目标**：把“AI coding 的经济影响”从定性争论变成**可复算、可调参、可做敏感性分析**的数值框架。

---

## 0. 这是什么模型（以及它不是什么）

这是一个**可调参的场景模型（scenario model）**：

- **输入**：组织规模（各角色人数）、人员全成本（fully-loaded）、AI 采用率（有效使用）、可被加速的工作占比（exposure）、有效 uplift（生产率变化的区间）、工具成本、enablement（培训/流程/集成）成本、安全/合规额外成本、返工成本等；以及可选的外部性（TTM 与质量）。
- **输出**：净节省/净增产的**区间**（P10/P50/P90/Mean），并提供“瓶颈迁移（capacity → delivery）”的启发式指标与拆解项。

它**不是**：

- 不是对某家公司未来的“事实预测”；
- 不是自动告诉你“会裁多少人/少招多少人”；
- 也不是用公开 benchmark 直接推导 ROI（benchmark ≠ 企业真实仓库 ROI）。

模型的目标是：把“定性争论”变成“可复算、可敏感性分析”的数值框架，帮助你回答：

> 在某个行业/组织约束下，要让 AI coding 的净收益为正，需要哪些条件？哪个因素最关键？  
> 哪些情况下会负净值（亏），亏在什么地方？

---

## 1. 模型的基本思想：把收益拆成“可计量”的四块

在代码里，核心等式是：

\[
\textbf{Net}
=\underbrace{\textbf{Gross}}_{\text{节省工时/产能价值}}
-\Big(\underbrace{\textbf{Tool}}_{\text{seat/推理}}
+\underbrace{\textbf{Enablement}}_{\text{培训/流程/集成（摊销）}}
+\underbrace{\textbf{Security}}_{\text{合规/隔离/审计（拆分+摊销）}}
+\underbrace{\textbf{Rework}}_{\text{AI 引入返工/验证/修复}}\Big)
+\underbrace{\textbf{Externalities}}_{\text{TTM + 质量外部性（可选）}}
\]

其中每一项都对应企业里真正要付的账单/成本中心。

---

## 2. 输入参数（每个参数在现实里代表什么）

以 `econ_model_v3.py` 的数据结构与 CLI 参数为准。模型按角色（Engineer / QA / SRE / PM / Design）显式建模，每个角色都有同构参数：

### 2.1 工程端（Engineers；其他角色同理）

- `fully_loaded_cost`：工程师全成本（年）  
  = 工资 + 奖金 + 福利 + 税费 + 管理摊销 + 办公/设备/IT/间接成本  
  Fully-loaded 的概念解释（示例）：https://eclub.mit.edu/2015/07/09/fully-loaded-cost-of-an-employee/  
  工资基准锚点（示例）：BLS Software Developers https://www.bls.gov/ooh/computer-and-information-technology/software-developers.htm

- `adoption`：实际活跃采用率（不是买 license；是“真的在工作流里用”）  
  普及度锚点（方向性）：Stack Overflow 2025 AI usage https://survey.stackoverflow.co/2025/ai

- `exposure`：工程师工作中可被 AI 加速的占比  
  例如 coding / debug / test / refactor / 脚手架 / 文档等。建议用内部数据校准，而不是凭感觉拍脑袋。

- `uplift=(low, mid, high)`：有效生产率变化，用三角分布做不确定性（允许为负）  
  - 上限型锚点：Copilot RCT 在特定任务上更快 55.8% https://arxiv.org/abs/2302.06590  
  - 下限型锚点：METR 2025 在某些真实代码库任务上 +19% 耗时 https://arxiv.org/abs/2507.09089

### 2.2 PM 端（用于“瓶颈迁移”的数值入口）

把 PM 放进模型不是为了证明“必须多招 PM”，而是为了把“瓶颈会不会迁移到需求/协调/审批”变成可检验的假设：

- PM 的 `adoption/exposure/uplift` 默认更保守；
- 在 **TTM 推导模式**中，PM 的收益会进入 “bottleneck” 折损项（见 4.5）。

---

## 3. 收益项 Gross 是怎么计算的（以及为什么用三角分布）

### 3.1 工程收益（核心）

脚本里（简化表达）：

\[
Gross_{eng}=(Engineers\cdot eng\_cost)\cdot adoption\cdot exposure\cdot U
\]

其中 \(U\) 是 uplift 随机变量。

更一般地，对任一角色 \(r\)：

\[
Gross_r=(Count_r\cdot Cost_r)\cdot Adoption_r\cdot Exposure_r\cdot U_r
\]
\[
Gross=\sum_r Gross_r
\]

### 3.2 为什么用三角分布（Triangular）

模型用 `uplift=(low, mid, high)` 并从三角分布抽样：

`rng.triangular(left=low, mode=mid, right=high, size=n)`

原因：
- 企业真实 uplift 往往没有足够历史数据拟合更复杂分布；
- 但你通常能给出三点：保守下界 / 最可能值 / 乐观上界；
- 三角分布适合“管理层可解释的区间假设”。

### 3.3 Monte Carlo 输出 P10/P50/P90

模型运行 \(n\) 次（例如 10k/20k），得到净值样本，然后输出：

- **P10**：更保守的结果  
- **P50**：中位数（典型情况）  
- **P90**：较乐观情况

---

## 4. 成本项（每一项在现实里代表什么）

### 4.1 Tool：seat/推理成本

\[
Tool = adopted\_seats \cdot seat\_cost\_per\_year
\]

### 4.2 Enablement：多期摊销（升级项）

\[
Enablement = adopted\_seats \cdot \frac{enablement\_cost\_per\_seat}{enablement\_years}
\]

### 4.3 Security：拆分为两类（升级项）

\[
Security=adopted\_seats\cdot security\_incremental\_per\_seat
+\frac{security\_program\_fixed\_cost}{security\_program\_years}
\]

治理/合规任务存在的公开证据：NIST SP 800-218A  
https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218A.pdf

### 4.4 Rework：返工比例

\[
Rework = rework\_rate \cdot \max(0, Gross)
\]

---

## 5. 外部性（TTM 与质量）与“瓶颈迁移”启发式

### 5.1 直接输入外部性

- `faster_delivery_value=(low, mid, high)`：更快上线带来的年化价值（按业务线估值）
- `defect_escape_cost_reduction=(low, mid, high)`：缺陷逃逸/事故成本降低的年化价值（按历史事故成本估值）

### 5.2 多业务线汇总（TTM 叠加）

当你提供 `product_lines_json`（每条业务线有 revenue/gm/elasticity）时，模型按业务线叠加 TTM 价值。

启发式交付提速：

\[
speedup \approx \max(0,eng\_gain\_frac)\cdot delivery\_translation \cdot \max(0, 1+pm\_gain\_frac)
\]

其中：
- \(eng\_gain\_frac\) 是工程“产能价值 / 工程劳动成本”的比值；
- \(pm\_gain\_frac\) 是 PM 端类似的比值（用于表达瓶颈是否迁移到 PM/需求端）；
- `delivery_translation` 明确表达：**产能提升不等于交付等比例提速**。

业务线 \(i\) 的 TTM 价值：

\[
TTM_i=(R_i\cdot GM_i)\cdot e_i \cdot speedup
\]
\[
TTM=\sum_i TTM_i
\]

---

## 6. Break-even 求解器（净值为 0 的阈值）

模型提供 break-even 求解器：自动求

- “净值为 0”时需要的 `adoption_eng`（采用率阈值）
- “净值为 0”时需要的 `uplift_multiplier`（整体 uplift 放大倍率）
- “净值为 0”时可承受的 `security_incremental_per_seat` 上限
- “净值为 0”时可承受的 `security_program_fixed_cost` 上限

方法：二分搜索 + 固定 seed Monte Carlo（目标统计量默认为 P50）。

---

## 7. 为什么会出现“负净值”（不是 bug）

负净值常见来源：

- uplift 低甚至为负（例如 METR 2025 所示的某些场景）；
- 合规/安全固定成本在小规模试点阶段被少量 seats 分摊导致单位成本极高；
- rework/验证链条吞噬产能节省；
- 产能无法转化为交付提速/收入/事故降低（瓶颈迁移到 PM/审批/依赖/需求）。

---



## 1. 你提出的 3 项升级在 V3 里的实现方式

### 1.1 enablement 多期摊销：`enablement_years`

**现实问题：** 培训、规范、评审机制、内部“AI Champion”、流程改造往往不是每年都重复发生，而是一次性投入（或头一两年更集中）。  
**模型实现：**  
- 你输入 `enablement_cost_per_seat`（一次性成本）  
- 用 `enablement_years` 把它折算为年化成本：

\[
\text{enablement\_annual}=\text{adopted\_seats}\times \frac{\text{enablement\_cost\_per\_seat}}{\text{enablement\_years}}
\]

---

### 1.2 security 拆成两类：`security_incremental_per_seat` + `security_program_fixed_cost`

**现实问题：** 合规成本通常分为两类：  
1) **随席位线性增长**：审计日志、DLP、权限/隔离、供应商 add-on、额外代码审查（时间就是钱）  
2) **项目级固定成本**：供应商安全评估、法律审阅、风险评估/红队、政策制定、数据治理、内部安全平台改造

**模型实现：**  
\[
\text{security\_inc}=\text{adopted\_seats}\times \text{security\_incremental\_per\_seat}
\]

\[
\text{security\_prog\_annual}=\frac{\text{security\_program\_fixed\_cost}}{\text{security\_program\_years}}
\]

我们在 README 里把 `security_program_years` 默认设为 3（与 enablement 类似），你也可以改成 1 表示只看“第一年真实现金流”。

**证据（为何要把安全合规成本显式建模）**：  
- NIST SP 800-218A 将 GenAI/双用途基础模型相关的软件开发实践扩展为一组更具体的安全任务和治理要求，说明“额外工作量/额外流程”在现实中真实存在：  
  https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218A.pdf

---

### 1.3 time-to-market 与质量外部性：`faster_delivery_value` + `defect_escape_cost_reduction`

你强调的点非常关键：**即使“劳动产能价值”为正，也可能因为无法转化成业务价值而“净值为负”；反过来，即使产能价值一般，TTM/质量外部性也可能把 ROI 拉回正值。**

V3 里这两项支持两种输入方式：

#### A) 直接输入（按业务线/事故成本估值）
- `faster_delivery_value = (low, mid, high)` 直接输入美元/年  
- `defect_escape_cost_reduction = (low, mid, high)` 直接输入美元/年  

适用：你已经有历史数据（例如：重大故障的年度损失、发布提前带来的 GM 改善等）。

#### B) 多业务线推导（TTM 叠加）：`product_lines_json`
对于 time-to-market，我们也提供“从业务线收入/毛利推导”的方式：

对每条业务线 \(i\)：
- 年收入 \(R_i\)
- 毛利率 \(GM_i\)
- “交付速度每提升 1%，毛利提升多少%” 的弹性 \(e_i\)

先估一个交付速度提升（cycle-time speedup）：
\[
\text{speedup} \approx \max(0, \text{eng\_gain\_frac}) \times \text{delivery\_translation} \times \max(0,1+\text{pm\_gain\_frac})
\]

再计算 TTM 价值并对所有业务线求和：
\[
\text{TTM\_value}=\sum_i (R_i \cdot GM_i) \cdot e_i \cdot \text{speedup}
\]

> **注意：** `delivery_translation`（默认 0.45）非常关键，它体现“产能提升”到“交付提速”之间的组织摩擦/瓶颈；  
> 如果你公司有严格审批、需求不足、或跨团队依赖极重，这个值应该更低。

---

## 2. 模型整体结构与数学定义

### 2.1 Role-level 产能价值（可为负）
每个角色（工程师/QA/SRE/PM/Design）都用同样形式：

- 年劳动成本：\(L = \text{count} \times \text{fully\_loaded\_cost}\)
- AI 影响到的劳动份额：\(L \cdot a \cdot x\)（adoption \(a\)、exposure \(x\)）
- uplift：三角分布 \(\Delta \sim \text{Tri}(low, mid, high)\)

\[
\text{gross\_value}=L \cdot a \cdot x \cdot \Delta
\]

> uplift 允许为负数：这用来表达你关心的“为什么某些行业/场景会负净值”，以及 METR 的发现：在某些任务上 AI 可能让资深开发者变慢。  
> 参考：METR 2025（熟悉开源代码库场景），AI 组任务耗时 +19%：https://arxiv.org/abs/2507.09089

### 2.2 Rework penalty（返工/校验/修 bug）
返工的来源包括：
- 生成代码需要更多 review/测试/修复
- 组织对 AI 输出不信任导致额外验证
- 产生更多小缺陷逃逸到后期

模型：只对正向产能价值施加一个比例：
\[
\text{rework}=\rho \cdot \max(0,\text{gross\_capacity})
\]

### 2.3 Costs：seat / enablement / security / fixed
\[
\text{tool\_cost}=\text{adopted\_seats} \cdot \text{seat\_cost\_per\_year}
\]
\[
\text{enablement\_annual}=\text{adopted\_seats}\cdot \frac{\text{enablement\_cost\_per\_seat}}{\text{enablement\_years}}
\]
\[
\text{security\_inc}=\text{adopted\_seats} \cdot \text{security\_incremental\_per\_seat}
\]
\[
\text{security\_prog\_annual}=\frac{\text{security\_program\_fixed\_cost}}{\text{security\_program\_years}}
\]

### 2.4 Net value
\[
\text{net}= \text{gross\_capacity} - \text{rework} - \text{tool\_cost} - \text{enablement\_annual} - \text{tool\_fixed} - \text{security\_inc} - \text{security\_prog\_annual} + \text{externalities}
\]

模型用 Monte Carlo 抽样（默认 20k 次）输出净值分布的 P10/P50/P90。

---

## 3. 为什么有的行业会出现“负净值”？（更贴近真实世界的解释）

负净值通常不是“AI 不行”，而是**组织/合规/流程**导致“产能 → 价值”的转换失败，或摩擦成本超过收益。

### 3.1 典型原因清单

1) **uplift 低甚至为负（真实存在）**  
- METR 的结果提示：在熟悉大型代码库/高约束任务中，开发者可能花更多时间验证/修正 AI 输出，从而变慢。  
  https://arxiv.org/abs/2507.09089

2) **AI adoption 没有达到“有效采用”**  
- “装了工具 ≠ 真正使用”；组织需要明确允许的场景、数据边界、审计、prompt 规范、代码评审策略。  
- Stack Overflow 2025 报告显示 AI 工具使用很普遍，但并不直接意味着“组织可规模化产出”。  
  https://survey.stackoverflow.co/2025/ai

3) **安全合规成本是硬约束且可非常大**  
- 对金融/医疗/政府/汽车航天而言，合规链条（供应商风险评估、数据隔离、审计、策略制定）会造成固定成本 + 持续性成本。  
- NIST SP 800-218A 的存在本身就是“治理和额外任务”的证据。  
  https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218A.pdf

4) **“产能价值”无法转化为“交付提速/收入提升”**  
- 如果需求/产品策略/审批流程是瓶颈，engineering 的 15% 提升未必会带来 15% 更快上线。  
- 我们用 `delivery_translation` 来显式建模这个衰减。

5) **工具成本在“多工具叠加 + usage 计费”下被低估**  
- 例如，GitHub Copilot Business 为 $19/seat/月（锚点），但企业常会叠加 IDE、agent、RAG、内部平台等成本。  
  https://docs.github.com/en/billing/concepts/product-billing/github-copilot-licenses

### 3.2 为什么某些行业默认更容易负净值（模型层面的“保守设定”）
- 芯片/EDA、汽车航天：验证/认证链条长，rework_rate 默认更高；uplift 低端允许为负；security 成本更高；delivery_translation 更低  
- 金融：security fixed & incremental 更高；uplift 分布更保守  
- 政府：adoption 低，delivery_translation 很低，fixed/security 占比更显著

---

## 4. Break-even 求解器（你要求的“净值=0 阈值”）

V3 支持对以下变量做求解（默认求 P50==0）：

- `adoption_eng`：工程师采用率达到多少，净值为 0
- `uplift_multiplier`：把所有角色 uplift 同比放大/缩小，净值为 0 的倍率
- `security_incremental_per_seat`：每席位安全合规增量成本上限（超过就亏）
- `security_program_fixed_cost`：项目级固定合规成本上限（超过就亏）

实现方法：对变量做**二分搜索**，每个候选点用固定 seed 的 Monte Carlo 估计指定统计量（P50/mean/P10/P90），直到接近 target。

---

## 5. 多业务线汇总（你要求的“按业务线叠加 TTM 价值”）

把多条业务线的 revenue/gm/elasticity 写到 JSON：

`examples/product_lines_example.json`

然后：

```bash
python econ_model_v3.py simulate \
  --template internet_bigtech --engineers 5000 \
  --product_lines_json examples/product_lines_example.json \
  --n 30000
```

模型会把每条业务线的 TTM 价值求和，并加入净值分布中。

---

## 6. 行业模板默认参数的“证据与理由”（你要求的 justify + sources）

> 重要声明：这些不是“行业真值”，只是“有公开证据支撑的区间锚点 + 工程/管理常识”构成的默认起点。  
> 最终要以你组织的真实数据校准。

### 6.1 adoption（采用率）与 agentic 成熟度差异
- Stack Overflow 2025：专业开发者 51% 每日使用 AI，整体 84% 使用或计划使用。  
  https://survey.stackoverflow.co/2025/ai
- McKinsey 2025（Exhibit 2/3）：不同业务函数/行业对“AI agents 达到 scaling”的比例差异巨大（例如 software engineering 在 technology 行业更高）。  
  https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/the%20state%20of%20ai/november%202025/the-state-of-ai-2025-agents-innovation_cmyk-v1.pdf

我们据此在模板里让：
- big tech adoption 更高  
- 金融/医疗/芯片/汽车航天 adoption 低一些，且 delivery_translation 更低（流程更重）

### 6.2 uplift（加速幅度）为何设得比“55.8% 更快”保守？
- Copilot RCT：某个编程任务快 55.8%（上限型证据）。  
  https://arxiv.org/abs/2302.06590
- METR 2025：在熟悉开源代码库任务上 AI 组更慢（-19%）。  
  https://arxiv.org/abs/2507.09089
- DORA 2024 摘要：AI adoption 伴随 throughput -1.5%、stability -7.2% 的统计关系（提示：若流程不改，收益可能被质量/流程摩擦抵消）。  
  https://cloud.google.com/blog/products/devops-sre/announcing-the-2024-dora-report

因此我们把 uplift 分布做成：
- mid uplift 多在 0.10~0.20（比 0.55 保守很多）
- 在合规/验证更重的行业，low 允许为负（反映“可能减速”）

### 6.3 seat/tool cost（工具成本）锚点
- GitHub Copilot Business 官方价格：$19/seat/月。  
  https://docs.github.com/en/billing/concepts/product-billing/github-copilot-licenses

模板中的 $25~$45/seat/月是一个“多工具叠加 + 内部平台/调用成本”的保守估计起点（你可以直接改）。

### 6.4 security & enablement 的默认水平为何不同
- NIST SP 800-218A 描述了 GenAI/双用途基础模型的安全开发实践与任务，支持“额外安全治理/流程成本”必须入模。  
  https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218A.pdf

行业差异的“方向性”依据：
- 金融/医疗/政府/汽车航天：数据敏感、监管要求、审计要求更强 → security fixed & incremental 更高
- 芯片/EDA：IP 极敏感、验证链条长 → security 与 rework 更高

---

## 7. 文件结构

- `econ_model_v3.py`：核心模型 + CLI（simulate / breakeven）
- `create_figures_v3.py`：生成图（net value by template、adoption-uplift heatmap）
- `examples/product_lines_example.json`：多业务线输入示例

---

## 8. 进一步建议（如何用你组织的数据把模型变“准”）

1) **从真实工时拆 exposure**：  
   把工时拆成“coding / debugging / code review / testing / incident / docs / meetings”等，并估每一类的 AI 可加速比例。

2) **用真实 adoption 曲线**：  
   统计 seat 的 DAU/WAU、AI 调用量、以及“有效使用”（例如：PR 中 AI 生成比例 + review 通过率）。

3) **把质量外部性做成你公司的真实数**：  
   历史事故损失（SLA 赔偿、业务损失、回滚/加班、品牌损失）→ `defect_escape_cost_reduction`

4) **把 TTM 价值做成业务线参数**：  
   revenue/gm/elasticity + 交付速度的真实提升（用 lead time、deployment frequency 等 DORA 指标）。

---

## 9. 免责声明

- 本模型用于战略/预算/ROI 讨论，不构成投资建议或保证。
- “行业模板”仅为默认起点；请务必用你自己的数据校准。


---

## 6.5 各行业模板默认值一览（便于审计/复现实验）

> 说明：下面表里的数值来自 `econ_model_v3.py` 的模板（工程师人数在模板里先写成 1000，CLI 会按比例缩放）。  
> 证据引用主要用于：**为何“方向/区间”合理**，并不声称这些就是行业平均。

**关键证据锚点：**
- uplift 上限（任务级）：Copilot RCT +55.8% 时间加速（用于设定 high 端/上限感知）  
  https://arxiv.org/abs/2302.06590
- uplift 下限（可能减速）：METR 2025 在某些真实代码库任务上 +19% 耗时（用于允许 low 为负）  
  https://arxiv.org/abs/2507.09089
- adoption 普遍性：Stack Overflow 2025（84% 使用或计划；51% 每日使用）  
  https://survey.stackoverflow.co/2025/ai
- 行业“agentic 成熟度”差异：McKinsey 2025（Exhibit 3 对 software engineering 的 scaling %）  
  https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/the%20state%20of%20ai/november%202025/the-state-of-ai-2025-agents-innovation_cmyk-v1.pdf
- 基础工资锚点：BLS Occupational Outlook（软件开发者职业工资）  
  https://www.bls.gov/ooh/computer-and-information-technology/software-developers.htm
- fully-loaded（“全成本”）为何要高于 base salary：福利、税、办公、管理、IT 等间接成本  
  一个公开、可引用的说明（示例）：MIT eClub 对 “fully-loaded cost” 的解释  
  https://eclub.mit.edu/2015/07/09/fully-loaded-cost-of-an-employee/

### 6.5.1 模板参数表（工程师=1000 的基准）

| Template | adoption_eng | exposure_eng | uplift_eng (low/mid/high) | rework_rate | seat_cost ($/mo) | enablement ($/seat) | security_inc ($/seat/yr) | security_fixed ($) | delivery_translation |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| internet_bigtech | 0.78 | 0.55 | (0.02,0.20,0.38) | 0.035 | 35 | 900 | 400 | 2.0M | 0.50 |
| finance | 0.68 | 0.50 | (-0.03,0.16,0.30) | 0.045 | 40 | 1200 | 900 | 5.0M | 0.42 |
| chips_eda | 0.55 | 0.40 | (-0.05,0.12,0.24) | 0.055 | 45 | 1500 | 1200 | 6.5M | 0.38 |
| auto_aero | 0.58 | 0.45 | (-0.04,0.12,0.25) | 0.055 | 35 | 1300 | 900 | 4.0M | 0.35 |
| consumer_retail_it | 0.62 | 0.50 | (-0.02,0.15,0.30) | 0.045 | 25 | 900 | 500 | 2.0M | 0.45 |
| healthcare | 0.50 | 0.45 | (-0.04,0.12,0.24) | 0.055 | 30 | 1200 | 1000 | 4.5M | 0.33 |
| government | 0.42 | 0.40 | (-0.05,0.10,0.20) | 0.060 | 25 | 1200 | 900 | 5.0M | 0.28 |
| telecom | 0.55 | 0.45 | (-0.03,0.13,0.26) | 0.050 | 30 | 1100 | 700 | 3.0M | 0.40 |

> seat_cost($/mo) 是 seat_cost_per_year/12 的展示形式，模板里设置为 $25~$45/月区间，参考 Copilot 等价格锚点，并预留“多工具叠加”的空间：  
> https://docs.github.com/en/billing/concepts/product-billing/github-copilot-licenses

### 6.5.2 每个模板“为何这么设”的一句话（带证据指向）

- **internet_bigtech**：adoption 更高，参考开发者 AI 工具普及度（Stack Overflow 2025）；agentic 成熟度更高（McKinsey Exhibit 3 tech 行业 software engineering scaling % 显著高于多数行业）。  
- **finance / healthcare / government / auto_aero**：合规与审计链条更重，因此 security_fixed + incremental 更高，且 uplift 的 low 允许为负以反映“额外验证/返工”可能抵消收益（METR、DORA 提醒）。  
- **chips_eda**：IP 与验证链条更重、开源数据更少，工具可直接加速的 exposure 相对低，rework_rate 更高；McKinsey Exhibit 3 把“advanced manufacturing（含 semiconductors）”的 agent scaling 显著低于 technology，也支持我们把 adoption/translation 设得更保守。  
- **consumer_retail_it**：seat_cost/固定成本更敏感（预算约束），但业务线多、TTM 价值潜在更高，因此建议用 product_lines_json 把多业务线 TTM 外部性显式加回去。  
- **telecom**：遗留系统+稳定性要求 => rework & translation 中等偏保守。

如果你愿意进一步把默认值“改得更像真实世界”，最有效的方法是：把你公司过去 6~12 个月的 DevEx/DORA 指标（lead time、deployment frequency、change failure rate、MTTR）以及事故成本（SLA/业务损失）直接放进外部性部分，而不是依赖模板。

