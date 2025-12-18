# -*- coding: utf-8 -*-
"""
AI Coding 经济影响预测模型 V3（可调参、可出图、可做 break-even 求解、支持多业务线汇总）
Time cutoff: 2025-12

-----------------
1) 这不是“事实预测”，而是一个“可解释的场景模型 / 可调参的估算器”：
   - 你输入组织参数（人数、全成本、采用率、uplift、工具成本、合规成本、enablement 等）
   - 模型输出净值的区间（P10/P50/P90），并拆解贡献项（劳动产能价值/工具成本/安全合规/返工/外部性等）

2) 为什么有时会出现“负净值”？
   - 采用率低 + exposure 小（可被加速的工作占比低）→ 产能价值小
   - rework（返工/审查/修复）偏高 或 uplift 低/为负（例如熟悉 codebase 的资深工程师可能被 AI 干扰）
   - 安全合规成本（增量 seat + 固定项目）或 enablement 一次性成本过高
   - “产能”无法转化为“交付速度/收入/成本节省”（组织瓶颈、流程、审批、供应链、需求不足）

-----------------
证据（用于默认参数/分布范围的“锚点”，不是对任何组织的保证）
- BLS Software Developers (wage baseline) https://www.bls.gov/ooh/computer-and-information-technology/software-developers.htm
- Fully-loaded cost explanation (example) https://eclub.mit.edu/2015/07/09/fully-loaded-cost-of-an-employee/：
- GitHub Copilot 随机对照实验：特定任务完成时间快 55.8%（上限型证据）：
  https://arxiv.org/abs/2302.06590
- METR 2025：在“熟悉的大型开源代码库”上，允许使用 AI 的组任务耗时反而 +19%：
  https://arxiv.org/abs/2507.09089
- 2024 DORA 报告（Google Cloud blog 摘要）：AI adoption 增加与 throughput -1.5%、stability -7.2% 相关：
  https://cloud.google.com/blog/products/devops-sre/announcing-the-2024-dora-report
- Stack Overflow Developer Survey 2025：84% 使用或计划使用 AI 工具；专业开发者 51% 每日使用：
  https://survey.stackoverflow.co/2025/ai
- McKinsey 2025 State of AI（Exhibit 2/3）：AI agents 在 software engineering/行业中的“达到 scaling”比例（成熟度差异）：
  https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/the%20state%20of%20ai/november%202025/the-state-of-ai-2025-agents-innovation_cmyk-v1.pdf
- NIST SP 800-218A：GenAI/双用途基础模型的安全开发/治理实践（支持“安全合规额外工作”这一成本项）：
  https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218A.pdf
- GitHub Copilot Business 官方价格（seat 成本锚点之一）：$19/seat/月：
  https://docs.github.com/en/billing/concepts/product-billing/github-copilot-licenses

-----------------
运行示例：
  # 1) 单一行业模板 + 自定义人数/采用率
  python econ_model_v3.py simulate --template internet_bigtech --engineers 5000 --adoption_eng 0.65 --n 20000

  # 2) 多业务线：把 TTM（time-to-market）价值按不同业务线 revenue/gm/elasticity 叠加
  python econ_model_v3.py simulate --template internet_bigtech --engineers 5000 --product_lines_json examples/product_lines_example.json --n 30000

  # 3) Break-even：求“净值=0 时所需 adoption（或 uplift）/ 可承受 security 上限”
  python econ_model_v3.py breakeven --template chips_eda --engineers 1200 --variable adoption_eng
  python econ_model_v3.py breakeven --template chips_eda --engineers 1200 --variable uplift_multiplier
  python econ_model_v3.py breakeven --template chips_eda --engineers 1200 --variable security_incremental_per_seat

作者备注：
- 默认模板是“行业平均 + 偏保守”的起点；强烈建议用你组织的历史数据校准：
  * 工程师/QA/SRE 真实 fully-loaded cost
  * 真实 adoption（按周活/日活）、真实 exposure（AI 真正能替代的工时占比）
  * 真实 defect/incident 成本与频率（外部性）
  * 工具链与数据/合规要求导致的真实 security fixed & incremental 成本
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple, List, Optional, Literal
import argparse
import json
import math
import numpy as np

Tri = Tuple[float, float, float]  # (low, mid, high)

def tri_sample(rng: np.random.Generator, tri: Tri, size: int) -> np.ndarray:
    """Triangular distribution sample with support for negative values."""
    low, mid, high = tri
    return rng.triangular(left=low, mode=mid, right=high, size=size)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

@dataclass
class RoleParams:
    """A role in the engineering org."""
    count: int
    fully_loaded_cost: float  # $/year per person (salary+bonus+benefits+tax+overhead)
    adoption: float           # fraction of people that actually use AI tooling effectively
    exposure: float           # fraction of their working time that is in-scope for acceleration
    uplift: Tri               # productivity improvement on the exposed time (can be negative)
    # Optional: role-specific rework penalty multiplier (e.g., junior-heavy orgs might see higher)
    rework_multiplier: float = 1.0

@dataclass
class ToolingCosts:
    """Tooling + enablement costs."""
    seat_cost_per_year: float              # $/adopted seat/year (license or internal infra cost)
    enablement_cost_per_seat: float        # one-time $/seat (training, rollout, playbooks, champions)
    enablement_years: int = 3              # amortization years for enablement (1/2/3/5 ...)
    other_fixed_cost_per_year: float = 0.0 # optional: AI platform team, eval infra, etc.

@dataclass
class SecurityCosts:
    """
    Security & compliance split:
      - incremental per seat (e.g., logging, DLP, vendor add-ons, extra review time)
      - program fixed cost (e.g., procurement+vendor review, risk assessment, policy, red teaming)
    """
    security_incremental_per_seat: float = 0.0      # $/adopted seat/year
    security_program_fixed_cost: float = 0.0        # one-time $ (amortized)
    security_program_years: int = 3                 # amortization years for fixed program

@dataclass
class Externalities:
    """
    Externalities / business-side value beyond “labor capacity”:
      - Faster delivery value (TTM)
      - Defect escape cost reduction (quality)
    Both are modeled as OPTIONAL and can be:
      A) direct dollar value distribution (triangular)
      B) derived from product lines (see ProductLine) and delivery speedup estimate
    """
    # A) direct (if provided, overrides product_lines-derived computation)
    faster_delivery_value: Optional[Tri] = None               # $/year
    defect_escape_cost_reduction: Optional[Tri] = None        # $/year

    # B) derived controls
    delivery_translation: float = 0.45  # how much of engineering capacity gain becomes cycle-time improvement (0~1)

@dataclass
class ProductLine:
    """
    One business line for TTM value estimation.
    - revenue: annual revenue ($)
    - gross_margin: (0~1) gross margin
    - elasticity_per_1pct_speedup: gross profit uplift (%) per 1% cycle-time reduction
      Example: 0.4 => if delivery is 1% faster, gross profit increases by 0.4% (approx.)
    """
    name: str
    revenue: float
    gross_margin: float
    elasticity_per_1pct_speedup: float

@dataclass
class Scenario:
    """Full scenario definition."""
    # Roles
    engineer: RoleParams
    qa: RoleParams
    sre: RoleParams
    pm: RoleParams
    design: RoleParams

    # Costs
    tooling: ToolingCosts
    security: SecurityCosts

    # Risk & friction
    rework_rate: float  # fraction of (affected labor * uplift_gain) that gets negated by verification/redo/bugs

    # Externalities
    externalities: Externalities = field(default_factory=Externalities)
    product_lines: List[ProductLine] = field(default_factory=list)

    # Bookkeeping
    currency: str = "USD"
    time_horizon_years: int = 1  # reporting horizon (net value per year). One-time costs amortized to this horizon.

# -----------------------
# Core simulation
# -----------------------
@dataclass
class SimResult:
    p10: float
    p50: float
    p90: float
    mean: float
    components_p50: Dict[str, float]
    assumptions: Dict[str, float]

def _annualize_one_time(cost: float, years: int) -> float:
    years = max(1, int(years))
    return cost / years

def _role_gross_value(rng: np.random.Generator, role: RoleParams, n: int) -> np.ndarray:
    labor = role.count * role.fully_loaded_cost
    affected = labor * clamp(role.adoption, 0, 1) * clamp(role.exposure, 0, 1)
    uplift = tri_sample(rng, role.uplift, n)
    # uplift applies only on affected time => capacity value
    return affected * uplift

def _estimate_delivery_speedup(eng_gain: float, pm_gain: float, ext: Externalities) -> float:
    """
    Very simple translation from "capacity gain" to "delivery cycle-time speedup".
    - Bottleneck: if PM/requirements can't keep up, effective speedup is limited.
    - ext.delivery_translation controls how much capacity becomes cycle-time.
    """
    # Convert to fractional throughput gain relative to baseline engineering labor
    # Here we treat eng_gain as $ of capacity; baseline labor approximated by eng labor.
    # The ratio is computed outside and passed in as eng_throughput_gain_frac.
    # In this function we only apply bottleneck and translation.
    bottleneck = min(1.0, pm_gain)  # pm_gain already normalized in caller (0~1)
    return clamp(ext.delivery_translation * bottleneck, 0.0, 1.0)

def simulate(s: Scenario, n: int = 20000, seed: int = 42) -> SimResult:
    rng = np.random.default_rng(seed)

    # Role-level gross capacity value (can be negative)
    eng_g = _role_gross_value(rng, s.engineer, n)
    qa_g  = _role_gross_value(rng, s.qa, n)
    sre_g = _role_gross_value(rng, s.sre, n)
    pm_g  = _role_gross_value(rng, s.pm, n)
    des_g = _role_gross_value(rng, s.design, n)

    gross_capacity = eng_g + qa_g + sre_g + pm_g + des_g

    # Rework penalty: apply on positive gross capacity (you don't "rework" negative uplift the same way)
    rework = s.rework_rate * np.maximum(0.0, gross_capacity)

    # Tooling costs (annual)
    adopted_seats = (
        s.engineer.count * clamp(s.engineer.adoption, 0, 1)
        + s.qa.count * clamp(s.qa.adoption, 0, 1)
        + s.sre.count * clamp(s.sre.adoption, 0, 1)
        + s.pm.count * clamp(s.pm.adoption, 0, 1)
        + s.design.count * clamp(s.design.adoption, 0, 1)
    )
    seat_cost = adopted_seats * s.tooling.seat_cost_per_year

    enablement_annual = adopted_seats * _annualize_one_time(s.tooling.enablement_cost_per_seat, s.tooling.enablement_years)
    tooling_fixed = s.tooling.other_fixed_cost_per_year

    # Security costs (annualized)
    security_inc = adopted_seats * s.security.security_incremental_per_seat
    security_prog = _annualize_one_time(s.security.security_program_fixed_cost, s.security.security_program_years)

    # Externalities
    ext_value = np.zeros(n)

    # TTM: direct override OR derived from product lines
    if s.externalities.faster_delivery_value is not None:
        ext_value += tri_sample(rng, s.externalities.faster_delivery_value, n)
    elif s.product_lines:
        # Estimate a delivery speedup fraction based on engineering capacity gain.
        eng_labor = s.engineer.count * s.engineer.fully_loaded_cost
        pm_labor  = s.pm.count * s.pm.fully_loaded_cost
        # Avoid division by zero
        eng_gain_frac = np.maximum(-1.0, np.minimum(1.0, eng_g / max(1.0, eng_labor)))
        pm_gain_frac  = np.maximum(-1.0, np.minimum(1.0, pm_g / max(1.0, pm_labor)))
        # Combine: take positive part and apply translation & bottleneck
        speedup_frac = np.maximum(0.0, eng_gain_frac) * clamp(s.externalities.delivery_translation, 0, 1) * np.maximum(0.0, np.minimum(1.0, 1.0 + pm_gain_frac))
        # speedup_frac is e.g., 0.05 => 5% faster cycle time (approx)
        for pl in s.product_lines:
            gp = pl.revenue * clamp(pl.gross_margin, 0, 1)
            # elasticity_per_1pct_speedup is in %gp per 1% speedup => multiply by (speedup_frac*100)/100 = speedup_frac
            ext_value += gp * pl.elasticity_per_1pct_speedup * speedup_frac

    # Quality: defect escape cost reduction
    if s.externalities.defect_escape_cost_reduction is not None:
        ext_value += tri_sample(rng, s.externalities.defect_escape_cost_reduction, n)

    # Net value distribution
    net = gross_capacity - rework - seat_cost - enablement_annual - tooling_fixed - security_inc - security_prog + ext_value

    def pct(x: np.ndarray, p: float) -> float:
        return float(np.percentile(x, p))

    p10, p50, p90 = pct(net, 10), pct(net, 50), pct(net, 90)
    mean = float(np.mean(net))

    # Component decomposition for p50 using median of each component (note: components are correlated; this is an approximation)
    comp = {
        "gross_capacity_value": float(np.median(gross_capacity)),
        "rework_penalty": -float(np.median(rework)),
        "seat_cost": -float(seat_cost),
        "enablement_annualized": -float(enablement_annual),
        "tooling_fixed": -float(tooling_fixed),
        "security_incremental": -float(security_inc),
        "security_program_annualized": -float(security_prog),
        "externalities_value": float(np.median(ext_value)),
    }

    assumptions = {
        "adopted_seats": float(adopted_seats),
        "enablement_years": float(s.tooling.enablement_years),
        "security_program_years": float(s.security.security_program_years),
        "delivery_translation": float(s.externalities.delivery_translation),
        "rework_rate": float(s.rework_rate),
    }

    return SimResult(p10=p10, p50=p50, p90=p90, mean=mean, components_p50=comp, assumptions=assumptions)

# -----------------------
# Break-even solver
# -----------------------
@dataclass
class BreakEvenResult:
    variable: str
    value: float
    stat: str
    target: float
    achieved: float
    note: str

def _stat_value(net_dist: np.ndarray, stat: str) -> float:
    stat = stat.lower()
    if stat in ("p50", "median"):
        return float(np.percentile(net_dist, 50))
    if stat == "mean":
        return float(np.mean(net_dist))
    if stat == "p10":
        return float(np.percentile(net_dist, 10))
    if stat == "p90":
        return float(np.percentile(net_dist, 90))
    raise ValueError(f"Unknown stat={stat}")

def _simulate_net_only(s: Scenario, n: int, seed: int) -> np.ndarray:
    """Return net distribution for solver (faster: minimal bookkeeping)."""
    rng = np.random.default_rng(seed)

    eng_g = _role_gross_value(rng, s.engineer, n)
    qa_g  = _role_gross_value(rng, s.qa, n)
    sre_g = _role_gross_value(rng, s.sre, n)
    pm_g  = _role_gross_value(rng, s.pm, n)
    des_g = _role_gross_value(rng, s.design, n)
    gross_capacity = eng_g + qa_g + sre_g + pm_g + des_g
    rework = s.rework_rate * np.maximum(0.0, gross_capacity)

    adopted_seats = (
        s.engineer.count * clamp(s.engineer.adoption, 0, 1)
        + s.qa.count * clamp(s.qa.adoption, 0, 1)
        + s.sre.count * clamp(s.sre.adoption, 0, 1)
        + s.pm.count * clamp(s.pm.adoption, 0, 1)
        + s.design.count * clamp(s.design.adoption, 0, 1)
    )
    seat_cost = adopted_seats * s.tooling.seat_cost_per_year
    enablement_annual = adopted_seats * _annualize_one_time(s.tooling.enablement_cost_per_seat, s.tooling.enablement_years)
    tooling_fixed = s.tooling.other_fixed_cost_per_year
    security_inc = adopted_seats * s.security.security_incremental_per_seat
    security_prog = _annualize_one_time(s.security.security_program_fixed_cost, s.security.security_program_years)

    ext_value = np.zeros(n)
    if s.externalities.faster_delivery_value is not None:
        ext_value += tri_sample(rng, s.externalities.faster_delivery_value, n)
    elif s.product_lines:
        eng_labor = s.engineer.count * s.engineer.fully_loaded_cost
        pm_labor  = s.pm.count * s.pm.fully_loaded_cost
        eng_gain_frac = np.maximum(-1.0, np.minimum(1.0, eng_g / max(1.0, eng_labor)))
        pm_gain_frac  = np.maximum(-1.0, np.minimum(1.0, pm_g / max(1.0, pm_labor)))
        speedup_frac = np.maximum(0.0, eng_gain_frac) * clamp(s.externalities.delivery_translation, 0, 1) * np.maximum(0.0, np.minimum(1.0, 1.0 + pm_gain_frac))
        for pl in s.product_lines:
            gp = pl.revenue * clamp(pl.gross_margin, 0, 1)
            ext_value += gp * pl.elasticity_per_1pct_speedup * speedup_frac
    if s.externalities.defect_escape_cost_reduction is not None:
        ext_value += tri_sample(rng, s.externalities.defect_escape_cost_reduction, n)

    net = gross_capacity - rework - seat_cost - enablement_annual - tooling_fixed - security_inc - security_prog + ext_value
    return net

def breakeven(
    s: Scenario,
    variable: Literal["adoption_eng","uplift_multiplier","security_incremental_per_seat","security_program_fixed_cost"],
    *,
    stat: str = "p50",
    target: float = 0.0,
    n: int = 8000,
    seed: int = 123,
    tol: float = 1e-3,
    max_iter: int = 40,
    linked_adoption: bool = True,
) -> BreakEvenResult:
    """
    Solve for knob value such that chosen statistic (e.g., p50) of net value hits target (default 0).
    Notes:
    - We assume monotonicity:
      * adoption_eng ↑ -> net tends to ↑ (not strictly if uplift negative; but in practice we constrain templates so mid uplift >= 0)
      * uplift_multiplier ↑ -> net ↑
      * security costs ↑ -> net ↓
    - Because Monte Carlo is noisy, we use a fixed seed per evaluation to make it stable enough for bisection.
    """

    def eval_at(x: float) -> float:
        ss = json.loads(json.dumps(s, default=lambda o: asdict(o)))  # deep copy via json
        ss = scenario_from_dict(ss)
        if variable == "adoption_eng":
            new_eng = clamp(x, 0.0, 1.0)
            old_eng = ss.engineer.adoption
            ss.engineer.adoption = new_eng
            if linked_adoption:
                # Heuristic: other roles follow engineering adoption with lags/clamps
                ratio = (new_eng / max(1e-6, old_eng)) if old_eng > 0 else 1.0
                ss.qa.adoption     = clamp(ss.qa.adoption * ratio, 0.0, 1.0)
                ss.sre.adoption    = clamp(ss.sre.adoption * ratio, 0.0, 1.0)
                ss.pm.adoption     = clamp(ss.pm.adoption * ratio * 0.85, 0.0, 1.0)
                ss.design.adoption = clamp(ss.design.adoption * ratio * 0.75, 0.0, 1.0)

        elif variable == "uplift_multiplier":
            m = max(0.0, x)
            def mul_tri(t: Tri) -> Tri:
                return (t[0]*m, t[1]*m, t[2]*m)
            ss.engineer.uplift = mul_tri(ss.engineer.uplift)
            ss.qa.uplift       = mul_tri(ss.qa.uplift)
            ss.sre.uplift      = mul_tri(ss.sre.uplift)
            ss.pm.uplift       = mul_tri(ss.pm.uplift)
            ss.design.uplift   = mul_tri(ss.design.uplift)

        elif variable == "security_incremental_per_seat":
            ss.security.security_incremental_per_seat = max(0.0, x)

        elif variable == "security_program_fixed_cost":
            ss.security.security_program_fixed_cost = max(0.0, x)

        net = _simulate_net_only(ss, n=n, seed=seed)
        return _stat_value(net, stat)

    # Pick bounds
    if variable == "adoption_eng":
        lo, hi = 0.05, 0.98
        direction = +1  # higher x => higher metric
    elif variable == "uplift_multiplier":
        lo, hi = 0.2, 5.0
        direction = +1
    elif variable == "security_incremental_per_seat":
        lo, hi = 0.0, 20_000.0
        direction = -1  # higher x => lower metric
    else:
        lo, hi = 0.0, 200_000_000.0
        direction = -1

    f_lo, f_hi = eval_at(lo), eval_at(hi)

    # Ensure bracket
    def is_ok(f: float) -> bool:
        return (f - target) * direction >= 0

    if is_ok(f_lo) and is_ok(f_hi):
        # already above target for both; choose lo as break-even (best-case)
        return BreakEvenResult(variable=variable, value=lo, stat=stat, target=target, achieved=f_lo, note="Already >= target at lower bound.")
    if (not is_ok(f_lo)) and (not is_ok(f_hi)):
        return BreakEvenResult(variable=variable, value=hi, stat=stat, target=target, achieved=f_hi, note="Even at upper bound cannot reach target (or is always below).")

    # Bisection
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = eval_at(mid)
        if abs(f_mid - target) <= tol * max(1.0, abs(target)):
            return BreakEvenResult(variable=variable, value=mid, stat=stat, target=target, achieved=f_mid, note="Converged.")
        if is_ok(f_mid):
            hi = mid
        else:
            lo = mid

    f_mid = eval_at((lo+hi)/2.0)
    return BreakEvenResult(variable=variable, value=(lo+hi)/2.0, stat=stat, target=target, achieved=f_mid, note="Max iterations reached; approximate solution.")

# -----------------------
# Templates + parsing
# -----------------------

def scenario_from_dict(d: Dict) -> Scenario:
    def rp(x): return RoleParams(**x)
    tooling = ToolingCosts(**d["tooling"])
    security = SecurityCosts(**d["security"])
    ext = Externalities(**d.get("externalities", {}))
    pls = [ProductLine(**pl) for pl in d.get("product_lines", [])]
    return Scenario(
        engineer=rp(d["engineer"]),
        qa=rp(d["qa"]),
        sre=rp(d["sre"]),
        pm=rp(d["pm"]),
        design=rp(d["design"]),
        tooling=tooling,
        security=security,
        rework_rate=d["rework_rate"],
        externalities=ext,
        product_lines=pls,
        currency=d.get("currency","USD"),
        time_horizon_years=d.get("time_horizon_years",1),
    )

def to_dict(s: Scenario) -> Dict:
    out = asdict(s)
    return out

def load_product_lines(path: str) -> List[ProductLine]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "product_lines" in data:
        data = data["product_lines"]
    pls = []
    for pl in data:
        pls.append(ProductLine(**pl))
    return pls

def make_templates() -> Dict[str, Scenario]:
    """
    注意：这些模板是“起点”，并且刻意偏保守：
    - uplift 的中位数通常显著低于 Copilot RCT 的 55.8%（因为 RCT 是特定任务的上限）
    - 某些行业允许 uplift 的低端为负（参考 METR 2025 的“可能减速”发现），用于反映：
      * 熟悉 codebase 的专家在 review/校验上花更多时间
      * 安全合规和审批链条带来的额外摩擦
    """

    def role(count, cost, adoption, exposure, uplift):
        return RoleParams(count=count, fully_loaded_cost=cost, adoption=adoption, exposure=exposure, uplift=uplift)

    # Baseline headcount composition (ratios) will be scaled by CLI inputs
    # Default ratios: QA 0.18, SRE 0.10, PM 0.16, Design 0.07 per engineer
    # These are rough; calibrate to your org.
    base = dict(qa_ratio=0.18, sre_ratio=0.10, pm_ratio=0.16, design_ratio=0.07)

    T = {}

    # 1) Internet / Big Tech (高工具成熟度、较高工程成本、较高 adoption、较高 exposure)
    T["internet_bigtech"] = Scenario(
        engineer=role(1000, 320_000, 0.78, 0.55, (0.02, 0.20, 0.38)),
        qa=role(180, 240_000, 0.72, 0.45, (0.01, 0.16, 0.30)),
        sre=role(100, 330_000, 0.70, 0.40, (0.00, 0.14, 0.28)),
        pm=role(160, 310_000, 0.60, 0.25, (0.00, 0.08, 0.16)),
        design=role(70, 280_000, 0.55, 0.20, (-0.02, 0.06, 0.12)),
        tooling=ToolingCosts(seat_cost_per_year=35*12, enablement_cost_per_seat=900, enablement_years=3, other_fixed_cost_per_year=1_500_000),
        security=SecurityCosts(security_incremental_per_seat=400, security_program_fixed_cost=2_000_000, security_program_years=3),
        rework_rate=0.035,
        externalities=Externalities(faster_delivery_value=None, defect_escape_cost_reduction=None, delivery_translation=0.50),
    )

    # 2) 金融（高合规、审计、数据隔离；adoption 较高但 agentic 更慢；安全成本更高）
    T["finance"] = Scenario(
        engineer=role(1000, 300_000, 0.68, 0.50, (-0.03, 0.16, 0.30)),
        qa=role(200, 230_000, 0.62, 0.45, (-0.02, 0.12, 0.26)),
        sre=role(120, 310_000, 0.60, 0.38, (-0.03, 0.10, 0.24)),
        pm=role(150, 290_000, 0.50, 0.22, (-0.02, 0.06, 0.12)),
        design=role(60, 260_000, 0.45, 0.18, (-0.03, 0.05, 0.10)),
        tooling=ToolingCosts(seat_cost_per_year=40*12, enablement_cost_per_seat=1200, enablement_years=3, other_fixed_cost_per_year=1_800_000),
        security=SecurityCosts(security_incremental_per_seat=900, security_program_fixed_cost=5_000_000, security_program_years=3),
        rework_rate=0.045,
        externalities=Externalities(delivery_translation=0.42),
    )

    # 3) 芯片/EDA/半导体（代码和 RTL/EDA 脚本很多不可公开，工具链复杂；合规/IP/验证成本高）
    T["chips_eda"] = Scenario(
        engineer=role(1000, 280_000, 0.55, 0.40, (-0.05, 0.12, 0.24)),
        qa=role(180, 220_000, 0.50, 0.38, (-0.05, 0.10, 0.22)),
        sre=role(90, 290_000, 0.48, 0.35, (-0.05, 0.08, 0.18)),
        pm=role(120, 270_000, 0.40, 0.18, (-0.03, 0.05, 0.10)),
        design=role(60, 250_000, 0.35, 0.15, (-0.04, 0.04, 0.08)),
        tooling=ToolingCosts(seat_cost_per_year=45*12, enablement_cost_per_seat=1500, enablement_years=3, other_fixed_cost_per_year=2_200_000),
        security=SecurityCosts(security_incremental_per_seat=1200, security_program_fixed_cost=6_500_000, security_program_years=3),
        rework_rate=0.055,
        externalities=Externalities(delivery_translation=0.38),
    )

    # 4) 汽车/航空航天（安全关键、合规/认证、流程重；adoption 中等，rework 风险更高）
    T["auto_aero"] = Scenario(
        engineer=role(1000, 240_000, 0.58, 0.45, (-0.04, 0.12, 0.25)),
        qa=role(220, 210_000, 0.55, 0.40, (-0.04, 0.10, 0.22)),
        sre=role(100, 260_000, 0.52, 0.35, (-0.04, 0.08, 0.18)),
        pm=role(130, 230_000, 0.45, 0.20, (-0.03, 0.05, 0.11)),
        design=role(70, 220_000, 0.40, 0.18, (-0.03, 0.05, 0.10)),
        tooling=ToolingCosts(seat_cost_per_year=35*12, enablement_cost_per_seat=1300, enablement_years=3, other_fixed_cost_per_year=1_600_000),
        security=SecurityCosts(security_incremental_per_seat=900, security_program_fixed_cost=4_000_000, security_program_years=3),
        rework_rate=0.055,
        externalities=Externalities(delivery_translation=0.35),
    )

    # 5) 消费品/零售 IT（成本更敏感、工具预算更紧；但业务线多，TTM 价值可能很高）
    T["consumer_retail_it"] = Scenario(
        engineer=role(1000, 200_000, 0.62, 0.50, (-0.02, 0.15, 0.30)),
        qa=role(200, 180_000, 0.58, 0.45, (-0.02, 0.12, 0.25)),
        sre=role(90, 230_000, 0.56, 0.38, (-0.03, 0.10, 0.22)),
        pm=role(180, 210_000, 0.50, 0.25, (-0.02, 0.07, 0.14)),
        design=role(90, 200_000, 0.48, 0.22, (-0.02, 0.07, 0.14)),
        tooling=ToolingCosts(seat_cost_per_year=25*12, enablement_cost_per_seat=900, enablement_years=3, other_fixed_cost_per_year=900_000),
        security=SecurityCosts(security_incremental_per_seat=500, security_program_fixed_cost=2_000_000, security_program_years=3),
        rework_rate=0.045,
        externalities=Externalities(delivery_translation=0.45),
    )

    # 6) 医疗（高合规、隐私；adoption 中低；安全成本高；质量外部性收益潜在更大）
    T["healthcare"] = Scenario(
        engineer=role(1000, 210_000, 0.50, 0.45, (-0.04, 0.12, 0.24)),
        qa=role(220, 190_000, 0.45, 0.42, (-0.04, 0.10, 0.22)),
        sre=role(90, 240_000, 0.45, 0.35, (-0.04, 0.08, 0.18)),
        pm=role(160, 230_000, 0.40, 0.20, (-0.03, 0.05, 0.10)),
        design=role(80, 220_000, 0.38, 0.18, (-0.03, 0.05, 0.10)),
        tooling=ToolingCosts(seat_cost_per_year=30*12, enablement_cost_per_seat=1200, enablement_years=3, other_fixed_cost_per_year=1_000_000),
        security=SecurityCosts(security_incremental_per_seat=1000, security_program_fixed_cost=4_500_000, security_program_years=3),
        rework_rate=0.055,
        externalities=Externalities(delivery_translation=0.33),
    )

    # 7) 政府（采购周期长、数据隔离；但 seat 成本也可能被统一采购拉低）
    T["government"] = Scenario(
        engineer=role(1000, 180_000, 0.42, 0.40, (-0.05, 0.10, 0.20)),
        qa=role(200, 170_000, 0.40, 0.35, (-0.05, 0.08, 0.18)),
        sre=role(80, 210_000, 0.38, 0.30, (-0.05, 0.07, 0.16)),
        pm=role(200, 200_000, 0.35, 0.18, (-0.04, 0.04, 0.09)),
        design=role(70, 190_000, 0.30, 0.15, (-0.04, 0.04, 0.08)),
        tooling=ToolingCosts(seat_cost_per_year=25*12, enablement_cost_per_seat=1200, enablement_years=3, other_fixed_cost_per_year=800_000),
        security=SecurityCosts(security_incremental_per_seat=900, security_program_fixed_cost=5_000_000, security_program_years=3),
        rework_rate=0.060,
        externalities=Externalities(delivery_translation=0.28),
    )

    # 8) 电信/运营商（大型遗留系统+高稳定性要求；adoption 中等，rework 中等）
    T["telecom"] = Scenario(
        engineer=role(1000, 220_000, 0.55, 0.45, (-0.03, 0.13, 0.26)),
        qa=role(220, 190_000, 0.52, 0.42, (-0.03, 0.11, 0.24)),
        sre=role(120, 260_000, 0.50, 0.35, (-0.03, 0.09, 0.20)),
        pm=role(150, 230_000, 0.45, 0.20, (-0.02, 0.05, 0.10)),
        design=role(70, 210_000, 0.40, 0.18, (-0.02, 0.05, 0.10)),
        tooling=ToolingCosts(seat_cost_per_year=30*12, enablement_cost_per_seat=1100, enablement_years=3, other_fixed_cost_per_year=1_200_000),
        security=SecurityCosts(security_incremental_per_seat=700, security_program_fixed_cost=3_000_000, security_program_years=3),
        rework_rate=0.050,
        externalities=Externalities(delivery_translation=0.40),
    )

    return T, base

TEMPLATES, _BASE_RATIOS = make_templates()

def scaled_scenario(template: str, engineers: int, overrides: Dict) -> Scenario:
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template: {template}. Available: {list(TEMPLATES)}")
    base_s = scenario_from_dict(to_dict(TEMPLATES[template]))

    # scale headcount with ratios
    qa = int(round(engineers * _BASE_RATIOS["qa_ratio"]))
    sre = int(round(engineers * _BASE_RATIOS["sre_ratio"]))
    pm = int(round(engineers * _BASE_RATIOS["pm_ratio"]))
    design = int(round(engineers * _BASE_RATIOS["design_ratio"]))

    base_s.engineer.count = int(engineers)
    base_s.qa.count = qa
    base_s.sre.count = sre
    base_s.pm.count = pm
    base_s.design.count = design

    # Apply overrides (flat)
    for k, v in overrides.items():
        if v is None:
            continue
        # role overrides: adoption_*, exposure_*, cost_*
        if k.startswith("adoption_"):
            role = k.replace("adoption_", "")
            getattr(base_s, role).adoption = float(v)
        elif k.startswith("exposure_"):
            role = k.replace("exposure_", "")
            getattr(base_s, role).exposure = float(v)
        elif k.startswith("cost_"):
            role = k.replace("cost_", "")
            getattr(base_s, role).fully_loaded_cost = float(v)
        elif k == "seat_cost_per_year":
            base_s.tooling.seat_cost_per_year = float(v)
        elif k == "enablement_cost_per_seat":
            base_s.tooling.enablement_cost_per_seat = float(v)
        elif k == "enablement_years":
            base_s.tooling.enablement_years = int(v)
        elif k == "other_fixed_cost_per_year":
            base_s.tooling.other_fixed_cost_per_year = float(v)
        elif k == "security_incremental_per_seat":
            base_s.security.security_incremental_per_seat = float(v)
        elif k == "security_program_fixed_cost":
            base_s.security.security_program_fixed_cost = float(v)
        elif k == "security_program_years":
            base_s.security.security_program_years = int(v)
        elif k == "rework_rate":
            base_s.rework_rate = float(v)
        elif k == "delivery_translation":
            base_s.externalities.delivery_translation = float(v)
        elif k == "faster_delivery_value":
            base_s.externalities.faster_delivery_value = tuple(v)
        elif k == "defect_escape_cost_reduction":
            base_s.externalities.defect_escape_cost_reduction = tuple(v)
        else:
            raise ValueError(f"Unknown override key: {k}")

    return base_s

# -----------------------
# CLI
# -----------------------
def _fmt_money(x: float) -> str:
    s = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e9: return f"{s}${x/1e9:.2f}B"
    if x >= 1e6: return f"{s}${x/1e6:.2f}M"
    if x >= 1e3: return f"{s}${x/1e3:.1f}K"
    return f"{s}${x:.0f}"

def cmd_simulate(args: argparse.Namespace) -> None:
    overrides = dict(
        adoption_engineer=args.adoption_eng,
        adoption_qa=args.adoption_qa,
        adoption_sre=args.adoption_sre,
        adoption_pm=args.adoption_pm,
        adoption_design=args.adoption_design,
        exposure_engineer=args.exposure_eng,
        exposure_qa=args.exposure_qa,
        exposure_sre=args.exposure_sre,
        exposure_pm=args.exposure_pm,
        exposure_design=args.exposure_design,
        cost_engineer=args.cost_eng,
        cost_qa=args.cost_qa,
        cost_sre=args.cost_sre,
        cost_pm=args.cost_pm,
        cost_design=args.cost_design,
        seat_cost_per_year=args.seat_cost_per_year,
        enablement_cost_per_seat=args.enablement_cost_per_seat,
        enablement_years=args.enablement_years,
        other_fixed_cost_per_year=args.other_fixed_cost_per_year,
        security_incremental_per_seat=args.security_incremental_per_seat,
        security_program_fixed_cost=args.security_program_fixed_cost,
        security_program_years=args.security_program_years,
        rework_rate=args.rework_rate,
        delivery_translation=args.delivery_translation,
    )

    s = scaled_scenario(args.template, args.engineers, overrides)

    if args.product_lines_json:
        s.product_lines = load_product_lines(args.product_lines_json)

    if args.faster_delivery_value:
        s.externalities.faster_delivery_value = tuple(args.faster_delivery_value)
    if args.defect_escape_cost_reduction:
        s.externalities.defect_escape_cost_reduction = tuple(args.defect_escape_cost_reduction)

    res = simulate(s, n=args.n, seed=args.seed)

    out = {
        "template": args.template,
        "engineers": args.engineers,
        "p10": res.p10,
        "p50": res.p50,
        "p90": res.p90,
        "mean": res.mean,
        "components_p50": res.components_p50,
        "assumptions": res.assumptions,
        "scenario": to_dict(s) if args.dump_scenario else None
    }

    print("\n=== AI Coding ROI (V3) ===")
    print(f"Template: {args.template} | Engineers: {args.engineers} | Seed: {args.seed} | N: {args.n}")
    print(f"Net value: P10={_fmt_money(res.p10)}  P50={_fmt_money(res.p50)}  P90={_fmt_money(res.p90)}  Mean={_fmt_money(res.mean)}")
    print("\n[Median decomposition ~approx]")
    for k, v in res.components_p50.items():
        print(f"  {k:28s}: {_fmt_money(v)}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.output_json}")

def cmd_breakeven(args: argparse.Namespace) -> None:
    overrides = dict(
        adoption_engineer=args.adoption_eng,
        adoption_qa=args.adoption_qa,
        adoption_sre=args.adoption_sre,
        adoption_pm=args.adoption_pm,
        adoption_design=args.adoption_design,
        exposure_engineer=args.exposure_eng,
        exposure_qa=args.exposure_qa,
        exposure_sre=args.exposure_sre,
        exposure_pm=args.exposure_pm,
        exposure_design=args.exposure_design,
        cost_engineer=args.cost_eng,
        cost_qa=args.cost_qa,
        cost_sre=args.cost_sre,
        cost_pm=args.cost_pm,
        cost_design=args.cost_design,
        seat_cost_per_year=args.seat_cost_per_year,
        enablement_cost_per_seat=args.enablement_cost_per_seat,
        enablement_years=args.enablement_years,
        other_fixed_cost_per_year=args.other_fixed_cost_per_year,
        security_incremental_per_seat=args.security_incremental_per_seat,
        security_program_fixed_cost=args.security_program_fixed_cost,
        security_program_years=args.security_program_years,
        rework_rate=args.rework_rate,
        delivery_translation=args.delivery_translation,
    )

    s = scaled_scenario(args.template, args.engineers, overrides)

    if args.product_lines_json:
        s.product_lines = load_product_lines(args.product_lines_json)

    if args.faster_delivery_value:
        s.externalities.faster_delivery_value = tuple(args.faster_delivery_value)
    if args.defect_escape_cost_reduction:
        s.externalities.defect_escape_cost_reduction = tuple(args.defect_escape_cost_reduction)

    be = breakeven(
        s,
        variable=args.variable,
        stat=args.stat,
        target=args.target,
        n=args.n,
        seed=args.seed,
        linked_adoption=(not args.unlinked_adoption),
    )
    print("\n=== Break-even solver (net stat == target) ===")
    print(json.dumps(asdict(be), ensure_ascii=False, indent=2))

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI Coding ROI model V3")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--template", required=True, choices=sorted(TEMPLATES.keys()))
        sp.add_argument("--engineers", type=int, required=True)
        sp.add_argument("--n", type=int, default=20000)
        sp.add_argument("--seed", type=int, default=42)

        # overrides: adoption/exposure/cost
        sp.add_argument("--adoption_eng", type=float, default=None)
        sp.add_argument("--adoption_qa", type=float, default=None)
        sp.add_argument("--adoption_sre", type=float, default=None)
        sp.add_argument("--adoption_pm", type=float, default=None)
        sp.add_argument("--adoption_design", type=float, default=None)

        sp.add_argument("--exposure_eng", type=float, default=None)
        sp.add_argument("--exposure_qa", type=float, default=None)
        sp.add_argument("--exposure_sre", type=float, default=None)
        sp.add_argument("--exposure_pm", type=float, default=None)
        sp.add_argument("--exposure_design", type=float, default=None)

        sp.add_argument("--cost_eng", type=float, default=None)
        sp.add_argument("--cost_qa", type=float, default=None)
        sp.add_argument("--cost_sre", type=float, default=None)
        sp.add_argument("--cost_pm", type=float, default=None)
        sp.add_argument("--cost_design", type=float, default=None)

        sp.add_argument("--seat_cost_per_year", type=float, default=None)
        sp.add_argument("--enablement_cost_per_seat", type=float, default=None)
        sp.add_argument("--enablement_years", type=int, default=None)
        sp.add_argument("--other_fixed_cost_per_year", type=float, default=None)

        sp.add_argument("--security_incremental_per_seat", type=float, default=None)
        sp.add_argument("--security_program_fixed_cost", type=float, default=None)
        sp.add_argument("--security_program_years", type=int, default=None)

        sp.add_argument("--rework_rate", type=float, default=None)
        sp.add_argument("--delivery_translation", type=float, default=None)

        # externalities / product lines
        sp.add_argument("--product_lines_json", type=str, default=None)
        sp.add_argument("--faster_delivery_value", type=float, nargs=3, default=None, metavar=("LOW","MID","HIGH"))
        sp.add_argument("--defect_escape_cost_reduction", type=float, nargs=3, default=None, metavar=("LOW","MID","HIGH"))

    sp1 = sub.add_parser("simulate", help="Run Monte Carlo simulation")
    add_common(sp1)
    sp1.add_argument("--output_json", type=str, default=None)
    sp1.add_argument("--dump_scenario", action="store_true")
    sp1.set_defaults(func=cmd_simulate)

    sp2 = sub.add_parser("breakeven", help="Solve break-even for a chosen variable")
    add_common(sp2)
    sp2.add_argument("--variable", required=True, choices=["adoption_eng","uplift_multiplier","security_incremental_per_seat","security_program_fixed_cost"])
    sp2.add_argument("--stat", type=str, default="p50", choices=["p10","p50","p90","mean"])
    sp2.add_argument("--target", type=float, default=0.0)
    sp2.add_argument("--unlinked_adoption", action="store_true", help="Do not auto-link other roles' adoption to engineering adoption")
    sp2.set_defaults(func=cmd_breakeven)

    return p

def main():
    p = build_parser()
    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
