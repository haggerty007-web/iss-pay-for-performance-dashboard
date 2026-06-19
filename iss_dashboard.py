"""
Real-Time ISS Pay-for-Performance Scoring Dashboard

A Streamlit working example for executive compensation advisors.
This model is a simplified analytical screen inspired by ISS pay-for-performance concepts.
It is not an official ISS model and should not be used as a substitute for ISS policy guidance,
company-specific proxy analysis, or professional judgment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ISS Pay-for-Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .small-note { color: #6b7280; font-size: 0.85rem; }
        .section-header {
            font-size: 1.25rem;
            font-weight: 700;
            border-bottom: 2px solid #4a9eff;
            padding-bottom: 0.35rem;
            margin: 1.25rem 0 0.75rem 0;
        }
        .risk-high { color: #d90429; font-weight: 700; }
        .risk-medium { color: #f59e0b; font-weight: 700; }
        .risk-low { color: #16a34a; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
@dataclass
class CompanyData:
    company_name: str = "MidCap Manufacturing Co."
    ticker: str = "MMC"
    fiscal_year: int = 2024

    # CEO compensation
    ceo_base_salary: float = 1_200_000
    ceo_annual_bonus: float = 1_800_000
    ceo_stock_awards: float = 8_500_000
    ceo_option_awards: float = 2_000_000
    ceo_other_comp: float = 350_000

    # Performance data, expressed as decimals, not percentages
    tsr_1yr: float = 0.08
    tsr_3yr: float = 0.18
    tsr_5yr: float = 0.45
    revenue_growth: float = 0.06
    ebitda_margin: float = 0.18
    eps_growth: float = 0.08
    roic: float = 0.10

    # Plan features
    has_clawback: bool = True
    has_stock_ownership: bool = True
    ownership_multiple: int = 6
    severance_multiple: float = 2.99
    has_single_trigger: bool = False
    has_excise_tax_gross_up: bool = False
    has_relative_tsr: bool = True
    lti_performance_pct: float = 0.60
    sti_has_caps: bool = True
    board_responsiveness: bool = True
    prior_say_on_pay: float = 0.88

    # Qualitative flags
    midcycle_grant: bool = False
    repricing_history: bool = False
    metric_changes: bool = False


class ISSScorer:
    """Simplified scoring engine for pay-for-performance dashboarding."""

    def __init__(self, company: CompanyData, peers: pd.DataFrame):
        self.company = company
        self.peers = peers.copy()

    def calculate_total_compensation(self) -> float:
        c = self.company
        return float(
            c.ceo_base_salary
            + c.ceo_annual_bonus
            + c.ceo_stock_awards
            + c.ceo_option_awards
            + c.ceo_other_comp
        )

    @staticmethod
    def percentile_rank(values: np.ndarray, score: float) -> float:
        values = np.asarray(values, dtype=float)
        return float((np.sum(values < score) + 0.5 * np.sum(values == score)) / len(values) * 100)

    def calculate_pay_rank(self) -> float:
        total_pay = self.calculate_total_compensation()
        all_pays = np.append(self.peers["total_pay"].to_numpy(dtype=float), total_pay)
        return self.percentile_rank(all_pays, total_pay)

    def calculate_tsr_rank(self) -> float:
        company_tsr = self.company.tsr_3yr
        all_tsrs = np.append(self.peers["tsr_3yr"].to_numpy(dtype=float), company_tsr)
        return self.percentile_rank(all_tsrs, company_tsr)

    def calculate_rda_score(self) -> float:
        # Negative values indicate pay rank exceeds performance rank.
        return self.calculate_tsr_rank() - self.calculate_pay_rank()

    def calculate_mom_score(self) -> float:
        peer_median = float(self.peers["total_pay"].median())
        return self.calculate_total_compensation() / max(peer_median, 1.0)

    def calculate_pta_score(self) -> float:
        # Approximation: compares company 5-year TSR against peer 3-year pay growth trend.
        peer_pay_trend = float(self.peers.get("pay_trend_score", pd.Series([0.0])).mean())
        raw = (self.company.tsr_5yr - peer_pay_trend) / 2.0
        return float(np.clip(raw, -1.0, 1.0))

    def quantitative_concern_level(self) -> str:
        rda = self.calculate_rda_score()
        mom = self.calculate_mom_score()
        pta = self.calculate_pta_score()

        high_triggers = int(rda < -30) + int(mom > 2.33) + int(pta < -0.35)
        medium_triggers = int(rda < -20) + int(mom > 1.75) + int(pta < -0.20)

        if high_triggers >= 2:
            return "HIGH"
        if high_triggers == 1 or medium_triggers >= 2:
            return "MEDIUM"
        return "LOW"

    def qualitative_score(self) -> Dict[str, float]:
        c = self.company
        total_pay = max(self.calculate_total_compensation(), 1.0)
        scores: Dict[str, float] = {}

        perf_pay_pct = (c.ceo_annual_bonus + c.ceo_stock_awards + c.ceo_option_awards) / total_pay
        scores["pay_mix"] = min(10.0, perf_pay_pct * 12.0)
        scores["lti_performance"] = min(10.0, c.lti_performance_pct * 14.0)

        gov_score = 10.0
        if c.has_single_trigger:
            gov_score -= 3.0
        if c.has_excise_tax_gross_up:
            gov_score -= 4.0
        if c.severance_multiple > 3.0:
            gov_score -= 2.0
        if c.midcycle_grant:
            gov_score -= 2.0
        if c.repricing_history:
            gov_score -= 3.0
        scores["governance"] = max(0.0, gov_score)

        ownership_score = 0.0
        if c.has_clawback:
            ownership_score += 4.0
        if c.has_stock_ownership:
            ownership_score += 3.0
        if c.ownership_multiple >= 6:
            ownership_score += 3.0
        elif c.ownership_multiple >= 3:
            ownership_score += 1.5
        scores["clawback_ownership"] = min(10.0, ownership_score)

        if c.prior_say_on_pay < 0.70:
            scores["board_response"] = 6.0 if c.board_responsiveness else 1.0
        elif c.prior_say_on_pay < 0.80:
            scores["board_response"] = 7.0 if c.board_responsiveness else 4.0
        else:
            scores["board_response"] = 9.0

        metric_score = 7.0
        if c.has_relative_tsr:
            metric_score += 2.0
        if c.sti_has_caps:
            metric_score += 1.0
        if c.metric_changes:
            metric_score -= 3.0
        scores["metric_rigor"] = min(10.0, max(0.0, metric_score))

        return scores

    def predict_say_on_pay_vote(self) -> float:
        quant = self.quantitative_concern_level()
        avg_qual = float(np.mean(list(self.qualitative_score().values())))
        base_vote = {"LOW": 0.91, "MEDIUM": 0.78, "HIGH": 0.62}[quant]
        qual_adjustment = (avg_qual - 5.0) * 0.015
        predicted = (base_vote + qual_adjustment) * 0.75 + self.company.prior_say_on_pay * 0.25
        return float(np.clip(predicted, 0.30, 0.99))

    def predict_iss_recommendation(self) -> Tuple[str, str]:
        vote_pct = self.predict_say_on_pay_vote()
        quant = self.quantitative_concern_level()
        if quant == "HIGH":
            return "AGAINST", "High quantitative concern"
        if quant == "MEDIUM" and vote_pct < 0.75:
            return "AGAINST", "Medium concern with weak predicted support"
        if quant == "MEDIUM":
            return "FOR, with concerns", "Medium quantitative concern"
        return "FOR", "Low quantitative concern"

    def generate_flags(self) -> List[Dict[str, str]]:
        c = self.company
        rda = self.calculate_rda_score()
        mom = self.calculate_mom_score()
        pta = self.calculate_pta_score()
        flags: List[Dict[str, str]] = []

        def add(severity: str, category: str, issue: str, recommendation: str, impact: str) -> None:
            flags.append(
                {
                    "Severity": severity,
                    "Category": category,
                    "Issue": issue,
                    "Recommendation": recommendation,
                    "Impact": impact,
                }
            )

        if rda < -30:
            add(
                "HIGH",
                "Pay-for-performance",
                f"RDA of {rda:.0f} indicates CEO pay rank materially exceeds TSR rank.",
                "Recalibrate grant values, strengthen performance vesting, or improve disclosure around performance context.",
                "Primary driver of potential negative pay-for-performance review.",
            )
        elif rda < -20:
            add(
                "MEDIUM",
                "Pay-for-performance",
                f"RDA of {rda:.0f} indicates some pay/performance misalignment.",
                "Monitor relative TSR positioning and evaluate forward LTI sizing.",
                "May contribute to a medium quantitative concern.",
            )

        if mom > 2.33:
            add(
                "HIGH",
                "Pay magnitude",
                f"CEO pay is {mom:.1f}x peer median.",
                "Benchmark total compensation against an objective peer group and consider rightsizing LTI awards.",
                "Potential quantitative concern trigger.",
            )
        elif mom > 1.75:
            add(
                "MEDIUM",
                "Pay magnitude",
                f"CEO pay is {mom:.1f}x peer median.",
                "Confirm that pay positioning is supported by size, complexity, performance, and retention rationale.",
                "May increase scrutiny if performance is below median.",
            )

        if pta < -0.35:
            add(
                "HIGH",
                "Pay-TSR alignment",
                f"Pay-TSR alignment score is {pta:.2f}.",
                "Revisit multi-year incentive trajectory and disclose links between pay outcomes and shareholder returns.",
                "Potential long-term alignment concern.",
            )

        if c.has_excise_tax_gross_up:
            add("HIGH", "Governance", "Excise tax gross-up provision is in place.", "Remove or sunset the gross-up provision.", "Negative qualitative factor.")
        if c.has_single_trigger:
            add("MEDIUM", "Change in control", "Single-trigger equity vesting is in place.", "Move to double-trigger vesting acceleration.", "Negative qualitative factor.")
        if c.severance_multiple > 3.0:
            add("MEDIUM", "Severance", f"CEO severance multiple is {c.severance_multiple:.2f}x.", "Reduce severance to 2.0x-2.99x base plus bonus.", "Negative qualitative factor.")
        if c.lti_performance_pct < 0.50:
            add("MEDIUM", "LTI mix", f"Performance-based LTI is {c.lti_performance_pct:.0%} of LTI.", "Increase performance-based LTI to at least 50%-60% of total LTI.", "Reduces qualitative score.")
        if c.midcycle_grant:
            add("MEDIUM", "Equity grants", "Off-cycle or mid-cycle equity grant is present.", "Provide robust disclosure on rationale, sizing, vesting, and one-time nature.", "Often attracts qualitative scrutiny.")
        if c.repricing_history:
            add("HIGH", "Equity plan practices", "Option repricing history is present.", "Avoid repricing without shareholder approval and explain historical facts clearly.", "Negative governance signal.")
        if not c.has_clawback:
            add("MEDIUM", "Clawback", "No clawback policy indicated.", "Adopt and disclose a compliant clawback policy.", "Governance and compliance concern.")
        if c.prior_say_on_pay < 0.70 and not c.board_responsiveness:
            add("HIGH", "Board responsiveness", f"Prior SOP vote was {c.prior_say_on_pay:.0%} without clear responsiveness.", "Engage major holders and disclose specific changes made in response.", "Major negative qualitative factor.")
        if c.metric_changes:
            add("LOW", "Metrics", "Incentive metrics changed year-over-year.", "Explain why the changes support strategy and rigor.", "May raise questions about goal rigor.")

        return flags


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
def generate_peer_data(n_peers: int = 15, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = [
        "Alpha Industrial",
        "Beta Manufacturing",
        "Gamma Corp",
        "Delta Systems",
        "Epsilon Holdings",
        "Zeta Industries",
        "Eta Partners",
        "Theta Group",
        "Iota Capital",
        "Kappa Tech",
        "Lambda Corp",
        "Mu Industries",
        "Nu Holdings",
        "Xi Systems",
        "Omicron Manufacturing",
        "Pi Products",
        "Rho Automation",
        "Sigma Components",
        "Tau Holdings",
        "Upsilon Systems",
    ]
    return pd.DataFrame(
        {
            "company": names[:n_peers],
            "total_pay": rng.lognormal(mean=15.75, sigma=0.38, size=n_peers),
            "tsr_3yr": rng.normal(loc=0.24, scale=0.22, size=n_peers),
            "tsr_1yr": rng.normal(loc=0.10, scale=0.25, size=n_peers),
            "revenue_growth": rng.normal(loc=0.06, scale=0.08, size=n_peers),
            "pay_trend_score": rng.normal(loc=0.08, scale=0.18, size=n_peers),
        }
    )


def parse_peer_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    required = {"company", "total_pay", "tsr_3yr"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Peer CSV is missing required columns: {', '.join(sorted(missing))}")
    for col in ["total_pay", "tsr_3yr", "tsr_1yr", "revenue_growth", "pay_trend_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "tsr_1yr" not in df.columns:
        df["tsr_1yr"] = np.nan
    if "revenue_growth" not in df.columns:
        df["revenue_growth"] = np.nan
    if "pay_trend_score" not in df.columns:
        df["pay_trend_score"] = 0.0
    return df.dropna(subset=["company", "total_pay", "tsr_3yr"])


def format_money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    return f"${value:,.0f}"


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def plot_pay_vs_tsr(company: CompanyData, peers: pd.DataFrame, scorer: ISSScorer) -> go.Figure:
    peer_pay_ranks = [ISSScorer.percentile_rank(peers["total_pay"].to_numpy(), x) for x in peers["total_pay"]]
    peer_tsr_ranks = [ISSScorer.percentile_rank(peers["tsr_3yr"].to_numpy(), x) for x in peers["tsr_3yr"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=peer_pay_ranks,
            y=peer_tsr_ranks,
            mode="markers",
            marker={"size": 10, "opacity": 0.75},
            name="Peers",
            text=peers["company"],
            customdata=np.stack([peers["total_pay"], peers["tsr_3yr"]], axis=-1),
            hovertemplate="<b>%{text}</b><br>Pay rank: %{x:.0f}th<br>TSR rank: %{y:.0f}th<br>Total pay: $%{customdata[0]:,.0f}<br>3Y TSR: %{customdata[1]:.0%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[scorer.calculate_pay_rank()],
            y=[scorer.calculate_tsr_rank()],
            mode="markers",
            marker={"size": 18, "symbol": "star", "line": {"width": 2}},
            name=company.company_name,
            hovertemplate=f"<b>{company.company_name}</b><br>Pay rank: {scorer.calculate_pay_rank():.0f}th<br>TSR rank: {scorer.calculate_tsr_rank():.0f}th<extra></extra>",
        )
    )
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line={"dash": "dash", "width": 1}, name="Alignment line"))
    fig.add_shape(type="rect", x0=50, y0=0, x1=100, y1=50, fillcolor="rgba(255,0,0,0.08)", line={"width": 0})
    fig.add_annotation(x=76, y=22, text="Concern zone", showarrow=False)
    fig.update_layout(
        title="CEO Pay Rank vs. 3-Year TSR Rank",
        xaxis_title="CEO Pay Percentile Rank",
        yaxis_title="3-Year TSR Percentile Rank",
        xaxis={"range": [0, 100]},
        yaxis={"range": [0, 100]},
        height=460,
        legend={"orientation": "h"},
    )
    return fig


def plot_pay_breakdown(company: CompanyData) -> go.Figure:
    labels = ["Base Salary", "Annual Bonus", "Stock Awards", "Option Awards", "Other"]
    values = [
        company.ceo_base_salary,
        company.ceo_annual_bonus,
        company.ceo_stock_awards,
        company.ceo_option_awards,
        company.ceo_other_comp,
    ]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.52, textinfo="label+percent"))
    fig.add_annotation(text=f"{format_money(sum(values))}<br>Total", x=0.5, y=0.5, showarrow=False, font={"size": 16})
    fig.update_layout(title="CEO Pay Mix", height=380, legend={"orientation": "h"})
    return fig


def plot_qualitative_radar(qual_scores: Dict[str, float]) -> go.Figure:
    display = {
        "pay_mix": "Pay Mix",
        "lti_performance": "LTI Performance",
        "governance": "Governance",
        "clawback_ownership": "Clawback / Ownership",
        "board_response": "Board Response",
        "metric_rigor": "Metric Rigor",
    }
    labels = [display.get(k, k) for k in qual_scores]
    values = list(qual_scores.values())
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values_closed, theta=labels_closed, fill="toself", name="Company"))
    fig.add_trace(go.Scatterpolar(r=[8] * len(labels_closed), theta=labels_closed, fill="toself", name="Reference score: 8"))
    fig.update_layout(
        title="Qualitative Factor Scores",
        polar={"radialaxis": {"visible": True, "range": [0, 10]}},
        height=430,
        legend={"orientation": "h"},
    )
    return fig


def plot_vote_gauge(predicted_vote: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=predicted_vote * 100,
            number={"suffix": "%"},
            delta={"reference": 90},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 70], "color": "rgba(220, 38, 38, 0.20)"},
                    {"range": [70, 80], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [80, 100], "color": "rgba(22, 163, 74, 0.20)"},
                ],
                "threshold": {"line": {"width": 3}, "thickness": 0.75, "value": 70},
            },
            title={"text": "Predicted Say-on-Pay FOR Vote"},
        )
    )
    fig.update_layout(height=330)
    return fig


def plot_peer_table(peers: pd.DataFrame) -> pd.DataFrame:
    out = peers.copy()
    out["total_pay"] = out["total_pay"].map(lambda x: f"${x:,.0f}")
    for col in ["tsr_3yr", "tsr_1yr", "revenue_growth", "pay_trend_score"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    return out


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
def build_sidebar() -> Tuple[CompanyData, pd.DataFrame]:
    st.sidebar.header("Company Information")
    c = CompanyData()
    c.company_name = st.sidebar.text_input("Company Name", c.company_name)
    c.ticker = st.sidebar.text_input("Ticker Symbol", c.ticker)
    c.fiscal_year = st.sidebar.number_input("Fiscal Year", min_value=2020, max_value=2030, value=c.fiscal_year, step=1)

    st.sidebar.header("CEO Compensation")
    c.ceo_base_salary = st.sidebar.number_input("Base Salary ($)", min_value=0, value=int(c.ceo_base_salary), step=50_000)
    c.ceo_annual_bonus = st.sidebar.number_input("Annual Bonus ($)", min_value=0, value=int(c.ceo_annual_bonus), step=50_000)
    c.ceo_stock_awards = st.sidebar.number_input("Stock Awards ($)", min_value=0, value=int(c.ceo_stock_awards), step=250_000)
    c.ceo_option_awards = st.sidebar.number_input("Option Awards ($)", min_value=0, value=int(c.ceo_option_awards), step=250_000)
    c.ceo_other_comp = st.sidebar.number_input("Other Compensation ($)", min_value=0, value=int(c.ceo_other_comp), step=10_000)

    st.sidebar.header("Performance Data")
    c.tsr_1yr = st.sidebar.slider("1-Year TSR", -0.50, 1.00, c.tsr_1yr, 0.01, format="%.0f%%")
    c.tsr_3yr = st.sidebar.slider("3-Year TSR", -0.50, 1.50, c.tsr_3yr, 0.01, format="%.0f%%")
    c.tsr_5yr = st.sidebar.slider("5-Year TSR", -0.50, 2.00, c.tsr_5yr, 0.01, format="%.0f%%")
    c.revenue_growth = st.sidebar.slider("Revenue Growth", -0.30, 0.60, c.revenue_growth, 0.01, format="%.0f%%")
    c.eps_growth = st.sidebar.slider("EPS Growth", -0.50, 1.00, c.eps_growth, 0.01, format="%.0f%%")
    c.roic = st.sidebar.slider("ROIC", -0.20, 0.50, c.roic, 0.01, format="%.0f%%")

    st.sidebar.header("Plan Features")
    c.lti_performance_pct = st.sidebar.slider("LTI % Performance-Based", 0.0, 1.0, c.lti_performance_pct, 0.05, format="%.0f%%")
    c.severance_multiple = st.sidebar.slider("Severance Multiple", 1.0, 5.0, c.severance_multiple, 0.25)
    c.ownership_multiple = st.sidebar.slider("Stock Ownership Guideline", 0, 10, c.ownership_multiple)
    c.prior_say_on_pay = st.sidebar.slider("Prior Year SOP Vote", 0.50, 1.00, c.prior_say_on_pay, 0.01, format="%.0f%%")
    c.has_clawback = st.sidebar.checkbox("Clawback Policy", c.has_clawback)
    c.has_stock_ownership = st.sidebar.checkbox("Stock Ownership Guidelines", c.has_stock_ownership)
    c.has_relative_tsr = st.sidebar.checkbox("Relative TSR Metric", c.has_relative_tsr)
    c.sti_has_caps = st.sidebar.checkbox("STI Plan Has Caps", c.sti_has_caps)
    c.board_responsiveness = st.sidebar.checkbox("Board Responsiveness Demonstrated", c.board_responsiveness)

    st.sidebar.header("Governance Flags")
    c.has_single_trigger = st.sidebar.checkbox("Single-Trigger CIC Vesting", c.has_single_trigger)
    c.has_excise_tax_gross_up = st.sidebar.checkbox("Excise Tax Gross-Up", c.has_excise_tax_gross_up)
    c.midcycle_grant = st.sidebar.checkbox("Mid-Cycle / Special Grant", c.midcycle_grant)
    c.repricing_history = st.sidebar.checkbox("Option Repricing History", c.repricing_history)
    c.metric_changes = st.sidebar.checkbox("Metric Changes", c.metric_changes)

    st.sidebar.header("Peer Group")
    uploaded = st.sidebar.file_uploader("Upload peer CSV", type=["csv"])
    if uploaded is not None:
        try:
            peers = parse_peer_upload(uploaded)
            st.sidebar.success(f"Loaded {len(peers)} peers")
        except Exception as exc:
            st.sidebar.error(str(exc))
            peers = generate_peer_data()
    else:
        n_peers = st.sidebar.slider("Sample peer count", 8, 20, 15)
        peers = generate_peer_data(n_peers=n_peers)

    return c, peers


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
def main() -> None:
    company, peers = build_sidebar()
    scorer = ISSScorer(company, peers)

    st.title("Real-Time ISS Pay-for-Performance Scoring Dashboard")
    st.caption("Executive compensation advisory dashboard | Simplified working model")
    st.markdown(
        "<span class='small-note'>This tool approximates pay-for-performance screening concepts for scenario modeling. It is not an official ISS model.</span>",
        unsafe_allow_html=True,
    )

    total_pay = scorer.calculate_total_compensation()
    pay_rank = scorer.calculate_pay_rank()
    tsr_rank = scorer.calculate_tsr_rank()
    rda = scorer.calculate_rda_score()
    mom = scorer.calculate_mom_score()
    pta = scorer.calculate_pta_score()
    quant = scorer.quantitative_concern_level()
    predicted_vote = scorer.predict_say_on_pay_vote()
    recommendation, reason = scorer.predict_iss_recommendation()

    risk_class = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[quant]

    st.markdown("<div class='section-header'>Current Risk Summary</div>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total CEO Pay", format_money(total_pay))
    kpi2.metric("Pay Rank", f"{pay_rank:.0f}th")
    kpi3.metric("TSR Rank", f"{tsr_rank:.0f}th")
    kpi4.markdown(f"**Quantitative Concern**<br><span class='{risk_class}'>{quant}</span>", unsafe_allow_html=True)
    kpi5.metric("Predicted SOP Vote", f"{predicted_vote:.0%}")

    rec1, rec2, rec3 = st.columns([1.2, 1, 1])
    rec1.info(f"Predicted ISS recommendation: **{recommendation}**")
    rec2.metric("RDA", f"{rda:.0f}", help="TSR percentile rank minus pay percentile rank")
    rec3.metric("MoM", f"{mom:.2f}x", help="CEO pay as a multiple of peer median")
    st.caption(reason)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Quantitative Tests", "Qualitative Review", "Flags & Fixes", "Peer Data"])

    with tab1:
        left, right = st.columns([1.3, 1])
        with left:
            st.plotly_chart(plot_pay_vs_tsr(company, peers, scorer), use_container_width=True)
        with right:
            st.plotly_chart(plot_vote_gauge(predicted_vote), use_container_width=True)
        st.plotly_chart(plot_pay_breakdown(company), use_container_width=True)

    with tab2:
        st.subheader("Quantitative Screen")
        quant_df = pd.DataFrame(
            [
                {"Test": "Relative Degree of Alignment", "Result": f"{rda:.0f}", "Interpretation": "Negative = pay rank exceeds TSR rank", "Indicative Concern": "High if below -30"},
                {"Test": "Multiple of Median", "Result": f"{mom:.2f}x", "Interpretation": "CEO pay vs peer median", "Indicative Concern": "High if above 2.33x"},
                {"Test": "Pay-TSR Alignment", "Result": f"{pta:.2f}", "Interpretation": "Approximate long-term alignment indicator", "Indicative Concern": "High if below -0.35"},
            ]
        )
        st.dataframe(quant_df, hide_index=True, use_container_width=True)
        st.plotly_chart(plot_pay_vs_tsr(company, peers, scorer), use_container_width=True)

    with tab3:
        qual_scores = scorer.qualitative_score()
        left, right = st.columns([1.2, 1])
        with left:
            st.plotly_chart(plot_qualitative_radar(qual_scores), use_container_width=True)
        with right:
            qdf = pd.DataFrame({"Factor": list(qual_scores.keys()), "Score": list(qual_scores.values())})
            qdf["Factor"] = qdf["Factor"].str.replace("_", " ").str.title()
            st.dataframe(qdf, hide_index=True, use_container_width=True)
            st.metric("Average Qualitative Score", f"{np.mean(list(qual_scores.values())):.1f} / 10")

    with tab4:
        st.subheader("Concern Flags and Recommendations")
        flags = scorer.generate_flags()
        if not flags:
            st.success("No material concern flags generated under the current assumptions.")
        else:
            flags_df = pd.DataFrame(flags)
            st.dataframe(flags_df, hide_index=True, use_container_width=True)
            st.download_button(
                "Download flags as CSV",
                flags_df.to_csv(index=False).encode("utf-8"),
                file_name="iss_dashboard_flags.csv",
                mime="text/csv",
            )

    with tab5:
        st.subheader("Peer Group Data")
        st.dataframe(plot_peer_table(peers), hide_index=True, use_container_width=True)
        template = pd.DataFrame(
            {
                "company": ["Peer A", "Peer B"],
                "total_pay": [10_000_000, 12_500_000],
                "tsr_3yr": [0.20, 0.35],
                "tsr_1yr": [0.08, 0.15],
                "revenue_growth": [0.05, 0.07],
                "pay_trend_score": [0.06, 0.08],
            }
        )
        st.download_button(
            "Download peer CSV template",
            template.to_csv(index=False).encode("utf-8"),
            file_name="peer_template.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
