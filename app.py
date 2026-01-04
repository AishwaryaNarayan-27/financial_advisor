import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import os
import json
import asyncio
import logging
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
import csv
from io import StringIO

import google.generativeai as genai

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Financial Coach",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# GEMINI SETUP (STREAMLIT CLOUD SAFE)
# -------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# -------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -------------------------------------------------------------------
class SpendingCategory(BaseModel):
    category: str
    amount: float
    percentage: Optional[float]

class SpendingRecommendation(BaseModel):
    category: str
    recommendation: str
    potential_savings: Optional[float]

class BudgetAnalysis(BaseModel):
    total_expenses: float
    monthly_income: Optional[float]
    spending_categories: List[SpendingCategory]
    recommendations: List[SpendingRecommendation]

class EmergencyFund(BaseModel):
    recommended_amount: float
    current_amount: Optional[float]
    current_status: str

class SavingsRecommendation(BaseModel):
    category: str
    amount: float
    rationale: Optional[str]

class AutomationTechnique(BaseModel):
    name: str
    description: str

class SavingsStrategy(BaseModel):
    emergency_fund: EmergencyFund
    recommendations: List[SavingsRecommendation]
    automation_techniques: Optional[List[AutomationTechnique]]

class Debt(BaseModel):
    name: str
    amount: float
    interest_rate: float
    min_payment: Optional[float]

class PayoffPlan(BaseModel):
    total_interest: float
    months_to_payoff: int
    monthly_payment: Optional[float]

class PayoffPlans(BaseModel):
    avalanche: PayoffPlan
    snowball: PayoffPlan

class DebtRecommendation(BaseModel):
    title: str
    description: str
    impact: Optional[str]

class DebtReduction(BaseModel):
    total_debt: float
    debts: List[Debt]
    payoff_plans: PayoffPlans
    recommendations: Optional[List[DebtRecommendation]]

# -------------------------------------------------------------------
# LLM PIPELINE (REPLACES ADK)
# -------------------------------------------------------------------
class FinanceAdvisorSystem:
    def __init__(self):
        self.model = model

    def _call_llm(self, system_prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
{system_prompt}

INPUT DATA (JSON):
{json.dumps(payload, indent=2)}

Return ONLY valid JSON. No markdown. No explanations.
"""
        response = self.model.generate_content(prompt)
        return json.loads(response.text)

    async def analyze_finances(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        budget_analysis = self._call_llm(
            system_prompt="""
You are a Budget Analysis Agent.
Categorize expenses, compute totals, percentages,
and give 3–5 concrete savings recommendations.
""",
            payload=financial_data
        )

        savings_strategy = self._call_llm(
            system_prompt="""
You are a Savings Strategy Agent.
Recommend emergency fund size, monthly savings allocation,
and automation techniques.
""",
            payload={
                "financial_data": financial_data,
                "budget_analysis": budget_analysis
            }
        )

        debt_reduction = self._call_llm(
            system_prompt="""
You are a Debt Reduction Agent.
Create avalanche and snowball payoff plans,
estimate interest and payoff duration.
""",
            payload={
                "financial_data": financial_data,
                "budget_analysis": budget_analysis,
                "savings_strategy": savings_strategy
            }
        )

        return {
            "budget_analysis": budget_analysis,
            "savings_strategy": savings_strategy,
            "debt_reduction": debt_reduction
        }

# -------------------------------------------------------------------
# CSV HELPERS
# -------------------------------------------------------------------
def validate_csv_format(file):
    content = file.read().decode("utf-8")
    file.seek(0)
    df = pd.read_csv(StringIO(content))
    for col in ["Date", "Category", "Amount"]:
        if col not in df.columns:
            return False, f"Missing column: {col}"
    df["Amount"] = df["Amount"].replace(r"[\$,]", "", regex=True).astype(float)
    return True, "Valid CSV"

def parse_csv_transactions(file):
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["Amount"] = df["Amount"].replace(r"[\$,]", "", regex=True).astype(float)
    return df

# -------------------------------------------------------------------
# DISPLAY FUNCTIONS
# -------------------------------------------------------------------
def display_budget_analysis(data):
    st.subheader("Spending Breakdown")
    fig = px.pie(
        values=[c["amount"] for c in data["spending_categories"]],
        names=[c["category"] for c in data["spending_categories"]]
    )
    st.plotly_chart(fig)

def display_savings_strategy(data):
    st.subheader("Savings Strategy")
    ef = data["emergency_fund"]
    st.metric("Emergency Fund Target", f"${ef['recommended_amount']:.2f}")
    for r in data["recommendations"]:
        st.write(f"**{r['category']}** — ${r['amount']:.2f}")

def display_debt_reduction(data):
    st.subheader("Debt Reduction")
    st.metric("Total Debt", f"${data['total_debt']:.2f}")

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
def main():
    st.title("💸 AI Financial Coach (Gemini-powered)")

    if not GEMINI_API_KEY:
        st.error("GOOGLE_API_KEY missing. Add it in Streamlit Secrets.")
        return

    income = st.number_input("Monthly Income", min_value=0.0, value=3000.0)
    dependants = st.number_input("Dependants", min_value=0, value=0)

    upload = st.file_uploader("Upload Expenses CSV", type=["csv"])
    transactions = None

    if upload:
        valid, msg = validate_csv_format(upload)
        if valid:
            transactions = parse_csv_transactions(upload)
            st.success("CSV loaded")
            st.dataframe(transactions.head())
        else:
            st.error(msg)

    analyze = st.button("Analyze My Finances")

    if analyze:
        payload = {
            "monthly_income": income,
            "dependants": dependants,
            "transactions": transactions.to_dict("records") if transactions is not None else [],
            "manual_expenses": None,
            "debts": []
        }

        system = FinanceAdvisorSystem()

        with st.spinner("Analyzing..."):
            results = asyncio.run(system.analyze_finances(payload))

        tabs = st.tabs(["Budget", "Savings", "Debt"])

        with tabs[0]:
            display_budget_analysis(results["budget_analysis"])

        with tabs[1]:
            display_savings_strategy(results["savings_strategy"])

        with tabs[2]:
            display_debt_reduction(results["debt_reduction"])

if __name__ == "__main__":
    main()
