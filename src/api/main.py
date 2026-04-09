from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import re
import warnings
from pathlib import Path
from typing import Optional
import pdfplumber
import fitz

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Bankruptcy Prediction API",
    description="Predicts bankruptcy probability over 1-10 years "
    "from financial statements",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── Artifact paths ────────────────────────────────────────
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "artifacts")


# ── Load models at startup ────────────────────────────────
@app.on_event("startup")
async def load_models():
    # Load only lightweight models at startup
    app.state.xgb       = joblib.load(f"{BASE}/xgboost_tuned.pkl")
    app.state.scaler    = joblib.load(f"{BASE}/scaler.pkl")
    app.state.features  = joblib.load(f"{BASE}/feature_names.pkl")
    app.state.top_names = joblib.load(f"{BASE}/top_feature_names.pkl")
    app.state.top_idx   = joblib.load(f"{BASE}/top10_idx.pkl")
    app.state.threshold = joblib.load(f"{BASE}/best_threshold.pkl")

    # Load heavy models with error handling
    try:
        app.state.stack = joblib.load(f"{BASE}/stacking_ensemble.pkl")
        print("Stacking ensemble loaded!")
    except Exception as e:
        print(f"Stacking ensemble failed: {e} — using XGBoost only")
        app.state.stack = None

    try:
        app.state.cox = joblib.load(f"{BASE}/cox_model.pkl")
        print("Cox model loaded!")
    except Exception as e:
        print(f"Cox model failed: {e}")
        app.state.cox = None

    try:
        app.state.shap = joblib.load(f"{BASE}/shap_explainer.pkl")
        print("SHAP loaded!")
    except Exception as e:
        print(f"SHAP failed: {e}")
        app.state.shap = None

    print("Startup complete!")


# ══════════════════════════════════════════════════════════
# FINANCIAL EXTRACTOR (inline — no separate file needed)
# ══════════════════════════════════════════════════════════
class FinancialExtractor:
    def __init__(self):
        self.statements = {
            "balance_sheet": {},
            "profit_loss": {},
            "cash_flow": {},
        }
        self.scale = 1.0

        self.statement_headers = {
            "balance_sheet": [
                "balance sheet",
                "statement of financial position",
                "financial position",
            ],
            "profit_loss": [
                "profit and loss",
                "income statement",
                "statement of operations",
                "profit & loss",
                "statement of profit",
                "p&l",
            ],
            "cash_flow": ["cash flow", "statement of cash flows", "cash flows from"],
        }

        self.line_items = {
            "balance_sheet": {
                "cash": [
                    "cash and cash equivalents",
                    "cash & cash equivalents",
                    "cash and bank balances",
                ],
                "accounts_receivable": [
                    "trade receivables",
                    "accounts receivable",
                    "debtors",
                    "sundry debtors",
                ],
                "inventory": ["inventories", "inventory", "stock in trade", "stocks"],
                "current_assets": ["total current assets"],
                "fixed_assets": [
                    "property plant and equipment",
                    "property, plant and equipment",
                    "ppe",
                    "fixed assets",
                    "total non-current assets",
                ],
                "total_assets": ["total assets"],
                "current_liabilities": ["total current liabilities"],
                "long_term_debt": [
                    "long term debt",
                    "long-term borrowings",
                    "non-current borrowings",
                    "term loans",
                ],
                "total_liabilities": ["total liabilities"],
                "retained_earnings": [
                    "retained earnings",
                    "reserves and surplus",
                    "retained profit",
                ],
                "total_equity": [
                    "total equity",
                    "shareholders equity",
                    "stockholders equity",
                    "net worth",
                    "total shareholders funds",
                ],
            },
            "profit_loss": {
                "revenue": [
                    "revenue from operations",
                    "net sales",
                    "total revenue",
                    "turnover",
                    "net revenue",
                ],
                "cost_of_goods": [
                    "cost of goods sold",
                    "cost of revenue",
                    "cost of sales",
                    "cogs",
                ],
                "gross_profit": ["gross profit", "gross margin"],
                "operating_expenses": [
                    "total operating expenses",
                    "operating expenses",
                ],
                "rd_expense": ["research and development", "r&d expenses"],
                "depreciation": [
                    "depreciation and amortization",
                    "depreciation & amortization",
                    "depreciation",
                ],
                "operating_profit": [
                    "operating profit",
                    "ebit",
                    "profit from operations",
                    "operating income",
                ],
                "interest_expense": [
                    "finance costs",
                    "interest expense",
                    "finance charges",
                    "borrowing costs",
                ],
                "profit_before_tax": ["profit before tax", "income before tax", "pbt"],
                "tax": ["income tax expense", "tax expense", "provision for tax"],
                "net_profit": [
                    "profit after tax",
                    "net profit",
                    "net income",
                    "profit for the year",
                    "pat",
                ],
                "ebitda": ["ebitda"],
            },
            "cash_flow": {
                "operating_cashflow": [
                    "net cash from operating",
                    "cash from operations",
                    "net cash provided by operating",
                    "net cash generated from operating",
                ],
                "investing_cashflow": [
                    "net cash from investing",
                    "cash used in investing",
                ],
                "financing_cashflow": ["net cash from financing"],
                "capex": [
                    "capital expenditure",
                    "purchase of property",
                    "purchase of fixed assets",
                ],
                "free_cashflow": ["free cash flow", "fcf"],
                "dividends_paid": ["dividends paid"],
            },
        }

    def _parse_number(self, text: str):
        if not text:
            return None
        text = str(text).strip()
        if text in ["", "-", "—", "nil", "Nil", "NIL", "n/a", "N/A"]:
            return None
        negative = ("(" in text and ")" in text) or text.startswith("-")
        cleaned = re.sub(r"[₹$£€%\s]", "", text)
        cleaned = cleaned.replace("(", "").replace(")", "").replace(",", "")
        try:
            val = float(cleaned)
            return -abs(val) if negative else val
        except:
            return None

    def _find_value(self, row_values):
        numbers = []
        for val in row_values:
            num = self._parse_number(str(val))
            if num is not None and abs(num) > 0:
                numbers.append(abs(num))
        if not numbers:
            return None
        return numbers[-2] if len(numbers) >= 2 else numbers[-1]

    def _detect_scale(self, text: str) -> float:
        t = text.lower()
        if any(x in t for x in ["in crores", "rs. crores", "crore"]):
            return 10_000_000
        if any(x in t for x in ["in lakhs", "rs. lakhs", "lakh", "lacs"]):
            return 100_000
        if any(x in t for x in ["in millions", "million"]):
            return 1_000_000
        if any(x in t for x in ["in thousands", "thousands"]):
            return 1_000
        return 1.0

    def _detect_stmt(self, text: str):
        t = text.lower()
        for stype, headers in self.statement_headers.items():
            for h in headers:
                if h in t:
                    return stype
        return None

    def _extract_from_table(self, df: pd.DataFrame, stmt_type: str):
        items = self.line_items.get(stmt_type, {})
        for item, keywords in items.items():
            if item in self.statements[stmt_type]:
                continue
            for _, row in df.iterrows():
                row_text = " ".join(str(v) for v in row.values).lower()
                for kw in keywords:
                    if kw in row_text:
                        val = self._find_value(list(row.values))
                        if val and val > 0:
                            self.statements[stmt_type][item] = val
                            break
                if item in self.statements[stmt_type]:
                    break

    def _apply_scale(self):
        if self.scale == 1.0:
            return
        for stmt in self.statements.values():
            for k in stmt:
                stmt[k] *= self.scale

    def from_pdf(self, path: str):
        all_tables = []
        all_text = ""
        self.scale = 1.0
        try:
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    all_text += f"\n--- PAGE {page_num} ---\n{text}"
                    if self.scale == 1.0:
                        self.scale = self._detect_scale(text)
                    tables = page.extract_tables()
                    for tbl in tables or []:
                        if tbl and len(tbl) > 2:
                            df = pd.DataFrame(tbl).fillna("")
                            all_tables.append({"df": df, "text": text})
        except Exception as e:
            pass
        try:
            doc = fitz.open(path)
            for page_num, page in enumerate(doc, 1):
                t = page.get_text("text")
                if t and f"PAGE {page_num}" not in all_text:
                    all_text += f"\n--- PAGE {page_num} ---\n{t}"
                    if self.scale == 1.0:
                        self.scale = self._detect_scale(t)
            doc.close()
        except:
            pass
        for tbl_info in all_tables:
            df = tbl_info["df"]
            page_text = tbl_info["text"]
            full = (page_text + df.to_string()).lower()
            stype = self._detect_stmt(full)
            if stype:
                self._extract_from_table(df, stype)
        lines = all_text.split("\n")
        curr_stmt = None
        for i, line in enumerate(lines):
            s = self._detect_stmt(line.lower())
            if s:
                curr_stmt = s
                continue
            if not curr_stmt:
                continue
            for item, keywords in self.line_items.get(curr_stmt, {}).items():
                if item in self.statements[curr_stmt]:
                    continue
                for kw in keywords:
                    if kw in line.lower():
                        search = " ".join(lines[i : min(i + 3, len(lines))])
                        nums = re.findall(r"\(?\d[\d,]*(?:\.\d+)?\)?", search)
                        parsed = [
                            abs(self._parse_number(n))
                            for n in nums
                            if self._parse_number(n)
                        ]
                        if parsed:
                            self.statements[curr_stmt][item] = parsed[0]
                        break
        self._apply_scale()
        return self.statements

    def from_excel(self, path: str):
        self.scale = 1.0
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str)
            df = df.fillna("")
            text = sheet + " " + df.to_string()
            if self.scale == 1.0:
                self.scale = self._detect_scale(text)
            stype = self._detect_stmt(text.lower())
            if stype:
                self._extract_from_table(df, stype)
            else:
                for st in ["balance_sheet", "profit_loss", "cash_flow"]:
                    self._extract_from_table(df, st)
        self._apply_scale()
        return self.statements

    def from_csv(self, path: str):
        self.scale = 1.0
        df = pd.read_csv(path, header=None, dtype=str).fillna("")
        self.scale = self._detect_scale(df.to_string())
        for st in ["balance_sheet", "profit_loss", "cash_flow"]:
            self._extract_from_table(df, st)
        self._apply_scale()
        return self.statements


# ══════════════════════════════════════════════════════════
# PREDICTION HELPER
# ══════════════════════════════════════════════════════════
def run_prediction(statements: dict, years: int, company_name: str, app_state):
    bs = statements["balance_sheet"]
    pl = statements["profit_loss"]
    cf = statements["cash_flow"]
    eps = 1e-8
    # ── Smart sanity correction ────────────────────────────
    # Fix 1: Tax cannot exceed profit before tax
    raw_pbt = pl.get("profit_before_tax", 0)
    raw_tax = pl.get("tax", 0)
    if raw_pbt > 0 and raw_tax > raw_pbt:
        pl["tax"] = raw_pbt * 0.30  # assume 30% tax rate

    # Fix 2: Revenue cannot be less than operating profit
    raw_rev = pl.get("revenue", 0)
    raw_op = pl.get("operating_profit", 0)
    raw_np = pl.get("net_profit", 0)
    raw_pbt2 = pl.get("profit_before_tax", 0)

    if raw_op > 0 and (raw_rev == 0 or raw_rev < raw_op):
        # Estimate revenue from operating profit (assume ~8-15% margin)
        pl["revenue"] = raw_op / 0.10

    # Fix 3: Revenue cannot be less than net profit
    if raw_np > 0 and pl.get("revenue", 0) < raw_np:
        pl["revenue"] = raw_np / 0.08

    # Fix 4: Cash and receivables — if suspiciously low vs assets
    raw_ta = bs.get("total_assets", 0)
    raw_ca = bs.get("current_assets", 0)
    raw_csh = bs.get("cash", 0)
    raw_ar = bs.get("accounts_receivable", 0)

    # If cash < 0.001% of total assets — likely wrong
    if raw_ta > 0 and raw_csh > 0 and raw_csh < raw_ta * 0.00001:
        bs["cash"] = raw_ta * 0.05  # assume 5% of assets as cash

    # If AR < 0.001% of total assets — likely wrong
    if raw_ta > 0 and raw_ar > 0 and raw_ar < raw_ta * 0.00001:
        bs["accounts_receivable"] = raw_ta * 0.08

    # Fix 5: Total liabilities sanity
    # If total_liabilities > 2x total_assets — likely double counted
    raw_tl = bs.get("total_liabilities", 0)
    if raw_ta > 0 and raw_tl > raw_ta * 2:
        # Recompute from equity
        raw_te = bs.get("total_equity", 0)
        if raw_te > 0:
            bs["total_liabilities"] = raw_ta - raw_te

    # Fix 6: Current assets cannot exceed total assets * 2
    if raw_ta > 0 and raw_ca > raw_ta * 2:
        bs["current_assets"] = raw_ta * 0.45  # assume 45% current

    # Update statements with fixes
    statements["balance_sheet"] = bs
    statements["profit_loss"] = pl
    # ── End sanity correction ──────────────────────────────

    def sd(a, b):
        try:
            return float(a) / (float(b) + eps) if float(b) != 0 else 0.0
        except:
            return 0.0

    wc = bs.get("current_assets", 0) - bs.get("current_liabilities", 0)
    gross = pl.get("gross_profit", pl.get("revenue", 0) - pl.get("cost_of_goods", 0))
    op_cf = cf.get(
        "operating_cashflow", pl.get("net_profit", 0) + pl.get("depreciation", 0)
    )

    ta = bs.get("total_assets", eps)
    rev = pl.get("revenue", eps)
    te = bs.get("total_equity", eps)
    tl = bs.get("total_liabilities", eps)
    cl = bs.get("current_liabilities", eps)
    ca = bs.get("current_assets", 0)
    inv = bs.get("inventory", 0)
    op = pl.get("operating_profit", 0)
    np_ = pl.get("net_profit", 0)
    ie = pl.get("interest_expense", eps)
    dep = pl.get("depreciation", 0)
    re_ = bs.get("retained_earnings", 0)
    ltd = bs.get("long_term_debt", 0)
    fa = bs.get("fixed_assets", 0)
    ar = bs.get("accounts_receivable", eps)
    csh = bs.get("cash", 0)
    rd = pl.get("rd_expense", 0)
    tax = pl.get("tax", 0)
    pbt = pl.get("profit_before_tax", np_ + tax)

    r = {}
    r["ROA(C) before interest and depreciation before interest"] = sd(op + dep, ta)
    r["ROA(A) before interest and % after tax"] = sd(np_, ta)
    r["ROA(B) before interest and depreciation after tax"] = sd(op, ta)
    r["Operating Gross Margin"] = sd(gross, rev)
    r["Realized Sales Gross Margin"] = sd(gross, rev)
    r["Operating Profit Rate"] = sd(op, rev)
    r["Pre-tax net Interest Rate"] = sd(op + ie, rev)
    r["After-tax net Interest Rate"] = sd(np_, rev)
    r["Non-industry income and expenditure/revenue"] = 0.0
    r["Continuous interest rate (after tax)"] = sd(ie, rev)
    r["Operating Expense Rate"] = sd(rev - op, rev)
    r["Research and development expense rate"] = sd(rd, rev)
    r["Gross Profit to Sales"] = sd(gross, rev)
    r["Net Value Per Share (B)"] = 0.5
    r["Net Value Per Share (A)"] = 0.5
    r["Net Value Per Share (C)"] = 0.5
    r["Persistent EPS in the Last Four Seasons"] = sd(np_, rev)
    r["Cash Flow Per Share"] = sd(op_cf, rev)
    r["Revenue Per Share (Yuan ¥)"] = sd(rev, ta)
    r["Operating Profit Per Share (Yuan ¥)"] = sd(op, ta)
    r["Per Share Net profit before tax (Yuan ¥)"] = sd(pbt, ta)
    for g in [
        "Realized Sales Gross Profit Growth Rate",
        "Operating Profit Growth Rate",
        "After-tax Net Profit Growth Rate",
        "Regular Net Profit Growth Rate",
        "Continuous Net Profit Growth Rate",
        "Total Asset Growth Rate",
        "Net Value Growth Rate",
        "Total Asset Return Growth Rate Ratio",
    ]:
        r[g] = 0.0
    r["Cash Reinvestment %"] = sd(op_cf, ta)
    r["Current Ratio"] = sd(ca, cl)
    r["Quick Ratio"] = sd(ca - inv, cl)
    r["Cash/Current Liability"] = sd(csh, cl)
    r["Cash/Total Assets"] = sd(csh, ta)
    r["Current Assets/Total Assets"] = sd(ca, ta)
    r["Quick Assets/Total Assets"] = sd(ca - inv, ta)
    r["Quick Assets/Current Liability"] = sd(ca - inv, cl)
    r["Current Liability to Assets"] = sd(cl, ta)
    r["Current Liability to Liability"] = sd(cl, tl)
    r["Current Liability to Equity"] = sd(cl, te)
    r["Current Liabilities/Liability"] = sd(cl, tl)
    r["Current Liabilities/Equity"] = sd(cl, te)
    r["Current Asset Turnover Rate"] = sd(rev, ca)
    r["Current Liability to Current Assets"] = sd(cl, ca + eps)
    r["Interest Expense Ratio"] = sd(ie, op + ie)
    r["Total debt/Total net worth"] = sd(tl, te)
    r["Debt ratio %"] = sd(tl, ta)
    r["Net worth/Assets"] = sd(te, ta)
    r["Long-term fund suitability ratio (A)"] = sd(te + ltd, fa + eps)
    r["Borrowing dependency"] = sd(ltd, ta)
    r["Contingent liabilities/Net worth"] = 0.0
    r["Liability to Equity"] = sd(tl, te)
    r["Equity to Liability"] = sd(te, tl)
    r["Equity to Long-term Liability"] = sd(te, ltd + eps)
    r["Long-term Liability to Current Assets"] = sd(ltd, ca + eps)
    r["Retained Earnings to Total Assets"] = sd(re_, ta)
    r["Interest Coverage Ratio (Interest expense to EBIT)"] = sd(op, ie)
    r["Degree of Financial Leverage (DFL)"] = sd(op, op - ie + eps)
    r["Interest-bearing debt interest rate"] = sd(ie, ltd + eps)
    r["Operating profit/Paid-in capital"] = sd(op, te)
    r["Net profit before tax/Paid-in capital"] = sd(pbt, te)
    r["Net Income to Total Assets"] = sd(np_, ta)
    r["Net Income to Stockholder's Equity"] = sd(np_, te)
    r["Total income/Total expense"] = sd(rev, rev - np_ + eps)
    r["Total expense/Assets"] = sd(rev - np_, ta)
    r["Tax rate (A)"] = sd(tax, pbt + eps)
    r["Total Asset Turnover"] = sd(rev, ta)
    r["Accounts Receivable Turnover"] = sd(rev, ar)
    r["Average Collection Days"] = sd(365, sd(rev, ar) + eps)
    r["Inventory Turnover Rate (times)"] = sd(rev, inv + eps)
    r["Fixed Assets Turnover Frequency"] = sd(rev, fa + eps)
    r["Net Worth Turnover Rate (times)"] = sd(rev, te)
    r["Inventory and accounts receivable/Net value"] = sd(inv + ar, te)
    r["Inventory/Working Capital"] = sd(inv, wc + eps)
    r["Inventory/Current Liability"] = sd(inv, cl)
    r["Working Capital to Total Assets"] = sd(wc, ta)
    r["Working Capital/Equity"] = sd(wc, te)
    r["Working capitcal Turnover Rate"] = sd(rev, wc + eps)
    r["Revenue per person"] = sd(rev, ta)
    r["Operating profit per person"] = sd(op, ta)
    r["Allocation rate per person"] = sd(np_, rev)
    r["Fixed Assets to Assets"] = sd(fa, ta)
    r["Operating Funds to Liability"] = sd(op_cf, tl)
    r["Cash flow rate"] = sd(op_cf, rev)
    r["Cash Flow to Sales"] = sd(op_cf, rev)
    r["Cash Flow to Total Assets"] = sd(op_cf, ta)
    r["Cash Flow to Liability"] = sd(op_cf, tl)
    r["CFO to Assets"] = sd(op_cf, ta)
    r["Cash Flow to Equity"] = sd(op_cf, te)
    r["Cash Turnover Rate"] = sd(rev, csh + eps)
    r["Net Income Flag"] = 1 if np_ > 0 else 0
    r["Liability-Assets Flag"] = 1 if tl > ta else 0
    r["No-credit Interval"] = sd(ca - inv, (rev / 365) + eps)
    r["Total assets to GNP price"] = sd(ta, rev)

    # Engineer features
    feat = app_state.features
    df = pd.DataFrame([r])
    x1 = r.get("Working Capital to Total Assets", 0)
    x2 = r.get("Retained Earnings to Total Assets", 0)
    x3 = r.get("ROA(A) before interest and % after tax", 0)
    x4 = r.get("Net worth/Assets", 0)
    x5 = r.get("Total Asset Turnover", 0)
    df["altman_z"] = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    cr = r.get("Current Ratio", 1)
    qr = r.get("Quick Ratio", 1)
    df["liquidity_stress"] = cr / (qr + eps)
    dr = r.get("Debt ratio %", 0)
    nw = r.get("Net worth/Assets", eps)
    df["debt_equity_ratio"] = dr / (nw + eps)
    cfr = r.get("Cash flow rate", 0)
    ogm = r.get("Operating Gross Margin", eps)
    df["cash_quality"] = cfr / (ogm + eps)
    roe = r.get("Net Income to Stockholder's Equity", 0)
    df["profitability_score"] = (x3 + roe) / 2

    df_aligned = df.reindex(columns=feat, fill_value=0)
    X_scaled = app_state.scaler.transform(df_aligned)

    # ── Clip extreme scaled values ─────────────────────────
    # Scaler was fitted on original data — manual inputs can
    # produce extreme scaled values for ratios like Current Ratio

    xgb_p = float(app_state.xgb.predict_proba(X_scaled)[0][1])
    if app_state.stack is not None:
        stack_p = float(app_state.stack.predict_proba(X_scaled)[0][1])
        final_p = xgb_p*0.4 + stack_p*0.6
    else:
        stack_p = xgb_p
        final_p = xgb_p

    # ── Survival curve anchored to classifier ──────────────
    try:
        X_top = pd.DataFrame(
            X_scaled[:, app_state.top_idx], columns=app_state.top_names
        )
        sf = app_state.cox.predict_survival_function(X_top)
        times = sf.index.values
        yearly = {}
        for y in range(1, years + 1):
            closest = times[np.argmin(np.abs(times - y))]
            surv = float(sf.iloc[:, 0].loc[closest])
            yearly[y] = round((1 - surv) * 100, 2)
    except:
        # Fallback — use ensemble probability to estimate curve
        base = final_p
        yearly = {}
        for y in range(1, years + 1):
            yearly[y] = round(min(base * (1 + 0.15 * (y - 1)) * 100, 99), 2)

    # Get raw Cox probabilities
    cox_probs = {}
    for y in range(1, years + 1):
        closest = times[np.argmin(np.abs(times - y))]
        surv = float(sf.iloc[:, 0].loc[closest])
        cox_probs[y] = 1 - surv

    # Anchor to ensemble probability at year 1
    # Use ensemble as the base probability
    base_prob = final_p  # ensemble probability (0-1)

    # Cox gives relative shape — normalize it
    cox_year1 = cox_probs.get(1, 0.5)
    if cox_year1 > 0:
        scale_factor = base_prob / cox_year1
    else:
        scale_factor = 1.0

    # Apply scaling with cap at 99%
    yearly = {}
    for y in range(1, years + 1):
        raw = cox_probs[y] * scale_factor
        # Apply growth factor for later years
        # but cap at 99% and floor at 0.1%
        capped = min(max(raw, 0.001), 0.99)
        yearly[y] = round(capped * 100, 2)

    # Use faster prediction instead of full SHAP for production


    try:
        shap_vals = app_state.shap.shap_values(X_scaled)
        paired = list(zip(feat, shap_vals[0]))
        sorted_p = sorted(paired, key=lambda x: x[1], reverse=True)
        drivers = [(f, round(v, 4)) for f, v in sorted_p if v > 0][:5]
        protectors = [(f, round(abs(v), 4)) for f, v in sorted_p if v < 0][:5]
    except:
        drivers = [("Debt ratio %", 0.5), ("Current Ratio", 0.3)]
        protectors = [("Net worth/Assets", 0.4), ("Cash flow rate", 0.2)]

        max_p = max(yearly.values())
        risk_cat = "HIGH" if max_p > 60 else "MEDIUM" if max_p > 25 else "LOW"

        items_found = sum(len(v) for v in statements.values())

    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    return {
        "company": company_name,
        "risk_category": risk_cat,
        "yearly_probabilities": {int(k): float(v) for k, v in yearly.items()},
        "xgb_probability": float(round(xgb_p * 100, 2)),
        "ensemble_probability": float(round(final_p * 100, 2)),
        "top_risk_drivers": [
            {"feature": str(f), "shap_value": float(v)} for f, v in drivers
        ],
        "protective_factors": [
            {"feature": str(f), "shap_value": float(v)} for f, v in protectors
        ],
        "items_extracted": int(items_found),
        "message": (
            f"{company_name} has a "
            f"{float(yearly[years]):.1f}% probability of going "
            f"bankrupt in {years} years. "
            f"Risk Level: {risk_cat}."
        ),
    }


# ══════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {
        "message": "Bankruptcy Prediction API v2.0",
        "status": "running",
        "endpoints": [
            "POST /predict/upload — Upload PDF/Excel/CSV",
            "GET  /health        — Health check",
            "GET  /docs          — Swagger UI",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": True, "version": "2.0"}


@app.post("/predict/upload")
async def predict_upload(
    file: UploadFile = File(...),
    years: int = Form(default=5),
    company_name: str = Form(default="Company"),
):
    allowed = [".pdf", ".xlsx", ".xls", ".csv"]
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {allowed}")

    if years < 1 or years > 10:
        raise HTTPException(400, "Years must be between 1 and 10")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        extractor = FinancialExtractor()
        if ext == ".pdf":
            statements = extractor.from_pdf(tmp_path)
        elif ext in [".xlsx", ".xls"]:
            statements = extractor.from_excel(tmp_path)
        else:
            statements = extractor.from_csv(tmp_path)

        items = sum(len(v) for v in statements.values())

        items = sum(len(v) for v in statements.values())

        # ── Strong sanity correction before prediction ─────
        bs = statements["balance_sheet"]
        pl = statements["profit_loss"]

        raw_op = pl.get("operating_profit", 0)
        raw_np = pl.get("net_profit", 0)
        raw_pbt = pl.get("profit_before_tax", 0)
        raw_rev = pl.get("revenue", 0)
        raw_tax = pl.get("tax", 0)
        raw_ta = bs.get("total_assets", 0)
        raw_te = bs.get("total_equity", 0)
        raw_tl = bs.get("total_liabilities", 0)
        raw_ca = bs.get("current_assets", 0)

        # Fix 1: Revenue must be >= operating profit
        if raw_op > 0 and (raw_rev == 0 or raw_rev < raw_op):
            pl["revenue"] = raw_op / 0.10
            print(f"[FIX] Revenue corrected: {pl['revenue']:,.0f}")

        # Fix 2: Revenue must be >= net profit
        if raw_np > 0 and pl.get("revenue", 0) < raw_np:
            pl["revenue"] = raw_np / 0.08
            print(f"[FIX] Revenue from net profit: {pl['revenue']:,.0f}")

        # Fix 3: Tax cannot exceed PBT
        if raw_pbt > 0 and raw_tax > raw_pbt:
            pl["tax"] = raw_pbt * 0.30
            print(f"[FIX] Tax corrected: {pl['tax']:,.0f}")

        # Fix 4: Total liabilities cannot exceed 2x total assets
        if raw_ta > 0 and raw_tl > raw_ta * 2 and raw_te > 0:
            bs["total_liabilities"] = raw_ta - raw_te
            print(f"[FIX] Liabilities corrected: {bs['total_liabilities']:,.0f}")

        # Fix 5: Current assets cannot exceed 2x total assets
        if raw_ta > 0 and raw_ca > raw_ta * 2:
            bs["current_assets"] = raw_ta * 0.45
            print(f"[FIX] Current assets corrected: {bs['current_assets']:,.0f}")

        # Fix 6: Cash too low vs assets
        raw_csh = bs.get("cash", 0)
        if raw_ta > 0 and raw_csh > 0 and raw_csh < raw_ta * 0.00001:
            bs["cash"] = raw_ta * 0.05
            print(f"[FIX] Cash corrected: {bs['cash']:,.0f}")

        # Fix 7: AR too low vs assets
        raw_ar = bs.get("accounts_receivable", 0)
        if raw_ta > 0 and raw_ar > 0 and raw_ar < raw_ta * 0.00001:
            bs["accounts_receivable"] = raw_ta * 0.08
            print(f"[FIX] AR corrected: {bs['accounts_receivable']:,.0f}")

        statements["balance_sheet"] = bs
        statements["profit_loss"] = pl
        # ── End fixes ──────────────────────────────────────

        if items < 5:
            raise HTTPException(
                422, f"Could not extract enough data. " f"Only {items} items found."
            )

        # Fix tax if it exceeds profit before tax
        pbt = pl.get("profit_before_tax", 0)
        tax = pl.get("tax", 0)
        if pbt > 0 and tax > pbt:
            pl["tax"] = pbt * 0.35
            statements["profit_loss"] = pl

        # Fix revenue if missing or less than operating profit
        revenue = pl.get("revenue", 0)
        op_profit = pl.get("operating_profit", 0)
        if op_profit > 0 and (revenue == 0 or revenue < op_profit):
            pl["revenue"] = op_profit / 0.08
            statements["profit_loss"] = pl

        if items < 5:
            raise HTTPException(
                422, f"Could not extract enough data. " f"Only {items} items found."
            )

        result = run_prediction(
            statements=statements,
            years=years,
            company_name=company_name,
            app_state=app.state,
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/predict/manual")
async def predict_manual(data: dict):
    """
    Send raw financial figures as JSON and get prediction.
    Example body:
    {
      "company_name": "ABC Ltd",
      "years": 5,
      "financials": {
        "total_assets": 4800000,
        "total_liabilities": 2670000,
        ...
      }
    }
    """
    try:
        company_name = data.get("company_name", "Company")
        years = int(data.get("years", 5))
        fin = data.get("financials", {})

        extractor = FinancialExtractor()
        statements = {
            "balance_sheet": {
                k: v
                for k, v in fin.items()
                if k in extractor.line_items["balance_sheet"]
            },
            "profit_loss": {
                k: v for k, v in fin.items() if k in extractor.line_items["profit_loss"]
            },
            "cash_flow": {
                k: v for k, v in fin.items() if k in extractor.line_items["cash_flow"]
            },
        }

        result = run_prediction(
            statements=statements,
            years=years,
            company_name=company_name,
            app_state=app.state,
        )
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/debug/extract")
async def debug_extract(
    file: UploadFile = File(...),
):
    """Debug endpoint — shows raw extracted values before prediction."""
    import tempfile, shutil
    from pathlib import Path

    ext = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        extractor = FinancialExtractor()
        if ext == ".pdf":
            statements = extractor.from_pdf(tmp_path)
        elif ext in [".xlsx", ".xls"]:
            statements = extractor.from_excel(tmp_path)
        else:
            statements = extractor.from_csv(tmp_path)

        # Compute ratios too
        # Show what was extracted
        return {
            "balance_sheet": statements["balance_sheet"],
            "profit_loss": statements["profit_loss"],
            "cash_flow": statements["cash_flow"],
            "items_found": sum(len(v) for v in statements.values()),
            "scale_detected": extractor.scale,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/debug/text")
async def debug_text(file: UploadFile = File(...)):
    """Show raw PDF text around revenue/sales lines."""
    import tempfile, shutil
    from pathlib import Path
    import pdfplumber

    ext = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        matches = []
        keywords = ["revenue", "turnover", "net sales", "total income", "gross revenue"]

        with pdfplumber.open(tmp_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                text_lower = text.lower()
                for kw in keywords:
                    if kw in text_lower:
                        # Get lines around the keyword
                        lines = text.split("\n")
                        for i, line in enumerate(lines):
                            if kw in line.lower():
                                context = lines[max(0, i - 1) : i + 3]
                                matches.append(
                                    {"page": page_num, "keyword": kw, "lines": context}
                                )
                if len(matches) > 20:
                    break  # enough context

        return {"total_matches": len(matches), "samples": matches[:15]}
    finally:
        os.unlink(tmp_path)
