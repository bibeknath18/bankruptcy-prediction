import { useState } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer
} from "recharts";

const API = "https://bankruptcy-prediction-production.up.railway.app";

interface ShapItem { feature: string; shap_value: number }
interface Result {
  company:              string;
  risk_category:        string;
  yearly_probabilities: { [year: number]: number };
  xgb_probability:      number;
  ensemble_probability: number;
  top_risk_drivers:     ShapItem[];
  protective_factors:   ShapItem[];
  items_extracted:      number;
  message:              string;
}

function riskColor(cat: string) {
  if (cat === "HIGH")   return "#ef4444";
  if (cat === "MEDIUM") return "#f59e0b";
  return "#10b981";
}

function riskBg(cat: string) {
  if (cat === "HIGH")   return "#fef2f2";
  if (cat === "MEDIUM") return "#fffbeb";
  return "#f0fdf4";
}

export default function App() {
  const [file,     setFile]     = useState<File | null>(null);
  const [years,    setYears]    = useState(5);
  const [company,  setCompany]  = useState("");
  const [loading,  setLoading]  = useState(false);
  const [result,   setResult]   = useState<Result | null>(null);
  const [error,    setError]    = useState("");
  const [dragOver, setDragOver] = useState(false);

  async function handleSubmit() {
    if (!file) { setError("Please upload a file first."); return; }
    setLoading(true);
    setError("");
    setResult(null);
    const form = new FormData();
    form.append("file",         file);
    form.append("years",        String(years));
    form.append("company_name", company.trim() || "Unknown Company");
    try {
      const endpoint = file.name.endsWith('.csv') 
        ? `${API}/predict/csv-direct` 
        : `${API}/predict/upload`;
      const res = await axios.post(endpoint, form, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setResult(res.data);
    } catch (e: any) {
      setError(e.response?.data?.detail || "Prediction failed. Check the server.");
    } finally {
      setLoading(false);
    }
  }

  const chartData = result
    ? Object.entries(result.yearly_probabilities).map(([yr, prob]) => ({
        year: `Year ${yr}`, probability: prob
      }))
    : [];

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  }

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      fontFamily: "'Segoe UI', sans-serif",
      padding: "2rem"
    }}>

      {/* ── Header ── */}
      <div style={{ textAlign: "center", marginBottom: "2rem" }}>
        <h1 style={{ fontSize: "2.5rem", fontWeight: 700, color: "#fff", margin: 0 }}>
          BankruptcyGuard
        </h1>
        <p style={{ color: "rgba(255,255,255,0.85)", fontSize: "1.1rem", marginTop: "0.5rem" }}>
          AI-powered bankruptcy risk prediction from financial statements
        </p>
      </div>

      {/* ── Upload Card ── */}
      <div style={{
        maxWidth: 780, margin: "0 auto",
        background: "#fff", borderRadius: 16,
        padding: "2rem", boxShadow: "0 20px 60px rgba(0,0,0,0.15)"
      }}>

        {/* Drop Zone */}
        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById("fileInput")?.click()}
          style={{
            border: `2px dashed ${dragOver ? "#6366f1" : "#d1d5db"}`,
            borderRadius: 12, padding: "2rem",
            textAlign: "center", cursor: "pointer",
            background: dragOver ? "#f0f0ff" : "#fafafa",
            transition: "all 0.2s", marginBottom: "1.5rem"
          }}
        >
          <div style={{ fontSize: "2.5rem" }}>📄</div>
          <p style={{ margin: "0.5rem 0", fontWeight: 600, color: "#374151" }}>
            {file ? file.name : "Drop your financial statement here"}
          </p>
          <p style={{ color: "#9ca3af", fontSize: "0.875rem", margin: 0 }}>
            PDF, Excel (.xlsx), or CSV
          </p>
          <input
            id="fileInput" type="file"
            accept=".pdf,.xlsx,.xls,.csv"
            style={{ display: "none" }}
            onChange={e => setFile(e.target.files?.[0] || null)}
          />
        </div>

        {/* Controls */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1.5rem" }}>
          <div>
            <label style={{ fontWeight: 600, color: "#374151", fontSize: "0.9rem" }}>
              Company Name
            </label>
            <input
              type="text"
              placeholder="e.g. Acme Corp"
              value={company}
              onChange={e => setCompany(e.target.value)}
              style={{
                width: "100%", marginTop: "0.4rem",
                padding: "0.6rem 0.8rem", borderRadius: 8,
                border: "1.5px solid #d1d5db", fontSize: "0.95rem",
                boxSizing: "border-box", outline: "none"
              }}
            />
          </div>
          <div>
            <label style={{ fontWeight: 600, color: "#374151", fontSize: "0.9rem" }}>
              Forecast Years: <span style={{ color: "#6366f1" }}>{years}</span>
            </label>
            <input
              type="range" min={1} max={10} value={years}
              onChange={e => setYears(Number(e.target.value))}
              style={{ width: "100%", marginTop: "0.8rem", accentColor: "#6366f1" }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "#9ca3af" }}>
              <span>1 yr</span><span>10 yrs</span>
            </div>
          </div>
        </div>

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={loading || !file}
          style={{
            width: "100%", padding: "0.85rem",
            background: loading || !file
              ? "#d1d5db"
              : "linear-gradient(135deg, #6366f1, #8b5cf6)",
            color: "#fff", border: "none", borderRadius: 10,
            fontSize: "1rem", fontWeight: 700,
            cursor: loading || !file ? "not-allowed" : "pointer"
          }}
        >
          {loading ? "Analyzing..." : "Predict Bankruptcy Risk"}
        </button>

        {/* Error */}
        {error && (
          <div style={{
            marginTop: "1rem", padding: "0.75rem 1rem",
            background: "#fef2f2", border: "1px solid #fca5a5",
            borderRadius: 8, color: "#dc2626", fontSize: "0.9rem"
          }}>
            {error}
          </div>
        )}
      </div>

      {/* ── Results ── */}
      {result && (
        <div style={{ maxWidth: 780, margin: "2rem auto" }}>

          {/* Risk Banner */}
          <div style={{
            background: riskBg(result.risk_category),
            border: `2px solid ${riskColor(result.risk_category)}`,
            borderRadius: 16, padding: "1.5rem 2rem",
            marginBottom: "1.5rem", textAlign: "center"
          }}>
            <div style={{ fontSize: "3rem" }}>
              {result.risk_category === "HIGH" ? "🔴" :
               result.risk_category === "MEDIUM" ? "🟡" : "🟢"}
            </div>
            <h2 style={{ margin: "0.5rem 0", color: riskColor(result.risk_category), fontSize: "1.8rem" }}>
              {result.risk_category} RISK
            </h2>
            <p style={{ color: "#374151", margin: 0 }}>{result.message}</p>
            <p style={{
              color: "#6b7280", fontSize: "0.78rem",
              marginTop: "0.75rem", padding: "0.5rem 1rem",
              background: "rgba(0,0,0,0.04)", borderRadius: 8
            }}>
              Note: Model trained on Taiwanese SME data (2000-2019).
              Results for large conglomerates may differ from credit agency ratings.
              Use alongside professional financial analysis.
            </p>
          </div>

          {/* Stat Cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "1rem", marginBottom: "1.5rem" }}>
            {[
              { label: "XGBoost Score",    value: `${result.xgb_probability.toFixed(1)}%`,      color: "#6366f1" },
              { label: "Ensemble Score",   value: `${result.ensemble_probability.toFixed(1)}%`,  color: "#8b5cf6" },
              { label: "Data Points Used", value: String(result.items_extracted),                color: "#10b981" },
            ].map(card => (
              <div key={card.label} style={{
                background: "#fff", borderRadius: 12,
                padding: "1.25rem", textAlign: "center",
                boxShadow: "0 4px 20px rgba(0,0,0,0.08)"
              }}>
                <div style={{ fontSize: "1.75rem", fontWeight: 700, color: card.color }}>{card.value}</div>
                <div style={{ color: "#6b7280", fontSize: "0.85rem", marginTop: "0.25rem" }}>{card.label}</div>
              </div>
            ))}
          </div>

          {/* Chart */}
          <div style={{
            background: "#fff", borderRadius: 16,
            padding: "1.5rem", marginBottom: "1.5rem",
            boxShadow: "0 4px 20px rgba(0,0,0,0.08)"
          }}>
            <h3 style={{ margin: "0 0 1rem", color: "#111827" }}>
              Bankruptcy Probability Over Time
            </h3>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                <XAxis dataKey="year" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v: any) => [`${Number(v).toFixed(1)}%`, "Probability"]} />
                <ReferenceLine y={25} stroke="#f59e0b" strokeDasharray="5 5"
                  label={{ value: "Medium", fontSize: 11, fill: "#f59e0b" }} />
                <ReferenceLine y={60} stroke="#ef4444" strokeDasharray="5 5"
                  label={{ value: "High", fontSize: 11, fill: "#ef4444" }} />
                <Line type="monotone" dataKey="probability"
                  stroke="#6366f1" strokeWidth={3}
                  dot={{ fill: "#6366f1", r: 5 }}
                  activeDot={{ r: 7 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* SHAP Panels */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1rem" }}>
            <div style={{ background: "#fff", borderRadius: 16, padding: "1.5rem", boxShadow: "0 4px 20px rgba(0,0,0,0.08)" }}>
              <h3 style={{ margin: "0 0 1rem", fontSize: "1rem", color: "#111827" }}>🔴 Top Risk Drivers</h3>
              {result.top_risk_drivers.map((item, i) => (
                <div key={i} style={{ marginBottom: "0.75rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "0.25rem" }}>
                    <span style={{ color: "#374151", fontWeight: 500 }}>
                      {item.feature.length > 28 ? item.feature.slice(0, 28) + "…" : item.feature}
                    </span>
                    <span style={{ color: "#ef4444", fontWeight: 700 }}>+{item.shap_value.toFixed(3)}</span>
                  </div>
                  <div style={{ background: "#fee2e2", borderRadius: 4, height: 6 }}>
                    <div style={{ background: "#ef4444", borderRadius: 4, height: 6, width: `${Math.min(item.shap_value * 60, 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>

            <div style={{ background: "#fff", borderRadius: 16, padding: "1.5rem", boxShadow: "0 4px 20px rgba(0,0,0,0.08)" }}>
              <h3 style={{ margin: "0 0 1rem", fontSize: "1rem", color: "#111827" }}>🟢 Protective Factors</h3>
              {result.protective_factors.map((item, i) => (
                <div key={i} style={{ marginBottom: "0.75rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "0.25rem" }}>
                    <span style={{ color: "#374151", fontWeight: 500 }}>
                      {item.feature.length > 28 ? item.feature.slice(0, 28) + "…" : item.feature}
                    </span>
                    <span style={{ color: "#10b981", fontWeight: 700 }}>-{item.shap_value.toFixed(3)}</span>
                  </div>
                  <div style={{ background: "#d1fae5", borderRadius: 4, height: 6 }}>
                    <div style={{ background: "#10b981", borderRadius: 4, height: 6, width: `${Math.min(item.shap_value * 60, 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Year Table */}
          <div style={{ background: "#fff", borderRadius: 16, padding: "1.5rem", boxShadow: "0 4px 20px rgba(0,0,0,0.08)" }}>
            <h3 style={{ margin: "0 0 1rem", color: "#111827" }}>Year-by-Year Breakdown</h3>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
              <thead>
                <tr style={{ background: "#f9fafb" }}>
                  {["Year","Probability","Risk Level","Visual"].map(h => (
                    <th key={h} style={{ padding: "0.6rem 1rem", textAlign: "left", color: "#6b7280", fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.yearly_probabilities).map(([yr, prob]) => {
                  const cat = Number(prob) > 60 ? "HIGH" : Number(prob) > 25 ? "MEDIUM" : "LOW";
                  return (
                    <tr key={yr} style={{ borderTop: "1px solid #f3f4f6" }}>
                      <td style={{ padding: "0.6rem 1rem", fontWeight: 600 }}>Year {yr}</td>
                      <td style={{ padding: "0.6rem 1rem", fontWeight: 700, color: riskColor(cat) }}>{Number(prob).toFixed(1)}%</td>
                      <td style={{ padding: "0.6rem 1rem" }}>
                        <span style={{ background: riskBg(cat), color: riskColor(cat), padding: "0.2rem 0.6rem", borderRadius: 6, fontSize: "0.8rem", fontWeight: 600 }}>
                          {cat}
                        </span>
                      </td>
                      <td style={{ padding: "0.6rem 1rem", width: "40%" }}>
                        <div style={{ background: "#f3f4f6", borderRadius: 4, height: 8 }}>
                          <div style={{ background: riskColor(cat), borderRadius: 4, height: 8, width: `${Math.min(Number(prob), 100)}%` }} />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

        </div>
      )}
    </div>
  );
}