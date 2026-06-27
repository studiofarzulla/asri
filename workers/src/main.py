"""
ASRI API - Cloudflare Python Workers Edition
Aggregated Systemic Risk Index for crypto/DeFi systemic risk monitoring

Uses native Workers API (no FastAPI - not available in Pyodide)
"""

from datetime import datetime
import json
from js import Response, Headers

ASRI_WEIGHTS = {
    "stablecoin_risk": 0.30,
    "defi_liquidity_risk": 0.25,
    "contagion_risk": 0.25,
    "arbitrage_opacity": 0.20,
}


# ============== Helper Functions ==============

def json_response(data: dict, status: int = 200) -> Response:
    """Create a JSON response."""
    headers = Headers.new(
        {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        }.items()
    )
    return Response.new(json.dumps(data), status=status, headers=headers)


def get_alert_level(asri: float) -> str:
    """Determine alert level based on ASRI value."""
    if asri < 30:
        return "low"
    elif asri < 50:
        return "moderate"
    elif asri < 70:
        return "elevated"
    return "critical"


def get_trend(current: float, avg: float) -> str:
    """Determine trend direction."""
    diff = current - avg
    if diff > 2:
        return "rising"
    elif diff < -2:
        return "falling"
    return "stable"


def compute_asri(sub_indices: dict) -> float:
    """Compute aggregate ASRI from sub-index components."""
    value = (
        ASRI_WEIGHTS["stablecoin_risk"] * sub_indices["stablecoin_risk"]
        + ASRI_WEIGHTS["defi_liquidity_risk"] * sub_indices["defi_liquidity_risk"]
        + ASRI_WEIGHTS["contagion_risk"] * sub_indices["contagion_risk"]
        + ASRI_WEIGHTS["arbitrage_opacity"] * sub_indices["arbitrage_opacity"]
    )
    return round(value, 1)


def options_response() -> Response:
    """Handle CORS preflight."""
    headers = Headers.new(
        {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        }.items()
    )
    return Response.new("", status=204, headers=headers)


# ============== Handlers ==============

def handle_root():
    """Root endpoint."""
    return json_response({
        "name": "ASRI API",
        "version": "2.1.0",
        "description": "Aggregated Systemic Risk Index",
        "canonical_base": "https://asri.dissensus.ai/api",
        "legacy_base": "https://api.dissensus.ai",
        "docs": "/docs",
        "paper_doi": "10.5281/zenodo.17918239",
        "methodology_profile": "paper_v2",
        "endpoints": {
            "current": "/asri/current",
            "timeseries": "/asri/timeseries",
            "methodology": "/asri/methodology",
            "regime": "/asri/regime",
            "validation": "/asri/validation",
            "subindex": "/asri/subindex/{name}",
        }
    })


def handle_health():
    """Health check endpoint."""
    return json_response({
        "status": "healthy",
        "version": "2.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "runtime": "cloudflare-python-workers",
        "methodology_profile": "paper_v2",
    })


async def handle_current(env):
    """Get current ASRI value and sub-indices from D1."""
    now = datetime.utcnow().isoformat()

    try:
        # Get latest record from D1
        result = await env.DB.prepare(
            "SELECT * FROM asri_daily ORDER BY date DESC LIMIT 1"
        ).all()

        rows = result.results.to_py() if hasattr(result.results, 'to_py') else list(result.results)

        if rows and len(rows) > 0:
            row = rows[0]
            sub_indices = {
                "stablecoin_risk": float(row["stablecoin_risk"]),
                "defi_liquidity_risk": float(row["defi_liquidity_risk"]),
                "contagion_risk": float(row["contagion_risk"]),
                "arbitrage_opacity": float(row["arbitrage_opacity"]),
            }
            asri = compute_asri(sub_indices)
            # Compute rolling average from reconciled values to avoid stale DB aggregates.
            avg_result = await env.DB.prepare(
                "SELECT stablecoin_risk, defi_liquidity_risk, contagion_risk, arbitrage_opacity FROM asri_daily ORDER BY date DESC LIMIT 30"
            ).all()
            avg_rows = avg_result.results.to_py() if hasattr(avg_result.results, "to_py") else list(avg_result.results)
            if avg_rows:
                avg_values = []
                for avg_row in avg_rows:
                    avg_values.append(
                        compute_asri(
                            {
                                "stablecoin_risk": float(avg_row["stablecoin_risk"]),
                                "defi_liquidity_risk": float(avg_row["defi_liquidity_risk"]),
                                "contagion_risk": float(avg_row["contagion_risk"]),
                                "arbitrage_opacity": float(avg_row["arbitrage_opacity"]),
                            }
                        )
                    )
                asri_30d_avg = round(sum(avg_values) / len(avg_values), 1)
            else:
                asri_30d_avg = asri
            trend = row["trend"] or get_trend(asri, asri_30d_avg)
            alert_level = get_alert_level(asri)
            return json_response({
                "timestamp": now,
                # Deterministic aggregation enforces equation consistency at read time.
                "asri": asri,
                "asri_30d_avg": asri_30d_avg,
                "trend": trend,
                "sub_indices": sub_indices,
                "alert_level": alert_level,
                "last_update": row["date"],
                "methodology_profile": "paper_v2",
            })
    except Exception as e:
        return json_response({"error": f"Database error: {str(e)}"}, status=500)

    # Fallback if no data
    return json_response({
        "timestamp": now,
        "asri": 0,
        "asri_30d_avg": 0,
        "trend": "unknown",
        "sub_indices": {},
        "alert_level": "unknown",
        "last_update": now,
        "warning": "No data available in database"
    })


async def handle_timeseries(env, start: str, end: str):
    """Get historical ASRI time series from D1."""
    try:
        # Query D1 for date range
        result = await env.DB.prepare(
            "SELECT * FROM asri_daily WHERE date >= ? AND date <= ? ORDER BY date ASC"
        ).bind(start, end).all()

        rows = result.results.to_py() if hasattr(result.results, 'to_py') else list(result.results)

        data = []
        for row in rows:
            sub_indices = {
                "stablecoin_risk": float(row["stablecoin_risk"]),
                "defi_liquidity_risk": float(row["defi_liquidity_risk"]),
                "contagion_risk": float(row["contagion_risk"]),
                "arbitrage_opacity": float(row["arbitrage_opacity"]),
            }
            data.append({
                "date": row["date"],
                "asri": compute_asri(sub_indices),
                "sub_indices": sub_indices,
            })

        return json_response({
            "data": data,
            "metadata": {
                "points": len(data),
                "frequency": "daily",
                "start": start,
                "end": end,
                "methodology_profile": "paper_v2",
            },
        })
    except Exception as e:
        return json_response({"error": f"Database error: {str(e)}"}, status=500)


def handle_methodology():
    """Get ASRI methodology documentation."""
    return json_response({
        "version": "2.1",
        "methodology_profile": "paper_v2",
        "weights": {
            "stablecoin_risk": 0.30,
            "defi_liquidity_risk": 0.25,
            "contagion_risk": 0.25,
            "arbitrage_opacity": 0.20,
        },
        "weight_derivation": {
            "theoretical": {
                "stablecoin_risk": 0.30,
                "defi_liquidity_risk": 0.25,
                "contagion_risk": 0.25,
                "arbitrage_opacity": 0.20,
            },
            "pca": {
                "stablecoin_risk": 0.176,
                "defi_liquidity_risk": 0.140,
                "contagion_risk": 0.362,
                "arbitrage_opacity": 0.322,
            },
        },
        "data_sources": [
            {"name": "DeFi Llama", "url": "https://defillama.com", "metrics": ["TVL", "protocol data", "stablecoin supply"]},
            {"name": "FRED", "url": "https://fred.stlouisfed.org", "metrics": ["treasury rates", "macro indicators"]},
            {"name": "CoinGecko", "url": "https://coingecko.com", "metrics": ["price data", "market cap"]},
        ],
        "update_frequency": "daily",
        "backtest_period": "2021-01-01 to 2024-12-31",
        "validation_results": {
            "crises_detected": "4/4 significant abnormal signal (t 5.47-32.64); fixed-threshold operational detection 3/4 (Terra/Luna missed in-sample); walk-forward 4/4 out-of-sample",
            "average_lead_time_days": 19,
            "walk_forward_lead_days": 26,
            "event_study_significance": "all p < 0.01",
            "structural_stability": "Chow p = 0.78 (stable)",
        },
        "documentation_url": "https://asri.dissensus.ai/docs",
        "paper_doi": "10.5281/zenodo.17918239",
    })


def handle_regime():
    """Get current market regime classification."""
    return json_response({
        "current_regime": 2,
        "regime_name": "Moderate",
        "probability": 0.78,
        "transition_probs": {
            "to_low_risk": 0.003,
            "stay_moderate": 0.992,
            "to_crisis": 0.005,
        },
    })


def handle_validation():
    """Get validation test results."""
    return json_response({
        "stationarity": {
            "asri": {"adf_stat": -4.05, "adf_p": 0.001, "kpss": 0.65, "conclusion": "stationary"},
            "stablecoin_risk": {"adf_stat": -3.76, "adf_p": 0.003, "conclusion": "trend-stationary"},
            "defi_liquidity": {"adf_stat": -4.34, "adf_p": 0.000, "conclusion": "stationary"},
            "contagion_risk": {"adf_stat": -2.73, "adf_p": 0.069, "kpss": 2.06, "conclusion": "non-stationary"},
            "regulatory_opacity": {"adf_stat": -3.71, "adf_p": 0.020, "conclusion": "trend-stationary"},
        },
        "event_study": {
            "terra_luna": {"date": "2022-05", "t_stat": 5.47, "p_value": 0.000, "lead_days": None, "detected_fixed_threshold": False, "significant": True},
            "celsius_3ac": {"date": "2022-06", "t_stat": 29.78, "p_value": 0.000, "lead_days": 19, "detected_fixed_threshold": True, "significant": True},
            "ftx_collapse": {"date": "2022-11", "t_stat": 32.64, "p_value": 0.000, "lead_days": 22, "detected_fixed_threshold": True, "significant": True},
            "svb_crisis": {"date": "2023-03", "t_stat": 26.91, "p_value": 0.000, "lead_days": 15, "detected_fixed_threshold": True, "significant": True},
            "summary": {"fixed_threshold_detection_rate": 0.75, "fixed_threshold_avg_lead": 18.7, "walk_forward_detection_rate": 1.0, "walk_forward_avg_lead": 25.75},
            "methodology_profile": "paper_canon",
        },
        "regime_model": {
            "n_regimes": 3,
            "regime_1": {"frequency": 0.198, "mean_risk": 31.8, "persistence": 0.997, "label": "Low Risk"},
            "regime_2": {"frequency": 0.563, "mean_risk": 36.8, "persistence": 0.992, "label": "Moderate"},
            "regime_3": {"frequency": 0.238, "mean_risk": 48.2, "persistence": 0.980, "label": "Crisis"},
        },
        "robustness": {
            "chow_test": {"statistic": 0.248, "critical": 3.002, "p_value": 0.78, "stable": True},
            "cusum": {"statistic": 4.715, "breaks_detected": True, "note": "breaks correspond to crisis events"},
        },
    })


async def handle_subindex(env, name: str):
    """Get individual sub-index information."""
    valid_names = {
        "stablecoin_risk": {
            "name": "Stablecoin Risk",
            "weight": 0.30,
            "current_value": None,
            "description": "Measures systemic risk from stablecoin peg deviations and concentration",
            "components": ["USDT peg deviation", "USDC peg deviation", "Stablecoin concentration"],
        },
        "defi_liquidity_risk": {
            "name": "DeFi Liquidity Risk",
            "weight": 0.25,
            "current_value": None,
            "description": "Measures liquidity fragility across DeFi protocols",
            "components": ["TVL volatility", "Protocol concentration", "Liquidity depth"],
        },
        "contagion_risk": {
            "name": "Contagion Risk",
            "weight": 0.25,
            "current_value": None,
            "description": "Measures cross-protocol and cross-chain exposure risk",
            "components": ["Cross-protocol exposure", "Bridge concentration", "Correlation clustering"],
        },
        "arbitrage_opacity": {
            "name": "Regulatory Opacity Risk",
            "weight": 0.20,
            "current_value": None,
            "description": "Measures regulatory uncertainty and policy opacity affecting the crypto-asset ecosystem",
            "components": ["Regulatory enforcement intensity", "Policy uncertainty", "Jurisdictional fragmentation"],
        },
    }

    if name not in valid_names:
        return json_response({
            "error": f"Invalid sub-index. Valid options: {list(valid_names.keys())}"
        }, status=400)

    try:
        result = await env.DB.prepare(
            "SELECT stablecoin_risk, defi_liquidity_risk, contagion_risk, arbitrage_opacity FROM asri_daily ORDER BY date DESC LIMIT 1"
        ).all()
        rows = result.results.to_py() if hasattr(result.results, "to_py") else list(result.results)
        if rows:
            valid_names[name]["current_value"] = float(rows[0][name])
        else:
            valid_names[name]["current_value"] = 0.0
    except Exception:
        valid_names[name]["current_value"] = 0.0

    return json_response(valid_names[name])


def handle_docs():
    """Return simple API documentation."""
    return json_response({
        "openapi": "3.0.0",
        "info": {
            "title": "ASRI API",
            "version": "2.1.0",
            "description": "Aggregated Systemic Risk Index - Unified crypto/DeFi systemic risk monitoring",
        },
        "paths": {
            "/": {"get": {"summary": "Root endpoint", "description": "Returns API info and available endpoints"}},
            "/health": {"get": {"summary": "Health check", "description": "Returns service health status"}},
            "/asri/current": {"get": {"summary": "Current ASRI", "description": "Returns current ASRI value and sub-indices"}},
            "/asri/timeseries": {"get": {"summary": "Historical data", "description": "Returns ASRI time series between dates", "parameters": ["start", "end"]}},
            "/asri/methodology": {"get": {"summary": "Methodology", "description": "Returns ASRI construction methodology"}},
            "/asri/regime": {"get": {"summary": "Regime classification", "description": "Returns current market regime"}},
            "/asri/validation": {"get": {"summary": "Validation results", "description": "Returns statistical validation tests"}},
            "/asri/subindex/{name}": {"get": {"summary": "Sub-index details", "description": "Returns individual sub-index information"}},
        },
        "methodology_profile": "paper_v2",
    })


# ============== Main Handler ==============

async def on_fetch(request, env):
    """Main request handler."""
    url = request.url
    path = url.split("?")[0].rstrip("/")
    method = request.method
    host = ""

    # Extract path from URL
    if "://" in path:
        host = path.split("/")[2].lower()
        path = "/" + "/".join(path.split("/")[3:])
    if not path:
        path = "/"

    # Canonicalize legacy dashboard URL.
    if host in {"dissensus.ai", "www.dissensus.ai"} and path.startswith("/asri"):
        headers = Headers.new({"Location": "https://asri.dissensus.ai/"}.items())
        return Response.new("", status=301, headers=headers)

    # Canonical routing support: asri.dissensus.ai/api/*
    if path == "/api":
        path = "/"
    elif path.startswith("/api/"):
        path = path[len("/api"):]

    if method == "OPTIONS":
        return options_response()

    # Parse query params
    query_params = {}
    if "?" in url:
        query_string = url.split("?")[1]
        for param in query_string.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)
                query_params[key] = value

    # Route handling
    if path == "/" or path == "":
        return handle_root()
    elif path == "/health":
        return handle_health()
    elif path == "/docs":
        return handle_docs()
    elif path == "/asri/current":
        return await handle_current(env)
    elif path == "/asri/timeseries":
        start = query_params.get("start", "2021-01-01")
        end = query_params.get("end", "2026-12-31")
        return await handle_timeseries(env, start, end)
    elif path == "/asri/methodology":
        return handle_methodology()
    elif path == "/asri/regime":
        return handle_regime()
    elif path == "/asri/validation":
        return handle_validation()
    elif path.startswith("/asri/subindex/"):
        name = path.split("/")[-1]
        return await handle_subindex(env, name)
    else:
        return json_response({"error": "Not found", "path": path}, status=404)
