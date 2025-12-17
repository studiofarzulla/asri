"""
Static Site Generator for ASRI Dashboard

Generates a static HTML dashboard from database data.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import structlog
from sqlalchemy import desc, select

from asri.models.asri import ASRIDaily
from asri.models.base import async_session

logger = structlog.get_logger()


def get_alert_color(level: str) -> str:
    """Get color for alert level."""
    colors = {
        "low": "#22c55e",      # green
        "moderate": "#eab308", # yellow
        "elevated": "#f97316", # orange
        "high": "#ef4444",     # red
        "critical": "#dc2626", # dark red
    }
    return colors.get(level.lower(), "#6b7280")


def get_risk_bar_color(value: float) -> str:
    """Get color based on risk value."""
    if value < 30:
        return "#22c55e"  # green
    elif value < 50:
        return "#eab308"  # yellow
    elif value < 70:
        return "#f97316"  # orange
    else:
        return "#ef4444"  # red


async def fetch_dashboard_data(max_history_days: int = 365) -> dict:
    """Fetch all data needed for dashboard."""
    async with async_session() as db:
        # Get latest record
        stmt = select(ASRIDaily).order_by(desc(ASRIDaily.date)).limit(1)
        result = await db.execute(stmt)
        latest = result.scalar_one_or_none()

        if not latest:
            return None

        # Get historical data (up to max_history_days or all available)
        cutoff_date = datetime.utcnow() - timedelta(days=max_history_days)
        stmt = (
            select(ASRIDaily)
            .where(ASRIDaily.date >= cutoff_date)
            .order_by(ASRIDaily.date)
        )
        result = await db.execute(stmt)
        history = result.scalars().all()

        return {
            'latest': latest,
            'history': history,
            'history_days': len(history),
        }


def generate_html(data: dict) -> str:
    """Generate the HTML dashboard."""
    latest = data['latest']
    history = data['history']

    alert_color = get_alert_color(latest.alert_level)

    # Prepare chart data
    chart_dates = [r.date.strftime("%Y-%m-%d") for r in history]
    chart_values = [round(r.asri, 1) for r in history]

    # Sub-indices for display
    sub_indices = [
        ("Stablecoin Risk", latest.stablecoin_risk, "Currency stability and concentration"),
        ("DeFi Liquidity Risk", latest.defi_liquidity_risk, "Protocol health and smart contract risk"),
        ("Contagion Risk", latest.contagion_risk, "Cross-market correlation and TradFi linkage"),
        ("Arbitrage Opacity", latest.arbitrage_opacity, "Regulatory sentiment and transparency"),
    ]

    sub_index_html = ""
    for name, value, desc in sub_indices:
        bar_color = get_risk_bar_color(value)
        bar_width = min(100, max(0, value))
        sub_index_html += f'''
        <div class="sub-index">
            <div class="sub-index-header">
                <span class="sub-index-name">{name}</span>
                <span class="sub-index-value">{value:.1f}</span>
            </div>
            <div class="sub-index-bar-bg">
                <div class="sub-index-bar" style="width: {bar_width}%; background: {bar_color};"></div>
            </div>
            <div class="sub-index-desc">{desc}</div>
        </div>
        '''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASRI - Aggregated Systemic Risk Index</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 2rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
        }}

        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .subtitle {{
            color: #94a3b8;
            font-size: 1.1rem;
        }}

        .main-score {{
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 1rem;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }}

        .asri-value {{
            font-size: 5rem;
            font-weight: 800;
            color: {alert_color};
            line-height: 1;
        }}

        .alert-badge {{
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 2rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1rem;
            background: {alert_color}22;
            color: {alert_color};
            border: 1px solid {alert_color}44;
        }}

        .meta-info {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1.5rem;
            color: #94a3b8;
            font-size: 0.9rem;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 1rem;
            padding: 1.5rem;
        }}

        .card h2 {{
            font-size: 1.1rem;
            color: #94a3b8;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .sub-index {{
            margin-bottom: 1.25rem;
        }}

        .sub-index:last-child {{
            margin-bottom: 0;
        }}

        .sub-index-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }}

        .sub-index-name {{
            font-weight: 500;
        }}

        .sub-index-value {{
            font-weight: 700;
            font-family: monospace;
        }}

        .sub-index-bar-bg {{
            height: 8px;
            background: rgba(148, 163, 184, 0.2);
            border-radius: 4px;
            overflow: hidden;
        }}

        .sub-index-bar {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}

        .sub-index-desc {{
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 0.25rem;
        }}

        .chart-container {{
            position: relative;
            height: 250px;
        }}

        footer {{
            text-align: center;
            color: #64748b;
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(148, 163, 184, 0.1);
        }}

        footer a {{
            color: #60a5fa;
            text-decoration: none;
        }}

        footer a:hover {{
            text-decoration: underline;
        }}

        .trend {{
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
        }}

        .trend-up {{ color: #ef4444; }}
        .trend-down {{ color: #22c55e; }}
        .trend-stable {{ color: #94a3b8; }}

        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}

            h1 {{
                font-size: 1.75rem;
            }}

            .asri-value {{
                font-size: 3.5rem;
            }}

            .meta-info {{
                flex-direction: column;
                gap: 0.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ASRI</h1>
            <p class="subtitle">Aggregated Systemic Risk Index for Crypto/DeFi Markets</p>
        </header>

        <div class="main-score">
            <div class="asri-value">{latest.asri:.1f}</div>
            <div class="alert-badge">{latest.alert_level.upper()} RISK</div>
            <div class="meta-info">
                <span>30-Day Avg: {latest.asri_30d_avg:.1f}</span>
                <span class="trend trend-{latest.trend}">
                    {"↑" if latest.trend == "increasing" else "↓" if latest.trend == "decreasing" else "→"}
                    {latest.trend.capitalize()}
                </span>
                <span>Updated: {latest.updated_at.strftime("%Y-%m-%d %H:%M UTC") if latest.updated_at else latest.created_at.strftime("%Y-%m-%d %H:%M UTC")}</span>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Sub-Indices</h2>
                {sub_index_html}
            </div>

            <div class="card">
                <h2>Historical Trend ({len(history)} days)</h2>
                <div class="chart-container">
                    <canvas id="trendChart"></canvas>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Methodology</h2>
            <p style="color: #94a3b8; line-height: 1.6;">
                ASRI aggregates four risk sub-indices using a weighted formula:
                Stablecoin Risk (30%), DeFi Liquidity Risk (25%), Contagion Risk (25%),
                and Arbitrage/Opacity Risk (20%). Data is sourced from DeFiLlama,
                FRED (Federal Reserve), CoinGecko, and news sentiment analysis.
            </p>
            <p style="color: #64748b; margin-top: 1rem; font-size: 0.9rem;">
                <a href="https://doi.org/10.5281/zenodo.17918239" target="_blank">Read the full methodology paper →</a>
            </p>
        </div>

        <footer>
            <p>
                Built by <a href="https://farzulla.org" target="_blank">Farzulla Research</a> |
                <a href="https://github.com/studiofarzulla/asri" target="_blank">Source Code</a> |
                <a href="https://resurrexi.dev" target="_blank">Documentation</a>
            </p>
            <p style="margin-top: 0.5rem;">
                Part of the <a href="https://resurrexi.io" target="_blank">Resurrexi Labs</a> research infrastructure
            </p>
        </footer>
    </div>

    <script>
        const ctx = document.getElementById('trendChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(chart_dates)},
                datasets: [{{
                    label: 'ASRI',
                    data: {json.dumps(chart_values)},
                    borderColor: '#60a5fa',
                    backgroundColor: 'rgba(96, 165, 250, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        min: 0,
                        max: 100,
                        grid: {{
                            color: 'rgba(148, 163, 184, 0.1)'
                        }},
                        ticks: {{
                            color: '#94a3b8'
                        }}
                    }},
                    x: {{
                        grid: {{
                            display: false
                        }},
                        ticks: {{
                            color: '#94a3b8',
                            maxRotation: 45
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>'''

    return html


async def generate_static_site(output_dir: Path | str) -> Path:
    """
    Generate the complete static site.

    Args:
        output_dir: Directory to write files to

    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating static site", output_dir=str(output_dir))

    # Fetch data
    data = await fetch_dashboard_data()

    if not data:
        logger.warning("No data available for static site generation")
        # Generate placeholder page
        html = '''<!DOCTYPE html>
<html><head><title>ASRI</title></head>
<body style="font-family: sans-serif; text-align: center; padding: 50px;">
<h1>ASRI Dashboard</h1>
<p>No data available yet. Run a calculation first.</p>
</body></html>'''
    else:
        html = generate_html(data)

    # Write index.html
    index_path = output_dir / "index.html"
    index_path.write_text(html)

    logger.info("Static site generated", index_path=str(index_path))

    return output_dir


if __name__ == "__main__":
    import asyncio

    output = asyncio.run(generate_static_site(Path("./static_site")))
    print(f"✅ Generated at: {output}")
