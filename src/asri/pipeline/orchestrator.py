"""
ASRI Pipeline Orchestrator

Coordinates the full data pipeline:
1. Fetch data from all sources (DeFiLlama, FRED)
2. Transform raw data into sub-index inputs
3. Calculate ASRI
4. Store results in database
"""

import asyncio
import os
from datetime import datetime, timedelta

import structlog
from dotenv import load_dotenv
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from asri.ingestion.coingecko import CoinGeckoClient, calculate_correlation
from asri.ingestion.defillama import DeFiLlamaClient
from asri.ingestion.fred import FREDConnector
from asri.ingestion.news import NewsAggregator
from asri.models.asri import ASRIDaily
from asri.models.base import async_session
from asri.pipeline.transform import (
    DataTransformer,
    transform_all_data,
)
from asri.signals.calculator import compute_asri

load_dotenv()
logger = structlog.get_logger()


class ASRIOrchestrator:
    """Orchestrates the full ASRI calculation pipeline."""

    # Weights for converting transformed inputs to sub-index scores
    STABLECOIN_WEIGHTS = {
        'tvl_ratio': 0.4,
        'treasury_stress': 0.3,
        'concentration_hhi': 0.2,
        'peg_volatility': 0.1,
    }

    DEFI_WEIGHTS = {
        'top10_concentration': 0.35,
        'tvl_volatility': 0.25,
        'smart_contract_risk': 0.20,
        'flash_loan_proxy': 0.10,
        'leverage_change': 0.10,
    }

    CONTAGION_WEIGHTS = {
        'rwa_growth_rate': 0.30,
        'bank_exposure': 0.25,
        'tradfi_linkage': 0.20,
        'crypto_equity_correlation': 0.15,
        'bridge_exploit_frequency': 0.10,
    }

    ARBITRAGE_WEIGHTS = {
        'unregulated_exposure': 0.25,
        'multi_issuer_risk': 0.25,
        'custody_concentration': 0.20,
        'regulatory_sentiment': 0.15,
        'transparency_score': 0.15,  # Note: inverted in calculation
    }

    def __init__(self, fred_api_key: str | None = None, coingecko_api_key: str | None = None):
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.coingecko_api_key = coingecko_api_key or os.getenv('COINGECKO_API_KEY')

        if not self.fred_api_key:
            raise ValueError("FRED_API_KEY required")

        self.defillama = DeFiLlamaClient()
        self.fred = FREDConnector(self.fred_api_key)
        self.coingecko = CoinGeckoClient(self.coingecko_api_key)
        self.news = NewsAggregator()
        self.transformer = DataTransformer()

    async def close(self):
        """Clean up connections."""
        await self.defillama.close()
        await self.coingecko.close()
        await self.news.close()

    async def fetch_all_data(self) -> dict:
        """Fetch data from all sources concurrently."""
        logger.info("Fetching data from all sources")

        # Fetch all data concurrently
        results = await asyncio.gather(
            self.defillama.get_total_tvl(),
            self.defillama.get_stablecoins(),
            self.defillama.get_protocols(),
            self.defillama.get_bridges(),
            self.defillama.get_tvl_history(),
            self.fred.fetch_series('DGS10', start_date='2024-01-01'),
            self.fred.fetch_series('VIXCLS', start_date='2024-01-01'),
            self.fred.fetch_series('T10Y2Y', start_date='2024-01-01'),
            self.fred.fetch_series('SP500', start_date='2024-09-01'),  # S&P500 for correlation
            self.coingecko.get_price_history('bitcoin', days=90),  # BTC for correlation
            self.news.calculate_regulatory_sentiment(),  # News-based regulatory sentiment
            return_exceptions=True
        )

        # Unpack results
        (
            total_tvl,
            stablecoins,
            protocols,
            bridges,
            tvl_history,
            dgs10_data,
            vix_data,
            spread_data,
            sp500_data,
            btc_history,
            news_sentiment,
        ) = results

        # Handle any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching data source {i}", error=str(result))

        # Extract latest FRED values
        def get_latest_fred_value(data, default=0.0):
            if isinstance(data, Exception):
                return default
            obs = data.get('observations', [])
            if obs:
                val = obs[-1].get('value', '.')
                return float(val) if val != '.' else default
            return default

        treasury_10y = get_latest_fred_value(dgs10_data, 4.0)
        vix = get_latest_fred_value(vix_data, 15.0)
        yield_spread = get_latest_fred_value(spread_data, 0.5)

        # Extract S&P500 prices for correlation
        def get_fred_price_series(data):
            if isinstance(data, Exception):
                return []
            obs = data.get('observations', [])
            prices = []
            for o in obs:
                val = o.get('value', '.')
                if val != '.':
                    try:
                        prices.append(float(val))
                    except ValueError:
                        pass
            return prices

        sp500_prices = get_fred_price_series(sp500_data)

        # Calculate BTC-S&P500 correlation
        if not isinstance(btc_history, Exception) and btc_history and sp500_prices:
            btc_prices = [price for _, price in btc_history]
            crypto_equity_corr = calculate_correlation(btc_prices, sp500_prices)
            logger.info(
                "Calculated BTC-S&P500 correlation",
                correlation=crypto_equity_corr,
                btc_points=len(btc_prices),
                sp500_points=len(sp500_prices),
            )
        else:
            crypto_equity_corr = 0.5  # Default if data unavailable
            logger.warning("Using default correlation (data unavailable)")

        # Calculate max historical TVL
        if not isinstance(tvl_history, Exception) and tvl_history:
            max_tvl = max(p.tvl for p in tvl_history)
            historical_tvls = [p.tvl for p in tvl_history[-30:]]  # Last 30 days
        else:
            max_tvl = total_tvl if not isinstance(total_tvl, Exception) else 100e9
            historical_tvls = None

        logger.info(
            "Data fetch complete",
            total_tvl=total_tvl if not isinstance(total_tvl, Exception) else "error",
            num_stables=len(stablecoins) if not isinstance(stablecoins, Exception) else "error",
            num_protocols=len(protocols) if not isinstance(protocols, Exception) else "error",
            treasury_10y=treasury_10y,
            vix=vix,
            yield_spread=yield_spread,
        )

        # Extract regulatory sentiment
        if isinstance(news_sentiment, Exception):
            reg_sentiment = {'score': 50.0, 'regulatory_count': 0, 'top_headlines': []}
        else:
            reg_sentiment = news_sentiment

        return {
            'total_tvl': total_tvl if not isinstance(total_tvl, Exception) else 100e9,
            'max_tvl': max_tvl,
            'stablecoins': stablecoins if not isinstance(stablecoins, Exception) else [],
            'protocols': protocols if not isinstance(protocols, Exception) else [],
            'bridges': bridges if not isinstance(bridges, Exception) else [],
            'tvl_history': historical_tvls,
            'treasury_10y': treasury_10y,
            'vix': vix,
            'yield_spread': yield_spread,
            'crypto_equity_corr': crypto_equity_corr,
            'regulatory_sentiment': reg_sentiment,
        }

    def calculate_sub_index(self, inputs, weights: dict) -> float:
        """Calculate a sub-index from inputs using weights."""
        total = 0.0
        for field, weight in weights.items():
            value = getattr(inputs, field, 50.0)
            # Invert transparency_score (high transparency = low risk)
            if field == 'transparency_score':
                value = 100 - value
            total += value * weight
        return min(100, max(0, total))

    async def calculate_asri(self) -> dict:
        """
        Calculate ASRI from live data.

        All inputs are now fetched automatically - no manual parameters needed!

        Returns:
            Dictionary with ASRI result and all metrics
        """
        # Fetch all data
        data = await self.fetch_all_data()

        # Use live data for all inputs
        crypto_equity_corr = data.get('crypto_equity_corr', 0.5)
        reg_data = data.get('regulatory_sentiment', {})
        regulatory_sentiment = reg_data.get('score', 50.0)

        logger.info(
            "Using live regulatory sentiment",
            score=regulatory_sentiment,
            regulatory_articles=reg_data.get('regulatory_count', 0),
        )

        # Transform data
        transformed = transform_all_data(
            stablecoins=data['stablecoins'],
            protocols=data['protocols'],
            bridges=data['bridges'],
            current_tvl=data['total_tvl'],
            max_historical_tvl=data['max_tvl'],
            treasury_10y_rate=data['treasury_10y'],
            vix=data['vix'],
            yield_curve_spread=data['yield_spread'],
            tvl_history=data['tvl_history'],
            crypto_equity_corr=crypto_equity_corr,
            regulatory_sentiment=regulatory_sentiment,
        )

        # Calculate sub-indices
        stablecoin_risk = self.calculate_sub_index(
            transformed.stablecoin_risk,
            self.STABLECOIN_WEIGHTS
        )
        defi_liquidity_risk = self.calculate_sub_index(
            transformed.defi_liquidity_risk,
            self.DEFI_WEIGHTS
        )
        contagion_risk = self.calculate_sub_index(
            transformed.contagion_risk,
            self.CONTAGION_WEIGHTS
        )
        arbitrage_opacity = self.calculate_sub_index(
            transformed.arbitrage_opacity_risk,
            self.ARBITRAGE_WEIGHTS
        )

        # Calculate aggregate ASRI
        result = compute_asri(
            stablecoin_risk=stablecoin_risk,
            defi_liquidity_risk=defi_liquidity_risk,
            contagion_risk=contagion_risk,
            arbitrage_opacity=arbitrage_opacity,
        )

        logger.info(
            "ASRI calculated",
            asri=result.asri,
            alert_level=result.alert_level,
            stablecoin_risk=stablecoin_risk,
            defi_liquidity_risk=defi_liquidity_risk,
            contagion_risk=contagion_risk,
            arbitrage_opacity=arbitrage_opacity,
        )

        return {
            'timestamp': datetime.utcnow(),
            'asri': result.asri,
            'asri_normalized': result.asri_normalized,
            'alert_level': result.alert_level,
            'sub_indices': {
                'stablecoin_risk': stablecoin_risk,
                'defi_liquidity_risk': defi_liquidity_risk,
                'contagion_risk': contagion_risk,
                'arbitrage_opacity': arbitrage_opacity,
            },
            'inputs': {
                'stablecoin': transformed.stablecoin_risk,
                'defi': transformed.defi_liquidity_risk,
                'contagion': transformed.contagion_risk,
                'arbitrage': transformed.arbitrage_opacity_risk,
            },
            'raw_metrics': transformed.raw_metrics,
            'fred_data': {
                'treasury_10y': data['treasury_10y'],
                'vix': data['vix'],
                'yield_spread': data['yield_spread'],
                'crypto_equity_corr': crypto_equity_corr,
            },
            'news_data': {
                'regulatory_sentiment': regulatory_sentiment,
                'articles_analyzed': reg_data.get('article_count', 0),
                'regulatory_articles': reg_data.get('regulatory_count', 0),
                'top_headlines': reg_data.get('top_headlines', [])[:3],
            }
        }

    async def save_to_db(self, result: dict, session: AsyncSession | None = None) -> ASRIDaily:
        """
        Save ASRI calculation result to database.

        Args:
            result: Result dictionary from calculate_asri()
            session: Optional existing session (creates new one if not provided)

        Returns:
            The created/updated ASRIDaily record
        """
        # Calculate 30-day average if we have history
        asri_30d_avg = None
        trend = "stable"

        async with async_session() as db:
            # Get last 30 days of data for average
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            stmt = select(ASRIDaily).where(ASRIDaily.date >= thirty_days_ago).order_by(ASRIDaily.date)
            history = await db.execute(stmt)
            history_records = history.scalars().all()

            if history_records:
                avg_values = [r.asri for r in history_records]
                asri_30d_avg = sum(avg_values) / len(avg_values)

                # Determine trend based on last value vs current
                if len(history_records) >= 1:
                    last_asri = history_records[-1].asri
                    delta = result['asri'] - last_asri
                    if delta > 2:
                        trend = "increasing"
                    elif delta < -2:
                        trend = "decreasing"

            # Check if we already have a record for today
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            stmt = select(ASRIDaily).where(ASRIDaily.date == today)
            existing = await db.execute(stmt)
            record = existing.scalar_one_or_none()

            if record:
                # Update existing record
                record.asri = result['asri']
                record.asri_normalized = result['asri_normalized']
                record.asri_30d_avg = asri_30d_avg or result['asri']
                record.trend = trend
                record.alert_level = result['alert_level']
                record.stablecoin_risk = result['sub_indices']['stablecoin_risk']
                record.defi_liquidity_risk = result['sub_indices']['defi_liquidity_risk']
                record.contagion_risk = result['sub_indices']['contagion_risk']
                record.arbitrage_opacity = result['sub_indices']['arbitrage_opacity']
                record.updated_at = datetime.utcnow()
                logger.info("Updated existing ASRI record", date=today)
            else:
                # Create new record
                record = ASRIDaily(
                    date=today,
                    asri=result['asri'],
                    asri_normalized=result['asri_normalized'],
                    asri_30d_avg=asri_30d_avg or result['asri'],
                    trend=trend,
                    alert_level=result['alert_level'],
                    stablecoin_risk=result['sub_indices']['stablecoin_risk'],
                    defi_liquidity_risk=result['sub_indices']['defi_liquidity_risk'],
                    contagion_risk=result['sub_indices']['contagion_risk'],
                    arbitrage_opacity=result['sub_indices']['arbitrage_opacity'],
                    created_at=datetime.utcnow(),
                )
                db.add(record)
                logger.info("Created new ASRI record", date=today)

            await db.commit()
            await db.refresh(record)
            return record

    async def calculate_and_save(self) -> dict:
        """
        Calculate ASRI and save to database.

        Returns:
            Dictionary with result and database record ID
        """
        result = await self.calculate_asri()
        record = await self.save_to_db(result)

        result['db_id'] = record.id
        result['trend'] = record.trend
        result['asri_30d_avg'] = record.asri_30d_avg

        logger.info(
            "ASRI calculated and saved",
            db_id=record.id,
            asri=result['asri'],
            trend=record.trend,
        )

        return result


async def run_live_calculation():
    """Run a live ASRI calculation and print results."""
    print("=" * 70)
    print("ASRI LIVE CALCULATION - FULLY AUTOMATED")
    print("Data Sources: DeFiLlama | FRED | CoinGecko | Google News")
    print("=" * 70)

    orchestrator = ASRIOrchestrator()

    try:
        result = await orchestrator.calculate_asri()

        print(f"\n{'='*70}")
        print(f"ASRI: {result['asri']:.1f} | Alert Level: {result['alert_level'].upper()}")
        print(f"{'='*70}")

        print("\nSub-Indices:")
        for name, value in result['sub_indices'].items():
            bar = "█" * int(value / 5) + "░" * (20 - int(value / 5))
            print(f"  {name:25} [{bar}] {value:.1f}")

        print("\nMacro Data (FRED + CoinGecko):")
        print(f"  10Y Treasury:     {result['fred_data']['treasury_10y']:.2f}%")
        print(f"  VIX:              {result['fred_data']['vix']:.1f}")
        print(f"  Yield Spread:     {result['fred_data']['yield_spread']:.2f}%")
        corr = result['fred_data']['crypto_equity_corr']
        corr_desc = "high" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "low"
        print(f"  BTC-S&P500 Corr:  {corr:.3f} ({corr_desc})")

        print("\nRegulatory Sentiment (Google News + NLP):")
        news = result['news_data']
        reg_level = "low" if news['regulatory_sentiment'] < 40 else "moderate" if news['regulatory_sentiment'] < 60 else "high"
        print(f"  Sentiment Score:  {news['regulatory_sentiment']:.1f}/100 ({reg_level} risk)")
        print(f"  Articles Analyzed: {news['articles_analyzed']} ({news['regulatory_articles']} regulatory)")

        if news['top_headlines']:
            print("\n  Top Headlines:")
            for h in news['top_headlines']:
                emoji = "🔴" if h['sentiment'] < -0.2 else "🟢" if h['sentiment'] > 0.2 else "🟡"
                title = h['title'][:55] + "..." if len(h['title']) > 55 else h['title']
                print(f"    {emoji} {title}")

        print("\nRaw Metrics:")
        for key, value in result['raw_metrics'].items():
            if isinstance(value, float) and value > 1e6:
                print(f"  {key}: ${value/1e9:.2f}B")
            else:
                print(f"  {key}: {value}")

        return result

    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(run_live_calculation())
