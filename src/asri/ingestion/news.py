"""
Crypto News Aggregator for Regulatory Sentiment

Fetches news from CryptoPanic and NewsAPI, applies sentiment analysis
to generate a regulatory sentiment score.
"""

import os
import re
from datetime import datetime, timedelta
from typing import Any

import httpx
import structlog
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()
logger = structlog.get_logger()


# Keywords for filtering regulatory news
REGULATORY_KEYWORDS = [
    'regulation', 'regulatory', 'sec', 'cftc', 'congress', 'senate',
    'legislation', 'law', 'legal', 'compliance', 'enforcement',
    'ban', 'crackdown', 'lawsuit', 'investigation', 'subpoena',
    'license', 'framework', 'policy', 'government', 'federal',
    'treasury', 'irs', 'tax', 'aml', 'kyc', 'sanctions',
    'gensler', 'warren', 'binance', 'coinbase', 'kraken',
    'stablecoin', 'cbdc', 'etf', 'approval', 'reject',
    'europe', 'mica', 'uk', 'fca', 'china', 'hong kong',
]

# Negative regulatory terms (increase risk)
NEGATIVE_TERMS = [
    'ban', 'crackdown', 'lawsuit', 'investigation', 'enforcement',
    'reject', 'denied', 'fraud', 'scam', 'illegal', 'criminal',
    'fine', 'penalty', 'warning', 'risk', 'concern', 'threat',
]

# Positive regulatory terms (decrease risk)
POSITIVE_TERMS = [
    'approval', 'approved', 'clarity', 'framework', 'support',
    'adoption', 'favorable', 'progress', 'landmark', 'milestone',
    'green light', 'license', 'authorized', 'compliant',
]


class NewsArticle:
    """Represents a news article with sentiment."""

    def __init__(
        self,
        title: str,
        source: str,
        url: str,
        published_at: datetime,
        sentiment: float = 0.0,
        is_regulatory: bool = False,
    ):
        self.title = title
        self.source = source
        self.url = url
        self.published_at = published_at
        self.sentiment = sentiment
        self.is_regulatory = is_regulatory


class GoogleNewsClient:
    """Client for Google News RSS feeds - no auth required."""

    # RSS feeds for crypto and regulation news
    FEEDS = [
        "https://news.google.com/rss/search?q=cryptocurrency+regulation&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=bitcoin+SEC&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=crypto+law+policy&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=stablecoin+regulation&hl=en-US&gl=US&ceid=US:en",
    ]

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self.client.aclose()

    async def get_news(self, limit: int = 50) -> list[dict]:
        """Fetch news from Google News RSS feeds."""
        import xml.etree.ElementTree as ET

        all_articles = []

        for feed_url in self.FEEDS:
            try:
                response = await self.client.get(feed_url)
                response.raise_for_status()

                # Parse RSS XML
                root = ET.fromstring(response.text)

                for item in root.findall(".//item"):
                    title = item.find("title")
                    link = item.find("link")
                    pub_date = item.find("pubDate")
                    source = item.find("source")

                    if title is not None and title.text:
                        all_articles.append({
                            "title": title.text,
                            "url": link.text if link is not None else "",
                            "published_at": pub_date.text if pub_date is not None else "",
                            "source": source.text if source is not None else "Google News",
                        })

            except Exception as e:
                logger.warning(f"Error fetching RSS feed", feed=feed_url, error=str(e))

        # Deduplicate by title
        seen = set()
        unique = []
        for article in all_articles:
            if article["title"] not in seen:
                seen.add(article["title"])
                unique.append(article)

        logger.info(f"Fetched {len(unique)} unique articles from Google News")
        return unique[:limit]


class CryptoPanicClient:
    """Client for CryptoPanic news API (requires auth token)."""

    BASE_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, auth_token: str | None = None):
        self.auth_token = auth_token or os.getenv('CRYPTOPANIC_AUTH_TOKEN')
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self.client.aclose()

    async def get_news(
        self,
        filter_type: str = "rising",
        kind: str = "news",
        limit: int = 50,
    ) -> list[dict]:
        """Fetch news from CryptoPanic (requires auth)."""
        if not self.auth_token:
            return []  # Skip if no auth

        params = {
            "auth_token": self.auth_token,
            "kind": kind,
            "filter": filter_type,
        }

        try:
            response = await self.client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])[:limit]
        except httpx.HTTPError as e:
            logger.warning("CryptoPanic API error", error=str(e))
            return []


class NewsAggregator:
    """Aggregates news from multiple sources and calculates sentiment."""

    def __init__(self):
        self.google_news = GoogleNewsClient()
        self.cryptopanic = CryptoPanicClient()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    async def close(self):
        await self.google_news.close()
        await self.cryptopanic.close()

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using VADER.

        Returns compound score from -1 (negative) to +1 (positive).
        """
        scores = self.sentiment_analyzer.polarity_scores(text)
        return scores['compound']

    def is_regulatory_news(self, title: str) -> bool:
        """Check if news is regulatory-related."""
        title_lower = title.lower()
        return any(keyword in title_lower for keyword in REGULATORY_KEYWORDS)

    def calculate_regulatory_weight(self, title: str) -> float:
        """
        Calculate how strongly regulatory the news is.

        Returns 0-1 weight based on keyword matches.
        """
        title_lower = title.lower()
        matches = sum(1 for kw in REGULATORY_KEYWORDS if kw in title_lower)
        # Cap at 5 matches for max weight
        return min(matches / 5, 1.0)

    def sentiment_to_risk(self, sentiment: float, is_negative_term: bool) -> float:
        """
        Convert sentiment to risk score.

        Negative sentiment about regulation = higher risk
        Positive sentiment about regulation = lower risk
        """
        # VADER gives -1 to +1, we want 0-100 risk
        # Negative news = higher risk, positive = lower risk
        # So we invert: -1 sentiment -> 100 risk, +1 sentiment -> 0 risk
        base_risk = (1 - sentiment) / 2 * 100

        # Boost risk if negative regulatory terms present
        if is_negative_term:
            base_risk = min(100, base_risk * 1.3)

        return base_risk

    async def fetch_all_news(self) -> list[NewsArticle]:
        """Fetch news from all sources."""
        articles = []

        # Primary: Google News RSS (no auth required)
        try:
            news = await self.google_news.get_news(limit=50)
            for item in news:
                title = item.get("title", "")
                if not title:
                    continue

                # Parse date
                pub_date = item.get("published_at", "")
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(pub_date)
                except (ValueError, TypeError, AttributeError):
                    dt = datetime.now()

                articles.append(NewsArticle(
                    title=title,
                    source=item.get("source", "Google News"),
                    url=item.get("url", ""),
                    published_at=dt,
                ))
        except Exception as e:
            logger.warning("Error fetching Google News", error=str(e))

        # Secondary: CryptoPanic if auth token available
        if self.cryptopanic.auth_token:
            for filter_type in ["rising", "bearish", "important"]:
                try:
                    news = await self.cryptopanic.get_news(filter_type=filter_type, limit=20)
                    for item in news:
                        title = item.get("title", "")
                        if not title:
                            continue

                        pub_date = item.get("published_at", "")
                        try:
                            dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                        except (ValueError, AttributeError):
                            dt = datetime.now()

                        articles.append(NewsArticle(
                            title=title,
                            source="CryptoPanic",
                            url=item.get("url", ""),
                            published_at=dt,
                        ))
                except Exception as e:
                    logger.warning(f"Error fetching CryptoPanic {filter_type}", error=str(e))

        return articles

    async def calculate_regulatory_sentiment(self) -> dict:
        """
        Calculate regulatory sentiment score from news.

        Returns:
            dict with score (0-100, higher = more risk),
            article count, and top headlines
        """
        articles = await self.fetch_all_news()

        if not articles:
            logger.warning("No news articles fetched, using neutral sentiment")
            return {
                'score': 50.0,
                'article_count': 0,
                'regulatory_count': 0,
                'avg_sentiment': 0.0,
                'top_headlines': [],
            }

        # Analyze each article
        regulatory_articles = []
        all_sentiments = []

        for article in articles:
            # Check if regulatory
            article.is_regulatory = self.is_regulatory_news(article.title)

            # Analyze sentiment
            article.sentiment = self.analyze_sentiment(article.title)
            all_sentiments.append(article.sentiment)

            if article.is_regulatory:
                regulatory_articles.append(article)

        # Calculate regulatory sentiment score
        if regulatory_articles:
            # Weight by how regulatory the news is
            weighted_risks = []
            for article in regulatory_articles:
                weight = self.calculate_regulatory_weight(article.title)
                has_negative = any(
                    term in article.title.lower()
                    for term in NEGATIVE_TERMS
                )
                risk = self.sentiment_to_risk(article.sentiment, has_negative)
                weighted_risks.append(risk * weight)

            # Average weighted risk
            reg_score = sum(weighted_risks) / len(weighted_risks) if weighted_risks else 50.0
        else:
            # No regulatory news = neutral
            reg_score = 50.0

        # Also factor in overall market sentiment
        avg_sentiment = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0
        market_mood = (1 - avg_sentiment) / 2 * 100  # Convert to 0-100 risk

        # Blend: 70% regulatory, 30% market mood
        final_score = reg_score * 0.7 + market_mood * 0.3

        # Get top regulatory headlines
        top_headlines = [
            {
                'title': a.title,
                'sentiment': a.sentiment,
                'source': a.source,
            }
            for a in sorted(
                regulatory_articles,
                key=lambda x: abs(x.sentiment),
                reverse=True
            )[:5]
        ]

        logger.info(
            "Calculated regulatory sentiment",
            score=final_score,
            regulatory_count=len(regulatory_articles),
            total_articles=len(articles),
            avg_sentiment=avg_sentiment,
        )

        return {
            'score': final_score,
            'article_count': len(articles),
            'regulatory_count': len(regulatory_articles),
            'avg_sentiment': avg_sentiment,
            'market_mood': market_mood,
            'top_headlines': top_headlines,
        }


async def get_regulatory_sentiment() -> float:
    """
    Convenience function to get regulatory sentiment score.

    Returns score from 0-100 (higher = more regulatory risk).
    """
    aggregator = NewsAggregator()
    try:
        result = await aggregator.calculate_regulatory_sentiment()
        return result['score']
    finally:
        await aggregator.close()


# Test function
async def test_news():
    """Test news aggregation and sentiment."""
    print("=" * 60)
    print("REGULATORY SENTIMENT ANALYSIS")
    print("=" * 60)

    aggregator = NewsAggregator()

    try:
        result = await aggregator.calculate_regulatory_sentiment()

        print(f"\n📊 Regulatory Sentiment Score: {result['score']:.1f}/100")
        print(f"   (Higher = More Regulatory Risk)")
        print(f"\n📰 Articles Analyzed: {result['article_count']}")
        print(f"   Regulatory News: {result['regulatory_count']}")
        print(f"   Avg Market Sentiment: {result['avg_sentiment']:.3f}")

        if result['top_headlines']:
            print("\n🔥 Top Regulatory Headlines:")
            for h in result['top_headlines']:
                emoji = "🔴" if h['sentiment'] < -0.2 else "🟢" if h['sentiment'] > 0.2 else "🟡"
                print(f"   {emoji} [{h['sentiment']:+.2f}] {h['title'][:70]}...")

        print("\n✅ News sentiment analysis complete!")
        return result

    finally:
        await aggregator.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_news())
