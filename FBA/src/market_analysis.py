"""
Amazon FBA Market Analysis

Comprehensive market and niche analysis including:
- Competitor analysis
- Market size estimation
- Trend detection
- Seasonality analysis
- Niche scoring
- Keyword demand analysis
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class TrendDirection(Enum):
    """Market trend directions."""
    STRONG_UP = "strong_uptrend"
    UP = "uptrend"
    STABLE = "stable"
    DOWN = "downtrend"
    STRONG_DOWN = "strong_downtrend"


class SeasonalityPattern(Enum):
    """Seasonality patterns."""
    Q4_PEAK = "q4_holiday_peak"          # Holiday season surge
    SUMMER_PEAK = "summer_peak"           # Summer months
    BACK_TO_SCHOOL = "back_to_school"     # Aug-Sep
    SPRING_PEAK = "spring_peak"           # Mar-May
    YEAR_ROUND = "year_round"             # No strong seasonality
    UNKNOWN = "unknown"


class CompetitionLevel(Enum):
    """Market competition levels."""
    VERY_LOW = "very_low"      # < 50 competitors
    LOW = "low"                # 50-200 competitors
    MODERATE = "moderate"      # 200-500 competitors
    HIGH = "high"              # 500-1000 competitors
    VERY_HIGH = "very_high"    # > 1000 competitors


@dataclass
class Competitor:
    """Represents a competitor in a niche."""
    asin: str
    title: str
    brand: str
    price: float
    rating: float
    review_count: int
    sales_rank: Optional[int]
    estimated_monthly_sales: int
    is_fba: bool
    is_amazon: bool  # Sold by Amazon
    listing_age_days: Optional[int] = None
    images_count: int = 0
    bullet_points_count: int = 0
    has_video: bool = False
    has_aplus: bool = False  # A+ Content


@dataclass
class NicheMetrics:
    """Aggregated metrics for a niche/market."""
    keyword: str
    total_results: int

    # Price metrics
    avg_price: float
    median_price: float
    min_price: float
    max_price: float
    price_std_dev: float

    # Review metrics
    avg_reviews: float
    median_reviews: int
    total_reviews: int
    avg_rating: float

    # Sales metrics
    total_estimated_revenue: float
    avg_monthly_sales: float
    top_10_revenue_share: float  # % of revenue from top 10

    # Competition metrics
    competitor_count: int
    fba_seller_percent: float
    amazon_seller_percent: float
    brand_concentration: float  # HHI for brands

    # Quality metrics
    avg_images: float
    avg_bullets: float
    video_percent: float
    aplus_percent: float


@dataclass
class MarketTrend:
    """Market trend analysis results."""
    direction: TrendDirection
    momentum: float  # -100 to +100
    growth_rate_monthly: float  # % per month
    volatility: float  # 0-100
    confidence: str  # low, medium, high


@dataclass
class SeasonalityAnalysis:
    """Seasonality analysis results."""
    pattern: SeasonalityPattern
    peak_months: List[int]
    low_months: List[int]
    monthly_indices: Dict[int, float]  # Month -> index (1.0 = average)
    yoy_growth: Optional[float]  # Year-over-year growth


@dataclass
class NicheScore:
    """Comprehensive niche opportunity score."""
    overall_score: float  # 0-100

    # Component scores (0-100)
    demand_score: float
    competition_score: float
    profit_potential_score: float
    entry_barrier_score: float
    trend_score: float

    # Flags
    recommended: bool
    warnings: List[str]
    opportunities: List[str]


@dataclass
class MarketReport:
    """Complete market analysis report."""
    keyword: str
    generated_at: str

    metrics: NicheMetrics
    trend: MarketTrend
    seasonality: SeasonalityAnalysis
    score: NicheScore

    top_competitors: List[Competitor]
    market_gaps: List[str]
    recommendations: List[str]


class MarketAnalyzer:
    """
    Comprehensive market and niche analysis.

    Analyzes competition, estimates market size, detects trends,
    and scores niche opportunities.
    """

    # Revenue estimation by BSR range (approximate)
    BSR_TO_MONTHLY_SALES = [
        (100, 3000),
        (500, 1500),
        (1000, 900),
        (2000, 600),
        (5000, 300),
        (10000, 150),
        (20000, 75),
        (50000, 30),
        (100000, 15),
        (200000, 8),
        (500000, 3),
        (1000000, 1),
    ]

    def __init__(self):
        """Initialize market analyzer."""
        pass

    def estimate_monthly_sales(self, bsr: int, category: str = 'all') -> int:
        """
        Estimate monthly sales from Best Sellers Rank.

        Args:
            bsr: Best Sellers Rank
            category: Product category (affects estimation)

        Returns:
            Estimated monthly unit sales
        """
        if bsr <= 0:
            return 0

        # Find bracketing BSR values
        for i, (threshold, sales) in enumerate(self.BSR_TO_MONTHLY_SALES):
            if bsr <= threshold:
                if i == 0:
                    return sales
                prev_threshold, prev_sales = self.BSR_TO_MONTHLY_SALES[i-1]
                # Linear interpolation
                ratio = (bsr - prev_threshold) / (threshold - prev_threshold)
                return int(prev_sales - ratio * (prev_sales - sales))

        # Beyond our table - extrapolate
        return max(1, int(1000000 / bsr))

    def analyze_competitors(self, competitors: List[Competitor]) -> NicheMetrics:
        """
        Analyze competitor data to generate niche metrics.

        Args:
            competitors: List of competitor products

        Returns:
            NicheMetrics with aggregated analysis
        """
        if not competitors:
            return None

        n = len(competitors)

        # Price analysis
        prices = [c.price for c in competitors if c.price > 0]
        prices_sorted = sorted(prices)

        avg_price = sum(prices) / len(prices) if prices else 0
        median_price = prices_sorted[len(prices)//2] if prices else 0
        min_price = min(prices) if prices else 0
        max_price = max(prices) if prices else 0

        # Price standard deviation
        if len(prices) > 1:
            variance = sum((p - avg_price) ** 2 for p in prices) / (len(prices) - 1)
            price_std = math.sqrt(variance)
        else:
            price_std = 0

        # Review analysis
        reviews = [c.review_count for c in competitors]
        ratings = [c.rating for c in competitors if c.rating > 0]

        avg_reviews = sum(reviews) / n
        reviews_sorted = sorted(reviews)
        median_reviews = reviews_sorted[n//2]
        total_reviews = sum(reviews)
        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        # Sales analysis
        monthly_sales = [c.estimated_monthly_sales for c in competitors]
        revenues = [c.estimated_monthly_sales * c.price for c in competitors]

        total_revenue = sum(revenues)
        avg_monthly_sales = sum(monthly_sales) / n

        # Top 10 concentration
        top_10_revenue = sum(sorted(revenues, reverse=True)[:10])
        top_10_share = (top_10_revenue / total_revenue * 100) if total_revenue > 0 else 0

        # Competition analysis
        fba_count = sum(1 for c in competitors if c.is_fba)
        amazon_count = sum(1 for c in competitors if c.is_amazon)

        # Brand concentration (Herfindahl-Hirschman Index)
        brand_sales = defaultdict(int)
        for c in competitors:
            brand_sales[c.brand] += c.estimated_monthly_sales

        total_sales = sum(monthly_sales)
        if total_sales > 0:
            hhi = sum((sales/total_sales * 100) ** 2 for sales in brand_sales.values())
        else:
            hhi = 0

        # Quality metrics
        images = [c.images_count for c in competitors]
        bullets = [c.bullet_points_count for c in competitors]
        video_count = sum(1 for c in competitors if c.has_video)
        aplus_count = sum(1 for c in competitors if c.has_aplus)

        return NicheMetrics(
            keyword="",  # Set by caller
            total_results=n,
            avg_price=round(avg_price, 2),
            median_price=round(median_price, 2),
            min_price=round(min_price, 2),
            max_price=round(max_price, 2),
            price_std_dev=round(price_std, 2),
            avg_reviews=round(avg_reviews, 1),
            median_reviews=median_reviews,
            total_reviews=total_reviews,
            avg_rating=round(avg_rating, 2),
            total_estimated_revenue=round(total_revenue, 2),
            avg_monthly_sales=round(avg_monthly_sales, 1),
            top_10_revenue_share=round(top_10_share, 1),
            competitor_count=n,
            fba_seller_percent=round(fba_count / n * 100, 1),
            amazon_seller_percent=round(amazon_count / n * 100, 1),
            brand_concentration=round(hhi, 1),
            avg_images=round(sum(images) / n, 1) if images else 0,
            avg_bullets=round(sum(bullets) / n, 1) if bullets else 0,
            video_percent=round(video_count / n * 100, 1),
            aplus_percent=round(aplus_count / n * 100, 1)
        )

    def analyze_trend(
        self,
        historical_data: List[Dict[str, Any]],
        period_days: int = 90
    ) -> MarketTrend:
        """
        Analyze market trend from historical data.

        Args:
            historical_data: List of {date, value} records
            period_days: Analysis period

        Returns:
            MarketTrend analysis
        """
        if not historical_data or len(historical_data) < 7:
            return MarketTrend(
                direction=TrendDirection.STABLE,
                momentum=0,
                growth_rate_monthly=0,
                volatility=0,
                confidence='low'
            )

        # Sort by date
        data = sorted(historical_data, key=lambda x: x['date'])
        values = [d['value'] for d in data]

        # Calculate trend using linear regression
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Calculate growth rate (monthly)
        if y_mean != 0:
            daily_growth = slope / y_mean
            monthly_growth = daily_growth * 30 * 100  # As percentage
        else:
            monthly_growth = 0

        # Calculate volatility (coefficient of variation)
        if n > 1:
            variance = sum((v - y_mean) ** 2 for v in values) / (n - 1)
            std_dev = math.sqrt(variance)
            volatility = (std_dev / y_mean * 100) if y_mean > 0 else 0
        else:
            volatility = 0

        # Determine trend direction
        if monthly_growth > 10:
            direction = TrendDirection.STRONG_UP
        elif monthly_growth > 3:
            direction = TrendDirection.UP
        elif monthly_growth < -10:
            direction = TrendDirection.STRONG_DOWN
        elif monthly_growth < -3:
            direction = TrendDirection.DOWN
        else:
            direction = TrendDirection.STABLE

        # Calculate momentum (-100 to +100)
        momentum = max(-100, min(100, monthly_growth * 5))

        # Confidence based on data points and consistency
        if n >= 60:
            confidence = 'high'
        elif n >= 30:
            confidence = 'medium'
        else:
            confidence = 'low'

        return MarketTrend(
            direction=direction,
            momentum=round(momentum, 1),
            growth_rate_monthly=round(monthly_growth, 2),
            volatility=round(min(100, volatility), 1),
            confidence=confidence
        )

    def analyze_seasonality(
        self,
        monthly_data: Dict[int, float]
    ) -> SeasonalityAnalysis:
        """
        Analyze seasonality patterns.

        Args:
            monthly_data: Dict of month (1-12) -> sales/demand value

        Returns:
            SeasonalityAnalysis results
        """
        if not monthly_data or len(monthly_data) < 6:
            return SeasonalityAnalysis(
                pattern=SeasonalityPattern.UNKNOWN,
                peak_months=[],
                low_months=[],
                monthly_indices={},
                yoy_growth=None
            )

        # Calculate average
        avg_value = sum(monthly_data.values()) / len(monthly_data)

        # Calculate seasonal indices
        indices = {}
        for month, value in monthly_data.items():
            indices[month] = round(value / avg_value, 2) if avg_value > 0 else 1.0

        # Find peaks and lows
        sorted_months = sorted(indices.items(), key=lambda x: x[1], reverse=True)
        peak_months = [m for m, idx in sorted_months[:3] if idx > 1.1]
        low_months = [m for m, idx in sorted_months[-3:] if idx < 0.9]

        # Detect pattern
        q4_months = {10, 11, 12}
        summer_months = {6, 7, 8}
        spring_months = {3, 4, 5}
        bts_months = {8, 9}

        peak_set = set(peak_months)

        if peak_set & q4_months and len(peak_set & q4_months) >= 2:
            pattern = SeasonalityPattern.Q4_PEAK
        elif peak_set & summer_months and len(peak_set & summer_months) >= 2:
            pattern = SeasonalityPattern.SUMMER_PEAK
        elif peak_set & bts_months:
            pattern = SeasonalityPattern.BACK_TO_SCHOOL
        elif peak_set & spring_months and len(peak_set & spring_months) >= 2:
            pattern = SeasonalityPattern.SPRING_PEAK
        elif max(indices.values()) - min(indices.values()) < 0.3:
            pattern = SeasonalityPattern.YEAR_ROUND
        else:
            pattern = SeasonalityPattern.UNKNOWN

        return SeasonalityAnalysis(
            pattern=pattern,
            peak_months=peak_months,
            low_months=low_months,
            monthly_indices=indices,
            yoy_growth=None
        )

    def get_competition_level(self, competitor_count: int) -> CompetitionLevel:
        """Determine competition level from competitor count."""
        if competitor_count < 50:
            return CompetitionLevel.VERY_LOW
        elif competitor_count < 200:
            return CompetitionLevel.LOW
        elif competitor_count < 500:
            return CompetitionLevel.MODERATE
        elif competitor_count < 1000:
            return CompetitionLevel.HIGH
        else:
            return CompetitionLevel.VERY_HIGH

    def calculate_niche_score(
        self,
        metrics: NicheMetrics,
        trend: Optional[MarketTrend] = None,
        avg_profit_margin: float = 30.0
    ) -> NicheScore:
        """
        Calculate comprehensive niche opportunity score.

        Args:
            metrics: Niche metrics from competitor analysis
            trend: Market trend data (optional)
            avg_profit_margin: Expected profit margin

        Returns:
            NicheScore with detailed breakdown
        """
        warnings = []
        opportunities = []

        # 1. DEMAND SCORE (0-100)
        # Based on total revenue and sales volume
        revenue_score = min(100, metrics.total_estimated_revenue / 10000)  # $1M = 100
        sales_score = min(100, metrics.avg_monthly_sales / 5)  # 500 sales = 100
        demand_score = (revenue_score * 0.6 + sales_score * 0.4)

        if demand_score < 30:
            warnings.append("Low market demand - may be difficult to generate sales")
        elif demand_score > 70:
            opportunities.append("Strong market demand with proven sales")

        # 2. COMPETITION SCORE (0-100, higher = less competition = better)
        # Consider competitor count, brand concentration, review barriers

        # Competitor count factor (fewer = better)
        if metrics.competitor_count < 100:
            count_score = 90
        elif metrics.competitor_count < 300:
            count_score = 70
        elif metrics.competitor_count < 500:
            count_score = 50
        elif metrics.competitor_count < 1000:
            count_score = 30
        else:
            count_score = 10

        # Brand concentration (lower HHI = better for new entrants)
        if metrics.brand_concentration < 1000:
            concentration_score = 90  # Fragmented market
        elif metrics.brand_concentration < 2500:
            concentration_score = 60  # Moderately concentrated
        else:
            concentration_score = 20  # Highly concentrated

        # Review barrier (lower avg = easier entry)
        if metrics.avg_reviews < 100:
            review_barrier_score = 90
        elif metrics.avg_reviews < 500:
            review_barrier_score = 60
        elif metrics.avg_reviews < 1000:
            review_barrier_score = 40
        else:
            review_barrier_score = 20

        competition_score = (
            count_score * 0.3 +
            concentration_score * 0.3 +
            review_barrier_score * 0.4
        )

        if competition_score < 40:
            warnings.append("High competition - established brands dominate")
        elif competition_score > 70:
            opportunities.append("Low competition with room for new entrants")

        # 3. PROFIT POTENTIAL SCORE (0-100)
        # Based on price point and margins

        # Price point score (mid-range preferred: $15-$50)
        if 15 <= metrics.avg_price <= 50:
            price_score = 90
        elif 10 <= metrics.avg_price <= 75:
            price_score = 70
        elif metrics.avg_price < 10:
            price_score = 30  # Low margins likely
        else:
            price_score = 50  # High price = higher risk

        # Revenue per product
        revenue_per_product = metrics.total_estimated_revenue / max(1, metrics.competitor_count)
        if revenue_per_product > 10000:
            rev_score = 90
        elif revenue_per_product > 5000:
            rev_score = 70
        elif revenue_per_product > 2000:
            rev_score = 50
        else:
            rev_score = 30

        profit_potential_score = (price_score * 0.5 + rev_score * 0.5)

        if metrics.avg_price < 15:
            warnings.append("Low price point may result in thin margins after FBA fees")
        if profit_potential_score > 70:
            opportunities.append("Good profit potential with healthy price points")

        # 4. ENTRY BARRIER SCORE (0-100, higher = lower barriers = better)

        # Listing quality (lower quality = easier to compete)
        quality_score = 100 - (
            (metrics.avg_images / 9 * 30) +  # Max 9 images
            (metrics.avg_bullets / 5 * 30) +  # Max 5 bullets
            (metrics.video_percent * 0.2) +
            (metrics.aplus_percent * 0.2)
        )
        quality_score = max(0, min(100, quality_score))

        # FBA penetration (lower = less sophisticated market)
        fba_score = 100 - metrics.fba_seller_percent

        # Amazon presence (no Amazon = better)
        amazon_score = 100 - (metrics.amazon_seller_percent * 2)
        amazon_score = max(0, amazon_score)

        entry_barrier_score = (
            quality_score * 0.4 +
            fba_score * 0.3 +
            amazon_score * 0.3
        )

        if metrics.amazon_seller_percent > 20:
            warnings.append("Amazon sells directly in this category")
        if entry_barrier_score > 60:
            opportunities.append("Low barriers to entry - listings can be improved")

        # 5. TREND SCORE (0-100)
        if trend:
            if trend.direction == TrendDirection.STRONG_UP:
                trend_score = 95
            elif trend.direction == TrendDirection.UP:
                trend_score = 75
            elif trend.direction == TrendDirection.STABLE:
                trend_score = 50
            elif trend.direction == TrendDirection.DOWN:
                trend_score = 25
            else:  # STRONG_DOWN
                trend_score = 5

            if trend.direction in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]:
                warnings.append(f"Market is declining ({trend.growth_rate_monthly:.1f}%/month)")
            elif trend.direction in [TrendDirection.UP, TrendDirection.STRONG_UP]:
                opportunities.append(f"Growing market ({trend.growth_rate_monthly:.1f}%/month)")
        else:
            trend_score = 50  # Neutral if no data

        # OVERALL SCORE
        overall_score = (
            demand_score * 0.25 +
            competition_score * 0.25 +
            profit_potential_score * 0.20 +
            entry_barrier_score * 0.15 +
            trend_score * 0.15
        )

        # Recommendation threshold
        recommended = overall_score >= 55 and len(warnings) < 3

        if overall_score >= 70:
            opportunities.insert(0, "HIGH OPPORTUNITY: Strong overall metrics")
        elif overall_score >= 55:
            opportunities.insert(0, "MODERATE OPPORTUNITY: Worth investigating")

        return NicheScore(
            overall_score=round(overall_score, 1),
            demand_score=round(demand_score, 1),
            competition_score=round(competition_score, 1),
            profit_potential_score=round(profit_potential_score, 1),
            entry_barrier_score=round(entry_barrier_score, 1),
            trend_score=round(trend_score, 1),
            recommended=recommended,
            warnings=warnings,
            opportunities=opportunities
        )

    def identify_market_gaps(
        self,
        competitors: List[Competitor],
        metrics: NicheMetrics
    ) -> List[str]:
        """
        Identify gaps and opportunities in the market.

        Args:
            competitors: List of competitors
            metrics: Niche metrics

        Returns:
            List of identified market gaps
        """
        gaps = []

        # Price gaps
        prices = sorted([c.price for c in competitors if c.price > 0])
        if len(prices) >= 3:
            # Find price bands with no products
            price_range = prices[-1] - prices[0]
            if price_range > 20:
                bands = [(prices[0], prices[0] + price_range/3),
                         (prices[0] + price_range/3, prices[0] + 2*price_range/3),
                         (prices[0] + 2*price_range/3, prices[-1])]

                band_counts = [0, 0, 0]
                for p in prices:
                    for i, (low, high) in enumerate(bands):
                        if low <= p <= high:
                            band_counts[i] += 1
                            break

                band_names = ['budget', 'mid-range', 'premium']
                for i, count in enumerate(band_counts):
                    if count < len(prices) * 0.15:  # Less than 15% in this band
                        low, high = bands[i]
                        gaps.append(f"Price gap in {band_names[i]} segment (${low:.0f}-${high:.0f})")

        # Quality gaps
        if metrics.avg_images < 6:
            gaps.append("Most listings have few images - opportunity to stand out visually")

        if metrics.video_percent < 20:
            gaps.append("Few competitors use video - video content would differentiate")

        if metrics.aplus_percent < 30:
            gaps.append("Low A+ Content adoption - enhanced content would stand out")

        # Review gaps
        low_review_high_sales = [
            c for c in competitors
            if c.review_count < 100 and c.estimated_monthly_sales > 50
        ]
        if low_review_high_sales:
            gaps.append(f"{len(low_review_high_sales)} products selling well with <100 reviews - beatable")

        # Rating gaps
        avg_rating = metrics.avg_rating
        if avg_rating < 4.3:
            gaps.append(f"Average rating only {avg_rating:.1f}★ - quality product could dominate")

        # Brand gaps
        brands = set(c.brand for c in competitors)
        if len(brands) > metrics.competitor_count * 0.7:
            gaps.append("Fragmented market with many small brands - no dominant player")

        # FBA gaps
        if metrics.fba_seller_percent < 50:
            gaps.append(f"Only {metrics.fba_seller_percent:.0f}% use FBA - Prime badge advantage")

        return gaps

    def generate_recommendations(
        self,
        metrics: NicheMetrics,
        score: NicheScore,
        gaps: List[str]
    ) -> List[str]:
        """
        Generate actionable recommendations.

        Args:
            metrics: Niche metrics
            score: Niche score
            gaps: Identified market gaps

        Returns:
            List of recommendations
        """
        recommendations = []

        # Based on score
        if score.overall_score >= 70:
            recommendations.append("STRONG BUY: This niche shows excellent potential")
        elif score.overall_score >= 55:
            recommendations.append("CONSIDER: Proceed with thorough product research")
        else:
            recommendations.append("CAUTION: High risk - consider alternatives")

        # Price strategy
        if metrics.avg_price > 30:
            recommendations.append(f"Target ${metrics.median_price:.0f} price point (market median)")
        elif metrics.avg_price < 20:
            recommendations.append("Focus on bundling or premium variants for better margins")

        # Competition strategy
        if score.competition_score < 50:
            recommendations.append("Differentiate strongly - existing players are established")
        else:
            recommendations.append("Standard launch strategy viable - market is accessible")

        # Quality strategy
        if metrics.avg_images < 7:
            recommendations.append("Invest in professional photography (7+ images)")

        if metrics.video_percent < 30:
            recommendations.append("Create product video for competitive advantage")

        # Review strategy
        if metrics.avg_reviews > 500:
            recommendations.append("Plan aggressive launch strategy to build reviews quickly")
        else:
            recommendations.append("Moderate review count - standard launch viable")

        # Timing
        if score.trend_score > 70:
            recommendations.append("Act quickly - market is growing")
        elif score.trend_score < 30:
            recommendations.append("Consider timing carefully - market declining")

        return recommendations

    def generate_report(
        self,
        keyword: str,
        competitors: List[Competitor],
        historical_data: Optional[List[Dict]] = None,
        monthly_seasonality: Optional[Dict[int, float]] = None
    ) -> MarketReport:
        """
        Generate comprehensive market analysis report.

        Args:
            keyword: Search keyword/niche
            competitors: List of competitor products
            historical_data: Optional historical trend data
            monthly_seasonality: Optional monthly sales data

        Returns:
            Complete MarketReport
        """
        # Analyze metrics
        metrics = self.analyze_competitors(competitors)
        metrics.keyword = keyword

        # Analyze trend
        trend = self.analyze_trend(historical_data) if historical_data else MarketTrend(
            direction=TrendDirection.STABLE,
            momentum=0,
            growth_rate_monthly=0,
            volatility=0,
            confidence='low'
        )

        # Analyze seasonality
        seasonality = self.analyze_seasonality(monthly_seasonality) if monthly_seasonality else SeasonalityAnalysis(
            pattern=SeasonalityPattern.UNKNOWN,
            peak_months=[],
            low_months=[],
            monthly_indices={},
            yoy_growth=None
        )

        # Calculate score
        score = self.calculate_niche_score(metrics, trend)

        # Find gaps
        gaps = self.identify_market_gaps(competitors, metrics)

        # Generate recommendations
        recommendations = self.generate_recommendations(metrics, score, gaps)

        # Get top competitors
        top_competitors = sorted(
            competitors,
            key=lambda c: c.estimated_monthly_sales,
            reverse=True
        )[:10]

        return MarketReport(
            keyword=keyword,
            generated_at=datetime.utcnow().isoformat(),
            metrics=metrics,
            trend=trend,
            seasonality=seasonality,
            score=score,
            top_competitors=top_competitors,
            market_gaps=gaps,
            recommendations=recommendations
        )


def create_sample_competitors() -> List[Competitor]:
    """Create sample competitor data for testing."""
    return [
        Competitor("B08ABC1234", "Premium Wireless Earbuds", "TechPro", 49.99, 4.5, 2847, 2500, 450, True, False, 365, 7, 5, True, True),
        Competitor("B09DEF5678", "Budget Bluetooth Earbuds", "ValueSound", 19.99, 4.1, 1523, 8500, 180, True, False, 180, 5, 4, False, False),
        Competitor("B07GHI9012", "Sport Earbuds Waterproof", "FitAudio", 34.99, 4.3, 856, 5000, 300, True, False, 540, 8, 5, True, False),
        Competitor("B08JKL3456", "Noise Canceling Earphones", "SilentPods", 79.99, 4.6, 3241, 1500, 600, True, False, 720, 9, 5, True, True),
        Competitor("B09MNO7890", "Kids Wireless Earbuds", "KidSound", 24.99, 4.2, 412, 15000, 120, True, False, 90, 6, 4, False, False),
        Competitor("B07PQR1234", "Gaming Earbuds Low Latency", "GameAudio", 44.99, 4.4, 1876, 3500, 380, True, False, 450, 7, 5, True, True),
        Competitor("B08STU5678", "True Wireless Earbuds", "SoundMax", 39.99, 4.0, 623, 7000, 200, True, False, 270, 5, 3, False, False),
        Competitor("B09VWX9012", "Earbuds with Charging Case", "PowerPods", 29.99, 4.3, 1045, 6000, 250, True, False, 150, 6, 4, True, False),
        Competitor("B07YZA3456", "Hi-Fi Earbuds Premium Sound", "AudioPhile", 89.99, 4.7, 521, 4000, 320, True, False, 600, 9, 5, True, True),
        Competitor("B08BCD7890", "Compact Earbuds", "MiniSound", 14.99, 3.9, 2156, 12000, 140, False, False, 420, 4, 3, False, False),
        Competitor("B09EFG1234", "Wireless Earbuds Amazon", "AmazonBasics", 22.99, 4.2, 8934, 1000, 800, True, True, 365, 6, 5, False, False),
        Competitor("B07HIJ5678", "Running Earbuds", "SportFit", 32.99, 4.1, 734, 9000, 160, True, False, 300, 5, 4, False, False),
    ]


if __name__ == "__main__":
    print("=" * 70)
    print("MARKET ANALYSIS MODULE")
    print("=" * 70)

    # Create analyzer
    analyzer = MarketAnalyzer()

    # Sample data
    competitors = create_sample_competitors()

    # Sample historical data (simulated)
    base_date = datetime(2024, 1, 1)
    historical_data = [
        {'date': base_date + timedelta(days=i), 'value': 100 + i * 2 + (i % 7) * 5}
        for i in range(90)
    ]

    # Sample seasonality data
    monthly_seasonality = {
        1: 85, 2: 80, 3: 90, 4: 95, 5: 100,
        6: 105, 7: 110, 8: 100, 9: 95,
        10: 110, 11: 140, 12: 160
    }

    # Generate report
    report = analyzer.generate_report(
        keyword="wireless earbuds",
        competitors=competitors,
        historical_data=historical_data,
        monthly_seasonality=monthly_seasonality
    )

    # Display report
    print(f"\n{'='*70}")
    print(f"MARKET REPORT: {report.keyword.upper()}")
    print(f"{'='*70}")
    print(f"Generated: {report.generated_at}")

    print(f"\n{'─'*70}")
    print("MARKET METRICS")
    print(f"{'─'*70}")
    m = report.metrics
    print(f"  Total Products:      {m.competitor_count}")
    print(f"  Avg Price:           ${m.avg_price:.2f} (range: ${m.min_price:.2f}-${m.max_price:.2f})")
    print(f"  Avg Reviews:         {m.avg_reviews:.0f} (total: {m.total_reviews:,})")
    print(f"  Avg Rating:          {m.avg_rating:.1f}★")
    print(f"  Est. Monthly Rev:    ${m.total_estimated_revenue:,.0f}")
    print(f"  FBA Sellers:         {m.fba_seller_percent:.0f}%")
    print(f"  Amazon Presence:     {m.amazon_seller_percent:.0f}%")
    print(f"  Top 10 Rev Share:    {m.top_10_revenue_share:.0f}%")

    print(f"\n{'─'*70}")
    print("TREND ANALYSIS")
    print(f"{'─'*70}")
    t = report.trend
    print(f"  Direction:           {t.direction.value}")
    print(f"  Monthly Growth:      {t.growth_rate_monthly:+.1f}%")
    print(f"  Momentum:            {t.momentum:+.0f}")
    print(f"  Volatility:          {t.volatility:.0f}%")

    print(f"\n{'─'*70}")
    print("SEASONALITY")
    print(f"{'─'*70}")
    s = report.seasonality
    print(f"  Pattern:             {s.pattern.value}")
    print(f"  Peak Months:         {s.peak_months}")
    print(f"  Low Months:          {s.low_months}")

    print(f"\n{'─'*70}")
    print("NICHE SCORE")
    print(f"{'─'*70}")
    sc = report.score
    print(f"  OVERALL:             {sc.overall_score:.0f}/100 {'✓ RECOMMENDED' if sc.recommended else '⚠ CAUTION'}")
    print(f"  ├─ Demand:           {sc.demand_score:.0f}/100")
    print(f"  ├─ Competition:      {sc.competition_score:.0f}/100")
    print(f"  ├─ Profit Potential: {sc.profit_potential_score:.0f}/100")
    print(f"  ├─ Entry Barriers:   {sc.entry_barrier_score:.0f}/100")
    print(f"  └─ Trend:            {sc.trend_score:.0f}/100")

    if sc.warnings:
        print(f"\n  ⚠ Warnings:")
        for w in sc.warnings:
            print(f"    • {w}")

    if sc.opportunities:
        print(f"\n  ✓ Opportunities:")
        for o in sc.opportunities:
            print(f"    • {o}")

    print(f"\n{'─'*70}")
    print("MARKET GAPS")
    print(f"{'─'*70}")
    for gap in report.market_gaps:
        print(f"  • {gap}")

    print(f"\n{'─'*70}")
    print("RECOMMENDATIONS")
    print(f"{'─'*70}")
    for rec in report.recommendations:
        print(f"  → {rec}")

    print(f"\n{'─'*70}")
    print("TOP 5 COMPETITORS")
    print(f"{'─'*70}")
    print(f"{'ASIN':<12} {'Brand':<12} {'Price':>8} {'Rating':>6} {'Reviews':>8} {'Sales/mo':>10}")
    for c in report.top_competitors[:5]:
        print(f"{c.asin:<12} {c.brand:<12} ${c.price:>6.2f} {c.rating:>5.1f}★ {c.review_count:>7,} {c.estimated_monthly_sales:>9,}")
