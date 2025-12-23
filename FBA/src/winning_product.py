#!/usr/bin/env python3
"""
Winning Product Finder for Amazon FBA

Identifies potentially profitable products using multi-criteria analysis:
- Market opportunity (demand, search volume, trends)
- Competition assessment (# sellers, review barriers, brand dominance)
- Profitability analysis (margins, fees, ROI potential)
- Risk evaluation (seasonality, complexity, barriers to entry)
- Overall "winning product" score

Combines insights from all FBA research modules.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime


class ProductCategory(Enum):
    HOME_KITCHEN = "Home & Kitchen"
    ELECTRONICS = "Electronics"
    SPORTS_OUTDOORS = "Sports & Outdoors"
    HEALTH_PERSONAL = "Health & Personal Care"
    BEAUTY = "Beauty"
    TOYS_GAMES = "Toys & Games"
    PET_SUPPLIES = "Pet Supplies"
    BABY = "Baby"
    OFFICE = "Office Products"
    GARDEN = "Garden & Outdoor"
    AUTOMOTIVE = "Automotive"
    TOOLS = "Tools & Home Improvement"
    OTHER = "Other"


class OpportunityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    LOW = "low"
    POOR = "poor"


class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class MarketData:
    """Market data for product analysis"""
    # Search & Demand
    main_keyword: str
    monthly_search_volume: int
    keyword_trend: str  # "growing", "stable", "declining"

    # Competition
    num_competitors: int
    avg_competitor_reviews: int
    avg_competitor_rating: float
    top_competitor_bsr: int
    num_branded_results: int  # How many page 1 results are big brands

    # Pricing
    avg_price: float
    price_range_low: float
    price_range_high: float

    # Category
    category: ProductCategory
    subcategory: str = ""

    # Seasonality
    peak_months: list[int] = field(default_factory=list)
    seasonality_variance: float = 0.0  # 0-1, higher = more seasonal


@dataclass
class ProductCosts:
    """Product cost structure"""
    estimated_cogs: float  # Cost of goods
    shipping_to_amazon: float = 0.50
    fba_fee_estimate: float = 5.00
    prep_cost: float = 0.25
    packaging_cost: float = 0.50

    @property
    def total_cost(self) -> float:
        return (self.estimated_cogs + self.shipping_to_amazon +
                self.fba_fee_estimate + self.prep_cost + self.packaging_cost)


@dataclass
class CriteriaScore:
    """Score for a single criterion"""
    name: str
    score: int  # 0-100
    weight: float
    weighted_score: float
    details: str
    status: str  # "pass", "warning", "fail"


@dataclass
class OpportunityAnalysis:
    """Market opportunity analysis"""
    demand_score: int
    search_volume_assessment: str
    trend_assessment: str
    market_size_estimate: float
    growth_potential: str


@dataclass
class CompetitionAnalysis:
    """Competition analysis"""
    competition_score: int
    barrier_to_entry: str
    review_gap_opportunity: bool
    brand_dominance: str
    differentiation_potential: str
    weaknesses_found: list[str]


@dataclass
class ProfitabilityAnalysis:
    """Profitability analysis"""
    profitability_score: int
    estimated_selling_price: float
    estimated_profit_per_unit: float
    estimated_margin: float
    estimated_roi: float
    break_even_units: int
    monthly_profit_potential: float


@dataclass
class RiskAnalysis:
    """Risk assessment"""
    risk_score: int  # Lower is better
    risk_level: RiskLevel
    risks: list[str]
    mitigations: list[str]
    deal_breakers: list[str]


@dataclass
class WinningProductScore:
    """Complete winning product analysis"""
    # Basic info
    product_idea: str
    category: ProductCategory
    analysis_date: datetime

    # Overall scores
    overall_score: int  # 0-100
    opportunity_level: OpportunityLevel
    recommendation: str

    # Individual criteria scores
    criteria_scores: list[CriteriaScore]

    # Detailed analysis
    opportunity: OpportunityAnalysis
    competition: CompetitionAnalysis
    profitability: ProfitabilityAnalysis
    risk: RiskAnalysis

    # Summary
    pros: list[str]
    cons: list[str]
    next_steps: list[str]

    # Estimates
    estimated_monthly_revenue: float
    estimated_monthly_profit: float
    estimated_startup_cost: float
    time_to_profit_months: int


class WinningProductFinder:
    """
    Analyzes products to identify winning opportunities.

    Scoring Criteria:
    1. Demand (20%): Search volume, trends, market size
    2. Competition (25%): # competitors, reviews, brands
    3. Profitability (25%): Margins, ROI, break-even
    4. Differentiation (15%): Room for improvement
    5. Risk (15%): Seasonality, complexity, barriers
    """

    # Scoring weights
    WEIGHTS = {
        "demand": 0.20,
        "competition": 0.25,
        "profitability": 0.25,
        "differentiation": 0.15,
        "risk": 0.15
    }

    # Thresholds for ideal product
    IDEAL_CRITERIA = {
        "min_search_volume": 3000,
        "max_competitors": 300,
        "max_avg_reviews": 300,
        "min_price": 15.00,
        "max_price": 50.00,
        "min_margin": 30.0,
        "max_top_bsr": 50000,
        "max_brand_dominance": 3,  # Max branded results on page 1
    }

    # Category-specific adjustments
    CATEGORY_FACTORS = {
        ProductCategory.HOME_KITCHEN: {"competition_mult": 1.2, "margin_target": 35},
        ProductCategory.ELECTRONICS: {"competition_mult": 1.3, "margin_target": 25},
        ProductCategory.SPORTS_OUTDOORS: {"competition_mult": 1.0, "margin_target": 35},
        ProductCategory.HEALTH_PERSONAL: {"competition_mult": 1.1, "margin_target": 40},
        ProductCategory.BEAUTY: {"competition_mult": 1.2, "margin_target": 45},
        ProductCategory.TOYS_GAMES: {"competition_mult": 1.1, "margin_target": 35},
        ProductCategory.PET_SUPPLIES: {"competition_mult": 0.9, "margin_target": 40},
        ProductCategory.BABY: {"competition_mult": 1.0, "margin_target": 35},
        ProductCategory.OFFICE: {"competition_mult": 0.8, "margin_target": 30},
        ProductCategory.GARDEN: {"competition_mult": 0.9, "margin_target": 35},
        ProductCategory.AUTOMOTIVE: {"competition_mult": 0.8, "margin_target": 30},
        ProductCategory.TOOLS: {"competition_mult": 0.9, "margin_target": 30},
        ProductCategory.OTHER: {"competition_mult": 1.0, "margin_target": 30},
    }

    def __init__(self):
        self.criteria_results: list[CriteriaScore] = []

    def analyze_product(self, product_idea: str, market: MarketData,
                       costs: ProductCosts, target_price: float = None) -> WinningProductScore:
        """
        Analyze a product idea and return winning product score.
        """
        self.criteria_results = []

        # Determine target price
        if target_price is None:
            target_price = market.avg_price

        # Calculate referral fee (15% typical)
        referral_fee = target_price * 0.15
        total_cost = costs.total_cost + referral_fee
        profit_per_unit = target_price - total_cost
        margin = (profit_per_unit / target_price) * 100 if target_price > 0 else 0

        # Analyze each criteria
        demand_score = self._score_demand(market)
        competition_score = self._score_competition(market)
        profitability_score = self._score_profitability(target_price, profit_per_unit, margin, market)
        differentiation_score = self._score_differentiation(market)
        risk_score = self._score_risk(market, margin)

        # Calculate overall score
        overall = int(
            demand_score.score * self.WEIGHTS["demand"] +
            competition_score.score * self.WEIGHTS["competition"] +
            profitability_score.score * self.WEIGHTS["profitability"] +
            differentiation_score.score * self.WEIGHTS["differentiation"] +
            (100 - risk_score.score) * self.WEIGHTS["risk"]  # Invert risk score
        )

        # Determine opportunity level
        if overall >= 80:
            opportunity = OpportunityLevel.EXCELLENT
            recommendation = "Strong opportunity - proceed with product validation"
        elif overall >= 65:
            opportunity = OpportunityLevel.GOOD
            recommendation = "Good potential - validate and differentiate"
        elif overall >= 50:
            opportunity = OpportunityLevel.MODERATE
            recommendation = "Moderate opportunity - needs strong differentiation"
        elif overall >= 35:
            opportunity = OpportunityLevel.LOW
            recommendation = "Challenging market - consider alternatives"
        else:
            opportunity = OpportunityLevel.POOR
            recommendation = "Not recommended - high risk, low reward"

        # Build detailed analyses
        opportunity_analysis = self._build_opportunity_analysis(market, demand_score)
        competition_analysis = self._build_competition_analysis(market, competition_score)
        profitability_analysis = self._build_profitability_analysis(
            target_price, costs, profit_per_unit, margin, market
        )
        risk_analysis = self._build_risk_analysis(market, margin, risk_score)

        # Generate pros/cons
        pros, cons = self._generate_pros_cons(market, margin, overall)

        # Next steps
        next_steps = self._generate_next_steps(opportunity, market)

        # Financial estimates
        est_monthly_units = self._estimate_monthly_sales(market, overall)
        est_monthly_revenue = est_monthly_units * target_price
        est_monthly_profit = est_monthly_units * profit_per_unit

        # Startup cost estimate
        initial_inventory = max(500, est_monthly_units * 2)
        startup_cost = initial_inventory * costs.estimated_cogs + 500  # + PPC budget

        # Time to profit
        if est_monthly_profit > 0:
            time_to_profit = max(1, int(startup_cost / est_monthly_profit) + 1)
        else:
            time_to_profit = 12

        return WinningProductScore(
            product_idea=product_idea,
            category=market.category,
            analysis_date=datetime.now(),
            overall_score=overall,
            opportunity_level=opportunity,
            recommendation=recommendation,
            criteria_scores=self.criteria_results,
            opportunity=opportunity_analysis,
            competition=competition_analysis,
            profitability=profitability_analysis,
            risk=risk_analysis,
            pros=pros,
            cons=cons,
            next_steps=next_steps,
            estimated_monthly_revenue=round(est_monthly_revenue, 2),
            estimated_monthly_profit=round(est_monthly_profit, 2),
            estimated_startup_cost=round(startup_cost, 2),
            time_to_profit_months=time_to_profit
        )

    def _score_demand(self, market: MarketData) -> CriteriaScore:
        """Score market demand"""
        score = 50  # Base score
        details = []

        # Search volume scoring
        vol = market.monthly_search_volume
        if vol >= 50000:
            score += 30
            details.append(f"Excellent search volume ({vol:,}/mo)")
        elif vol >= 20000:
            score += 25
            details.append(f"Strong search volume ({vol:,}/mo)")
        elif vol >= 10000:
            score += 20
            details.append(f"Good search volume ({vol:,}/mo)")
        elif vol >= 5000:
            score += 10
            details.append(f"Moderate search volume ({vol:,}/mo)")
        elif vol >= 3000:
            score += 5
            details.append(f"Low but viable search volume ({vol:,}/mo)")
        else:
            score -= 20
            details.append(f"Low search volume ({vol:,}/mo) - limited demand")

        # Trend scoring
        if market.keyword_trend == "growing":
            score += 15
            details.append("Growing trend")
        elif market.keyword_trend == "stable":
            score += 5
            details.append("Stable demand")
        else:
            score -= 10
            details.append("Declining trend")

        # BSR check (proxy for sales velocity)
        if market.top_competitor_bsr < 10000:
            score += 5
            details.append("Strong category sales velocity")
        elif market.top_competitor_bsr > 100000:
            score -= 10
            details.append("Low category sales velocity")

        score = max(0, min(100, score))
        status = "pass" if score >= 60 else "warning" if score >= 40 else "fail"

        result = CriteriaScore(
            name="Market Demand",
            score=score,
            weight=self.WEIGHTS["demand"],
            weighted_score=score * self.WEIGHTS["demand"],
            details="; ".join(details),
            status=status
        )
        self.criteria_results.append(result)
        return result

    def _score_competition(self, market: MarketData) -> CriteriaScore:
        """Score competition level"""
        score = 50
        details = []

        # Number of competitors
        if market.num_competitors < 100:
            score += 25
            details.append(f"Low competition ({market.num_competitors} sellers)")
        elif market.num_competitors < 200:
            score += 15
            details.append(f"Moderate competition ({market.num_competitors} sellers)")
        elif market.num_competitors < 500:
            score += 5
            details.append(f"Competitive market ({market.num_competitors} sellers)")
        else:
            score -= 15
            details.append(f"Highly competitive ({market.num_competitors}+ sellers)")

        # Review barrier
        if market.avg_competitor_reviews < 100:
            score += 25
            details.append(f"Low review barrier (avg {market.avg_competitor_reviews})")
        elif market.avg_competitor_reviews < 300:
            score += 15
            details.append(f"Moderate review barrier (avg {market.avg_competitor_reviews})")
        elif market.avg_competitor_reviews < 500:
            score += 5
            details.append(f"High review barrier (avg {market.avg_competitor_reviews})")
        else:
            score -= 15
            details.append(f"Very high review barrier (avg {market.avg_competitor_reviews})")

        # Brand dominance
        if market.num_branded_results <= 2:
            score += 15
            details.append("Low brand dominance")
        elif market.num_branded_results <= 4:
            score += 5
            details.append("Moderate brand presence")
        else:
            score -= 15
            details.append("High brand dominance - difficult entry")

        # Rating quality (opportunity if avg is low)
        if market.avg_competitor_rating < 4.0:
            score += 10
            details.append(f"Quality gap opportunity (avg rating {market.avg_competitor_rating})")
        elif market.avg_competitor_rating < 4.3:
            score += 5
            details.append("Room for quality differentiation")

        # Apply category factor
        cat_factor = self.CATEGORY_FACTORS.get(market.category, {}).get("competition_mult", 1.0)
        score = int(score / cat_factor)

        score = max(0, min(100, score))
        status = "pass" if score >= 55 else "warning" if score >= 35 else "fail"

        result = CriteriaScore(
            name="Competition",
            score=score,
            weight=self.WEIGHTS["competition"],
            weighted_score=score * self.WEIGHTS["competition"],
            details="; ".join(details),
            status=status
        )
        self.criteria_results.append(result)
        return result

    def _score_profitability(self, price: float, profit: float,
                            margin: float, market: MarketData) -> CriteriaScore:
        """Score profitability potential"""
        score = 50
        details = []

        # Margin scoring
        target_margin = self.CATEGORY_FACTORS.get(market.category, {}).get("margin_target", 30)

        if margin >= target_margin + 15:
            score += 30
            details.append(f"Excellent margin ({margin:.1f}%)")
        elif margin >= target_margin + 5:
            score += 20
            details.append(f"Strong margin ({margin:.1f}%)")
        elif margin >= target_margin:
            score += 10
            details.append(f"Good margin ({margin:.1f}%)")
        elif margin >= target_margin - 10:
            score += 0
            details.append(f"Acceptable margin ({margin:.1f}%)")
        elif margin >= 15:
            score -= 10
            details.append(f"Low margin ({margin:.1f}%) - optimize costs")
        else:
            score -= 25
            details.append(f"Poor margin ({margin:.1f}%) - not viable")

        # Price point scoring
        if 20 <= price <= 50:
            score += 15
            details.append(f"Sweet spot price (${price:.2f})")
        elif 15 <= price <= 75:
            score += 10
            details.append(f"Acceptable price point (${price:.2f})")
        elif price < 15:
            score -= 10
            details.append(f"Low price point (${price:.2f}) - volume needed")
        else:
            score -= 5
            details.append(f"Premium price (${price:.2f}) - harder to launch")

        # Profit per unit
        if profit >= 10:
            score += 10
            details.append(f"Strong profit/unit (${profit:.2f})")
        elif profit >= 5:
            score += 5
            details.append(f"Decent profit/unit (${profit:.2f})")
        elif profit < 3:
            score -= 10
            details.append(f"Low profit/unit (${profit:.2f})")

        score = max(0, min(100, score))
        status = "pass" if score >= 60 else "warning" if score >= 40 else "fail"

        result = CriteriaScore(
            name="Profitability",
            score=score,
            weight=self.WEIGHTS["profitability"],
            weighted_score=score * self.WEIGHTS["profitability"],
            details="; ".join(details),
            status=status
        )
        self.criteria_results.append(result)
        return result

    def _score_differentiation(self, market: MarketData) -> CriteriaScore:
        """Score differentiation potential"""
        score = 50
        details = []

        # Rating gap (low avg rating = opportunity)
        if market.avg_competitor_rating < 3.8:
            score += 30
            details.append("Significant quality gap - easy differentiation")
        elif market.avg_competitor_rating < 4.2:
            score += 20
            details.append("Quality improvement opportunity exists")
        elif market.avg_competitor_rating < 4.5:
            score += 10
            details.append("Some room for quality differentiation")
        else:
            score -= 5
            details.append("High quality bar - need unique angle")

        # Price range (wide range = room for positioning)
        price_spread = market.price_range_high - market.price_range_low
        price_spread_pct = price_spread / market.avg_price if market.avg_price > 0 else 0

        if price_spread_pct > 1.0:
            score += 15
            details.append("Wide price range - multiple positioning options")
        elif price_spread_pct > 0.5:
            score += 10
            details.append("Good price spread for positioning")
        else:
            score += 0
            details.append("Narrow price range")

        # Low brand dominance = easier differentiation
        if market.num_branded_results <= 2:
            score += 10
            details.append("Open market for new brands")
        elif market.num_branded_results >= 5:
            score -= 10
            details.append("Brand-dominated - differentiation harder")

        score = max(0, min(100, score))
        status = "pass" if score >= 55 else "warning" if score >= 35 else "fail"

        result = CriteriaScore(
            name="Differentiation",
            score=score,
            weight=self.WEIGHTS["differentiation"],
            weighted_score=score * self.WEIGHTS["differentiation"],
            details="; ".join(details),
            status=status
        )
        self.criteria_results.append(result)
        return result

    def _score_risk(self, market: MarketData, margin: float) -> CriteriaScore:
        """Score risk level (higher score = more risk)"""
        score = 30  # Base risk
        details = []

        # Seasonality risk
        if market.seasonality_variance > 0.5:
            score += 25
            details.append("High seasonality risk")
        elif market.seasonality_variance > 0.3:
            score += 15
            details.append("Moderate seasonality")
        elif market.seasonality_variance > 0.1:
            score += 5
            details.append("Low seasonality")
        else:
            details.append("Evergreen demand")

        # Competition risk
        if market.num_competitors > 500:
            score += 20
            details.append("Saturated market risk")
        elif market.num_competitors > 200:
            score += 10
            details.append("Competitive pressure risk")

        # Margin risk
        if margin < 20:
            score += 20
            details.append("Thin margin risk")
        elif margin < 30:
            score += 10
            details.append("Margin compression risk")

        # Price war risk (if low avg price)
        if market.avg_price < 20:
            score += 10
            details.append("Price war risk in low-price segment")

        # Review barrier risk
        if market.avg_competitor_reviews > 500:
            score += 15
            details.append("High review barrier risk")

        # Brand risk
        if market.num_branded_results >= 5:
            score += 15
            details.append("Brand dominance risk")

        score = max(0, min(100, score))
        status = "pass" if score <= 40 else "warning" if score <= 60 else "fail"

        result = CriteriaScore(
            name="Risk Level",
            score=score,
            weight=self.WEIGHTS["risk"],
            weighted_score=(100 - score) * self.WEIGHTS["risk"],  # Inverted for overall
            details="; ".join(details),
            status=status
        )
        self.criteria_results.append(result)
        return result

    def _build_opportunity_analysis(self, market: MarketData,
                                   demand_score: CriteriaScore) -> OpportunityAnalysis:
        """Build detailed opportunity analysis"""
        # Market size estimate
        avg_price = market.avg_price
        est_conversion = 0.05  # 5% search to sale
        est_market_size = market.monthly_search_volume * est_conversion * avg_price * 12

        # Growth potential
        if market.keyword_trend == "growing" and market.monthly_search_volume > 10000:
            growth = "High growth potential"
        elif market.keyword_trend == "growing":
            growth = "Moderate growth potential"
        elif market.keyword_trend == "stable":
            growth = "Stable market"
        else:
            growth = "Declining market"

        return OpportunityAnalysis(
            demand_score=demand_score.score,
            search_volume_assessment=f"{market.monthly_search_volume:,} monthly searches",
            trend_assessment=market.keyword_trend.title(),
            market_size_estimate=round(est_market_size, 0),
            growth_potential=growth
        )

    def _build_competition_analysis(self, market: MarketData,
                                   competition_score: CriteriaScore) -> CompetitionAnalysis:
        """Build detailed competition analysis"""
        # Barrier to entry
        if market.avg_competitor_reviews < 100 and market.num_competitors < 200:
            barrier = "Low"
        elif market.avg_competitor_reviews < 300 and market.num_competitors < 500:
            barrier = "Moderate"
        else:
            barrier = "High"

        # Review gap
        review_gap = market.avg_competitor_reviews < 200

        # Brand dominance
        if market.num_branded_results <= 2:
            brand_dom = "Low - open market"
        elif market.num_branded_results <= 4:
            brand_dom = "Moderate"
        else:
            brand_dom = "High - brand controlled"

        # Differentiation
        if market.avg_competitor_rating < 4.0:
            diff_potential = "High - quality gap exists"
        elif market.avg_competitor_rating < 4.3:
            diff_potential = "Moderate - room for improvement"
        else:
            diff_potential = "Low - need unique angle"

        # Weaknesses
        weaknesses = []
        if market.avg_competitor_rating < 4.2:
            weaknesses.append("Quality issues in existing products")
        if market.avg_competitor_reviews < 200:
            weaknesses.append("Low review counts - new entrants viable")
        if market.num_branded_results <= 2:
            weaknesses.append("No dominant brands")

        return CompetitionAnalysis(
            competition_score=competition_score.score,
            barrier_to_entry=barrier,
            review_gap_opportunity=review_gap,
            brand_dominance=brand_dom,
            differentiation_potential=diff_potential,
            weaknesses_found=weaknesses
        )

    def _build_profitability_analysis(self, price: float, costs: ProductCosts,
                                     profit: float, margin: float,
                                     market: MarketData) -> ProfitabilityAnalysis:
        """Build detailed profitability analysis"""
        roi = (profit / costs.estimated_cogs) * 100 if costs.estimated_cogs > 0 else 0

        # Break-even units (assuming $500 PPC + initial costs)
        monthly_fixed = 300  # PPC estimate
        break_even = int(monthly_fixed / max(0.01, profit)) if profit > 0 else 9999

        # Monthly profit potential
        est_units = int(market.monthly_search_volume * 0.02 * 0.1)  # 2% conversion, 10% market share
        monthly_profit = est_units * profit

        return ProfitabilityAnalysis(
            profitability_score=self.criteria_results[2].score if len(self.criteria_results) > 2 else 50,
            estimated_selling_price=round(price, 2),
            estimated_profit_per_unit=round(profit, 2),
            estimated_margin=round(margin, 1),
            estimated_roi=round(roi, 1),
            break_even_units=break_even,
            monthly_profit_potential=round(monthly_profit, 2)
        )

    def _build_risk_analysis(self, market: MarketData, margin: float,
                            risk_score: CriteriaScore) -> RiskAnalysis:
        """Build detailed risk analysis"""
        risks = []
        mitigations = []
        deal_breakers = []

        # Identify risks and mitigations
        if market.seasonality_variance > 0.3:
            risks.append("Seasonal demand fluctuations")
            mitigations.append("Plan inventory for peak seasons, diversify products")

        if market.num_competitors > 300:
            risks.append("High competition")
            mitigations.append("Focus on differentiation and niche targeting")

        if margin < 25:
            risks.append("Low profit margins")
            mitigations.append("Optimize sourcing, consider premium positioning")

        if market.avg_competitor_reviews > 500:
            risks.append("High review barrier")
            mitigations.append("Implement aggressive review strategy, use Vine")

        if market.num_branded_results >= 5:
            risks.append("Brand-dominated market")
            mitigations.append("Target underserved niches within category")

        # Deal breakers
        if margin < 15:
            deal_breakers.append("Margins too low for sustainable business")
        if market.monthly_search_volume < 1000:
            deal_breakers.append("Insufficient demand")
        if market.keyword_trend == "declining" and market.monthly_search_volume < 5000:
            deal_breakers.append("Declining market with low volume")

        # Risk level
        if risk_score.score >= 70:
            level = RiskLevel.VERY_HIGH
        elif risk_score.score >= 55:
            level = RiskLevel.HIGH
        elif risk_score.score >= 40:
            level = RiskLevel.MODERATE
        elif risk_score.score >= 25:
            level = RiskLevel.LOW
        else:
            level = RiskLevel.VERY_LOW

        return RiskAnalysis(
            risk_score=risk_score.score,
            risk_level=level,
            risks=risks,
            mitigations=mitigations,
            deal_breakers=deal_breakers
        )

    def _generate_pros_cons(self, market: MarketData, margin: float,
                           overall: int) -> tuple[list[str], list[str]]:
        """Generate pros and cons lists"""
        pros = []
        cons = []

        # Demand
        if market.monthly_search_volume >= 10000:
            pros.append(f"Strong demand ({market.monthly_search_volume:,} searches/mo)")
        elif market.monthly_search_volume < 3000:
            cons.append(f"Limited demand ({market.monthly_search_volume:,} searches/mo)")

        # Trend
        if market.keyword_trend == "growing":
            pros.append("Growing market trend")
        elif market.keyword_trend == "declining":
            cons.append("Declining market trend")

        # Competition
        if market.num_competitors < 150:
            pros.append(f"Low competition ({market.num_competitors} sellers)")
        elif market.num_competitors > 400:
            cons.append(f"High competition ({market.num_competitors} sellers)")

        # Reviews
        if market.avg_competitor_reviews < 150:
            pros.append("Low review barrier to entry")
        elif market.avg_competitor_reviews > 400:
            cons.append("High review barrier")

        # Brands
        if market.num_branded_results <= 2:
            pros.append("Low brand dominance")
        elif market.num_branded_results >= 5:
            cons.append("Market controlled by established brands")

        # Margin
        if margin >= 35:
            pros.append(f"Strong profit margin ({margin:.0f}%)")
        elif margin < 25:
            cons.append(f"Low profit margin ({margin:.0f}%)")

        # Quality gap
        if market.avg_competitor_rating < 4.0:
            pros.append("Quality gap opportunity")
        elif market.avg_competitor_rating >= 4.5:
            cons.append("High quality bar from competitors")

        # Seasonality
        if market.seasonality_variance > 0.4:
            cons.append("High seasonality")
        elif market.seasonality_variance < 0.15:
            pros.append("Evergreen demand")

        return pros[:5], cons[:5]

    def _generate_next_steps(self, opportunity: OpportunityLevel,
                            market: MarketData) -> list[str]:
        """Generate recommended next steps"""
        steps = []

        if opportunity in [OpportunityLevel.EXCELLENT, OpportunityLevel.GOOD]:
            steps = [
                "1. Validate demand with deeper keyword research",
                "2. Analyze top 10 competitor listings for weaknesses",
                "3. Source samples from 3-5 suppliers",
                "4. Calculate exact landed costs",
                "5. Develop differentiation strategy",
                "6. Create product mockups for validation",
                "7. Plan launch strategy and PPC budget"
            ]
        elif opportunity == OpportunityLevel.MODERATE:
            steps = [
                "1. Identify specific niche within category",
                "2. Research differentiation opportunities",
                "3. Analyze if unique angle is viable",
                "4. Consider adjacent product opportunities",
                "5. Validate with small test order if proceeding"
            ]
        else:
            steps = [
                "1. Consider alternative product ideas",
                "2. If proceeding, identify specific underserved niche",
                "3. Validate unique value proposition",
                "4. Calculate minimum viable differentiation"
            ]

        return steps

    def _estimate_monthly_sales(self, market: MarketData, score: int) -> int:
        """Estimate monthly sales potential"""
        # Base estimate from search volume
        base_units = int(market.monthly_search_volume * 0.02 * 0.05)  # 2% CVR, 5% market share

        # Adjust by score
        multiplier = 0.5 + (score / 100)  # 0.5x to 1.5x

        # Adjust by competition
        if market.num_competitors < 100:
            multiplier *= 1.3
        elif market.num_competitors > 400:
            multiplier *= 0.7

        return max(50, int(base_units * multiplier))


def create_sample_market_data() -> MarketData:
    """Create sample market data for testing"""
    return MarketData(
        main_keyword="silicone ice cube tray",
        monthly_search_volume=22000,
        keyword_trend="stable",
        num_competitors=180,
        avg_competitor_reviews=245,
        avg_competitor_rating=4.1,
        top_competitor_bsr=8500,
        num_branded_results=2,
        avg_price=14.99,
        price_range_low=8.99,
        price_range_high=24.99,
        category=ProductCategory.HOME_KITCHEN,
        subcategory="Kitchen Storage",
        peak_months=[6, 7, 8],
        seasonality_variance=0.25
    )


def create_sample_costs() -> ProductCosts:
    """Create sample cost structure"""
    return ProductCosts(
        estimated_cogs=2.50,
        shipping_to_amazon=0.40,
        fba_fee_estimate=4.25,
        prep_cost=0.20,
        packaging_cost=0.30
    )


if __name__ == "__main__":
    print("=" * 70)
    print("WINNING PRODUCT FINDER")
    print("=" * 70)

    finder = WinningProductFinder()
    market = create_sample_market_data()
    costs = create_sample_costs()

    result = finder.analyze_product(
        product_idea="Premium Silicone Ice Cube Tray with Lid",
        market=market,
        costs=costs,
        target_price=15.99
    )

    # Overall Score
    print(f"\n{'─' * 70}")
    print(f"PRODUCT: {result.product_idea}")
    print(f"{'─' * 70}")

    print(f"\n  OVERALL SCORE: {result.overall_score}/100")
    print(f"  OPPORTUNITY:   {result.opportunity_level.value.upper()}")
    print(f"\n  {result.recommendation}")

    # Criteria Breakdown
    print(f"\n{'─' * 70}")
    print("CRITERIA BREAKDOWN")
    print(f"{'─' * 70}")
    for criteria in result.criteria_scores:
        status_icon = "✓" if criteria.status == "pass" else "⚠" if criteria.status == "warning" else "✗"
        print(f"  {status_icon} {criteria.name:20} {criteria.score:3}/100 (weight: {criteria.weight:.0%})")
        print(f"      {criteria.details[:65]}...")

    # Market Opportunity
    print(f"\n{'─' * 70}")
    print("MARKET OPPORTUNITY")
    print(f"{'─' * 70}")
    print(f"  Search Volume:    {result.opportunity.search_volume_assessment}")
    print(f"  Trend:            {result.opportunity.trend_assessment}")
    print(f"  Est. Market Size: ${result.opportunity.market_size_estimate:,.0f}/year")
    print(f"  Growth Potential: {result.opportunity.growth_potential}")

    # Competition
    print(f"\n{'─' * 70}")
    print("COMPETITION ANALYSIS")
    print(f"{'─' * 70}")
    print(f"  Barrier to Entry:    {result.competition.barrier_to_entry}")
    print(f"  Brand Dominance:     {result.competition.brand_dominance}")
    print(f"  Review Gap:          {'Yes' if result.competition.review_gap_opportunity else 'No'}")
    print(f"  Differentiation:     {result.competition.differentiation_potential}")
    if result.competition.weaknesses_found:
        print(f"  Competitor Weaknesses:")
        for w in result.competition.weaknesses_found:
            print(f"    • {w}")

    # Profitability
    print(f"\n{'─' * 70}")
    print("PROFITABILITY")
    print(f"{'─' * 70}")
    print(f"  Selling Price:       ${result.profitability.estimated_selling_price:.2f}")
    print(f"  Profit/Unit:         ${result.profitability.estimated_profit_per_unit:.2f}")
    print(f"  Margin:              {result.profitability.estimated_margin:.1f}%")
    print(f"  ROI:                 {result.profitability.estimated_roi:.0f}%")
    print(f"  Break-even:          {result.profitability.break_even_units} units/month")

    # Risk Assessment
    print(f"\n{'─' * 70}")
    print("RISK ASSESSMENT")
    print(f"{'─' * 70}")
    print(f"  Risk Level: {result.risk.risk_level.value.upper()}")
    if result.risk.risks:
        print(f"  Risks:")
        for r in result.risk.risks[:3]:
            print(f"    ⚠ {r}")
    if result.risk.deal_breakers:
        print(f"  Deal Breakers:")
        for db in result.risk.deal_breakers:
            print(f"    ✗ {db}")

    # Pros & Cons
    print(f"\n{'─' * 70}")
    print("PROS & CONS")
    print(f"{'─' * 70}")
    print("  PROS:")
    for pro in result.pros:
        print(f"    ✓ {pro}")
    print("  CONS:")
    for con in result.cons:
        print(f"    ✗ {con}")

    # Financial Projections
    print(f"\n{'─' * 70}")
    print("FINANCIAL PROJECTIONS")
    print(f"{'─' * 70}")
    print(f"  Est. Monthly Revenue:  ${result.estimated_monthly_revenue:,.2f}")
    print(f"  Est. Monthly Profit:   ${result.estimated_monthly_profit:,.2f}")
    print(f"  Startup Cost:          ${result.estimated_startup_cost:,.2f}")
    print(f"  Time to Profit:        ~{result.time_to_profit_months} months")

    # Next Steps
    print(f"\n{'─' * 70}")
    print("NEXT STEPS")
    print(f"{'─' * 70}")
    for step in result.next_steps[:5]:
        print(f"  {step}")

    print(f"\n{'=' * 70}")
