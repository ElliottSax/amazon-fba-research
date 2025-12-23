#!/usr/bin/env python3
"""
Product Launch Strategy Generator for Amazon FBA

Combines insights from all modules to create a comprehensive launch plan:
- Market positioning
- Pricing strategy
- PPC campaign structure
- Inventory planning
- Timeline and milestones
- Risk assessment
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime, timedelta


class LaunchPhase(Enum):
    PRE_LAUNCH = "pre_launch"
    LAUNCH = "launch"
    GROWTH = "growth"
    OPTIMIZATION = "optimization"
    MAINTENANCE = "maintenance"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ProductInfo:
    """Product information for launch planning"""
    asin: Optional[str]
    title: str
    category: str
    unit_cost: float
    target_price: float
    weight_lbs: float
    dimensions: tuple[float, float, float]  # L x W x H in inches
    initial_inventory: int = 500
    lead_time_days: int = 45
    target_monthly_sales: int = 300


@dataclass
class MarketContext:
    """Market context from analysis"""
    competitor_count: int
    avg_competitor_price: float
    avg_competitor_rating: float
    avg_competitor_reviews: int
    top_competitor_bsr: int
    keyword_search_volume: int
    estimated_monthly_market_revenue: float
    seasonality_peak_months: list[int] = field(default_factory=list)
    is_trending: bool = False


@dataclass
class LaunchMilestone:
    """Key milestone in launch timeline"""
    phase: LaunchPhase
    name: str
    target_date: datetime
    description: str
    success_criteria: str
    is_critical: bool = False


@dataclass
class PricingStrategy:
    """Pricing strategy for launch phases"""
    launch_price: float
    growth_price: float
    target_price: float
    discount_pct_launch: float
    promotion_type: str
    price_vs_competitors: str  # below, at, above
    rationale: str


@dataclass
class InventoryPlan:
    """Inventory planning for launch"""
    initial_order: int
    reorder_point: int
    reorder_quantity: int
    safety_stock_days: int
    estimated_first_reorder_date: datetime
    monthly_projections: dict[int, int] = field(default_factory=dict)  # month -> units
    total_6_month_units: int = 0


@dataclass
class PPCPlan:
    """PPC campaign plan for launch"""
    daily_budget_launch: float
    daily_budget_growth: float
    daily_budget_maintenance: float
    target_acos: float
    break_even_acos: float
    campaign_structure: list[dict]
    estimated_monthly_ad_spend: float
    estimated_attributed_sales: float


@dataclass
class RiskAssessment:
    """Risk assessment for launch"""
    risk: str
    level: RiskLevel
    probability: str  # low, medium, high
    impact: str  # low, medium, high
    mitigation: str


@dataclass
class LaunchStrategy:
    """Complete launch strategy"""
    product: ProductInfo
    market: MarketContext

    executive_summary: str
    launch_type: str  # aggressive, moderate, conservative
    estimated_success_probability: float

    pricing: PricingStrategy
    inventory: InventoryPlan
    ppc: PPCPlan

    milestones: list[LaunchMilestone]
    risks: list[RiskAssessment]
    action_items: list[str]

    projected_revenue_6mo: float
    projected_profit_6mo: float
    break_even_month: int
    roi_6mo: float


class LaunchStrategyGenerator:
    """Generates comprehensive product launch strategies"""

    # Category-specific launch benchmarks
    CATEGORY_BENCHMARKS = {
        "Electronics": {"review_target": 50, "acos_target": 30, "conversion_rate": 0.08},
        "Home & Kitchen": {"review_target": 30, "acos_target": 25, "conversion_rate": 0.10},
        "Sports & Outdoors": {"review_target": 25, "acos_target": 28, "conversion_rate": 0.09},
        "Beauty": {"review_target": 40, "acos_target": 22, "conversion_rate": 0.12},
        "Health": {"review_target": 35, "acos_target": 25, "conversion_rate": 0.11},
        "Toys & Games": {"review_target": 30, "acos_target": 30, "conversion_rate": 0.08},
        "default": {"review_target": 30, "acos_target": 28, "conversion_rate": 0.10}
    }

    def __init__(self):
        self.launch_date = datetime.now() + timedelta(days=14)  # 2 weeks from now

    def generate_strategy(self, product: ProductInfo, market: MarketContext) -> LaunchStrategy:
        """Generate complete launch strategy"""

        # Determine launch type based on competition
        launch_type = self._determine_launch_type(market)

        # Calculate profit margins
        fba_fee = self._estimate_fba_fee(product)
        referral_fee = product.target_price * 0.15  # 15% typical
        profit_per_unit = product.target_price - product.unit_cost - fba_fee - referral_fee

        # Generate strategy components
        pricing = self._generate_pricing_strategy(product, market, launch_type)
        inventory = self._generate_inventory_plan(product, market)
        ppc = self._generate_ppc_plan(product, market, pricing, profit_per_unit)
        milestones = self._generate_milestones(launch_type)
        risks = self._assess_risks(product, market)
        action_items = self._generate_action_items(launch_type)

        # Calculate projections
        projections = self._calculate_projections(
            product, market, pricing, ppc, profit_per_unit, inventory
        )

        # Generate executive summary
        summary = self._generate_executive_summary(
            product, market, launch_type, projections, risks
        )

        return LaunchStrategy(
            product=product,
            market=market,
            executive_summary=summary,
            launch_type=launch_type,
            estimated_success_probability=projections["success_probability"],
            pricing=pricing,
            inventory=inventory,
            ppc=ppc,
            milestones=milestones,
            risks=risks,
            action_items=action_items,
            projected_revenue_6mo=projections["revenue_6mo"],
            projected_profit_6mo=projections["profit_6mo"],
            break_even_month=projections["break_even_month"],
            roi_6mo=projections["roi_6mo"]
        )

    def _determine_launch_type(self, market: MarketContext) -> str:
        """Determine optimal launch approach"""
        # Aggressive: Low competition, trending market
        # Moderate: Normal competition
        # Conservative: High competition, established players

        competition_score = 0

        if market.competitor_count > 50:
            competition_score += 3
        elif market.competitor_count > 20:
            competition_score += 2
        else:
            competition_score += 1

        if market.avg_competitor_reviews > 500:
            competition_score += 2
        elif market.avg_competitor_reviews > 200:
            competition_score += 1

        if market.top_competitor_bsr < 5000:
            competition_score += 2

        if competition_score >= 5:
            return "conservative"
        elif competition_score >= 3:
            return "moderate"
        else:
            return "aggressive"

    def _estimate_fba_fee(self, product: ProductInfo) -> float:
        """Estimate FBA fulfillment fee"""
        # Simplified tier-based estimation
        volume = product.dimensions[0] * product.dimensions[1] * product.dimensions[2]
        weight = product.weight_lbs

        if weight <= 0.5 and volume <= 200:
            return 3.22  # Small standard
        elif weight <= 1 and volume <= 500:
            return 4.75  # Large standard
        elif weight <= 2:
            return 5.79
        else:
            return 7.17 + (weight - 2) * 0.4

    def _generate_pricing_strategy(self, product: ProductInfo, market: MarketContext,
                                   launch_type: str) -> PricingStrategy:
        """Generate pricing strategy"""
        avg_price = market.avg_competitor_price

        if launch_type == "aggressive":
            # Price below competitors, heavy discounting
            launch_price = round(avg_price * 0.85, 2)
            discount_pct = 20
            promotion = "Lightning Deal + Coupon"
            position = "below"
            rationale = "Aggressive penetration pricing to gain market share quickly"
        elif launch_type == "conservative":
            # Price at or slightly below, modest promotion
            launch_price = round(avg_price * 0.95, 2)
            discount_pct = 10
            promotion = "5% Coupon"
            position = "at"
            rationale = "Competitive pricing with focus on profitability over volume"
        else:
            # Moderate approach
            launch_price = round(avg_price * 0.90, 2)
            discount_pct = 15
            promotion = "15% Coupon + Subscribe & Save"
            position = "below"
            rationale = "Balanced approach: competitive pricing with modest promotions"

        # Ensure launch price is above cost
        min_price = product.unit_cost * 1.5
        launch_price = max(launch_price, min_price)

        return PricingStrategy(
            launch_price=launch_price,
            growth_price=round(launch_price * 1.05, 2),
            target_price=product.target_price,
            discount_pct_launch=discount_pct,
            promotion_type=promotion,
            price_vs_competitors=position,
            rationale=rationale
        )

    def _generate_inventory_plan(self, product: ProductInfo,
                                 market: MarketContext) -> InventoryPlan:
        """Generate inventory plan"""
        monthly_target = product.target_monthly_sales

        # Conservative first month (lower sales during ramp-up)
        month_1 = int(monthly_target * 0.3)
        month_2 = int(monthly_target * 0.6)
        month_3 = int(monthly_target * 0.8)
        month_4 = monthly_target
        month_5 = monthly_target
        month_6 = monthly_target

        # Adjust for seasonality
        if market.seasonality_peak_months:
            now = datetime.now().month
            for m in [1, 2, 3, 4, 5, 6]:
                actual_month = (now + m - 1) % 12 + 1
                if actual_month in market.seasonality_peak_months:
                    if m == 4:
                        month_4 = int(month_4 * 1.3)
                    elif m == 5:
                        month_5 = int(month_5 * 1.3)
                    elif m == 6:
                        month_6 = int(month_6 * 1.3)

        projections = {1: month_1, 2: month_2, 3: month_3, 4: month_4, 5: month_5, 6: month_6}
        total = sum(projections.values())

        # Safety stock = 2 weeks of peak month sales
        safety_days = 14
        safety_stock = int(max(projections.values()) / 30 * safety_days)

        # Reorder when inventory = lead_time sales + safety stock
        daily_sales = max(projections.values()) / 30
        reorder_point = int(daily_sales * product.lead_time_days + safety_stock)
        reorder_qty = int(max(projections.values()) * 1.5)  # 1.5 months supply

        # Estimate first reorder date
        days_to_reorder = (product.initial_inventory - reorder_point) / max(1, month_2/30)
        first_reorder = self.launch_date + timedelta(days=int(days_to_reorder))

        return InventoryPlan(
            initial_order=product.initial_inventory,
            reorder_point=reorder_point,
            reorder_quantity=reorder_qty,
            safety_stock_days=safety_days,
            estimated_first_reorder_date=first_reorder,
            monthly_projections=projections,
            total_6_month_units=total
        )

    def _generate_ppc_plan(self, product: ProductInfo, market: MarketContext,
                          pricing: PricingStrategy, profit_per_unit: float) -> PPCPlan:
        """Generate PPC campaign plan"""
        benchmarks = self.CATEGORY_BENCHMARKS.get(
            product.category, self.CATEGORY_BENCHMARKS["default"]
        )

        # Calculate break-even ACOS
        margin_pct = profit_per_unit / pricing.launch_price
        break_even_acos = margin_pct * 100
        target_acos = min(break_even_acos * 0.7, benchmarks["acos_target"])

        # Estimate CPC based on competition
        if market.competitor_count > 30:
            est_cpc = 1.20
        elif market.competitor_count > 15:
            est_cpc = 0.90
        else:
            est_cpc = 0.65

        # Calculate budgets
        conversion_rate = benchmarks["conversion_rate"]
        target_orders_launch = product.target_monthly_sales * 0.3  # 30% from PPC at launch
        clicks_needed = target_orders_launch / conversion_rate
        daily_spend_launch = (clicks_needed * est_cpc) / 30

        campaigns = [
            {
                "name": "Exact Match - High Intent",
                "type": "Sponsored Products",
                "match_type": "exact",
                "budget_share": 0.40,
                "strategy": "Focus on converting high-intent keywords"
            },
            {
                "name": "Phrase Match - Discovery",
                "type": "Sponsored Products",
                "match_type": "phrase",
                "budget_share": 0.30,
                "strategy": "Discover new converting keywords"
            },
            {
                "name": "Auto Campaign - Research",
                "type": "Sponsored Products",
                "match_type": "auto",
                "budget_share": 0.20,
                "strategy": "Find Amazon-suggested keywords"
            },
            {
                "name": "Sponsored Brands",
                "type": "Sponsored Brands",
                "match_type": "mixed",
                "budget_share": 0.10,
                "strategy": "Build brand awareness (add after 10 reviews)"
            }
        ]

        monthly_spend = daily_spend_launch * 30
        attributed_sales = (monthly_spend / target_acos * 100) if target_acos > 0 else 0

        return PPCPlan(
            daily_budget_launch=round(daily_spend_launch, 2),
            daily_budget_growth=round(daily_spend_launch * 1.5, 2),
            daily_budget_maintenance=round(daily_spend_launch * 2, 2),
            target_acos=round(target_acos, 1),
            break_even_acos=round(break_even_acos, 1),
            campaign_structure=campaigns,
            estimated_monthly_ad_spend=round(monthly_spend, 2),
            estimated_attributed_sales=round(attributed_sales, 2)
        )

    def _generate_milestones(self, launch_type: str) -> list[LaunchMilestone]:
        """Generate launch milestones"""
        milestones = []
        base = self.launch_date

        # Pre-launch (2 weeks before)
        milestones.append(LaunchMilestone(
            phase=LaunchPhase.PRE_LAUNCH,
            name="Listing Optimization",
            target_date=base - timedelta(days=14),
            description="Complete title, bullets, A+ content, and images",
            success_criteria="Listing score 9+/10 on tools like Helium 10",
            is_critical=True
        ))

        milestones.append(LaunchMilestone(
            phase=LaunchPhase.PRE_LAUNCH,
            name="Inventory Check-in",
            target_date=base - timedelta(days=7),
            description="Confirm inventory received at FBA warehouse",
            success_criteria="Inventory shows as fulfillable",
            is_critical=True
        ))

        # Launch (Day 1-30)
        milestones.append(LaunchMilestone(
            phase=LaunchPhase.LAUNCH,
            name="Go Live",
            target_date=base,
            description="Product live, PPC campaigns active, promotions enabled",
            success_criteria="First orders received within 24 hours",
            is_critical=True
        ))

        milestones.append(LaunchMilestone(
            phase=LaunchPhase.LAUNCH,
            name="First Reviews",
            target_date=base + timedelta(days=14),
            description="First organic reviews from verified purchases",
            success_criteria="5+ reviews with 4.0+ average rating"
        ))

        if launch_type == "aggressive":
            review_target = 30
            review_days = 30
        else:
            review_target = 15
            review_days = 45

        milestones.append(LaunchMilestone(
            phase=LaunchPhase.LAUNCH,
            name="Review Milestone",
            target_date=base + timedelta(days=review_days),
            description=f"Reach {review_target} reviews",
            success_criteria=f"{review_target}+ reviews, 4.0+ rating",
            is_critical=True
        ))

        # Growth (Day 31-90)
        milestones.append(LaunchMilestone(
            phase=LaunchPhase.GROWTH,
            name="PPC Optimization",
            target_date=base + timedelta(days=45),
            description="Optimize bids, harvest keywords, launch Sponsored Brands",
            success_criteria="ACOS at or below target"
        ))

        milestones.append(LaunchMilestone(
            phase=LaunchPhase.GROWTH,
            name="Organic Ranking",
            target_date=base + timedelta(days=60),
            description="Establish organic rankings for main keywords",
            success_criteria="Page 1-2 organic ranking for 5+ keywords"
        ))

        milestones.append(LaunchMilestone(
            phase=LaunchPhase.GROWTH,
            name="Inventory Reorder",
            target_date=base + timedelta(days=45),
            description="Place first reorder to prevent stockout",
            success_criteria="Reorder placed with buffer before stockout"
        ))

        # Optimization (Day 91-120)
        milestones.append(LaunchMilestone(
            phase=LaunchPhase.OPTIMIZATION,
            name="Price Optimization",
            target_date=base + timedelta(days=90),
            description="Adjust price toward target, reduce promotions",
            success_criteria="Price within 5% of target while maintaining velocity"
        ))

        milestones.append(LaunchMilestone(
            phase=LaunchPhase.OPTIMIZATION,
            name="Profitability Target",
            target_date=base + timedelta(days=120),
            description="Achieve target profit margins",
            success_criteria="Positive monthly profit, TACOS under 15%"
        ))

        # Maintenance (Day 121+)
        milestones.append(LaunchMilestone(
            phase=LaunchPhase.MAINTENANCE,
            name="Steady State",
            target_date=base + timedelta(days=180),
            description="Product at steady state with organic sales majority",
            success_criteria="50%+ organic sales, positive cash flow"
        ))

        return milestones

    def _assess_risks(self, product: ProductInfo, market: MarketContext) -> list[RiskAssessment]:
        """Assess launch risks"""
        risks = []

        # Competition risk
        if market.competitor_count > 30:
            risks.append(RiskAssessment(
                risk="High Competition",
                level=RiskLevel.HIGH,
                probability="high",
                impact="medium",
                mitigation="Differentiate through superior listing quality, target long-tail keywords, focus on niche sub-categories"
            ))
        elif market.competitor_count > 15:
            risks.append(RiskAssessment(
                risk="Moderate Competition",
                level=RiskLevel.MEDIUM,
                probability="medium",
                impact="medium",
                mitigation="Ensure competitive pricing and strong review velocity"
            ))

        # Review barrier risk
        if market.avg_competitor_reviews > 300:
            risks.append(RiskAssessment(
                risk="High Review Barrier",
                level=RiskLevel.HIGH,
                probability="high",
                impact="high",
                mitigation="Implement aggressive review request strategy, consider Amazon Vine, focus on insert cards"
            ))

        # Price war risk
        if market.competitor_count > 20:
            risks.append(RiskAssessment(
                risk="Price War",
                level=RiskLevel.MEDIUM,
                probability="medium",
                impact="high",
                mitigation="Monitor competitor prices daily, have margin buffer, differentiate on quality not price"
            ))

        # Inventory risk
        if product.lead_time_days > 60:
            risks.append(RiskAssessment(
                risk="Long Lead Time",
                level=RiskLevel.MEDIUM,
                probability="medium",
                impact="high",
                mitigation="Order conservatively, maintain safety stock, consider domestic supplier backup"
            ))

        # Seasonality risk
        if market.seasonality_peak_months:
            risks.append(RiskAssessment(
                risk="Seasonal Demand",
                level=RiskLevel.MEDIUM,
                probability="high",
                impact="medium",
                mitigation="Time launch before peak season, ensure inventory for peaks, plan for off-season"
            ))

        # Cash flow risk
        initial_investment = product.initial_inventory * product.unit_cost
        if initial_investment > 10000:
            risks.append(RiskAssessment(
                risk="High Initial Investment",
                level=RiskLevel.MEDIUM,
                probability="medium",
                impact="high",
                mitigation="Start with smaller inventory, validate product-market fit before scaling"
            ))

        # Always add these standard risks
        risks.append(RiskAssessment(
            risk="Negative Reviews",
            level=RiskLevel.MEDIUM,
            probability="low",
            impact="high",
            mitigation="Ensure product quality, proactive customer service, quick response to issues"
        ))

        risks.append(RiskAssessment(
            risk="Stockout During Launch",
            level=RiskLevel.HIGH,
            probability="low",
            impact="high",
            mitigation="Conservative sales projections, early reorder trigger, air shipping backup"
        ))

        return sorted(risks, key=lambda x: (x.level == RiskLevel.HIGH, x.level == RiskLevel.MEDIUM), reverse=True)

    def _generate_action_items(self, launch_type: str) -> list[str]:
        """Generate pre-launch action items"""
        items = [
            "Complete keyword research and prioritize top 20 keywords",
            "Finalize main image and 6 lifestyle/infographic images",
            "Write optimized title, 5 bullet points, and A+ content",
            "Set up PPC campaigns in draft mode",
            "Create coupon/promotion in Seller Central",
            "Prepare review request email sequence",
            "Set up inventory alerts and reorder triggers",
            "Configure competitor price monitoring",
            "Prepare customer service response templates",
            "Test product page on mobile and desktop"
        ]

        if launch_type == "aggressive":
            items.extend([
                "Apply for Amazon Vine program",
                "Schedule Lightning Deal for week 2",
                "Prepare social media launch campaign"
            ])

        return items

    def _calculate_projections(self, product: ProductInfo, market: MarketContext,
                               pricing: PricingStrategy, ppc: PPCPlan,
                               profit_per_unit: float, inventory: InventoryPlan) -> dict:
        """Calculate financial projections"""
        # 6-month revenue
        units_6mo = inventory.total_6_month_units
        avg_price = (pricing.launch_price + pricing.growth_price + pricing.target_price) / 3
        revenue_6mo = units_6mo * avg_price

        # 6-month costs
        cogs = units_6mo * product.unit_cost
        ad_spend = ppc.estimated_monthly_ad_spend * 6 * 0.7  # Decreases over time
        fba_fees = units_6mo * self._estimate_fba_fee(product)
        referral_fees = revenue_6mo * 0.15

        # Profit
        profit_6mo = revenue_6mo - cogs - ad_spend - fba_fees - referral_fees

        # Initial investment
        initial_investment = product.initial_inventory * product.unit_cost

        # ROI
        roi = (profit_6mo / initial_investment) * 100 if initial_investment > 0 else 0

        # Break-even month (simplified)
        monthly_units = [inventory.monthly_projections.get(m, 0) for m in range(1, 7)]
        cumulative_profit = 0
        break_even = 6
        for month, units in enumerate(monthly_units, 1):
            # Adjust pricing over time
            if month <= 2:
                price = pricing.launch_price
            elif month <= 4:
                price = pricing.growth_price
            else:
                price = pricing.target_price

            monthly_revenue = units * price
            monthly_profit = monthly_revenue * 0.25  # ~25% margin after all costs
            cumulative_profit += monthly_profit

            if cumulative_profit >= initial_investment and break_even == 6:
                break_even = month

        # Success probability (simplified scoring)
        success_score = 70  # Base

        if market.competitor_count < 20:
            success_score += 10
        elif market.competitor_count > 40:
            success_score -= 10

        if market.avg_competitor_reviews < 200:
            success_score += 10
        elif market.avg_competitor_reviews > 500:
            success_score -= 15

        if market.is_trending:
            success_score += 10

        if profit_per_unit > pricing.launch_price * 0.3:
            success_score += 5

        success_score = max(20, min(95, success_score))

        return {
            "revenue_6mo": round(revenue_6mo, 2),
            "profit_6mo": round(profit_6mo, 2),
            "break_even_month": break_even,
            "roi_6mo": round(roi, 1),
            "success_probability": success_score / 100
        }

    def _generate_executive_summary(self, product: ProductInfo, market: MarketContext,
                                    launch_type: str, projections: dict,
                                    risks: list[RiskAssessment]) -> str:
        """Generate executive summary"""
        high_risks = [r for r in risks if r.level == RiskLevel.HIGH]

        summary = f"""
LAUNCH STRATEGY: {launch_type.upper()} APPROACH

Product: {product.title}
Category: {product.category}
Target Price: ${product.target_price:.2f}

Market Assessment:
- {market.competitor_count} competitors identified
- Average competitor price: ${market.avg_competitor_price:.2f}
- Search volume: {market.keyword_search_volume:,} monthly searches
- Estimated market size: ${market.estimated_monthly_market_revenue:,.0f}/month

Financial Projections (6 months):
- Projected Revenue: ${projections['revenue_6mo']:,.0f}
- Projected Profit: ${projections['profit_6mo']:,.0f}
- ROI: {projections['roi_6mo']:.0f}%
- Break-even: Month {projections['break_even_month']}

Success Probability: {projections['success_probability']*100:.0f}%

Key Risks: {len(high_risks)} high-priority risks identified
{chr(10).join([f'  - {r.risk}' for r in high_risks]) if high_risks else '  - No critical risks identified'}

Recommendation: {'PROCEED' if projections['success_probability'] >= 0.6 else 'PROCEED WITH CAUTION' if projections['success_probability'] >= 0.4 else 'RECONSIDER'}
""".strip()

        return summary


def create_sample_input() -> tuple[ProductInfo, MarketContext]:
    """Create sample product and market data"""
    product = ProductInfo(
        asin=None,
        title="Premium Wireless Earbuds with Active Noise Cancellation",
        category="Electronics",
        unit_cost=8.50,
        target_price=34.99,
        weight_lbs=0.3,
        dimensions=(4.5, 3.0, 1.5),
        initial_inventory=500,
        lead_time_days=45,
        target_monthly_sales=300
    )

    market = MarketContext(
        competitor_count=25,
        avg_competitor_price=32.99,
        avg_competitor_rating=4.2,
        avg_competitor_reviews=450,
        top_competitor_bsr=5500,
        keyword_search_volume=85000,
        estimated_monthly_market_revenue=2500000,
        seasonality_peak_months=[11, 12],
        is_trending=True
    )

    return product, market


if __name__ == "__main__":
    print("=" * 70)
    print("PRODUCT LAUNCH STRATEGY GENERATOR")
    print("=" * 70)

    generator = LaunchStrategyGenerator()
    product, market = create_sample_input()

    strategy = generator.generate_strategy(product, market)

    # Executive Summary
    print("\n" + strategy.executive_summary)

    # Pricing Strategy
    print("\n" + "-" * 70)
    print("PRICING STRATEGY")
    print("-" * 70)
    print(f"  Launch Price:  ${strategy.pricing.launch_price:.2f}")
    print(f"  Growth Price:  ${strategy.pricing.growth_price:.2f}")
    print(f"  Target Price:  ${strategy.pricing.target_price:.2f}")
    print(f"  Launch Discount: {strategy.pricing.discount_pct_launch}%")
    print(f"  Promotion: {strategy.pricing.promotion_type}")
    print(f"  Rationale: {strategy.pricing.rationale}")

    # Inventory Plan
    print("\n" + "-" * 70)
    print("INVENTORY PLAN")
    print("-" * 70)
    print(f"  Initial Order: {strategy.inventory.initial_order} units")
    print(f"  Reorder Point: {strategy.inventory.reorder_point} units")
    print(f"  Reorder Quantity: {strategy.inventory.reorder_quantity} units")
    print(f"  Safety Stock: {strategy.inventory.safety_stock_days} days")
    print(f"  Est. First Reorder: {strategy.inventory.estimated_first_reorder_date.strftime('%Y-%m-%d')}")
    print(f"\n  Monthly Projections:")
    for month, units in strategy.inventory.monthly_projections.items():
        bar = "█" * (units // 30)
        print(f"    Month {month}: {units:4d} units {bar}")
    print(f"  Total (6mo): {strategy.inventory.total_6_month_units} units")

    # PPC Plan
    print("\n" + "-" * 70)
    print("PPC PLAN")
    print("-" * 70)
    print(f"  Daily Budget (Launch): ${strategy.ppc.daily_budget_launch:.2f}")
    print(f"  Daily Budget (Growth): ${strategy.ppc.daily_budget_growth:.2f}")
    print(f"  Daily Budget (Maint.): ${strategy.ppc.daily_budget_maintenance:.2f}")
    print(f"  Target ACOS: {strategy.ppc.target_acos}%")
    print(f"  Break-even ACOS: {strategy.ppc.break_even_acos}%")
    print(f"\n  Campaign Structure:")
    for camp in strategy.ppc.campaign_structure:
        print(f"    • {camp['name']}: {camp['budget_share']*100:.0f}% budget")

    # Milestones
    print("\n" + "-" * 70)
    print("LAUNCH MILESTONES")
    print("-" * 70)
    current_phase = None
    for m in strategy.milestones:
        if m.phase != current_phase:
            print(f"\n  [{m.phase.value.upper()}]")
            current_phase = m.phase
        critical = " ⚡" if m.is_critical else ""
        print(f"    {m.target_date.strftime('%Y-%m-%d')}: {m.name}{critical}")

    # Risks
    print("\n" + "-" * 70)
    print("RISK ASSESSMENT")
    print("-" * 70)
    for risk in strategy.risks[:5]:
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[risk.level.value]
        print(f"  {icon} {risk.risk}")
        print(f"      Probability: {risk.probability} | Impact: {risk.impact}")
        print(f"      Mitigation: {risk.mitigation[:60]}...")

    # Action Items
    print("\n" + "-" * 70)
    print("PRE-LAUNCH ACTION ITEMS")
    print("-" * 70)
    for i, item in enumerate(strategy.action_items, 1):
        print(f"  {i:2d}. {item}")

    print("\n" + "=" * 70)
    print("STRATEGY GENERATION COMPLETE")
    print("=" * 70)
