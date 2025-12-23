"""
Amazon PPC/Advertising Analysis

Estimate advertising costs, ACOS, and campaign performance for product launches.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class CampaignType(Enum):
    """Amazon PPC campaign types."""
    SPONSORED_PRODUCTS = "sponsored_products"
    SPONSORED_BRANDS = "sponsored_brands"
    SPONSORED_DISPLAY = "sponsored_display"


class TargetingType(Enum):
    """PPC targeting strategies."""
    AUTO = "automatic"
    MANUAL_KEYWORD = "manual_keyword"
    MANUAL_PRODUCT = "manual_product"
    CATEGORY = "category"


class MatchType(Enum):
    """Keyword match types."""
    BROAD = "broad"
    PHRASE = "phrase"
    EXACT = "exact"


@dataclass
class KeywordBid:
    """Keyword with bid information."""
    keyword: str
    match_type: MatchType
    suggested_bid: float
    bid_range_low: float
    bid_range_high: float
    estimated_impressions: int
    estimated_clicks: int
    competition_level: str  # low, medium, high


@dataclass
class CampaignBudget:
    """Campaign budget recommendation."""
    daily_budget: float
    monthly_budget: float

    # Expected metrics
    estimated_impressions: int
    estimated_clicks: int
    estimated_orders: int
    estimated_sales: float

    # Efficiency metrics
    estimated_acos: float  # Advertising Cost of Sales
    estimated_tacos: float  # Total ACOS (including organic)
    estimated_roas: float  # Return on Ad Spend

    # Breakdown
    campaign_type: CampaignType
    targeting_type: TargetingType


@dataclass
class PPCStrategy:
    """Complete PPC strategy recommendation."""
    product_price: float
    target_acos: float
    break_even_acos: float

    # Budget recommendations
    launch_phase_budget: CampaignBudget  # First 30 days
    growth_phase_budget: CampaignBudget  # Days 31-90
    maintenance_budget: CampaignBudget   # Ongoing

    # Keyword strategy
    primary_keywords: List[KeywordBid]
    secondary_keywords: List[KeywordBid]
    negative_keywords: List[str]

    # Campaign structure
    recommended_campaigns: List[Dict[str, Any]]

    # Projections
    monthly_ad_spend: float
    projected_monthly_sales: float
    projected_profit_after_ads: float


@dataclass
class PPCPerformanceMetrics:
    """PPC performance tracking metrics."""
    impressions: int
    clicks: int
    ctr: float  # Click-through rate
    cpc: float  # Cost per click
    spend: float
    orders: int
    sales: float
    acos: float
    roas: float
    conversion_rate: float


class PPCAnalyzer:
    """
    Amazon PPC cost and performance analyzer.

    Estimates advertising costs, recommends budgets,
    and projects campaign performance.
    """

    # Average metrics by category (US marketplace)
    CATEGORY_BENCHMARKS = {
        'electronics': {'cpc': 0.85, 'cvr': 0.10, 'ctr': 0.35},
        'home_kitchen': {'cpc': 0.65, 'cvr': 0.12, 'ctr': 0.40},
        'beauty': {'cpc': 0.75, 'cvr': 0.09, 'ctr': 0.38},
        'toys': {'cpc': 0.55, 'cvr': 0.11, 'ctr': 0.42},
        'sports': {'cpc': 0.60, 'cvr': 0.10, 'ctr': 0.38},
        'clothing': {'cpc': 0.45, 'cvr': 0.08, 'ctr': 0.32},
        'health': {'cpc': 0.70, 'cvr': 0.11, 'ctr': 0.36},
        'pet_supplies': {'cpc': 0.55, 'cvr': 0.12, 'ctr': 0.40},
        'office': {'cpc': 0.50, 'cvr': 0.13, 'ctr': 0.38},
        'default': {'cpc': 0.65, 'cvr': 0.10, 'ctr': 0.38},
    }

    # Competition multipliers for CPC
    COMPETITION_MULTIPLIERS = {
        'very_low': 0.6,
        'low': 0.8,
        'medium': 1.0,
        'high': 1.3,
        'very_high': 1.6,
    }

    def __init__(self, category: str = 'default'):
        """
        Initialize PPC analyzer.

        Args:
            category: Product category for benchmarks
        """
        self.category = category.lower().replace(' ', '_')
        self.benchmarks = self.CATEGORY_BENCHMARKS.get(
            self.category,
            self.CATEGORY_BENCHMARKS['default']
        )

    def calculate_break_even_acos(
        self,
        product_price: float,
        product_cost: float,
        fba_fees: float,
        other_costs: float = 0
    ) -> float:
        """
        Calculate break-even ACOS (max ACOS before losing money).

        Break-even ACOS = Profit Margin %

        Args:
            product_price: Selling price
            product_cost: Cost of goods
            fba_fees: Total FBA fees
            other_costs: Any other costs

        Returns:
            Break-even ACOS as percentage
        """
        total_cost = product_cost + fba_fees + other_costs
        profit = product_price - total_cost

        if product_price <= 0:
            return 0

        break_even_acos = (profit / product_price) * 100
        return max(0, round(break_even_acos, 1))

    def estimate_cpc(
        self,
        keyword: str,
        competition_level: str = 'medium',
        match_type: MatchType = MatchType.BROAD
    ) -> Tuple[float, float, float]:
        """
        Estimate CPC for a keyword.

        Args:
            keyword: Target keyword
            competition_level: Competition level
            match_type: Keyword match type

        Returns:
            Tuple of (suggested_bid, low_range, high_range)
        """
        base_cpc = self.benchmarks['cpc']

        # Apply competition multiplier
        multiplier = self.COMPETITION_MULTIPLIERS.get(competition_level, 1.0)

        # Match type adjustments
        match_multipliers = {
            MatchType.BROAD: 0.85,
            MatchType.PHRASE: 1.0,
            MatchType.EXACT: 1.15,
        }
        match_mult = match_multipliers.get(match_type, 1.0)

        # Keyword length adjustment (longer = cheaper)
        word_count = len(keyword.split())
        length_mult = 1.0 - (min(word_count - 1, 4) * 0.08)

        suggested = base_cpc * multiplier * match_mult * length_mult

        # Range is typically ±30%
        low = suggested * 0.7
        high = suggested * 1.4

        return (round(suggested, 2), round(low, 2), round(high, 2))

    def estimate_campaign_metrics(
        self,
        daily_budget: float,
        avg_cpc: float,
        product_price: float,
        conversion_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Estimate campaign performance metrics.

        Args:
            daily_budget: Daily ad spend
            avg_cpc: Average cost per click
            product_price: Product selling price
            conversion_rate: Expected conversion rate (or use benchmark)

        Returns:
            Dictionary of estimated metrics
        """
        cvr = conversion_rate or self.benchmarks['cvr']
        ctr = self.benchmarks['ctr']

        # Calculate daily metrics
        daily_clicks = daily_budget / avg_cpc
        daily_impressions = daily_clicks / (ctr / 100)
        daily_orders = daily_clicks * cvr
        daily_sales = daily_orders * product_price

        # Monthly extrapolation
        monthly_clicks = daily_clicks * 30
        monthly_impressions = daily_impressions * 30
        monthly_orders = daily_orders * 30
        monthly_sales = daily_sales * 30
        monthly_spend = daily_budget * 30

        # Efficiency metrics
        acos = (monthly_spend / monthly_sales * 100) if monthly_sales > 0 else 0
        roas = (monthly_sales / monthly_spend) if monthly_spend > 0 else 0

        return {
            'daily_impressions': int(daily_impressions),
            'daily_clicks': round(daily_clicks, 1),
            'daily_orders': round(daily_orders, 2),
            'daily_sales': round(daily_sales, 2),
            'monthly_impressions': int(monthly_impressions),
            'monthly_clicks': int(monthly_clicks),
            'monthly_orders': int(monthly_orders),
            'monthly_sales': round(monthly_sales, 2),
            'monthly_spend': round(monthly_spend, 2),
            'acos': round(acos, 1),
            'roas': round(roas, 2),
            'ctr': ctr,
            'cvr': cvr * 100,
        }

    def generate_keyword_bids(
        self,
        keywords: List[str],
        competition_levels: Optional[Dict[str, str]] = None
    ) -> List[KeywordBid]:
        """
        Generate bid recommendations for keywords.

        Args:
            keywords: List of target keywords
            competition_levels: Optional dict of keyword -> competition level

        Returns:
            List of KeywordBid recommendations
        """
        competition_levels = competition_levels or {}
        bids = []

        for keyword in keywords:
            comp_level = competition_levels.get(keyword, 'medium')

            # Generate bids for each match type
            for match_type in [MatchType.EXACT, MatchType.PHRASE, MatchType.BROAD]:
                suggested, low, high = self.estimate_cpc(keyword, comp_level, match_type)

                # Estimate impressions/clicks based on match type
                base_impressions = 10000
                if match_type == MatchType.EXACT:
                    impressions = int(base_impressions * 0.3)
                elif match_type == MatchType.PHRASE:
                    impressions = int(base_impressions * 0.5)
                else:
                    impressions = base_impressions

                clicks = int(impressions * self.benchmarks['ctr'] / 100)

                bids.append(KeywordBid(
                    keyword=keyword,
                    match_type=match_type,
                    suggested_bid=suggested,
                    bid_range_low=low,
                    bid_range_high=high,
                    estimated_impressions=impressions,
                    estimated_clicks=clicks,
                    competition_level=comp_level
                ))

        return bids

    def recommend_budget(
        self,
        product_price: float,
        target_acos: float,
        target_daily_sales: int = 5,
        campaign_type: CampaignType = CampaignType.SPONSORED_PRODUCTS,
        targeting_type: TargetingType = TargetingType.MANUAL_KEYWORD
    ) -> CampaignBudget:
        """
        Recommend campaign budget based on targets.

        Args:
            product_price: Product selling price
            target_acos: Target ACOS percentage
            target_daily_sales: Target sales per day
            campaign_type: Type of campaign
            targeting_type: Targeting strategy

        Returns:
            CampaignBudget recommendation
        """
        cvr = self.benchmarks['cvr']
        ctr = self.benchmarks['ctr']
        cpc = self.benchmarks['cpc']

        # Calculate required clicks for target sales
        required_daily_clicks = target_daily_sales / cvr

        # Calculate budget from target ACOS
        # ACOS = Ad Spend / Sales → Ad Spend = ACOS * Sales
        target_daily_sales_revenue = target_daily_sales * product_price
        max_daily_spend = (target_acos / 100) * target_daily_sales_revenue

        # Adjust based on actual CPC needs
        cpc_based_budget = required_daily_clicks * cpc

        # Use the lower of the two to stay within ACOS target
        daily_budget = min(max_daily_spend, cpc_based_budget * 1.2)
        daily_budget = max(10, daily_budget)  # Minimum $10/day

        # Calculate expected metrics
        metrics = self.estimate_campaign_metrics(
            daily_budget, cpc, product_price, cvr
        )

        return CampaignBudget(
            daily_budget=round(daily_budget, 2),
            monthly_budget=round(daily_budget * 30, 2),
            estimated_impressions=metrics['monthly_impressions'],
            estimated_clicks=metrics['monthly_clicks'],
            estimated_orders=metrics['monthly_orders'],
            estimated_sales=metrics['monthly_sales'],
            estimated_acos=metrics['acos'],
            estimated_tacos=metrics['acos'] * 0.7,  # Assumes 30% organic
            estimated_roas=metrics['roas'],
            campaign_type=campaign_type,
            targeting_type=targeting_type
        )

    def generate_ppc_strategy(
        self,
        product_price: float,
        product_cost: float,
        fba_fees: float,
        keywords: List[str],
        competition_level: str = 'medium'
    ) -> PPCStrategy:
        """
        Generate complete PPC strategy for product launch.

        Args:
            product_price: Selling price
            product_cost: Cost of goods
            fba_fees: FBA fees
            keywords: Target keywords
            competition_level: Overall competition level

        Returns:
            Complete PPCStrategy
        """
        # Calculate break-even ACOS
        break_even = self.calculate_break_even_acos(
            product_price, product_cost, fba_fees
        )

        # Target ACOS at 70% of break-even for profit
        target_acos = break_even * 0.7
        target_acos = max(15, min(50, target_acos))  # Reasonable bounds

        # Generate keyword bids
        comp_levels = {kw: competition_level for kw in keywords}
        all_bids = self.generate_keyword_bids(keywords, comp_levels)

        # Split into primary (exact/phrase) and secondary (broad)
        primary = [b for b in all_bids if b.match_type in [MatchType.EXACT, MatchType.PHRASE]]
        secondary = [b for b in all_bids if b.match_type == MatchType.BROAD]

        # Generate budgets for each phase
        launch_budget = self.recommend_budget(
            product_price, target_acos * 1.5,  # Higher ACOS during launch
            target_daily_sales=3
        )

        growth_budget = self.recommend_budget(
            product_price, target_acos * 1.2,
            target_daily_sales=5
        )

        maintenance_budget = self.recommend_budget(
            product_price, target_acos,
            target_daily_sales=8
        )

        # Recommend campaign structure
        campaigns = [
            {
                'name': 'Exact Match - High Intent',
                'type': CampaignType.SPONSORED_PRODUCTS.value,
                'targeting': TargetingType.MANUAL_KEYWORD.value,
                'match_types': ['exact'],
                'daily_budget': round(launch_budget.daily_budget * 0.4, 2),
                'priority': 'high',
            },
            {
                'name': 'Phrase Match - Discovery',
                'type': CampaignType.SPONSORED_PRODUCTS.value,
                'targeting': TargetingType.MANUAL_KEYWORD.value,
                'match_types': ['phrase'],
                'daily_budget': round(launch_budget.daily_budget * 0.3, 2),
                'priority': 'medium',
            },
            {
                'name': 'Auto Campaign - Research',
                'type': CampaignType.SPONSORED_PRODUCTS.value,
                'targeting': TargetingType.AUTO.value,
                'match_types': ['auto'],
                'daily_budget': round(launch_budget.daily_budget * 0.3, 2),
                'priority': 'medium',
            },
        ]

        # Generate negative keywords
        negative_keywords = self._generate_negative_keywords(keywords)

        # Calculate projections
        monthly_spend = maintenance_budget.monthly_budget
        projected_sales = maintenance_budget.estimated_sales

        # Profit after ads
        units_sold = int(projected_sales / product_price)
        profit_per_unit = product_price - product_cost - fba_fees
        gross_profit = units_sold * profit_per_unit
        profit_after_ads = gross_profit - monthly_spend

        return PPCStrategy(
            product_price=product_price,
            target_acos=target_acos,
            break_even_acos=break_even,
            launch_phase_budget=launch_budget,
            growth_phase_budget=growth_budget,
            maintenance_budget=maintenance_budget,
            primary_keywords=primary[:20],
            secondary_keywords=secondary[:10],
            negative_keywords=negative_keywords,
            recommended_campaigns=campaigns,
            monthly_ad_spend=monthly_spend,
            projected_monthly_sales=projected_sales,
            projected_profit_after_ads=round(profit_after_ads, 2)
        )

    def _generate_negative_keywords(self, keywords: List[str]) -> List[str]:
        """Generate common negative keywords to exclude."""
        # Common negative keywords to exclude irrelevant traffic
        common_negatives = [
            'free', 'cheap', 'used', 'refurbished', 'broken',
            'repair', 'fix', 'manual', 'instructions', 'how to',
            'diy', 'homemade', 'alternative', 'substitute',
            'wholesale', 'bulk', 'sample', 'return', 'refund'
        ]

        # Add brand-specific negatives (competitors)
        # In production, this would come from actual competitor data

        return common_negatives

    def calculate_tacos(
        self,
        ad_spend: float,
        ad_sales: float,
        organic_sales: float
    ) -> float:
        """
        Calculate Total ACOS (including organic sales).

        TACoS = Ad Spend / Total Sales

        Args:
            ad_spend: Total advertising spend
            ad_sales: Sales from ads
            organic_sales: Organic sales

        Returns:
            TACoS percentage
        """
        total_sales = ad_sales + organic_sales
        if total_sales <= 0:
            return 0
        return round((ad_spend / total_sales) * 100, 1)

    def project_organic_growth(
        self,
        current_bsr: int,
        ad_sales_per_month: int,
        months: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Project organic sales growth from PPC momentum.

        Args:
            current_bsr: Current Best Sellers Rank
            ad_sales_per_month: Monthly sales from ads
            months: Projection period

        Returns:
            Monthly projections
        """
        projections = []
        bsr = current_bsr
        organic_ratio = 0.3  # Starting organic ratio

        for month in range(1, months + 1):
            # BSR improves with sales
            total_sales = ad_sales_per_month * (1 + organic_ratio)
            bsr = max(100, int(bsr * 0.85))  # BSR improves ~15%/month

            # Organic ratio grows as BSR improves
            organic_ratio = min(0.7, organic_ratio + 0.07)

            organic_sales = int(ad_sales_per_month * organic_ratio)

            projections.append({
                'month': month,
                'bsr': bsr,
                'ad_sales': ad_sales_per_month,
                'organic_sales': organic_sales,
                'total_sales': ad_sales_per_month + organic_sales,
                'organic_ratio': round(organic_ratio * 100, 1),
            })

        return projections


if __name__ == "__main__":
    print("=" * 70)
    print("PPC ANALYZER")
    print("=" * 70)

    analyzer = PPCAnalyzer(category='electronics')

    # Product details
    product_price = 34.99
    product_cost = 8.00
    fba_fees = 6.50

    # Calculate break-even
    break_even = analyzer.calculate_break_even_acos(
        product_price, product_cost, fba_fees
    )

    print(f"\nProduct: ${product_price}")
    print(f"Cost: ${product_cost} + ${fba_fees} FBA")
    print(f"Break-even ACOS: {break_even}%")

    # Generate strategy
    keywords = [
        "wireless earbuds",
        "bluetooth earbuds",
        "earbuds with microphone",
        "sport earbuds",
        "noise canceling earbuds"
    ]

    strategy = analyzer.generate_ppc_strategy(
        product_price=product_price,
        product_cost=product_cost,
        fba_fees=fba_fees,
        keywords=keywords,
        competition_level='medium'
    )

    print(f"\n{'─'*70}")
    print("PPC STRATEGY")
    print(f"{'─'*70}")
    print(f"  Target ACOS:     {strategy.target_acos:.1f}%")
    print(f"  Break-even ACOS: {strategy.break_even_acos:.1f}%")

    print(f"\n  LAUNCH PHASE (Days 1-30):")
    lb = strategy.launch_phase_budget
    print(f"    Daily Budget:  ${lb.daily_budget}")
    print(f"    Monthly Budget: ${lb.monthly_budget}")
    print(f"    Est. Orders:   {lb.estimated_orders}/month")
    print(f"    Est. ACOS:     {lb.estimated_acos}%")

    print(f"\n  GROWTH PHASE (Days 31-90):")
    gb = strategy.growth_phase_budget
    print(f"    Daily Budget:  ${gb.daily_budget}")
    print(f"    Monthly Budget: ${gb.monthly_budget}")
    print(f"    Est. Orders:   {gb.estimated_orders}/month")
    print(f"    Est. ACOS:     {gb.estimated_acos}%")

    print(f"\n  MAINTENANCE PHASE:")
    mb = strategy.maintenance_budget
    print(f"    Daily Budget:  ${mb.daily_budget}")
    print(f"    Monthly Budget: ${mb.monthly_budget}")
    print(f"    Est. Orders:   {mb.estimated_orders}/month")
    print(f"    Est. ACOS:     {mb.estimated_acos}%")

    print(f"\n{'─'*70}")
    print("CAMPAIGN STRUCTURE")
    print(f"{'─'*70}")
    for camp in strategy.recommended_campaigns:
        print(f"  • {camp['name']}")
        print(f"    Budget: ${camp['daily_budget']}/day | Priority: {camp['priority']}")

    print(f"\n{'─'*70}")
    print("KEYWORD BIDS (Top 5 Exact Match)")
    print(f"{'─'*70}")
    exact_bids = [b for b in strategy.primary_keywords if b.match_type == MatchType.EXACT][:5]
    print(f"  {'Keyword':<30} {'Bid':>8} {'Range':>15}")
    for bid in exact_bids:
        print(f"  {bid.keyword:<30} ${bid.suggested_bid:>5.2f}   ${bid.bid_range_low:.2f}-${bid.bid_range_high:.2f}")

    print(f"\n{'─'*70}")
    print("PROJECTIONS")
    print(f"{'─'*70}")
    print(f"  Monthly Ad Spend:      ${strategy.monthly_ad_spend:,.2f}")
    print(f"  Projected Sales:       ${strategy.projected_monthly_sales:,.2f}")
    print(f"  Profit After Ads:      ${strategy.projected_profit_after_ads:,.2f}")

    # Organic growth projection
    print(f"\n{'─'*70}")
    print("6-MONTH ORGANIC GROWTH PROJECTION")
    print(f"{'─'*70}")
    projections = analyzer.project_organic_growth(
        current_bsr=50000,
        ad_sales_per_month=mb.estimated_orders
    )
    print(f"  {'Month':<8} {'BSR':>8} {'Ad Sales':>10} {'Organic':>10} {'Total':>10} {'Org %':>8}")
    for p in projections:
        print(f"  {p['month']:<8} {p['bsr']:>8,} {p['ad_sales']:>10} {p['organic_sales']:>10} {p['total_sales']:>10} {p['organic_ratio']:>7}%")
