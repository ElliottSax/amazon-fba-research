#!/usr/bin/env python3
"""
Supplier Sourcing Helper for Amazon FBA Research

Helps evaluate and compare suppliers:
- Cost analysis (landed cost calculation)
- MOQ and pricing tier analysis
- Quality scoring
- Lead time evaluation
- Risk assessment
- Supplier comparison
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime


class SupplierType(Enum):
    MANUFACTURER = "manufacturer"
    TRADING_COMPANY = "trading_company"
    WHOLESALER = "wholesaler"
    DISTRIBUTOR = "distributor"


class CommunicationRating(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"


class ShippingMethod(Enum):
    SEA_FREIGHT = "sea_freight"
    AIR_FREIGHT = "air_freight"
    EXPRESS = "express"  # DHL, FedEx, etc.
    RAIL = "rail"


@dataclass
class PricingTier:
    """Pricing tier from supplier"""
    min_quantity: int
    max_quantity: Optional[int]
    unit_price: float
    currency: str = "USD"


@dataclass
class ShippingQuote:
    """Shipping cost estimate"""
    method: ShippingMethod
    cost_per_unit: float
    cost_per_kg: float
    transit_days: int
    min_volume: Optional[float] = None  # CBM for sea freight


@dataclass
class SupplierInfo:
    """Supplier information"""
    name: str
    supplier_type: SupplierType
    country: str
    city: str

    # Pricing
    pricing_tiers: list[PricingTier]
    moq: int
    sample_cost: float = 0

    # Lead times
    sample_lead_time_days: int = 7
    production_lead_time_days: int = 30
    shipping_lead_time_days: int = 30

    # Quality
    has_certifications: list[str] = field(default_factory=list)
    inspection_available: bool = True
    years_in_business: int = 0

    # Communication
    communication: CommunicationRating = CommunicationRating.GOOD
    english_level: str = "good"
    response_time_hours: int = 24

    # Payment
    payment_terms: list[str] = field(default_factory=list)
    accepts_paypal: bool = False
    accepts_trade_assurance: bool = False

    # Other
    alibaba_url: Optional[str] = None
    notes: str = ""


@dataclass
class LandedCostBreakdown:
    """Complete landed cost calculation"""
    supplier_name: str
    quantity: int

    # Product costs
    unit_price: float
    total_product_cost: float

    # Shipping
    shipping_method: ShippingMethod
    shipping_cost: float
    shipping_per_unit: float

    # Import costs
    import_duty_rate: float
    import_duty: float
    customs_fees: float

    # Other costs
    inspection_cost: float
    prep_cost: float
    other_costs: float

    # Totals
    total_landed_cost: float
    landed_cost_per_unit: float

    # Analysis
    margin_at_price: dict[float, float] = field(default_factory=dict)


@dataclass
class SupplierScore:
    """Supplier evaluation score"""
    supplier_name: str

    # Individual scores (0-100)
    price_score: int
    quality_score: int
    reliability_score: int
    communication_score: int
    terms_score: int

    # Overall
    overall_score: int
    grade: str  # A, B, C, D, F
    recommendation: str

    # Pros and cons
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)


@dataclass
class SupplierComparison:
    """Comparison of multiple suppliers"""
    suppliers: list[SupplierInfo]
    scores: list[SupplierScore]
    landed_costs: list[LandedCostBreakdown]
    best_price: str
    best_quality: str
    best_overall: str
    recommendation: str


class SupplierAnalyzer:
    """Analyzes and compares suppliers"""

    # Import duty rates by category (simplified)
    DUTY_RATES = {
        "electronics": 0.0,  # Most electronics duty-free
        "textiles": 0.12,
        "toys": 0.0,
        "home_goods": 0.05,
        "sporting_goods": 0.05,
        "beauty": 0.0,
        "default": 0.05
    }

    # Shipping cost estimates per method
    SHIPPING_ESTIMATES = {
        ShippingMethod.SEA_FREIGHT: {"per_cbm": 150, "per_kg": 0.15, "days": 35},
        ShippingMethod.AIR_FREIGHT: {"per_kg": 4.50, "days": 7},
        ShippingMethod.EXPRESS: {"per_kg": 8.00, "days": 5},
        ShippingMethod.RAIL: {"per_cbm": 100, "per_kg": 0.25, "days": 25},
    }

    def __init__(self, category: str = "default"):
        self.category = category
        self.duty_rate = self.DUTY_RATES.get(category, self.DUTY_RATES["default"])

    def calculate_landed_cost(
        self,
        supplier: SupplierInfo,
        quantity: int,
        shipping_method: ShippingMethod,
        product_weight_kg: float,
        product_volume_cbm: float,
        inspection: bool = True,
        prep_per_unit: float = 0.50
    ) -> LandedCostBreakdown:
        """Calculate total landed cost"""

        # Get unit price for quantity
        unit_price = self._get_price_for_quantity(supplier.pricing_tiers, quantity)
        total_product = unit_price * quantity

        # Shipping cost
        shipping_cost = self._calculate_shipping(
            shipping_method, product_weight_kg * quantity, product_volume_cbm * quantity
        )
        shipping_per_unit = shipping_cost / quantity

        # Import costs
        import_duty = total_product * self.duty_rate
        customs_fees = 50 + (total_product * 0.003)  # MPF + HMF estimates

        # Inspection
        inspection_cost = 300 if inspection and quantity >= 500 else 0

        # Prep and other
        prep_cost = prep_per_unit * quantity
        other_costs = 0

        # Total
        total = (
            total_product +
            shipping_cost +
            import_duty +
            customs_fees +
            inspection_cost +
            prep_cost +
            other_costs
        )

        landed_per_unit = total / quantity

        # Calculate margins at different prices
        margins = {}
        for price in [landed_per_unit * 1.5, landed_per_unit * 2, landed_per_unit * 2.5, landed_per_unit * 3]:
            price = round(price, 2)
            margin = ((price - landed_per_unit) / price) * 100
            margins[price] = round(margin, 1)

        return LandedCostBreakdown(
            supplier_name=supplier.name,
            quantity=quantity,
            unit_price=unit_price,
            total_product_cost=round(total_product, 2),
            shipping_method=shipping_method,
            shipping_cost=round(shipping_cost, 2),
            shipping_per_unit=round(shipping_per_unit, 2),
            import_duty_rate=self.duty_rate,
            import_duty=round(import_duty, 2),
            customs_fees=round(customs_fees, 2),
            inspection_cost=inspection_cost,
            prep_cost=round(prep_cost, 2),
            other_costs=other_costs,
            total_landed_cost=round(total, 2),
            landed_cost_per_unit=round(landed_per_unit, 2),
            margin_at_price=margins
        )

    def _get_price_for_quantity(self, tiers: list[PricingTier], quantity: int) -> float:
        """Get price for a specific quantity"""
        applicable_price = tiers[0].unit_price if tiers else 0

        for tier in sorted(tiers, key=lambda x: x.min_quantity):
            if quantity >= tier.min_quantity:
                if tier.max_quantity is None or quantity <= tier.max_quantity:
                    applicable_price = tier.unit_price

        return applicable_price

    def _calculate_shipping(self, method: ShippingMethod,
                           total_weight_kg: float,
                           total_volume_cbm: float) -> float:
        """Calculate shipping cost"""
        estimates = self.SHIPPING_ESTIMATES[method]

        if method == ShippingMethod.SEA_FREIGHT:
            # Sea freight uses greater of weight or volume
            weight_cost = total_weight_kg * estimates["per_kg"]
            volume_cost = max(1, total_volume_cbm) * estimates["per_cbm"]
            return max(weight_cost, volume_cost, 500)  # Minimum $500

        elif method == ShippingMethod.RAIL:
            weight_cost = total_weight_kg * estimates["per_kg"]
            volume_cost = max(1, total_volume_cbm) * estimates["per_cbm"]
            return max(weight_cost, volume_cost, 400)

        else:
            # Air/Express use weight
            return total_weight_kg * estimates["per_kg"]

    def score_supplier(self, supplier: SupplierInfo,
                      landed_cost: LandedCostBreakdown,
                      target_landed_cost: float) -> SupplierScore:
        """Score a supplier"""
        pros = []
        cons = []

        # Price score (0-100)
        price_ratio = target_landed_cost / max(0.01, landed_cost.landed_cost_per_unit)
        price_score = min(100, int(price_ratio * 70))

        if landed_cost.landed_cost_per_unit <= target_landed_cost:
            pros.append(f"Competitive pricing (${landed_cost.landed_cost_per_unit:.2f}/unit)")
        else:
            cons.append(f"Above target cost (${landed_cost.landed_cost_per_unit:.2f} vs ${target_landed_cost:.2f})")

        # Quality score
        quality_score = 50
        if supplier.has_certifications:
            quality_score += len(supplier.has_certifications) * 10
            pros.append(f"Certifications: {', '.join(supplier.has_certifications[:3])}")

        if supplier.supplier_type == SupplierType.MANUFACTURER:
            quality_score += 15
            pros.append("Direct manufacturer")
        elif supplier.supplier_type == SupplierType.TRADING_COMPANY:
            quality_score -= 10
            cons.append("Trading company (not direct)")

        if supplier.years_in_business >= 10:
            quality_score += 10
            pros.append(f"{supplier.years_in_business} years in business")
        elif supplier.years_in_business < 3:
            quality_score -= 10
            cons.append("Less than 3 years in business")

        if supplier.inspection_available:
            quality_score += 5

        quality_score = min(100, quality_score)

        # Reliability score
        reliability_score = 60

        total_lead_time = (
            supplier.production_lead_time_days +
            supplier.shipping_lead_time_days
        )

        if total_lead_time <= 45:
            reliability_score += 20
            pros.append(f"Fast lead time ({total_lead_time} days)")
        elif total_lead_time > 75:
            reliability_score -= 15
            cons.append(f"Long lead time ({total_lead_time} days)")

        if supplier.moq <= 500:
            reliability_score += 10
            pros.append(f"Low MOQ ({supplier.moq} units)")
        elif supplier.moq > 2000:
            reliability_score -= 10
            cons.append(f"High MOQ ({supplier.moq} units)")

        reliability_score = min(100, reliability_score)

        # Communication score
        comm_scores = {
            CommunicationRating.EXCELLENT: 95,
            CommunicationRating.GOOD: 80,
            CommunicationRating.AVERAGE: 60,
            CommunicationRating.POOR: 30
        }
        communication_score = comm_scores[supplier.communication]

        if supplier.response_time_hours <= 12:
            communication_score = min(100, communication_score + 10)
            pros.append("Fast response time")
        elif supplier.response_time_hours > 48:
            communication_score -= 15
            cons.append("Slow response time")

        # Terms score
        terms_score = 50

        if supplier.accepts_trade_assurance:
            terms_score += 20
            pros.append("Accepts Trade Assurance")

        if supplier.accepts_paypal:
            terms_score += 10

        if "30% deposit" in supplier.payment_terms or "T/T" in supplier.payment_terms:
            terms_score += 10

        if supplier.sample_cost == 0:
            terms_score += 10
            pros.append("Free samples available")

        terms_score = min(100, terms_score)

        # Overall score (weighted)
        overall = int(
            price_score * 0.30 +
            quality_score * 0.25 +
            reliability_score * 0.20 +
            communication_score * 0.15 +
            terms_score * 0.10
        )

        # Grade
        if overall >= 85:
            grade = "A"
            recommendation = "Highly recommended - consider as primary supplier"
        elif overall >= 70:
            grade = "B"
            recommendation = "Good option - order samples to verify quality"
        elif overall >= 55:
            grade = "C"
            recommendation = "Acceptable - proceed with caution"
        elif overall >= 40:
            grade = "D"
            recommendation = "Below average - look for alternatives"
        else:
            grade = "F"
            recommendation = "Not recommended"

        return SupplierScore(
            supplier_name=supplier.name,
            price_score=price_score,
            quality_score=quality_score,
            reliability_score=reliability_score,
            communication_score=communication_score,
            terms_score=terms_score,
            overall_score=overall,
            grade=grade,
            recommendation=recommendation,
            pros=pros,
            cons=cons
        )

    def compare_suppliers(
        self,
        suppliers: list[SupplierInfo],
        quantity: int,
        shipping_method: ShippingMethod,
        product_weight_kg: float,
        product_volume_cbm: float,
        target_landed_cost: float
    ) -> SupplierComparison:
        """Compare multiple suppliers"""

        landed_costs = []
        scores = []

        for supplier in suppliers:
            cost = self.calculate_landed_cost(
                supplier, quantity, shipping_method,
                product_weight_kg, product_volume_cbm
            )
            landed_costs.append(cost)

            score = self.score_supplier(supplier, cost, target_landed_cost)
            scores.append(score)

        # Find best in categories
        best_price = min(landed_costs, key=lambda x: x.landed_cost_per_unit).supplier_name
        best_quality = max(scores, key=lambda x: x.quality_score).supplier_name
        best_overall = max(scores, key=lambda x: x.overall_score).supplier_name

        # Recommendation
        best_score = max(scores, key=lambda x: x.overall_score)
        if best_score.overall_score >= 70:
            recommendation = f"Recommend {best_overall} as primary supplier. Order samples first."
        elif best_score.overall_score >= 50:
            recommendation = f"{best_overall} is the best option but score is marginal. Consider additional sourcing."
        else:
            recommendation = "No strong candidates. Recommend expanding supplier search."

        return SupplierComparison(
            suppliers=suppliers,
            scores=scores,
            landed_costs=landed_costs,
            best_price=best_price,
            best_quality=best_quality,
            best_overall=best_overall,
            recommendation=recommendation
        )

    def estimate_minimum_order_value(self, landed_cost_per_unit: float,
                                     target_margin: float,
                                     monthly_sales: int) -> dict:
        """Estimate minimum order based on cash flow"""

        # Target selling price
        selling_price = landed_cost_per_unit / (1 - target_margin / 100)

        # 3-month supply recommendation
        three_month_units = monthly_sales * 3
        order_value = three_month_units * landed_cost_per_unit

        return {
            "recommended_units": three_month_units,
            "order_value": round(order_value, 2),
            "target_selling_price": round(selling_price, 2),
            "expected_margin": target_margin,
            "monthly_revenue": round(monthly_sales * selling_price, 2),
            "monthly_profit": round(monthly_sales * (selling_price - landed_cost_per_unit), 2)
        }


def create_sample_suppliers() -> list[SupplierInfo]:
    """Create sample suppliers for testing"""
    return [
        SupplierInfo(
            name="Shenzhen Electronics Co.",
            supplier_type=SupplierType.MANUFACTURER,
            country="China",
            city="Shenzhen",
            pricing_tiers=[
                PricingTier(100, 499, 5.50),
                PricingTier(500, 999, 4.80),
                PricingTier(1000, None, 4.20),
            ],
            moq=500,
            sample_cost=25,
            production_lead_time_days=25,
            shipping_lead_time_days=35,
            has_certifications=["ISO9001", "CE", "FCC"],
            years_in_business=12,
            communication=CommunicationRating.EXCELLENT,
            response_time_hours=8,
            payment_terms=["30% deposit, 70% before shipping", "T/T"],
            accepts_trade_assurance=True
        ),
        SupplierInfo(
            name="Guangzhou Trading Ltd.",
            supplier_type=SupplierType.TRADING_COMPANY,
            country="China",
            city="Guangzhou",
            pricing_tiers=[
                PricingTier(100, 499, 5.00),
                PricingTier(500, 999, 4.50),
                PricingTier(1000, None, 4.00),
            ],
            moq=300,
            sample_cost=0,
            production_lead_time_days=30,
            shipping_lead_time_days=35,
            has_certifications=["CE"],
            years_in_business=5,
            communication=CommunicationRating.GOOD,
            response_time_hours=24,
            payment_terms=["50% deposit", "PayPal", "T/T"],
            accepts_paypal=True,
            accepts_trade_assurance=True
        ),
        SupplierInfo(
            name="Ningbo Quality Tech",
            supplier_type=SupplierType.MANUFACTURER,
            country="China",
            city="Ningbo",
            pricing_tiers=[
                PricingTier(500, 999, 5.20),
                PricingTier(1000, 2999, 4.50),
                PricingTier(3000, None, 3.90),
            ],
            moq=1000,
            sample_cost=50,
            production_lead_time_days=20,
            shipping_lead_time_days=30,
            has_certifications=["ISO9001", "ISO14001", "CE", "FCC", "RoHS"],
            years_in_business=18,
            communication=CommunicationRating.GOOD,
            response_time_hours=16,
            payment_terms=["30% deposit, 70% before shipping"],
            accepts_trade_assurance=True
        ),
    ]


if __name__ == "__main__":
    print("=" * 70)
    print("SUPPLIER SOURCING ANALYZER")
    print("=" * 70)

    analyzer = SupplierAnalyzer(category="electronics")
    suppliers = create_sample_suppliers()

    # Compare suppliers
    comparison = analyzer.compare_suppliers(
        suppliers=suppliers,
        quantity=1000,
        shipping_method=ShippingMethod.SEA_FREIGHT,
        product_weight_kg=0.15,  # 150g per unit
        product_volume_cbm=0.001,  # Small product
        target_landed_cost=10.00
    )

    print(f"\nComparing {len(suppliers)} suppliers for 1,000 units:")

    print("\n" + "-" * 70)
    print("LANDED COST COMPARISON")
    print("-" * 70)
    for cost in comparison.landed_costs:
        print(f"\n  {cost.supplier_name}")
        print(f"    Unit Price:      ${cost.unit_price:.2f}")
        print(f"    Shipping/Unit:   ${cost.shipping_per_unit:.2f}")
        print(f"    Landed Cost:     ${cost.landed_cost_per_unit:.2f}")
        print(f"    Total Order:     ${cost.total_landed_cost:,.2f}")

    print("\n" + "-" * 70)
    print("SUPPLIER SCORES")
    print("-" * 70)
    for score in comparison.scores:
        print(f"\n  {score.supplier_name}: {score.overall_score}/100 (Grade: {score.grade})")
        print(f"    Price: {score.price_score} | Quality: {score.quality_score} | "
              f"Reliability: {score.reliability_score}")
        if score.pros:
            print(f"    ✓ {score.pros[0]}")
        if score.cons:
            print(f"    ✗ {score.cons[0]}")

    print("\n" + "-" * 70)
    print("RECOMMENDATION")
    print("-" * 70)
    print(f"  Best Price:    {comparison.best_price}")
    print(f"  Best Quality:  {comparison.best_quality}")
    print(f"  Best Overall:  {comparison.best_overall}")
    print(f"\n  {comparison.recommendation}")

    # Detailed analysis of best supplier
    print("\n" + "-" * 70)
    print(f"DETAILED: {comparison.best_overall}")
    print("-" * 70)
    best_cost = next(c for c in comparison.landed_costs if c.supplier_name == comparison.best_overall)
    best_score = next(s for s in comparison.scores if s.supplier_name == comparison.best_overall)

    print(f"\n  Cost Breakdown:")
    print(f"    Product:     ${best_cost.total_product_cost:,.2f}")
    print(f"    Shipping:    ${best_cost.shipping_cost:,.2f}")
    print(f"    Import Duty: ${best_cost.import_duty:,.2f}")
    print(f"    Customs:     ${best_cost.customs_fees:,.2f}")
    print(f"    Inspection:  ${best_cost.inspection_cost:,.2f}")
    print(f"    Prep:        ${best_cost.prep_cost:,.2f}")
    print(f"    TOTAL:       ${best_cost.total_landed_cost:,.2f}")

    print(f"\n  Margin Analysis:")
    for price, margin in sorted(best_cost.margin_at_price.items()):
        print(f"    @ ${price:.2f}: {margin:.1f}% margin")

    print(f"\n  Pros:")
    for pro in best_score.pros:
        print(f"    ✓ {pro}")
    print(f"\n  Cons:")
    for con in best_score.cons:
        print(f"    ✗ {con}")

    # Order recommendation
    print("\n" + "-" * 70)
    print("ORDER RECOMMENDATION")
    print("-" * 70)
    order_rec = analyzer.estimate_minimum_order_value(
        best_cost.landed_cost_per_unit,
        target_margin=35,
        monthly_sales=200
    )
    print(f"  For 200 units/month with 35% target margin:")
    print(f"    Recommended Order:  {order_rec['recommended_units']} units (3-month supply)")
    print(f"    Order Value:        ${order_rec['order_value']:,.2f}")
    print(f"    Target Sell Price:  ${order_rec['target_selling_price']:.2f}")
    print(f"    Monthly Revenue:    ${order_rec['monthly_revenue']:,.2f}")
    print(f"    Monthly Profit:     ${order_rec['monthly_profit']:,.2f}")

    print("\n" + "=" * 70)
