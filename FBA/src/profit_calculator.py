"""
Amazon FBA Profit Calculator

Calculates profitability including all FBA fees, costs, and margins.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class ProductSize(Enum):
    """Amazon product size tiers."""
    SMALL_STANDARD = "small_standard"
    LARGE_STANDARD = "large_standard"
    SMALL_OVERSIZE = "small_oversize"
    MEDIUM_OVERSIZE = "medium_oversize"
    LARGE_OVERSIZE = "large_oversize"
    SPECIAL_OVERSIZE = "special_oversize"


@dataclass
class ProductDimensions:
    """Product physical dimensions."""
    length_inches: float
    width_inches: float
    height_inches: float
    weight_lbs: float

    @property
    def dimensional_weight(self) -> float:
        """Calculate dimensional weight (length × width × height / 139)."""
        return (self.length_inches * self.width_inches * self.height_inches) / 139

    @property
    def billable_weight(self) -> float:
        """Billable weight is greater of actual or dimensional."""
        return max(self.weight_lbs, self.dimensional_weight)

    @property
    def girth(self) -> float:
        """Girth = 2 × (width + height)."""
        return 2 * (self.width_inches + self.height_inches)

    @property
    def length_plus_girth(self) -> float:
        """Length + girth for oversize determination."""
        return self.length_inches + self.girth

    def get_size_tier(self) -> ProductSize:
        """Determine Amazon size tier based on dimensions."""
        l, w, h = self.length_inches, self.width_inches, self.height_inches
        wt = self.weight_lbs

        # Sort dimensions
        dims = sorted([l, w, h], reverse=True)
        longest, median, shortest = dims

        # Small standard: ≤15" × 12" × 0.75", ≤1 lb
        if longest <= 15 and median <= 12 and shortest <= 0.75 and wt <= 1:
            return ProductSize.SMALL_STANDARD

        # Large standard: ≤18" × 14" × 8", ≤20 lb
        if longest <= 18 and median <= 14 and shortest <= 8 and wt <= 20:
            return ProductSize.LARGE_STANDARD

        # Small oversize: ≤60" × 30", ≤70 lb, length+girth ≤130"
        if longest <= 60 and median <= 30 and wt <= 70 and self.length_plus_girth <= 130:
            return ProductSize.SMALL_OVERSIZE

        # Medium oversize: ≤108", ≤150 lb, length+girth ≤130"
        if longest <= 108 and wt <= 150 and self.length_plus_girth <= 130:
            return ProductSize.MEDIUM_OVERSIZE

        # Large oversize: ≤108", ≤150 lb, length+girth ≤165"
        if longest <= 108 and wt <= 150 and self.length_plus_girth <= 165:
            return ProductSize.LARGE_OVERSIZE

        # Special oversize
        return ProductSize.SPECIAL_OVERSIZE


@dataclass
class FBAFees:
    """Breakdown of all FBA fees."""
    referral_fee: float
    fba_fulfillment_fee: float
    monthly_storage_fee: float
    closing_fee: float = 0.0  # For media items

    @property
    def total_fees(self) -> float:
        return self.referral_fee + self.fba_fulfillment_fee + self.monthly_storage_fee + self.closing_fee


@dataclass
class ProfitResult:
    """Complete profit calculation result."""
    # Revenue
    sale_price: float

    # Costs
    product_cost: float
    shipping_to_fba: float
    fba_fees: FBAFees
    other_costs: float

    # Calculations
    @property
    def total_cost(self) -> float:
        return self.product_cost + self.shipping_to_fba + self.fba_fees.total_fees + self.other_costs

    @property
    def profit(self) -> float:
        return self.sale_price - self.total_cost

    @property
    def profit_margin(self) -> float:
        if self.sale_price <= 0:
            return 0.0
        return (self.profit / self.sale_price) * 100

    @property
    def roi(self) -> float:
        """Return on Investment."""
        invested = self.product_cost + self.shipping_to_fba + self.other_costs
        if invested <= 0:
            return 0.0
        return (self.profit / invested) * 100


class FBAProfitCalculator:
    """
    Calculate Amazon FBA profitability.

    Includes:
    - Referral fees (by category)
    - FBA fulfillment fees (by size tier)
    - Monthly storage fees
    - Profit margins and ROI
    """

    # Referral fee percentages by category (US marketplace)
    REFERRAL_FEES = {
        'default': 0.15,  # 15%
        'amazon_device_accessories': 0.45,
        'appliances': 0.15,
        'automotive': 0.12,
        'baby_products': 0.15,
        'beauty': 0.15,
        'books': 0.15,
        'camera': 0.08,
        'cell_phone_devices': 0.08,
        'clothing': 0.17,
        'computers': 0.08,
        'consumer_electronics': 0.08,
        'electronics_accessories': 0.15,
        'furniture': 0.15,
        'grocery': 0.15,
        'health_personal_care': 0.15,
        'home_garden': 0.15,
        'jewelry': 0.20,
        'kitchen': 0.15,
        'luggage': 0.15,
        'music': 0.15,
        'musical_instruments': 0.15,
        'office_products': 0.15,
        'outdoors': 0.15,
        'personal_computers': 0.08,
        'pet_supplies': 0.15,
        'shoes': 0.15,
        'software': 0.15,
        'sports': 0.15,
        'tools': 0.15,
        'toys': 0.15,
        'video_games': 0.15,
        'watches': 0.16,
    }

    # Minimum referral fees
    MIN_REFERRAL_FEE = 0.30  # $0.30 minimum

    # FBA Fulfillment fees (2024 rates, US)
    # Format: (weight_threshold_oz, fee)
    FBA_FEES_SMALL_STANDARD = [
        (4, 3.22),
        (8, 3.40),
        (12, 3.58),
        (16, 3.77),
    ]

    FBA_FEES_LARGE_STANDARD = [
        (4, 3.86),
        (8, 4.08),
        (12, 4.24),
        (16, 4.75),
        (24, 5.40),
        (32, 5.69),
        (48, 6.10),
        (64, 6.62),
        (80, 7.17),
        (96, 7.72),
        (112, 8.13),
        (128, 8.66),
        (144, 9.28),
        (160, 9.28),  # Up to 10 lbs base
        (320, 9.28),  # Over 3 lbs, add $0.16/4oz
    ]

    # Monthly storage fees (per cubic foot)
    STORAGE_FEE_STANDARD = {
        'jan_sep': 0.87,
        'oct_dec': 2.40,
    }

    STORAGE_FEE_OVERSIZE = {
        'jan_sep': 0.56,
        'oct_dec': 1.40,
    }

    def __init__(self, category: str = 'default'):
        """
        Initialize calculator.

        Args:
            category: Product category for referral fee calculation
        """
        self.category = category.lower().replace(' ', '_')

    def get_referral_fee(self, sale_price: float) -> float:
        """Calculate referral fee for sale price."""
        rate = self.REFERRAL_FEES.get(self.category, self.REFERRAL_FEES['default'])
        fee = sale_price * rate
        return max(fee, self.MIN_REFERRAL_FEE)

    def get_fba_fee(self, dimensions: ProductDimensions) -> float:
        """Calculate FBA fulfillment fee based on size tier."""
        size_tier = dimensions.get_size_tier()
        weight_oz = dimensions.weight_lbs * 16

        if size_tier == ProductSize.SMALL_STANDARD:
            for threshold, fee in self.FBA_FEES_SMALL_STANDARD:
                if weight_oz <= threshold:
                    return fee
            return self.FBA_FEES_SMALL_STANDARD[-1][1]

        elif size_tier == ProductSize.LARGE_STANDARD:
            for threshold, fee in self.FBA_FEES_LARGE_STANDARD:
                if weight_oz <= threshold:
                    return fee
            # Over 10 lbs: base + $0.16 per 4oz over 3 lbs
            base_fee = 9.28
            extra_weight = max(0, weight_oz - 48)  # Over 3 lbs
            extra_fee = (extra_weight / 4) * 0.16
            return base_fee + extra_fee

        elif size_tier == ProductSize.SMALL_OVERSIZE:
            # Base fee + weight-based fee
            return 9.73 + max(0, dimensions.billable_weight - 1) * 0.42

        elif size_tier == ProductSize.MEDIUM_OVERSIZE:
            return 19.05 + max(0, dimensions.billable_weight - 1) * 0.42

        elif size_tier == ProductSize.LARGE_OVERSIZE:
            return 89.98 + max(0, dimensions.billable_weight - 90) * 0.83

        else:  # Special oversize
            return 158.49 + max(0, dimensions.billable_weight - 90) * 0.83

    def get_storage_fee(
        self,
        dimensions: ProductDimensions,
        month: int = 6,
        units: int = 1
    ) -> float:
        """
        Calculate monthly storage fee.

        Args:
            dimensions: Product dimensions
            month: Month (1-12) for seasonal rates
            units: Number of units stored
        """
        # Calculate cubic feet
        cubic_feet = (
            dimensions.length_inches *
            dimensions.width_inches *
            dimensions.height_inches
        ) / 1728  # cubic inches to cubic feet

        size_tier = dimensions.get_size_tier()
        is_oversize = size_tier not in [ProductSize.SMALL_STANDARD, ProductSize.LARGE_STANDARD]

        # Determine rate based on season
        if 10 <= month <= 12:
            rate = self.STORAGE_FEE_OVERSIZE['oct_dec'] if is_oversize else self.STORAGE_FEE_STANDARD['oct_dec']
        else:
            rate = self.STORAGE_FEE_OVERSIZE['jan_sep'] if is_oversize else self.STORAGE_FEE_STANDARD['jan_sep']

        return cubic_feet * rate * units

    def calculate_profit(
        self,
        sale_price: float,
        product_cost: float,
        dimensions: ProductDimensions,
        shipping_to_fba: float = 0.0,
        other_costs: float = 0.0,
        storage_month: int = 6
    ) -> ProfitResult:
        """
        Calculate complete profit breakdown.

        Args:
            sale_price: Amazon selling price
            product_cost: Cost to acquire product
            dimensions: Product dimensions
            shipping_to_fba: Cost to ship to FBA warehouse
            other_costs: Any other costs (packaging, prep, etc.)
            storage_month: Month for storage fee calculation

        Returns:
            ProfitResult with full breakdown
        """
        referral_fee = self.get_referral_fee(sale_price)
        fba_fee = self.get_fba_fee(dimensions)
        storage_fee = self.get_storage_fee(dimensions, storage_month)

        fees = FBAFees(
            referral_fee=referral_fee,
            fba_fulfillment_fee=fba_fee,
            monthly_storage_fee=storage_fee
        )

        return ProfitResult(
            sale_price=sale_price,
            product_cost=product_cost,
            shipping_to_fba=shipping_to_fba,
            fba_fees=fees,
            other_costs=other_costs
        )

    def find_minimum_price(
        self,
        product_cost: float,
        dimensions: ProductDimensions,
        target_margin: float = 30.0,
        shipping_to_fba: float = 0.0,
        other_costs: float = 0.0
    ) -> Dict[str, Any]:
        """
        Find minimum sale price to achieve target margin.

        Args:
            product_cost: Cost to acquire product
            dimensions: Product dimensions
            target_margin: Target profit margin percentage
            shipping_to_fba: Shipping cost to FBA
            other_costs: Other costs

        Returns:
            Dict with minimum price and breakdown
        """
        # Binary search for price
        low, high = product_cost, product_cost * 10

        for _ in range(50):  # Max iterations
            mid = (low + high) / 2
            result = self.calculate_profit(
                mid, product_cost, dimensions, shipping_to_fba, other_costs
            )

            if abs(result.profit_margin - target_margin) < 0.1:
                break
            elif result.profit_margin < target_margin:
                low = mid
            else:
                high = mid

        result = self.calculate_profit(
            mid, product_cost, dimensions, shipping_to_fba, other_costs
        )

        return {
            'minimum_price': round(mid, 2),
            'profit': round(result.profit, 2),
            'margin': round(result.profit_margin, 1),
            'roi': round(result.roi, 1),
            'fees_breakdown': {
                'referral': round(result.fba_fees.referral_fee, 2),
                'fulfillment': round(result.fba_fees.fba_fulfillment_fee, 2),
                'storage': round(result.fba_fees.monthly_storage_fee, 2),
            }
        }


def quick_profit_check(
    sale_price: float,
    product_cost: float,
    weight_lbs: float,
    category: str = 'default'
) -> Dict[str, Any]:
    """
    Quick profit estimate using standard dimensions.

    Args:
        sale_price: Selling price on Amazon
        product_cost: Your cost per unit
        weight_lbs: Product weight in pounds
        category: Amazon category

    Returns:
        Quick profit estimate
    """
    # Estimate dimensions based on weight
    if weight_lbs <= 1:
        dims = ProductDimensions(10, 8, 2, weight_lbs)
    elif weight_lbs <= 3:
        dims = ProductDimensions(14, 10, 4, weight_lbs)
    elif weight_lbs <= 10:
        dims = ProductDimensions(16, 12, 6, weight_lbs)
    else:
        dims = ProductDimensions(18, 14, 8, weight_lbs)

    calc = FBAProfitCalculator(category)
    result = calc.calculate_profit(sale_price, product_cost, dims)

    return {
        'sale_price': sale_price,
        'product_cost': product_cost,
        'total_fees': round(result.fba_fees.total_fees, 2),
        'profit': round(result.profit, 2),
        'margin': round(result.profit_margin, 1),
        'roi': round(result.roi, 1),
        'viable': result.profit_margin >= 20,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("FBA PROFIT CALCULATOR")
    print("=" * 60)

    # Example product
    dims = ProductDimensions(
        length_inches=12,
        width_inches=8,
        height_inches=4,
        weight_lbs=1.5
    )

    print(f"\nProduct Dimensions: {dims.length_inches}\" × {dims.width_inches}\" × {dims.height_inches}\"")
    print(f"Weight: {dims.weight_lbs} lbs")
    print(f"Size Tier: {dims.get_size_tier().value}")

    calc = FBAProfitCalculator(category='home_garden')

    # Calculate profit
    result = calc.calculate_profit(
        sale_price=29.99,
        product_cost=8.00,
        dimensions=dims,
        shipping_to_fba=1.50
    )

    print(f"\n{'='*40}")
    print("PROFIT BREAKDOWN")
    print(f"{'='*40}")
    print(f"Sale Price:          ${result.sale_price:>8.2f}")
    print(f"{'─'*40}")
    print(f"Product Cost:        ${result.product_cost:>8.2f}")
    print(f"Shipping to FBA:     ${result.shipping_to_fba:>8.2f}")
    print(f"Referral Fee:        ${result.fba_fees.referral_fee:>8.2f}")
    print(f"FBA Fulfillment:     ${result.fba_fees.fba_fulfillment_fee:>8.2f}")
    print(f"Storage (monthly):   ${result.fba_fees.monthly_storage_fee:>8.2f}")
    print(f"{'─'*40}")
    print(f"Total Cost:          ${result.total_cost:>8.2f}")
    print(f"{'='*40}")
    print(f"PROFIT:              ${result.profit:>8.2f}")
    print(f"Margin:              {result.profit_margin:>8.1f}%")
    print(f"ROI:                 {result.roi:>8.1f}%")

    # Find minimum price
    print(f"\n{'='*40}")
    print("MINIMUM PRICE FOR 30% MARGIN")
    print(f"{'='*40}")
    min_price = calc.find_minimum_price(
        product_cost=8.00,
        dimensions=dims,
        target_margin=30.0,
        shipping_to_fba=1.50
    )
    print(f"Minimum Price: ${min_price['minimum_price']}")
    print(f"Profit: ${min_price['profit']}")
    print(f"Margin: {min_price['margin']}%")

    # Quick check
    print(f"\n{'='*40}")
    print("QUICK PROFIT CHECK")
    print(f"{'='*40}")
    quick = quick_profit_check(24.99, 6.00, 0.8, 'toys')
    print(f"Sale: ${quick['sale_price']} | Cost: ${quick['product_cost']}")
    print(f"Fees: ${quick['total_fees']} | Profit: ${quick['profit']}")
    print(f"Margin: {quick['margin']}% | ROI: {quick['roi']}%")
    print(f"Viable: {'✓ Yes' if quick['viable'] else '✗ No'}")
