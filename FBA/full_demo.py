#!/usr/bin/env python3
"""
Full demo of FBA Product Research Pipeline

Demonstrates all features with mock data (no API keys required).
"""
import sys
import json
from datetime import datetime

print("=" * 70)
print("FBA PRODUCT RESEARCH PIPELINE - FULL DEMO")
print("=" * 70)

# ============================================================================
# 1. INVENTORY FORECASTING DEMO
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: INVENTORY FORECASTING & REORDER OPTIMIZATION")
print("=" * 70)

from inventory_forecast import InventoryForecaster, InventoryItem

# Create sample inventory
products = [
    InventoryItem(
        sku="EARBUDS-001",
        asin="B08PZHWJS5",
        name="Wireless Earbuds Pro",
        current_stock=45,  # Low stock!
        unit_cost=15.00,
        lead_time_days=21,
        sales_history=[8, 12, 9, 11, 10, 13, 8, 11, 9, 12, 10, 14, 9, 11, 10,
                       12, 8, 13, 9, 11, 10, 12, 9, 14, 11, 10, 12, 9, 13, 10],
        reorder_cost=150.00
    ),
    InventoryItem(
        sku="CHARGER-002",
        asin="B09V3KXJPB",
        name="USB-C Fast Charger",
        current_stock=500,  # Overstocked
        unit_cost=8.00,
        lead_time_days=14,
        sales_history=[3, 4, 2, 5, 3, 4, 2, 3, 4, 5, 2, 3, 4, 3, 5],
        reorder_cost=100.00
    ),
    InventoryItem(
        sku="CABLE-003",
        asin="B07XYZ1234",
        name="Premium USB Cable 3-Pack",
        current_stock=180,
        unit_cost=5.00,
        lead_time_days=10,
        sales_history=[6, 7, 5, 8, 6, 7, 5, 6, 8, 7, 5, 6, 7, 8, 6],
        reorder_cost=75.00
    ),
]

forecaster = InventoryForecaster(service_level=0.95)

print("\nInventory Analysis:")
print("-" * 70)
print(f"{'SKU':<15} {'Stock':>7} {'ROP':>6} {'EOQ':>6} {'Days':>6} {'Urgency':<12} {'Action'}")
print("-" * 70)

for item in products:
    rec = forecaster.generate_reorder_recommendation(item)

    # Determine action
    if rec.reorder_urgency == 'critical':
        action = f"ORDER {rec.recommended_order_qty} NOW!"
    elif rec.reorder_urgency == 'soon':
        action = f"Order {rec.recommended_order_qty} soon"
    elif rec.reorder_urgency == 'overstocked':
        action = "Consider promotion"
    else:
        action = "OK"

    print(f"{item.sku:<15} {rec.current_stock:>7} {rec.reorder_point:>6} "
          f"{rec.economic_order_quantity:>6} {rec.days_of_stock:>6.0f} "
          f"{rec.reorder_urgency:<12} {action}")

# Full analysis
analysis = forecaster.analyze_inventory(products)
print("\n" + "-" * 70)
print("Summary:")
print(f"  Total SKUs: {analysis['summary']['total_skus']}")
print(f"  Critical: {analysis['summary']['critical_count']}")
print(f"  Reorder Soon: {analysis['summary']['reorder_soon_count']}")
print(f"  Overstocked: {analysis['summary']['overstocked_count']}")
print(f"  Total Inventory Value: ${analysis['summary']['total_inventory_value']:,.2f}")

# ============================================================================
# 2. SP-API REPORT TYPES DEMO
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: SP-API REPORT TYPES (FREE BULK DATA)")
print("=" * 70)

from sp_api_client import REPORT_TYPES

print("\nAvailable Reports (No per-request fees!):")
for category, reports in REPORT_TYPES.items():
    print(f"\n  {category.upper()}:")
    for report_type, description in reports.items():
        print(f"    • {report_type}")
        print(f"      {description}")

# ============================================================================
# 3. PRODUCT RESEARCH DEMO (Mock Data)
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: PRODUCT RESEARCH & OPPORTUNITY ANALYSIS")
print("=" * 70)

from research_pipeline import ProductResearchResult, FBAResearchPipeline

# Create mock research results (simulating what we'd get from APIs)
mock_products = [
    ProductResearchResult(
        asin='B08PZHWJS5',
        timestamp=datetime.now().isoformat(),
        title='Premium Wireless Earbuds with ANC',
        brand='TechPro',
        category=['Electronics', 'Headphones', 'Earbuds'],
        current_price=49.99,
        avg_price_90d=54.99,
        min_price_90d=39.99,
        max_price_90d=64.99,
        buy_box_price=49.99,
        sales_rank=2500,
        estimated_monthly_sales=450,
        new_offer_count=12,
        fba_seller_count=8,
        rating=4.4,
        review_count=2847,
        price_volatility='medium',
        competition_level='medium',
        sales_velocity='fast',
        price_history={},
        keepa_stats={},
        sp_api_data={}
    ),
    ProductResearchResult(
        asin='B09V3KXJPB',
        timestamp=datetime.now().isoformat(),
        title='65W USB-C GaN Charger',
        brand='PowerMax',
        category=['Electronics', 'Chargers'],
        current_price=24.99,
        avg_price_90d=27.99,
        min_price_90d=19.99,
        max_price_90d=29.99,
        buy_box_price=24.99,
        sales_rank=8500,
        estimated_monthly_sales=180,
        new_offer_count=25,
        fba_seller_count=15,
        rating=4.6,
        review_count=5632,
        price_volatility='low',
        competition_level='high',
        sales_velocity='moderate',
        price_history={},
        keepa_stats={},
        sp_api_data={}
    ),
    ProductResearchResult(
        asin='B07NEWITEM1',
        timestamp=datetime.now().isoformat(),
        title='Silicone Kitchen Utensil Set',
        brand='HomeChef',
        category=['Kitchen', 'Utensils'],
        current_price=19.99,
        avg_price_90d=19.99,
        min_price_90d=17.99,
        max_price_90d=22.99,
        buy_box_price=19.99,
        sales_rank=45000,
        estimated_monthly_sales=35,
        new_offer_count=3,
        fba_seller_count=2,
        rating=4.7,
        review_count=156,
        price_volatility='low',
        competition_level='low',
        sales_velocity='slow',
        price_history={},
        keepa_stats={},
        sp_api_data={}
    ),
]

# Create pipeline for analysis (won't make API calls)
pipeline = FBAResearchPipeline.__new__(FBAResearchPipeline)
pipeline.marketplace = 'US'
pipeline.sp_api = None
pipeline.keepa = None
pipeline.forecaster = forecaster

print("\nProduct Opportunity Analysis:")
print("-" * 70)

for product in mock_products:
    analysis = pipeline.analyze_opportunity(product)

    print(f"\n{product.title[:50]}")
    print(f"  ASIN: {product.asin}")
    print(f"  Price: ${product.current_price} | BSR: #{product.sales_rank:,}")
    print(f"  Monthly Sales: ~{product.estimated_monthly_sales} | Competition: {product.competition_level}")
    print(f"  Rating: {product.rating}★ ({product.review_count:,} reviews)")
    print(f"\n  Scores:")
    for metric, score in analysis['scores'].items():
        bar = "█" * (score // 10) + "░" * (10 - score // 10)
        print(f"    {metric:<18} {bar} {score}")
    print(f"\n  OVERALL: {analysis['overall_score']}/100")
    print(f"  RECOMMENDATION: {analysis['recommendation']}")

# ============================================================================
# 4. EXPORT DEMO
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: EXPORT CAPABILITIES")
print("=" * 70)

# Show what export would look like
sample_export = {
    'generated_at': datetime.now().isoformat(),
    'marketplace': 'US',
    'product_count': len(mock_products),
    'products': [
        {
            'asin': p.asin,
            'title': p.title,
            'current_price': p.current_price,
            'sales_rank': p.sales_rank,
            'opportunity_score': pipeline.analyze_opportunity(p)['overall_score'],
            'recommendation': pipeline.analyze_opportunity(p)['recommendation']
        }
        for p in mock_products
    ]
}

print("\nSample Export (JSON):")
print(json.dumps(sample_export, indent=2, default=str))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DEMO COMPLETE")
print("=" * 70)
print("""
This demo showed:

1. INVENTORY FORECASTING
   - Demand prediction using exponential smoothing
   - Safety stock calculation (95% service level)
   - Economic Order Quantity (EOQ) optimization
   - Reorder point recommendations

2. SP-API REPORTS (FREE)
   - 18 report types available
   - No per-request fees for bulk data
   - Inventory, sales, FBA, and analytics reports

3. PRODUCT RESEARCH
   - Opportunity scoring (demand, competition, price stability)
   - Recommendation engine (STRONG/MODERATE/WEAK/NOT_RECOMMENDED)
   - Multi-metric analysis

4. EXPORT
   - JSON export with full analysis
   - Ready for dashboards or spreadsheets

NEXT STEPS:
   1. Configure SP-API credentials (see SETUP.md)
   2. Get Keepa API key for price history
   3. Run: from research_pipeline import FBAResearchPipeline
          pipeline = FBAResearchPipeline()
          result = pipeline.research_product('B08PZHWJS5')
""")

# ============================================================================
# 5. PROFIT CALCULATOR DEMO
# ============================================================================
print("\n" + "=" * 70)
print("BONUS: FBA PROFIT CALCULATOR")
print("=" * 70)

from profit_calculator import FBAProfitCalculator, ProductDimensions, quick_profit_check

# Example products to analyze
test_products = [
    {"name": "Wireless Earbuds", "price": 29.99, "cost": 8.00, "weight": 0.3, "category": "electronics_accessories"},
    {"name": "Kitchen Utensil Set", "price": 24.99, "cost": 6.50, "weight": 1.2, "category": "kitchen"},
    {"name": "Phone Case", "price": 14.99, "cost": 2.50, "weight": 0.1, "category": "cell_phone_devices"},
]

print("\nProduct Profitability Analysis:")
print("-" * 70)
print(f"{'Product':<25} {'Price':>8} {'Cost':>7} {'Fees':>7} {'Profit':>8} {'Margin':>8} {'Viable'}")
print("-" * 70)

for p in test_products:
    result = quick_profit_check(p['price'], p['cost'], p['weight'], p['category'])
    viable = "✓" if result['viable'] else "✗"
    print(f"{p['name']:<25} ${p['price']:>6.2f} ${p['cost']:>5.2f} ${result['total_fees']:>5.2f} "
          f"${result['profit']:>6.2f} {result['margin']:>6.1f}% {viable:>6}")

print("-" * 70)
print("\n✓ = Margin ≥ 20% (viable for FBA)")
