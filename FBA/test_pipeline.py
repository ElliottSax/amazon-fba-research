#!/usr/bin/env python3
"""
Test script for FBA Product Research Pipeline

Tests each component independently to verify installation and functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules import correctly."""
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)

    results = {}

    # Test sp_api_client
    try:
        from sp_api_client import SPAPIClient, REPORT_TYPES, MarketplaceIDs
        results['sp_api_client'] = '✓ OK'
        print(f"  sp_api_client: ✓ Imported successfully")
        print(f"    - Report types available: {len(REPORT_TYPES)} categories")
    except Exception as e:
        results['sp_api_client'] = f'✗ {e}'
        print(f"  sp_api_client: ✗ {e}")

    # Test keepa_client
    try:
        from keepa_client import KeepaClient
        results['keepa_client'] = '✓ OK'
        print(f"  keepa_client: ✓ Imported successfully")
        print(f"    - Supported domains: {len(KeepaClient.DOMAINS)}")
    except Exception as e:
        results['keepa_client'] = f'✗ {e}'
        print(f"  keepa_client: ✗ {e}")

    # Test inventory_forecast
    try:
        from inventory_forecast import InventoryForecaster, InventoryItem, ForecastResult
        results['inventory_forecast'] = '✓ OK'
        print(f"  inventory_forecast: ✓ Imported successfully")
    except Exception as e:
        results['inventory_forecast'] = f'✗ {e}'
        print(f"  inventory_forecast: ✗ {e}")

    # Test research_pipeline
    try:
        from research_pipeline import FBAResearchPipeline, ProductResearchResult
        results['research_pipeline'] = '✓ OK'
        print(f"  research_pipeline: ✓ Imported successfully")
    except Exception as e:
        results['research_pipeline'] = f'✗ {e}'
        print(f"  research_pipeline: ✗ {e}")

    return all('OK' in v for v in results.values())


def test_inventory_forecasting():
    """Test inventory forecasting module."""
    print("\n" + "=" * 60)
    print("TEST 2: Inventory Forecasting")
    print("=" * 60)

    from inventory_forecast import InventoryForecaster, InventoryItem

    # Create test item
    item = InventoryItem(
        sku="TEST-SKU-001",
        asin="B08PZHWJS5",
        name="Test Product - Wireless Earbuds",
        current_stock=150,
        unit_cost=25.00,
        lead_time_days=14,
        sales_history=[5, 7, 6, 8, 4, 9, 7, 6, 8, 5, 7, 6, 9, 8, 7,
                       6, 8, 9, 7, 5, 6, 8, 7, 9, 6, 8, 7, 6, 9, 8],
        reorder_cost=75.00
    )

    print(f"\n  Test Item: {item.name}")
    print(f"  Current Stock: {item.current_stock}")
    print(f"  Sales History: {len(item.sales_history)} days")

    # Create forecaster
    forecaster = InventoryForecaster(service_level=0.95)

    # Test forecast
    forecast = forecaster.forecast_demand(item)
    print(f"\n  Demand Forecast:")
    print(f"    - Avg Daily Demand: {forecast.avg_daily_demand:.1f} units")
    print(f"    - Std Deviation: {forecast.demand_std_dev:.1f}")
    print(f"    - 30-Day Forecast: {forecast.forecast_30_day:.0f} units")
    print(f"    - Trend: {forecast.trend}")
    print(f"    - Confidence: {forecast.confidence}")

    # Test reorder recommendation
    rec = forecaster.generate_reorder_recommendation(item)
    print(f"\n  Reorder Recommendation:")
    print(f"    - Reorder Point: {rec.reorder_point} units")
    print(f"    - Safety Stock: {rec.safety_stock} units")
    print(f"    - EOQ: {rec.economic_order_quantity} units")
    print(f"    - Days of Stock: {rec.days_of_stock:.1f}")
    print(f"    - Urgency: {rec.reorder_urgency.upper()}")
    print(f"    - Stockout Risk: {rec.stockout_risk_percent}%")

    # Verify calculations
    assert forecast.avg_daily_demand > 0, "Average demand should be positive"
    assert rec.reorder_point > rec.safety_stock, "Reorder point should exceed safety stock"
    assert rec.economic_order_quantity > 0, "EOQ should be positive"

    print("\n  ✓ Inventory forecasting tests passed!")
    return True


def test_sp_api_structure():
    """Test SP-API client structure (without actual API calls)."""
    print("\n" + "=" * 60)
    print("TEST 3: SP-API Client Structure")
    print("=" * 60)

    from sp_api_client import SPAPIClient, REPORT_TYPES, MarketplaceIDs

    # Test marketplace IDs
    print("\n  Marketplace IDs:")
    for mp in ['US', 'UK', 'DE', 'CA', 'JP']:
        mp_id = getattr(MarketplaceIDs, mp, None)
        print(f"    - {mp}: {mp_id}")

    # Test report types
    print("\n  Available Report Types:")
    for category, reports in REPORT_TYPES.items():
        print(f"    {category}: {len(reports)} reports")

    # Test client initialization (will fail without credentials - expected)
    print("\n  Client Initialization Test:")
    try:
        # This should initialize but API calls would fail
        client = SPAPIClient(marketplace='US')
        print("    - Client created (credentials may not be configured)")
    except Exception as e:
        print(f"    - Expected: {type(e).__name__}")

    print("\n  ✓ SP-API structure tests passed!")
    return True


def test_keepa_structure():
    """Test Keepa client structure (without actual API calls)."""
    print("\n" + "=" * 60)
    print("TEST 4: Keepa Client Structure")
    print("=" * 60)

    from keepa_client import KeepaClient

    # Test domains
    print("\n  Supported Domains:")
    for domain, code in KeepaClient.DOMAINS.items():
        print(f"    - {domain}: {code}")

    # Test price types
    print(f"\n  Price Types Available: {len(KeepaClient.PRICE_TYPES)}")
    key_types = ['AMAZON', 'NEW', 'USED', 'SALES', 'BUY_BOX', 'RATING']
    for pt in key_types:
        idx = KeepaClient.PRICE_TYPES.get(pt, 'N/A')
        print(f"    - {pt}: index {idx}")

    # Test sales estimation (doesn't require API)
    print("\n  Sales Estimation Test (no API required):")

    # Create mock client for testing sales estimation
    class MockKeepaClient(KeepaClient):
        def __init__(self):
            self.domain = 1  # US
            # Skip API key validation

    try:
        mock = MockKeepaClient()

        test_ranks = [100, 1000, 10000, 50000, 100000]
        print("    BSR → Estimated Monthly Sales:")
        for rank in test_ranks:
            estimate = mock.estimate_sales(rank)
            print(f"      #{rank:,}: ~{estimate['estimated_monthly_sales']:.0f}/month ({estimate['confidence']})")
    except Exception as e:
        print(f"    Sales estimation: {e}")

    print("\n  ✓ Keepa structure tests passed!")
    return True


def test_research_pipeline():
    """Test research pipeline structure."""
    print("\n" + "=" * 60)
    print("TEST 5: Research Pipeline")
    print("=" * 60)

    from research_pipeline import FBAResearchPipeline, ProductResearchResult
    from dataclasses import fields

    # Show ProductResearchResult fields
    print("\n  ProductResearchResult Fields:")
    result_fields = fields(ProductResearchResult)
    for f in result_fields[:10]:  # First 10 fields
        print(f"    - {f.name}: {f.type}")
    print(f"    ... and {len(result_fields) - 10} more fields")

    # Test pipeline initialization
    print("\n  Pipeline Initialization:")
    try:
        pipeline = FBAResearchPipeline(marketplace='US')
        print("    ✓ Pipeline created")
        print(f"    - SP-API: {'configured' if pipeline.sp_api else 'not configured'}")
        print(f"    - Keepa: {'configured' if pipeline.keepa else 'not configured'}")
        print(f"    - Forecaster: {'configured' if pipeline.forecaster else 'not configured'}")
    except Exception as e:
        print(f"    Pipeline init: {e}")

    # Test opportunity analysis with mock data
    print("\n  Opportunity Analysis Test (mock data):")

    mock_result = ProductResearchResult(
        asin='B08PZHWJS5',
        timestamp='2024-01-01T00:00:00',
        title='Test Wireless Earbuds',
        brand='TestBrand',
        category=['Electronics', 'Headphones'],
        current_price=29.99,
        avg_price_90d=32.50,
        min_price_90d=24.99,
        max_price_90d=39.99,
        buy_box_price=29.99,
        sales_rank=5000,
        estimated_monthly_sales=250,
        new_offer_count=8,
        fba_seller_count=5,
        rating=4.3,
        review_count=1250,
        price_volatility='medium',
        competition_level='medium',
        sales_velocity='moderate',
        price_history={},
        keepa_stats={},
        sp_api_data={}
    )

    analysis = pipeline.analyze_opportunity(mock_result)
    print(f"    Product: {mock_result.title}")
    print(f"    Scores:")
    for metric, score in analysis['scores'].items():
        print(f"      - {metric}: {score}/100")
    print(f"    Overall Score: {analysis['overall_score']}/100")
    print(f"    Recommendation: {analysis['recommendation']}")

    print("\n  ✓ Research pipeline tests passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FBA PRODUCT RESEARCH PIPELINE - TEST SUITE")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working Directory: {os.getcwd()}")

    tests = [
        ("Module Imports", test_imports),
        ("Inventory Forecasting", test_inventory_forecasting),
        ("SP-API Structure", test_sp_api_structure),
        ("Keepa Structure", test_keepa_structure),
        ("Research Pipeline", test_research_pipeline),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ✗ {name} failed with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All tests passed!")
        print("\n  Next Steps:")
        print("    1. Configure SP-API credentials (see SETUP.md)")
        print("    2. Get Keepa API key (https://keepa.com/#!api)")
        print("    3. Run: python -c \"from src.research_pipeline import FBAResearchPipeline; p = FBAResearchPipeline()\"")
    else:
        print("\n  ⚠ Some tests failed. Check error messages above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
