#!/usr/bin/env python3
"""
Comprehensive Integration Test for FBA Research Toolkit

Tests all modules work together correctly.
"""

import sys
sys.path.insert(0, 'src')

print("=" * 70)
print("FBA RESEARCH TOOLKIT - INTEGRATION TEST")
print("=" * 70)

# Track results
results = []

# Test 1: Market Analysis
print("\n[1/9] Market Analysis Module")
print("-" * 70)
try:
    from market_analysis import MarketAnalyzer, create_sample_competitors
    from datetime import datetime, timedelta

    analyzer = MarketAnalyzer()
    competitors = create_sample_competitors()
    metrics = analyzer.analyze_competitors(competitors)

    base_date = datetime(2024, 1, 1)
    historical = [{'date': base_date + timedelta(days=i), 'value': 100 + i * 1.5} for i in range(60)]
    trend = analyzer.analyze_trend(historical)

    monthly_data = {1: 80, 2: 85, 3: 90, 4: 95, 5: 100, 6: 105, 7: 110, 8: 100, 9: 95, 10: 120, 11: 150, 12: 170}
    seasonality = analyzer.analyze_seasonality(monthly_data)

    score = analyzer.calculate_niche_score(metrics, trend)

    print(f"  ✓ Competitors analyzed: {metrics.competitor_count}")
    print(f"  ✓ Trend: {trend.direction.value}")
    print(f"  ✓ Niche score: {score.overall_score}/100")
    results.append(("Market Analysis", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results.append(("Market Analysis", False))

# Test 2: Keyword Research
print("\n[2/9] Keyword Research Module")
print("-" * 70)
try:
    from keyword_research import KeywordResearcher

    researcher = KeywordResearcher()
    keywords = researcher.generate_long_tail_keywords("yoga mat", max_keywords=20)
    result = researcher.research_keyword("yoga mat", base_volume=25000, max_keywords=30)
    backend = researcher.suggest_backend_keywords(result.keywords)

    print(f"  ✓ Long-tail keywords: {len(keywords)}")
    print(f"  ✓ Research result: {len(result.keywords)} keywords")
    print(f"  ✓ Backend keywords: {len(backend.encode())} bytes")
    results.append(("Keyword Research", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results.append(("Keyword Research", False))

# Test 3: PPC Analyzer
print("\n[3/9] PPC Analyzer Module")
print("-" * 70)
try:
    from ppc_analyzer import PPCAnalyzer

    ppc = PPCAnalyzer()
    break_even = ppc.calculate_break_even_acos(
        product_price=29.99, product_cost=8.0, fba_fees=5.50
    )
    keywords = ["wireless earbuds", "bluetooth earbuds", "sport earbuds"]
    strategy = ppc.generate_ppc_strategy(
        product_price=29.99, product_cost=8.0, fba_fees=5.50,
        keywords=keywords, competition_level="medium"
    )
    projection = ppc.project_organic_growth(
        current_bsr=50000, ad_sales_per_month=200, months=6
    )

    print(f"  ✓ Break-even ACOS: {break_even:.1f}%")
    print(f"  ✓ Strategy generated: {strategy.target_acos}% target ACOS")
    print(f"  ✓ 6-month projection: {len(projection)} months")
    results.append(("PPC Analyzer", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results.append(("PPC Analyzer", False))

# Test 4: Review Analyzer
print("\n[4/9] Review Analyzer Module")
print("-" * 70)
try:
    from review_analyzer import ReviewAnalyzer, create_sample_reviews

    analyzer = ReviewAnalyzer()
    reviews = create_sample_reviews()
    result = analyzer.analyze_reviews("B0TEST123", reviews)

    print(f"  ✓ Reviews analyzed: {result.total_reviews}")
    print(f"  ✓ Sentiment distribution: {len(result.sentiment_distribution)} categories")
    print(f"  ✓ Positive themes: {len(result.top_positive_themes)}")
    print(f"  ✓ Quality issues: {len(result.quality_issues)}")
    print(f"  ✓ Marketing angles: {len(result.marketing_angles)}")
    results.append(("Review Analyzer", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results.append(("Review Analyzer", False))

# Test 5: Competitor Monitor
print("\n[5/9] Competitor Monitor Module")
print("-" * 70)
try:
    from competitor_monitor import CompetitorMonitor, create_sample_data

    monitor = CompetitorMonitor()
    snapshots = create_sample_data()

    alerts = []
    for snap in sorted(snapshots, key=lambda x: x.timestamp):
        new_alerts = monitor.record_snapshot(snap)
        alerts.extend(new_alerts)

    position = monitor.calculate_market_position("B0YOURS01")
    threats = monitor.get_threat_summary()

    print(f"  ✓ Snapshots processed: {len(snapshots)}")
    print(f"  ✓ Alerts generated: {len(alerts)}")
    print(f"  ✓ Profiles created: {len(monitor.profiles)}")
    print(f"  ✓ Market position: #{position.bsr_rank if position else 'N/A'} BSR")
    results.append(("Competitor Monitor", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results.append(("Competitor Monitor", False))

# Test 6: Launch Strategy
print("\n[6/9] Launch Strategy Module")
print("-" * 70)
try:
    from launch_strategy import LaunchStrategyGenerator, create_sample_input

    generator = LaunchStrategyGenerator()
    product, market = create_sample_input()
    strategy = generator.generate_strategy(product, market)

    print(f"  ✓ Launch type: {strategy.launch_type}")
    print(f"  ✓ Success probability: {strategy.estimated_success_probability*100:.0f}%")
    print(f"  ✓ 6-month revenue: ${strategy.projected_revenue_6mo:,.0f}")
    print(f"  ✓ 6-month profit: ${strategy.projected_profit_6mo:,.0f}")
    print(f"  ✓ Break-even: Month {strategy.break_even_month}")
    print(f"  ✓ Milestones: {len(strategy.milestones)}")
    print(f"  ✓ Risks identified: {len(strategy.risks)}")
    results.append(("Launch Strategy", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results.append(("Launch Strategy", False))

# Test 7: Listing Optimizer
print("\n[7/9] Listing Optimizer Module")
print("-" * 70)
try:
    from listing_optimizer import ListingOptimizer, create_sample_listing

    optimizer = ListingOptimizer()
    listing = create_sample_listing()
    keywords = ["wireless earbuds", "bluetooth earbuds", "sport earbuds"]

    result = optimizer.analyze_listing(listing, keywords)

    print(f"  ✓ Overall score: {result.overall_score}/100 (Grade: {result.grade})")
    print(f"  ✓ Section scores: {len(result.section_scores)} sections")
    print(f"  ✓ Issues found: {len(result.all_issues)}")
    print(f"  ✓ Keyword analysis: {len(result.keyword_analysis)} keywords")
    results.append(("Listing Optimizer", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results.append(("Listing Optimizer", False))

# Test 8: Sales Analytics
print("\n[8/9] Sales Analytics Module")
print("-" * 70)
try:
    from sales_analytics import SalesAnalyzer, create_sample_sales_data

    sales_data, costs = create_sample_sales_data()
    analyzer = SalesAnalyzer()
    analyzer.load_sales_data(sales_data, costs)

    report = analyzer.generate_report()
    perf = analyzer.analyze_product_performance("B0TEST", "Test Product", 200)
    econ = analyzer.calculate_unit_economics(29.99)

    print(f"  ✓ Report generated: {report.total_days} days")
    print(f"  ✓ Total revenue: ${report.total_revenue:,.2f}")
    print(f"  ✓ Product performance: {perf.velocity_trend} trend")
    print(f"  ✓ Unit economics: {econ['profit']['margin_pct']}% margin")
    results.append(("Sales Analytics", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results.append(("Sales Analytics", False))

# Test 9: Supplier Sourcing
print("\n[9/9] Supplier Sourcing Module")
print("-" * 70)
try:
    from supplier_sourcing import SupplierAnalyzer, create_sample_suppliers, ShippingMethod

    analyzer = SupplierAnalyzer(category="electronics")
    suppliers = create_sample_suppliers()

    comparison = analyzer.compare_suppliers(
        suppliers=suppliers,
        quantity=1000,
        shipping_method=ShippingMethod.SEA_FREIGHT,
        product_weight_kg=0.15,
        product_volume_cbm=0.001,
        target_landed_cost=10.00
    )

    print(f"  ✓ Suppliers compared: {len(suppliers)}")
    print(f"  ✓ Best price: {comparison.best_price}")
    print(f"  ✓ Best overall: {comparison.best_overall}")
    print(f"  ✓ Landed costs calculated: {len(comparison.landed_costs)}")
    results.append(("Supplier Sourcing", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    results.append(("Supplier Sourcing", False))

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

passed = sum(1 for _, ok in results if ok)
total = len(results)

for name, ok in results:
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  [{status}] {name}")

print(f"\nResult: {passed}/{total} modules passed")

if passed == total:
    print("\n✓ ALL TESTS PASSED!")
else:
    print(f"\n✗ {total - passed} tests failed")
    sys.exit(1)

print("=" * 70)
