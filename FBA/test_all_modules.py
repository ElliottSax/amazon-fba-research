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
print("\n[1/6] Market Analysis Module")
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
print("\n[2/6] Keyword Research Module")
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
print("\n[3/6] PPC Analyzer Module")
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
print("\n[4/6] Review Analyzer Module")
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
print("\n[5/6] Competitor Monitor Module")
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
print("\n[6/6] Launch Strategy Module")
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
