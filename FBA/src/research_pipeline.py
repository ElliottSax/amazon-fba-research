"""
FBA Product Research Pipeline - Compliant Stack

Combines:
- SP-API (Official Amazon API)
- Reports API (Free bulk data)
- Keepa (Price history)
- Inventory Forecasting

Usage:
    pipeline = FBAResearchPipeline()
    results = pipeline.research_product('B08PZHWJS5')
    pipeline.export_report(results, 'report.json')
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass, asdict

# Local imports
from sp_api_client import SPAPIClient, REPORT_TYPES
from keepa_client import KeepaClient
from inventory_forecast import InventoryForecaster, InventoryItem, ForecastResult


@dataclass
class ProductResearchResult:
    """Complete product research result."""
    asin: str
    timestamp: str

    # Basic info
    title: str
    brand: str
    category: List[str]

    # Pricing
    current_price: Optional[float]
    avg_price_90d: Optional[float]
    min_price_90d: Optional[float]
    max_price_90d: Optional[float]
    buy_box_price: Optional[float]

    # Competition
    sales_rank: Optional[int]
    estimated_monthly_sales: int
    new_offer_count: int
    fba_seller_count: int

    # Reviews
    rating: float
    review_count: int

    # Profitability indicators
    price_volatility: str  # 'low', 'medium', 'high'
    competition_level: str  # 'low', 'medium', 'high'
    sales_velocity: str  # 'slow', 'moderate', 'fast'

    # Raw data
    price_history: Dict[str, Any]
    keepa_stats: Dict[str, Any]
    sp_api_data: Dict[str, Any]


class FBAResearchPipeline:
    """
    Complete FBA product research pipeline.

    Combines multiple data sources for comprehensive product analysis.
    """

    def __init__(
        self,
        marketplace: str = 'US',
        sp_api_credentials: Optional[str] = None,
        keepa_api_key: Optional[str] = None
    ):
        """
        Initialize research pipeline.

        Args:
            marketplace: Amazon marketplace (US, UK, DE, etc.)
            sp_api_credentials: Path to SP-API credentials
            keepa_api_key: Keepa API key
        """
        self.marketplace = marketplace

        # Initialize clients (may fail if not configured)
        self.sp_api = None
        self.keepa = None

        try:
            self.sp_api = SPAPIClient(
                marketplace=marketplace,
                credentials_path=sp_api_credentials
            )
            print("✓ SP-API client initialized")
        except Exception as e:
            print(f"⚠ SP-API not configured: {e}")

        try:
            self.keepa = KeepaClient(
                api_key=keepa_api_key,
                domain=marketplace
            )
            print("✓ Keepa client initialized")
        except Exception as e:
            print(f"⚠ Keepa not configured: {e}")

        self.forecaster = InventoryForecaster(service_level=0.95)
        print("✓ Forecaster initialized")

    def research_product(self, asin: str) -> ProductResearchResult:
        """
        Comprehensive product research.

        Args:
            asin: Amazon Standard Identification Number

        Returns:
            ProductResearchResult with all analysis
        """
        print(f"\nResearching ASIN: {asin}")
        print("=" * 50)

        # Initialize result containers
        sp_api_data = {}
        keepa_stats = {}
        price_history = {}

        # 1. Get SP-API catalog data (if available)
        if self.sp_api:
            print("Fetching SP-API catalog data...")
            try:
                sp_api_data = self.sp_api.get_catalog_item(asin)
            except Exception as e:
                print(f"  SP-API error: {e}")

        # 2. Get Keepa price history and stats
        if self.keepa:
            print("Fetching Keepa price history...")
            try:
                keepa_stats = self.keepa.get_product_stats(asin, days=90)
                price_history = self.keepa.get_price_history(asin)
            except Exception as e:
                print(f"  Keepa error: {e}")

        # 3. Extract and combine data
        result = self._build_result(asin, sp_api_data, keepa_stats, price_history)

        print(f"\nResearch complete for: {result.title or asin}")
        return result

    def _build_result(
        self,
        asin: str,
        sp_api_data: Dict,
        keepa_stats: Dict,
        price_history: Dict
    ) -> ProductResearchResult:
        """Build research result from collected data."""

        # Extract from SP-API
        summaries = sp_api_data.get('summaries', [{}])
        summary = summaries[0] if summaries else {}

        # Extract from Keepa
        current = keepa_stats.get('current', {})
        avg = keepa_stats.get('avg', {})
        min_vals = keepa_stats.get('min', {})
        max_vals = keepa_stats.get('max', {})

        # Get title (prefer SP-API, fallback to Keepa)
        title = summary.get('itemName', keepa_stats.get('title', ''))

        # Get brand
        brand = summary.get('brand', keepa_stats.get('brand', ''))

        # Get category
        category = keepa_stats.get('category', [])
        if not category and 'classifications' in sp_api_data:
            category = [c.get('displayName', '') for c in sp_api_data.get('classifications', [])]

        # Pricing
        current_price = current.get('amazon_price') or current.get('buy_box_price')
        avg_price = avg.get('amazon_price') or avg.get('buy_box_price')
        min_price = min_vals.get('amazon_price')
        max_price = max_vals.get('amazon_price')
        buy_box = current.get('buy_box_price')

        # Competition
        sales_rank = current.get('sales_rank')
        offer_counts = keepa_stats.get('offer_counts', {})
        new_offers = offer_counts.get('new', 0) or 0

        # Estimate sales
        if sales_rank:
            sales_estimate = self.keepa.estimate_sales(sales_rank) if self.keepa else {}
            monthly_sales = int(sales_estimate.get('estimated_monthly_sales', 0))
        else:
            monthly_sales = 0

        # Reviews
        rating = current.get('rating', 0) or 0
        review_count = current.get('review_count', 0) or 0

        # Calculate indicators
        price_volatility = self._calc_price_volatility(min_price, max_price, avg_price)
        competition = self._calc_competition_level(new_offers, sales_rank)
        velocity = self._calc_sales_velocity(monthly_sales)

        return ProductResearchResult(
            asin=asin,
            timestamp=datetime.utcnow().isoformat(),
            title=title,
            brand=brand,
            category=category if isinstance(category, list) else [category],
            current_price=current_price,
            avg_price_90d=avg_price,
            min_price_90d=min_price,
            max_price_90d=max_price,
            buy_box_price=buy_box,
            sales_rank=sales_rank,
            estimated_monthly_sales=monthly_sales,
            new_offer_count=new_offers,
            fba_seller_count=0,  # Would need offers API
            rating=rating,
            review_count=review_count,
            price_volatility=price_volatility,
            competition_level=competition,
            sales_velocity=velocity,
            price_history=price_history,
            keepa_stats=keepa_stats,
            sp_api_data=sp_api_data
        )

    def _calc_price_volatility(
        self,
        min_price: Optional[float],
        max_price: Optional[float],
        avg_price: Optional[float]
    ) -> str:
        """Calculate price volatility indicator."""
        if not all([min_price, max_price, avg_price]) or avg_price == 0:
            return 'unknown'

        range_pct = (max_price - min_price) / avg_price * 100

        if range_pct > 30:
            return 'high'
        elif range_pct > 15:
            return 'medium'
        return 'low'

    def _calc_competition_level(
        self,
        offer_count: int,
        sales_rank: Optional[int]
    ) -> str:
        """Calculate competition level indicator."""
        if offer_count > 20:
            return 'high'
        elif offer_count > 5:
            return 'medium'
        return 'low'

    def _calc_sales_velocity(self, monthly_sales: int) -> str:
        """Calculate sales velocity indicator."""
        if monthly_sales > 300:
            return 'fast'
        elif monthly_sales > 100:
            return 'moderate'
        return 'slow'

    def research_batch(
        self,
        asins: List[str],
        delay_seconds: float = 1.0
    ) -> List[ProductResearchResult]:
        """
        Research multiple products.

        Args:
            asins: List of ASINs to research
            delay_seconds: Delay between requests (rate limiting)

        Returns:
            List of research results
        """
        import time

        results = []
        total = len(asins)

        for i, asin in enumerate(asins, 1):
            print(f"\n[{i}/{total}] Processing {asin}")
            try:
                result = self.research_product(asin)
                results.append(result)
            except Exception as e:
                print(f"Error researching {asin}: {e}")

            if i < total:
                time.sleep(delay_seconds)

        return results

    def get_my_listings(self, output_path: Optional[str] = None) -> Any:
        """
        Get your merchant listings via Reports API (FREE).

        This is the recommended way to get your catalog data.
        """
        if not self.sp_api:
            print("SP-API not configured")
            return None

        print("Requesting merchant listings report (this may take a few minutes)...")
        return self.sp_api.get_merchant_listings(output_path)

    def get_sales_report(
        self,
        days: int = 30,
        output_path: Optional[str] = None
    ) -> Any:
        """
        Get sales and traffic report via Reports API (FREE).
        """
        if not self.sp_api:
            print("SP-API not configured")
            return None

        print(f"Requesting sales report for last {days} days...")
        return self.sp_api.get_sales_traffic_report(days, output_path)

    def analyze_opportunity(self, result: ProductResearchResult) -> Dict[str, Any]:
        """
        Analyze product opportunity based on research.

        Args:
            result: Product research result

        Returns:
            Opportunity analysis with scores and recommendations
        """
        scores = {}

        # Demand score (based on sales velocity)
        if result.sales_velocity == 'fast':
            scores['demand'] = 90
        elif result.sales_velocity == 'moderate':
            scores['demand'] = 60
        else:
            scores['demand'] = 30

        # Competition score (lower is better)
        if result.competition_level == 'low':
            scores['competition'] = 90
        elif result.competition_level == 'medium':
            scores['competition'] = 60
        else:
            scores['competition'] = 30

        # Price stability score
        if result.price_volatility == 'low':
            scores['price_stability'] = 90
        elif result.price_volatility == 'medium':
            scores['price_stability'] = 60
        else:
            scores['price_stability'] = 30

        # Review quality score
        if result.rating >= 4.5 and result.review_count >= 100:
            scores['review_quality'] = 90
        elif result.rating >= 4.0 and result.review_count >= 50:
            scores['review_quality'] = 70
        elif result.rating >= 3.5:
            scores['review_quality'] = 50
        else:
            scores['review_quality'] = 30

        # Overall opportunity score
        overall = (
            scores['demand'] * 0.35 +
            scores['competition'] * 0.25 +
            scores['price_stability'] * 0.20 +
            scores['review_quality'] * 0.20
        )

        # Recommendation
        if overall >= 75:
            recommendation = 'STRONG_OPPORTUNITY'
        elif overall >= 55:
            recommendation = 'MODERATE_OPPORTUNITY'
        elif overall >= 40:
            recommendation = 'WEAK_OPPORTUNITY'
        else:
            recommendation = 'NOT_RECOMMENDED'

        return {
            'asin': result.asin,
            'title': result.title,
            'scores': scores,
            'overall_score': round(overall, 1),
            'recommendation': recommendation,
            'key_metrics': {
                'monthly_sales': result.estimated_monthly_sales,
                'price': result.current_price,
                'competitors': result.new_offer_count,
                'rating': result.rating,
                'reviews': result.review_count,
            }
        }

    def export_report(
        self,
        results: List[ProductResearchResult],
        output_path: str,
        include_analysis: bool = True
    ):
        """
        Export research results to JSON.

        Args:
            results: Research results to export
            output_path: Output file path
            include_analysis: Include opportunity analysis
        """
        if isinstance(results, ProductResearchResult):
            results = [results]

        export_data = {
            'generated_at': datetime.utcnow().isoformat(),
            'marketplace': self.marketplace,
            'product_count': len(results),
            'products': []
        }

        for result in results:
            product_data = asdict(result)

            if include_analysis:
                product_data['opportunity_analysis'] = self.analyze_opportunity(result)

            export_data['products'].append(product_data)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"Report exported to: {output_path}")


def main():
    """Example usage of the research pipeline."""
    print("FBA Product Research Pipeline")
    print("=" * 60)
    print("\nCompliant Stack:")
    print("  • SP-API (Official Amazon API)")
    print("  • Reports API (Free bulk data)")
    print("  • Keepa (Price history)")
    print("  • Inventory Forecasting")

    print("\n" + "=" * 60)
    print("SETUP REQUIRED:")
    print("=" * 60)

    print("""
1. SP-API Setup:
   - Register as Amazon Developer: https://developer.amazonservices.com/
   - Create App in Seller Central: Apps & Services > Develop Apps
   - Save credentials to ~/.sp-api/credentials.yml

2. Keepa Setup:
   - Subscribe to Keepa API: https://keepa.com/#!api
   - Set KEEPA_API_KEY environment variable

3. Install dependencies:
   pip install python-amazon-sp-api keepa

Example usage:

    from research_pipeline import FBAResearchPipeline

    # Initialize
    pipeline = FBAResearchPipeline(marketplace='US')

    # Research a product
    result = pipeline.research_product('B08PZHWJS5')

    # Analyze opportunity
    analysis = pipeline.analyze_opportunity(result)
    print(f"Score: {analysis['overall_score']}")
    print(f"Recommendation: {analysis['recommendation']}")

    # Get your listings (FREE via Reports API)
    listings = pipeline.get_my_listings('my_listings.csv')

    # Batch research
    results = pipeline.research_batch(['B08PZHWJS5', 'B09V3KXJPB'])
    pipeline.export_report(results, 'research_report.json')
""")


if __name__ == "__main__":
    main()
