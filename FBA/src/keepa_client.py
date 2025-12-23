"""
Keepa API Client - Price History & Product Tracking

Provides historical pricing data, sales rank history, and deal detection.
Requires Keepa API subscription: https://keepa.com/#!api
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
import json

try:
    import keepa
except ImportError:
    raise ImportError(
        "keepa library required. Install with:\n"
        "pip install keepa"
    )

try:
    import numpy as np
except ImportError:
    np = None


class KeepaClient:
    """
    Keepa API client for Amazon price history and product tracking.

    Keepa provides:
    - Historical price data (Amazon, 3rd party, used)
    - Sales rank history
    - Buy Box statistics
    - Deal and price drop detection
    - Product availability tracking
    """

    # Keepa domain codes
    DOMAINS = {
        'US': 1,   # amazon.com
        'UK': 2,   # amazon.co.uk
        'DE': 3,   # amazon.de
        'FR': 4,   # amazon.fr
        'JP': 5,   # amazon.co.jp
        'CA': 6,   # amazon.ca
        'IT': 8,   # amazon.it
        'ES': 9,   # amazon.es
        'IN': 10,  # amazon.in
        'MX': 11,  # amazon.com.mx
        'AU': 13,  # amazon.com.au
    }

    # Price type indices in Keepa data
    PRICE_TYPES = {
        'AMAZON': 0,           # Amazon price
        'NEW': 1,              # Lowest 3rd party new price
        'USED': 2,             # Lowest 3rd party used price
        'SALES': 3,            # Sales rank
        'LISTPRICE': 4,        # List price
        'COLLECTIBLE': 5,      # Collectible price
        'REFURBISHED': 6,      # Refurbished price
        'NEW_FBM': 7,          # New FBM shipping price
        'LIGHTNING_DEAL': 8,   # Lightning deal price
        'WAREHOUSE': 9,        # Amazon Warehouse price
        'NEW_FBA': 10,         # New price FBA
        'COUNT_NEW': 11,       # New offer count
        'COUNT_USED': 12,      # Used offer count
        'COUNT_REFURBISHED': 13,
        'COUNT_COLLECTIBLE': 14,
        'EXTRA_INFO_UPDATES': 15,
        'RATING': 16,          # Product rating
        'COUNT_REVIEWS': 17,   # Review count
        'BUY_BOX': 18,         # Buy Box price
        'USED_NEW_SHIPPING': 19,
        'USED_VERY_GOOD': 20,
        'USED_GOOD': 21,
        'USED_ACCEPTABLE': 22,
        'COLLECTIBLE_NEW': 23,
        'COLLECTIBLE_VERY_GOOD': 24,
        'COLLECTIBLE_GOOD': 25,
        'COLLECTIBLE_ACCEPTABLE': 26,
        'REFURBISHED_SHIPPING': 27,
        'TRADE_IN': 30,        # Trade-in price
        'RENTAL': 31,          # Rental price
    }

    def __init__(self, api_key: Optional[str] = None, domain: str = 'US'):
        """
        Initialize Keepa client.

        Args:
            api_key: Keepa API key (or set KEEPA_API_KEY env var)
            domain: Amazon domain (US, UK, DE, etc.)
        """
        self.api_key = api_key or os.environ.get('KEEPA_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Keepa API key required. Set KEEPA_API_KEY env var or pass api_key."
            )

        self.domain = self.DOMAINS.get(domain.upper(), 1)
        self.api = keepa.Keepa(self.api_key)

    def _keepa_time_to_datetime(self, keepa_minutes: int) -> datetime:
        """Convert Keepa time (minutes since 2011-01-01) to datetime."""
        base = datetime(2011, 1, 1)
        return base + timedelta(minutes=keepa_minutes)

    def _price_to_dollars(self, price: int) -> Optional[float]:
        """Convert Keepa price (cents) to dollars. -1 means unavailable."""
        if price < 0:
            return None
        return price / 100.0

    def query_products(
        self,
        asins: Union[str, List[str]],
        stats: int = 180,
        history: bool = True,
        offers: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Query product data for one or more ASINs.

        Args:
            asins: Single ASIN or list of ASINs (max 100)
            stats: Days for statistics calculation (default 180)
            history: Include price history
            offers: Number of offers to retrieve (0 = none)

        Returns:
            List of product data dictionaries
        """
        if isinstance(asins, str):
            asins = [asins]

        products = self.api.query(
            asins,
            domain=self.domain,
            stats=stats,
            history=history,
            offers=offers
        )

        return products

    def get_price_history(
        self,
        asin: str,
        price_types: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Get price history for a product.

        Args:
            asin: Product ASIN
            price_types: Price types to retrieve (default: AMAZON, NEW, SALES)

        Returns:
            Dictionary with price type -> list of {date, price} records
        """
        if price_types is None:
            price_types = ['AMAZON', 'NEW', 'SALES', 'BUY_BOX']

        products = self.query_products(asin)
        if not products:
            return {}

        product = products[0]
        result = {}

        for price_type in price_types:
            type_idx = self.PRICE_TYPES.get(price_type.upper())
            if type_idx is None:
                continue

            csv_data = product.get('csv', [])
            if type_idx >= len(csv_data) or csv_data[type_idx] is None:
                continue

            history_data = csv_data[type_idx]
            records = []

            # Keepa stores [time, value, time, value, ...]
            for i in range(0, len(history_data) - 1, 2):
                time_val = history_data[i]
                price_val = history_data[i + 1]

                if price_type == 'SALES':
                    # Sales rank is stored directly
                    value = price_val if price_val > 0 else None
                else:
                    value = self._price_to_dollars(price_val)

                if value is not None:
                    records.append({
                        'date': self._keepa_time_to_datetime(time_val),
                        'value': value
                    })

            result[price_type] = records

        return result

    def get_product_stats(self, asin: str, days: int = 90) -> Dict[str, Any]:
        """
        Get statistical summary for a product.

        Args:
            asin: Product ASIN
            days: Days for statistics calculation

        Returns:
            Product statistics including avg price, sales rank, etc.
        """
        products = self.query_products(asin, stats=days)
        if not products:
            return {}

        product = products[0]
        stats = product.get('stats', {})

        return {
            'asin': asin,
            'title': product.get('title', ''),
            'brand': product.get('brand', ''),
            'product_group': product.get('productGroup', ''),
            'category': product.get('categoryTree', []),
            'stats_period_days': days,

            # Current prices
            'current': {
                'amazon_price': self._price_to_dollars(stats.get('current', [None] * 19)[0] or -1),
                'new_price': self._price_to_dollars(stats.get('current', [None] * 19)[1] or -1),
                'buy_box_price': self._price_to_dollars(stats.get('current', [None] * 19)[18] or -1),
                'sales_rank': stats.get('current', [None] * 19)[3],
                'review_count': stats.get('current', [None] * 19)[17],
                'rating': (stats.get('current', [None] * 19)[16] or 0) / 10.0,
            },

            # Average prices over period
            'avg': {
                'amazon_price': self._price_to_dollars(stats.get('avg', [None] * 19)[0] or -1),
                'new_price': self._price_to_dollars(stats.get('avg', [None] * 19)[1] or -1),
                'buy_box_price': self._price_to_dollars(stats.get('avg', [None] * 19)[18] or -1),
                'sales_rank': stats.get('avg', [None] * 19)[3],
            },

            # Price ranges
            'min': {
                'amazon_price': self._price_to_dollars(stats.get('min', [None] * 19)[0] or -1),
                'new_price': self._price_to_dollars(stats.get('min', [None] * 19)[1] or -1),
                'sales_rank': stats.get('min', [None] * 19)[3],
            },
            'max': {
                'amazon_price': self._price_to_dollars(stats.get('max', [None] * 19)[0] or -1),
                'new_price': self._price_to_dollars(stats.get('max', [None] * 19)[1] or -1),
                'sales_rank': stats.get('max', [None] * 19)[3],
            },

            # Offer counts
            'offer_counts': {
                'new': stats.get('current', [None] * 19)[11],
                'used': stats.get('current', [None] * 19)[12],
            },

            # Availability
            'amazon_in_stock': product.get('availabilityAmazon', -1) == 0,
            'fba_fees': product.get('fbaFees', {}),
        }

    def find_deals(
        self,
        category_id: Optional[int] = None,
        price_drop_percent: int = 20,
        min_rating: float = 4.0,
        min_reviews: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Find products with recent price drops.

        Args:
            category_id: Amazon category ID to search
            price_drop_percent: Minimum price drop percentage
            min_rating: Minimum product rating
            min_reviews: Minimum review count

        Returns:
            List of deal products
        """
        deals = self.api.deals(
            deal_parms={
                'page': 0,
                'domainId': self.domain,
                'excludeCategories': [],
                'includeCategories': [category_id] if category_id else [],
                'priceTypes': [0],  # Amazon price
                'deltaRange': [price_drop_percent, 100],
                'deltaPercentRange': [price_drop_percent, 100],
                'salesRankRange': [1, 100000],
                'currentRange': [100, 1000000],  # $1 - $10,000
                'minRating': int(min_rating * 10),
            }
        )

        return deals.get('deals', [])

    def search_products(
        self,
        search_term: str,
        product_type: int = 0,  # 0 = all
        sort_by: int = 0,  # 0 = best match
        min_price: int = 0,
        max_price: int = -1
    ) -> List[Dict[str, Any]]:
        """
        Search for products by keyword.

        Args:
            search_term: Search query
            product_type: Product type filter
            sort_by: Sort order (0=best, 1=price asc, 2=price desc)
            min_price: Minimum price in cents
            max_price: Maximum price in cents (-1 = no limit)

        Returns:
            List of matching products
        """
        results = self.api.search(
            search_term,
            domain=self.domain,
            product_type=product_type,
            sort_by=sort_by,
            current_range=[min_price, max_price] if max_price > 0 else None
        )

        return results

    def get_best_sellers(
        self,
        category_id: int,
        range_type: int = 0  # 0 = current, 1 = 30 day avg, 2 = 90 day avg
    ) -> List[str]:
        """
        Get best sellers for a category.

        Args:
            category_id: Amazon category ID
            range_type: Time range for ranking

        Returns:
            List of ASINs
        """
        return self.api.best_sellers(
            category=category_id,
            domain=self.domain,
            range_type=range_type
        )

    def estimate_sales(
        self,
        sales_rank: int,
        category: str = 'all'
    ) -> Dict[str, Any]:
        """
        Estimate monthly sales from sales rank.

        Based on general BSR-to-sales correlations.
        Note: These are rough estimates. Actual sales vary by category.

        Args:
            sales_rank: Amazon Best Sellers Rank
            category: Product category for calibration

        Returns:
            Estimated sales data
        """
        # General BSR to sales estimation (US marketplace)
        # These are approximate - real values vary significantly by category
        if sales_rank <= 100:
            estimated_daily = 100 - (sales_rank * 0.5)
        elif sales_rank <= 1000:
            estimated_daily = 50 - (sales_rank - 100) * 0.04
        elif sales_rank <= 10000:
            estimated_daily = 15 - (sales_rank - 1000) * 0.0012
        elif sales_rank <= 100000:
            estimated_daily = 5 - (sales_rank - 10000) * 0.00004
        else:
            estimated_daily = max(0.1, 1 - (sales_rank - 100000) * 0.000001)

        estimated_daily = max(0.1, estimated_daily)

        return {
            'sales_rank': sales_rank,
            'estimated_daily_sales': round(estimated_daily, 1),
            'estimated_monthly_sales': round(estimated_daily * 30, 0),
            'confidence': 'low' if sales_rank > 50000 else 'medium' if sales_rank > 5000 else 'high',
            'note': 'Estimates vary significantly by category. Use Keepa/JungleScout for accurate data.'
        }

    def plot_product(self, asin: str, output_path: Optional[str] = None):
        """
        Generate price history plot for a product.

        Args:
            asin: Product ASIN
            output_path: Optional path to save plot
        """
        products = self.query_products(asin)
        if not products:
            print(f"No data found for ASIN: {asin}")
            return

        try:
            keepa.plot_product(products[0], output_path)
        except Exception as e:
            print(f"Plot error: {e}")
            print("Ensure matplotlib is installed: pip install matplotlib")


if __name__ == "__main__":
    print("Keepa Client - Price History & Product Tracking")
    print("\nAvailable price types:")
    for name, idx in KeepaClient.PRICE_TYPES.items():
        print(f"  {idx:2d}: {name}")
    print("\nSupported domains:")
    for domain, code in KeepaClient.DOMAINS.items():
        print(f"  {domain}: {code}")
