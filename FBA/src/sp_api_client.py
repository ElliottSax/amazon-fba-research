"""
Amazon SP-API Client - Compliant Product Research

Uses python-amazon-sp-api for official API access.
Prioritizes Reports API (free) over GET requests (paid).
"""

import os
import json
import gzip
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    from sp_api.api import Catalog, Reports, Orders, Products
    from sp_api.base import Marketplaces, ReportType, SellingApiException
    from sp_api.base.reportTypes import ReportType
except ImportError:
    raise ImportError(
        "python-amazon-sp-api required. Install with:\n"
        "pip install python-amazon-sp-api"
    )

import requests


class MarketplaceIDs:
    """Amazon Marketplace identifiers."""
    US = "ATVPDKIKX0DER"
    CA = "A2EUQ1WTGCTBG2"
    MX = "A1AM78C64UM0Y8"
    UK = "A1F83G8C2ARO7P"
    DE = "A1PA6795UKMFR9"
    FR = "A13V1IB3VIYBER"
    IT = "APJ6JRA9NG5V4"
    ES = "A1RKKUPIHCS9HS"
    JP = "A1VC38T7YXB528"
    AU = "A39IBJ37TRP1C6"


class SPAPIClient:
    """
    Compliant Amazon SP-API client for product research.

    Prioritizes Reports API (free bulk data) over individual GET requests.
    """

    def __init__(
        self,
        marketplace: str = "US",
        credentials_path: Optional[str] = None
    ):
        """
        Initialize SP-API client.

        Args:
            marketplace: Marketplace code (US, UK, DE, etc.)
            credentials_path: Path to credentials.yml (default: ~/.sp-api/credentials.yml)
        """
        self.marketplace = self._get_marketplace(marketplace)
        self.marketplace_id = getattr(MarketplaceIDs, marketplace.upper(), MarketplaceIDs.US)

        if credentials_path:
            os.environ['SP_API_CREDENTIALS_FILE'] = credentials_path

    def _get_marketplace(self, code: str) -> Marketplaces:
        """Convert marketplace code to Marketplaces enum."""
        mapping = {
            "US": Marketplaces.US,
            "CA": Marketplaces.CA,
            "MX": Marketplaces.MX,
            "UK": Marketplaces.UK,
            "DE": Marketplaces.DE,
            "FR": Marketplaces.FR,
            "IT": Marketplaces.IT,
            "ES": Marketplaces.ES,
            "JP": Marketplaces.JP,
            "AU": Marketplaces.AU,
        }
        return mapping.get(code.upper(), Marketplaces.US)

    # =========================================================================
    # CATALOG API - Product Research (Paid per request)
    # =========================================================================

    def search_catalog(
        self,
        keywords: str,
        included_data: Optional[List[str]] = None,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        Search Amazon catalog by keywords.

        NOTE: This uses GET requests which incur SP-API usage fees.
        For bulk data, use Reports API methods instead.

        Args:
            keywords: Search terms
            included_data: Data to include (summaries, images, salesRanks, etc.)
            page_size: Results per page (max 20)

        Returns:
            Catalog search results
        """
        if included_data is None:
            included_data = [
                'summaries',
                'salesRanks',
                'images',
                'productTypes'
            ]

        catalog = Catalog(marketplace=self.marketplace)

        try:
            response = catalog.search_catalog_items(
                keywords=keywords,
                marketplaceIds=[self.marketplace_id],
                includedData=included_data,
                pageSize=page_size
            )
            return response.payload
        except SellingApiException as e:
            print(f"Catalog search error: {e}")
            return {"items": [], "error": str(e)}

    def get_catalog_item(
        self,
        asin: str,
        included_data: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get detailed catalog item by ASIN.

        Args:
            asin: Amazon Standard Identification Number
            included_data: Data to include

        Returns:
            Product details
        """
        if included_data is None:
            included_data = [
                'attributes',
                'dimensions',
                'identifiers',
                'images',
                'productTypes',
                'relationships',
                'salesRanks',
                'summaries'
            ]

        catalog = Catalog(marketplace=self.marketplace)

        try:
            response = catalog.get_catalog_item(
                asin=asin,
                marketplaceIds=[self.marketplace_id],
                includedData=included_data
            )
            return response.payload
        except SellingApiException as e:
            print(f"Get catalog item error: {e}")
            return {"error": str(e)}

    # =========================================================================
    # REPORTS API - Free Bulk Data (Recommended)
    # =========================================================================

    def create_report(
        self,
        report_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """
        Request a report from Amazon.

        Common report types for product research:
        - GET_MERCHANT_LISTINGS_ALL_DATA: All your listings
        - GET_MERCHANT_LISTINGS_DATA: Active listings
        - GET_FBA_INVENTORY_PLANNING_DATA: FBA inventory planning
        - GET_SALES_AND_TRAFFIC_REPORT: Sales and traffic metrics
        - GET_BRAND_ANALYTICS_SEARCH_TERMS_REPORT: Search term analytics

        Args:
            report_type: Report type enum value
            start_time: Report data start time
            end_time: Report data end time

        Returns:
            Report ID for tracking
        """
        reports = Reports(marketplace=self.marketplace)

        params = {
            'reportType': report_type,
            'marketplaceIds': [self.marketplace_id]
        }

        if start_time:
            params['dataStartTime'] = start_time.isoformat()
        if end_time:
            params['dataEndTime'] = end_time.isoformat()

        try:
            response = reports.create_report(**params)
            report_id = response.payload.get('reportId')
            print(f"Report requested: {report_id}")
            return report_id
        except SellingApiException as e:
            print(f"Create report error: {e}")
            return None

    def get_report_status(self, report_id: str) -> Dict[str, Any]:
        """Check report processing status."""
        reports = Reports(marketplace=self.marketplace)

        try:
            response = reports.get_report(report_id)
            return response.payload
        except SellingApiException as e:
            print(f"Get report status error: {e}")
            return {"error": str(e)}

    def wait_for_report(
        self,
        report_id: str,
        timeout_minutes: int = 30,
        poll_interval: int = 30
    ) -> Optional[str]:
        """
        Wait for report to complete and return document ID.

        Args:
            report_id: Report ID from create_report
            timeout_minutes: Max wait time
            poll_interval: Seconds between status checks

        Returns:
            Report document ID when ready, None on timeout/failure
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while time.time() - start_time < timeout_seconds:
            status = self.get_report_status(report_id)

            processing_status = status.get('processingStatus', '')
            print(f"Report {report_id}: {processing_status}")

            if processing_status == 'DONE':
                return status.get('reportDocumentId')
            elif processing_status in ('CANCELLED', 'FATAL'):
                print(f"Report failed: {processing_status}")
                return None

            time.sleep(poll_interval)

        print(f"Report timeout after {timeout_minutes} minutes")
        return None

    def download_report(
        self,
        report_document_id: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Download report document content.

        Args:
            report_document_id: Document ID from completed report
            output_path: Optional file path to save report

        Returns:
            Report content as string
        """
        reports = Reports(marketplace=self.marketplace)

        try:
            response = reports.get_report_document(
                report_document_id,
                download=True
            )

            content = response.payload

            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    if isinstance(content, dict):
                        json.dump(content, f, indent=2)
                    else:
                        f.write(str(content))
                print(f"Report saved to: {output_path}")

            return content

        except SellingApiException as e:
            print(f"Download report error: {e}")
            return None

    def get_merchant_listings(self, output_path: Optional[str] = None) -> Any:
        """
        Convenience method: Get all merchant listings via Reports API (FREE).

        This is the recommended way to get your catalog data.
        """
        report_id = self.create_report('GET_MERCHANT_LISTINGS_ALL_DATA')
        if not report_id:
            return None

        doc_id = self.wait_for_report(report_id)
        if not doc_id:
            return None

        return self.download_report(doc_id, output_path)

    def get_fba_inventory(self, output_path: Optional[str] = None) -> Any:
        """
        Convenience method: Get FBA inventory planning data (FREE).
        """
        report_id = self.create_report('GET_FBA_INVENTORY_PLANNING_DATA')
        if not report_id:
            return None

        doc_id = self.wait_for_report(report_id)
        if not doc_id:
            return None

        return self.download_report(doc_id, output_path)

    def get_sales_traffic_report(
        self,
        days_back: int = 30,
        output_path: Optional[str] = None
    ) -> Any:
        """
        Convenience method: Get sales and traffic report (FREE).

        Args:
            days_back: Number of days of data (max 730)
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)

        report_id = self.create_report(
            'GET_SALES_AND_TRAFFIC_REPORT',
            start_time=start_time,
            end_time=end_time
        )
        if not report_id:
            return None

        doc_id = self.wait_for_report(report_id)
        if not doc_id:
            return None

        return self.download_report(doc_id, output_path)

    # =========================================================================
    # ORDERS API
    # =========================================================================

    def get_orders(
        self,
        days_back: int = 7,
        order_statuses: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get recent orders.

        Args:
            days_back: Number of days to look back
            order_statuses: Filter by status (Pending, Shipped, etc.)
        """
        orders = Orders(marketplace=self.marketplace)

        created_after = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

        try:
            response = orders.get_orders(
                CreatedAfter=created_after,
                OrderStatuses=order_statuses,
                MarketplaceIds=[self.marketplace_id]
            )
            return response.payload.get('Orders', [])
        except SellingApiException as e:
            print(f"Get orders error: {e}")
            return []


# =============================================================================
# REPORT TYPES REFERENCE
# =============================================================================

REPORT_TYPES = {
    # Inventory Reports
    'inventory': {
        'GET_MERCHANT_LISTINGS_ALL_DATA': 'All listings (active + inactive)',
        'GET_MERCHANT_LISTINGS_DATA': 'Active listings only',
        'GET_MERCHANT_LISTINGS_INACTIVE_DATA': 'Inactive listings',
        'GET_FBA_INVENTORY_PLANNING_DATA': 'FBA inventory planning',
        'GET_FBA_MYI_UNSUPPRESSED_INVENTORY_DATA': 'FBA inventory',
        'GET_RESTOCK_INVENTORY_RECOMMENDATIONS_REPORT': 'Restock recommendations',
    },

    # Sales Reports
    'sales': {
        'GET_SALES_AND_TRAFFIC_REPORT': 'Sales and traffic by ASIN/date',
        'GET_FLAT_FILE_ALL_ORDERS_DATA_BY_ORDER_DATE_GENERAL': 'All orders',
        'GET_AMAZON_FULFILLED_SHIPMENTS_DATA_GENERAL': 'FBA shipments',
    },

    # FBA Reports
    'fba': {
        'GET_FBA_FULFILLMENT_CUSTOMER_RETURNS_DATA': 'Customer returns',
        'GET_FBA_FULFILLMENT_REMOVAL_ORDER_DETAIL_DATA': 'Removal orders',
        'GET_FBA_STORAGE_FEE_CHARGES_DATA': 'Storage fees',
        'GET_FBA_ESTIMATED_FBA_FEES_TXT_DATA': 'FBA fee estimates',
    },

    # Brand Analytics (requires Brand Registry)
    'brand_analytics': {
        'GET_BRAND_ANALYTICS_SEARCH_TERMS_REPORT': 'Search term report',
        'GET_BRAND_ANALYTICS_MARKET_BASKET_REPORT': 'Market basket analysis',
        'GET_BRAND_ANALYTICS_REPEAT_PURCHASE_REPORT': 'Repeat purchase behavior',
    },

    # B2B Product Opportunities
    'b2b': {
        'GET_B2B_PRODUCT_OPPORTUNITIES_RECOMMENDED_FOR_YOU': 'B2B opportunities',
        'GET_B2B_PRODUCT_OPPORTUNITIES_NOT_YET_ON_AMAZON': 'Products not on Amazon',
    }
}


if __name__ == "__main__":
    # Example usage
    print("SP-API Client initialized")
    print("\nAvailable Report Types:")
    for category, reports in REPORT_TYPES.items():
        print(f"\n{category.upper()}:")
        for report_type, description in reports.items():
            print(f"  {report_type}: {description}")
