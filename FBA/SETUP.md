# FBA Product Research - Compliant Stack Setup

## Overview

This stack uses **only official and compliant** methods for Amazon product research:

| Component | Purpose | Cost |
|-----------|---------|------|
| SP-API | Official Amazon API | Per-request fees |
| Reports API | Bulk data export | **FREE** |
| Keepa API | Price history | Subscription |
| Local Forecasting | Demand prediction | Free |

---

## 1. Amazon SP-API Setup

### Step 1: Register as Developer

1. Go to [Seller Central](https://sellercentral.amazon.com/)
2. Navigate to **Apps & Services** → **Develop Apps**
3. Register as a developer (if not already)

### Step 2: Create Application

1. Click **Add new app client**
2. Fill in application details:
   - App name: "FBA Research Tool"
   - API Type: SP-API
3. Select required roles:
   - ✅ **Product Listing** - For catalog access
   - ✅ **Reports** - For free bulk data
   - ✅ **Brand Analytics** (if brand registered)

### Step 3: Get Credentials

After app creation, note down:
- **LWA App ID**: `amzn1.application-oa2-client.xxxxx`
- **LWA Client Secret**: `xxxxxxxx`
- **Refresh Token**: Generated during self-authorization

### Step 4: Configure Credentials

Create `~/.sp-api/credentials.yml`:

```yaml
refresh_token: 'Atzr|your-refresh-token'
lwa_app_id: 'amzn1.application-oa2-client.xxxxx'
lwa_client_secret: 'your-secret'
```

Or use environment variables:
```bash
export SP_API_REFRESH_TOKEN='Atzr|your-refresh-token'
export SP_API_LWA_APP_ID='amzn1.application-oa2-client.xxxxx'
export SP_API_LWA_CLIENT_SECRET='your-secret'
```

---

## 2. Keepa API Setup

### Step 1: Subscribe to Keepa API

1. Go to [Keepa API](https://keepa.com/#!api)
2. Choose a subscription plan (starts ~€15/month)
3. Get your API key

### Step 2: Configure

```bash
export KEEPA_API_KEY='your-keepa-api-key'
```

---

## 3. Installation

```bash
# Navigate to FBA folder
cd /mnt/e/projects/amazon/FBA

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Quick Start

### Basic Product Research

```python
from src.research_pipeline import FBAResearchPipeline

# Initialize (will warn if credentials missing)
pipeline = FBAResearchPipeline(marketplace='US')

# Research a product
result = pipeline.research_product('B08PZHWJS5')

# View key metrics
print(f"Title: {result.title}")
print(f"Price: ${result.current_price}")
print(f"Sales Rank: {result.sales_rank}")
print(f"Monthly Sales: ~{result.estimated_monthly_sales}")
print(f"Competition: {result.competition_level}")

# Get opportunity score
analysis = pipeline.analyze_opportunity(result)
print(f"Score: {analysis['overall_score']}/100")
print(f"Recommendation: {analysis['recommendation']}")
```

### Get Your Listings (FREE)

```python
# Uses Reports API - no per-request fees!
listings = pipeline.get_my_listings('output/my_listings.csv')
```

### Get Sales Report (FREE)

```python
# Last 30 days sales and traffic
sales = pipeline.get_sales_report(days=30, output_path='output/sales.json')
```

### Batch Research

```python
asins = ['B08PZHWJS5', 'B09V3KXJPB', 'B07XYZ1234']
results = pipeline.research_batch(asins)
pipeline.export_report(results, 'output/research_report.json')
```

---

## 5. Cost Optimization

### Use Reports API (FREE) Instead of GET Requests

```python
from src.sp_api_client import SPAPIClient, REPORT_TYPES

client = SPAPIClient()

# FREE: Bulk listing data
client.get_merchant_listings('listings.csv')

# FREE: Inventory data
client.get_fba_inventory('inventory.csv')

# FREE: Sales data
client.get_sales_traffic_report(days=30)

# PAID: Individual catalog lookups (use sparingly)
# client.get_catalog_item('B08PZHWJS5')  # Costs money!
```

### Available Free Reports

| Report Type | Description |
|------------|-------------|
| `GET_MERCHANT_LISTINGS_ALL_DATA` | All your listings |
| `GET_FBA_INVENTORY_PLANNING_DATA` | FBA inventory |
| `GET_SALES_AND_TRAFFIC_REPORT` | Sales metrics |
| `GET_RESTOCK_INVENTORY_RECOMMENDATIONS_REPORT` | Restock alerts |
| `GET_FBA_STORAGE_FEE_CHARGES_DATA` | Storage fees |

---

## 6. Inventory Forecasting

```python
from src.inventory_forecast import InventoryForecaster, InventoryItem

# Create item with sales history
item = InventoryItem(
    sku="MY-SKU-001",
    asin="B08PZHWJS5",
    name="My Product",
    current_stock=150,
    unit_cost=25.00,
    lead_time_days=14,
    sales_history=[5, 7, 6, 8, 4, 9, 7, 6, 8, 5, 7, 6, 9, 8, 7],
)

# Get recommendations
forecaster = InventoryForecaster(service_level=0.95)
rec = forecaster.generate_reorder_recommendation(item)

print(f"Reorder Point: {rec.reorder_point} units")
print(f"Safety Stock: {rec.safety_stock} units")
print(f"Days of Stock: {rec.days_of_stock}")
print(f"Urgency: {rec.reorder_urgency}")
print(f"Order Quantity: {rec.recommended_order_qty}")
```

---

## 7. File Structure

```
FBA/
├── config/
│   ├── credentials.example.yml    # SP-API credentials template
│   └── .env.example               # Environment variables template
├── src/
│   ├── __init__.py
│   ├── sp_api_client.py           # SP-API wrapper
│   ├── keepa_client.py            # Keepa price history
│   ├── inventory_forecast.py      # Demand forecasting
│   └── research_pipeline.py       # Main pipeline
├── output/                        # Generated reports
├── requirements.txt
├── SETUP.md                       # This file
└── PRODUCT_RESEARCH_TOOLS.md      # Tool reference
```

---

## 8. Troubleshooting

### SP-API Authentication Error

```
SellingApiException: Access to requested resource is denied
```

**Solution**: Check credentials and ensure app has required roles in Seller Central.

### Keepa API Error

```
ValueError: Keepa API key required
```

**Solution**: Set `KEEPA_API_KEY` environment variable or pass to constructor.

### Rate Limiting

```
SellingApiException: QuotaExceeded
```

**Solution**: Use Reports API for bulk data. Add delays between requests.

---

## 9. Resources

- [SP-API Documentation](https://developer-docs.amazon.com/sp-api)
- [python-amazon-sp-api Docs](https://python-amazon-sp-api.readthedocs.io/)
- [Keepa API](https://keepa.com/#!api)
- [Report Type Values](https://developer-docs.amazon.com/sp-api/docs/report-type-values)
