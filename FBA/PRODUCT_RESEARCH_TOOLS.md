# Amazon FBA Product Research Tools & Resources

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FBA PRODUCT RESEARCH STACK                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  TOOL DISCOVERY │    │ DATA EXTRACTION │    │ COMPLIANT ACCESS│         │
│  │                 │    │                 │    │                 │         │
│  │ awesome-amazon- │    │ amazon-product- │    │ python-amazon-  │         │
│  │ seller          │───▶│ api (TOS RISK)  │ OR │ sp-api          │         │
│  │                 │    │                 │    │                 │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                     │                       │                   │
│           ▼                     ▼                       ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SUPPORTING TOOLS                                  │   │
│  │  • Keepa API (Price History)    • Keyword Scrapers                  │   │
│  │  • Inventory Forecasting        • Competitor Analysis               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Tool Discovery: ScaleLeap/awesome-amazon-seller

**Repository:** https://github.com/ScaleLeap/awesome-amazon-seller

A curated list of 50+ tools and resources for Amazon sellers. Key categories:

### Product Research & Analytics
| Tool | Description | URL |
|------|-------------|-----|
| Jungle Scout | Product tracking with demand/sales estimates | junglescout.com |
| Helium10 | Multi-tool suite for keyword research | helium10.com |
| DataHawk | End-to-end platform with AI guidance | datahawk.co |
| ASINspector | Sales trend data and BSR tracking | asinspector.com |
| SellerApp | Product research, alerts, PPC analysis | sellerapp.com |

### Repricing & Pricing
| Tool | Description | URL |
|------|-------------|-----|
| BQool | Scheduled repricing, buy box targeting | bqool.com |
| Splitly | Algorithmic split testing & price optimization | splitly.com |
| Prisync | Price monitoring with dynamic pricing | prisync.com |

### Keyword Research
| Tool | Description | URL |
|------|-------------|-----|
| Keyword Tool | Uses Amazon autocomplete | keywordtool.io/amazon |
| MerchantWords | Buyer search phrase identification | merchantwords.com |
| Sonar (Free) | Keyword research by Sellics | sonar-tool.com |

### Inventory & Forecasting
| Tool | Description | URL |
|------|-------------|-----|
| Eva | Price management, replenishment, analytics | eva.guru |
| ForecastRx | Inventory demand forecasting | forecastrx.com |
| SellerLegend | Real-time orders, KPI dashboards | sellerlegend.com |

### Free Resources
- **FBA Catalog** - Software directory: fbacatalog.com
- **FBA Monthly** - Newsletter: fbamonthly.com
- **r/FulfillmentByAmazon** - Reddit community
- **500K ASIN Lists** - Free weekly datasets: app.turbopiranha.com/Download/bestselleritems

---

## 2. Data Extraction: drawrowfly/amazon-product-api

**Repository:** https://github.com/drawrowfly/amazon-product-api

⚠️ **TOS RISK WARNING**: Scraping Amazon directly violates their Terms of Service. Use at your own risk. Consider SP-API for compliant access.

### Installation
```bash
npm i -g amazon-buddy
```

### CLI Commands
```bash
# Search products
amazon-buddy products -k 'wireless earbuds' -n 50 --country US --filetype json

# Get product reviews
amazon-buddy reviews B08PZHWJS5 -n 100 --min-rating 1 --max-rating 3

# Single product details
amazon-buddy asin B08PZHWJS5

# List categories
amazon-buddy categories --country US

# Filter by discount only
amazon-buddy products -k 'headphones' -n 40 --discount

# Sponsored products only
amazon-buddy products -k 'gaming mouse' -n 40 --sponsored
```

### Node.js Integration
```javascript
const amazonScraper = require('amazon-buddy');

(async () => {
  // Product search
  const products = await amazonScraper.products({
    keyword: 'yoga mat',
    number: 100,
    country: 'US',
    category: 'sports'
  });

  // Get reviews (filter by rating)
  const reviews = await amazonScraper.reviews({
    asin: 'B08PZHWJS5',
    number: 200,
    rating: [1, 2, 3]  // Only 1-3 star reviews
  });

  // Single product details
  const details = await amazonScraper.asin({
    asin: 'B08PZHWJS5'
  });

  console.log(products);
})();
```

### Output Fields
**Products**: position, asin, price, reviews_count, rating, url, sponsored, prime, title, thumbnail

**Reviews**: id, date, author, rating, title, text

### Related Tools by drawrowfly
- **amazon-keyword-scraper-go** - Keyword suggestions in Go
- **amazon-keyword-scraper-ts** - Keyword scraper in TypeScript

---

## 3. Compliant Access: Amazon SP-API

### Official SDK
**Repository:** https://github.com/amzn/selling-partner-api-sdk

### Python Wrapper (Recommended)
**Repository:** https://github.com/saleweaver/python-amazon-sp-api

### Installation
```bash
pip install python-amazon-sp-api
pip install "python-amazon-sp-api[aws]"  # AWS Secrets Manager support
```

### Authentication Setup
Create `~/.sp-api/credentials.yml`:
```yaml
refresh_token: 'your-refresh-token'
lwa_app_id: 'your-lwa-app-id'
lwa_client_secret: 'your-lwa-client-secret'
```

### Key APIs for Product Research

#### Catalog Items API (2022-04-01)
```python
from sp_api.api import Catalog
from sp_api.base import Marketplaces

# Search catalog by keywords
catalog = Catalog(marketplace=Marketplaces.US)
result = catalog.search_catalog_items(
    keywords='wireless earbuds',
    includedData=['summaries', 'salesRanks', 'images']
)

# Get item by ASIN
item = catalog.get_catalog_item(
    asin='B08PZHWJS5',
    includedData=['attributes', 'dimensions', 'identifiers', 'images',
                  'productTypes', 'relationships', 'salesRanks', 'summaries']
)
```

#### Reports API (Free Alternative to GET requests)
```python
from sp_api.api import Reports
from sp_api.base import ReportType

reports = Reports()

# Request merchant listings report
report = reports.create_report(
    reportType=ReportType.GET_MERCHANT_LISTINGS_ALL_DATA
)

# B2B Product Opportunities (for product research)
b2b_report = reports.create_report(
    reportType='GET_B2B_PRODUCT_OPPORTUNITIES_RECOMMENDED_FOR_YOU'
)
```

#### Orders API
```python
from sp_api.api import Orders
from datetime import datetime, timedelta

orders = Orders()
result = orders.get_orders(
    CreatedAfter=(datetime.utcnow() - timedelta(days=30)).isoformat()
)
```

### Important: SP-API Pricing
Amazon charges usage fees for GET requests. Use Reports API when possible:
- Reports are generally free
- GET requests incur per-call charges
- Batch operations when possible

### Documentation
- Official: https://developer-docs.amazon.com/sp-api
- Python wrapper: https://python-amazon-sp-api.readthedocs.io/

---

## 4. Price History: Keepa API

**Repository:** https://github.com/akaszynski/keepa

### Installation
```bash
pip install keepa
```

### Requirements
- Keepa API subscription required (keepa.com)
- Access key from Keepa account

### Usage
```python
import keepa

# Initialize API
api = keepa.Keepa('your-api-key')

# Query product by ASIN
products = api.query(['B08PZHWJS5', 'B09V3KXJPB'])

# Access price history
for product in products:
    print(f"Title: {product['title']}")
    print(f"Current Amazon Price: {product['data']['AMAZON'][-1] / 100}")
    print(f"Sales Rank: {product['salesRankReference']}")

# Get price history arrays
prices = products[0]['data']
# Keys: AMAZON, NEW, USED, SALES, LISTPRICE, COLLECTIBLE, REFURBISHED, etc.
```

### Plotting
```python
import keepa
api = keepa.Keepa('your-api-key')
products = api.query('B08PZHWJS5')
keepa.plot_product(products[0])
```

### Data Available
- Amazon price history
- Marketplace (3rd party new) prices
- Used prices
- Sales rank history
- Lightning deal prices
- Buy Box statistics
- Review counts over time

---

## 5. Keyword Research Tools

### amazon-keyword-scraper-go
**Repository:** https://github.com/drawrowfly/amazon-keyword-scraper-go

```bash
# Build
go build -o amazon-keyword-scraper

# Run
./amazon-keyword-scraper -keyword "yoga mat" -country US
```

Generates hundreds of long-tail keywords from 1 seed keyword.

### omkarcloud/amazon-scraper
**Repository:** https://github.com/omkarcloud/amazon-scraper

Features:
- Competitor listing analysis
- Review sentiment extraction
- Inventory tracking
- Free until Dec 2025

---

## 6. Inventory & Demand Forecasting

### supplychainpy
**Repository:** https://github.com/KevinFasusi/supplychainpy

Python library for supply chain analysis replacing Excel/VBA workflows.

```bash
pip install supplychainpy
```

```python
from supplychainpy import model_inventory
from supplychainpy.inventory.summarise import Inventory

# Analyze inventory
orders = model_inventory.analyse(
    file_path="inventory.csv",
    z_value=Decimal(1.28),  # 90% service level
    reorder_cost=Decimal(400),
    file_type="csv"
)
```

### inventory-forecast
**Repository:** https://github.com/shanayamalik/inventory-forecast

ML-based forecasting using Random Forest and XGBoost.

---

## 7. Competitor Analysis

### amazon-search-rank
**Repository:** https://github.com/rafalf/amazon-search-rank

Track ASIN positions for keywords over time.

### amazon-bestsellers-scraper
**Repository:** https://github.com/m-murasovs/amazon-bestsellers-scraper

Scrape Best Sellers lists for trend analysis.

---

## Recommended Stack by Use Case

### Budget-Conscious Seller
1. **ScaleLeap awesome list** - Free tool discovery
2. **Sonar** - Free keyword research
3. **amazon-buddy** - Free scraping (TOS risk)
4. **FBA Catalog** - Free software discovery

### Compliant Enterprise Setup
1. **python-amazon-sp-api** - Official API access
2. **Reports API** - Cost-effective bulk data
3. **Keepa API** - Historical pricing
4. **supplychainpy** - Inventory optimization

### Product Research Flow
```
1. Identify niche (awesome-amazon-seller resources)
         │
         ▼
2. Keyword research (Sonar, amazon-keyword-scraper)
         │
         ▼
3. Product data (SP-API Catalog or amazon-buddy)
         │
         ▼
4. Price history (Keepa API)
         │
         ▼
5. Competitor analysis (amazon-search-rank)
         │
         ▼
6. Demand forecasting (supplychainpy)
```

---

## Legal Considerations

| Method | TOS Compliant | Cost | Data Access |
|--------|---------------|------|-------------|
| SP-API | ✅ Yes | Per-request fees | Full (your account) |
| Reports API | ✅ Yes | Usually free | Bulk data |
| Keepa API | ✅ Yes | Subscription | Price history |
| amazon-buddy | ❌ No (scraping) | Free | Public data |
| amazon-scraper | ❌ No (scraping) | Free (until 2025) | Public data |

**Recommendation**: Use SP-API + Reports for compliant access. Use scraping tools for personal research only, understanding the risks.

---

## Quick Start

```bash
# Clone this folder's examples
cd /mnt/e/projects/amazon/FBA

# Install compliant tools
pip install python-amazon-sp-api keepa supplychainpy

# Install scraping tools (TOS risk)
npm i -g amazon-buddy
```

---

## Sources

- [ScaleLeap/awesome-amazon-seller](https://github.com/ScaleLeap/awesome-amazon-seller)
- [drawrowfly/amazon-product-api](https://github.com/drawrowfly/amazon-product-api)
- [saleweaver/python-amazon-sp-api](https://github.com/saleweaver/python-amazon-sp-api)
- [amzn/selling-partner-api-models](https://github.com/amzn/selling-partner-api-models)
- [akaszynski/keepa](https://github.com/akaszynski/keepa)
- [KevinFasusi/supplychainpy](https://github.com/KevinFasusi/supplychainpy)
- [drawrowfly/amazon-keyword-scraper-go](https://github.com/drawrowfly/amazon-keyword-scraper-go)
- [omkarcloud/amazon-scraper](https://github.com/omkarcloud/amazon-scraper)
