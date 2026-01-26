# Amazon KDP/FBA Publishing Ecosystem

Complete toolkit for Amazon product research and KDP book publishing.

## 🎯 Three Integrated Systems

| Project | Purpose | Status |
|---------|---------|--------|
| **[FBA Research](FBA/)** | Product research & inventory | ✅ Production |
| **[Coloring Books](coloring-books/)** | Adult coloring book generator | ✅ v2.0 |
| **[Puzzle Books](kdp-quality-pipeline/)** | Puzzle book generator & validator | ✅ Production |

📖 **[Complete Ecosystem Guide](ECOSYSTEM_GUIDE.md)** - Integration workflows and business models

## 🚀 Quick Start with Master Control

```bash
# Interactive menu for all systems
python3 master_control.py

# Or launch directly into a specific system
python3 master_control.py --system coloring
python3 master_control.py --system puzzle
python3 master_control.py --system fba
```

---

# FBA Product Research Pipeline

A compliant toolkit for Amazon FBA product research using official APIs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESEARCH PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │   SP-API     │   │   Reports    │   │    Keepa     │        │
│  │  (Catalog)   │   │    API       │   │  (History)   │        │
│  │  Paid/req    │   │    FREE      │   │ Subscription │        │
│  └──────────────┘   └──────────────┘   └──────────────┘        │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              INVENTORY FORECASTER                         │  │
│  │  • Safety Stock    • Reorder Point    • EOQ              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              OPPORTUNITY ANALYSIS                         │  │
│  │  • Demand Score    • Competition    • Recommendation     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Features

| Component | Description | Cost |
|-----------|-------------|------|
| **SP-API Client** | Official Amazon Selling Partner API | Per-request |
| **Reports API** | 18 free bulk data report types | **FREE** |
| **Keepa Integration** | Price history, BSR tracking | ~€15/mo |
| **Inventory Forecasting** | EOQ, safety stock, reorder points | Free |
| **Opportunity Scoring** | Multi-factor product analysis | Free |

## Quick Start

```bash
# Clone
git clone https://github.com/ElliottSax/amazon-fba-research.git
cd amazon-fba-research

# Install
pip install -r FBA/requirements.txt

# Test
python FBA/quick_test.py

# Demo
python FBA/full_demo.py
```

## Usage

### Inventory Forecasting (No API Required)

```python
from FBA.src.inventory_forecast import InventoryForecaster, InventoryItem

item = InventoryItem(
    sku="MY-SKU-001",
    asin="B08PZHWJS5",
    name="My Product",
    current_stock=150,
    unit_cost=25.00,
    lead_time_days=14,
    sales_history=[5, 7, 6, 8, 4, 9, 7, 6, 8, 5, 7, 6, 9, 8, 7],
)

forecaster = InventoryForecaster(service_level=0.95)
rec = forecaster.generate_reorder_recommendation(item)

print(f"Reorder Point: {rec.reorder_point} units")
print(f"Safety Stock: {rec.safety_stock} units")
print(f"EOQ: {rec.economic_order_quantity} units")
print(f"Urgency: {rec.reorder_urgency}")
```

### Product Research (Requires API Keys)

```python
from FBA.src.research_pipeline import FBAResearchPipeline

pipeline = FBAResearchPipeline(marketplace='US')

# Research a product
result = pipeline.research_product('B08PZHWJS5')

# Analyze opportunity
analysis = pipeline.analyze_opportunity(result)
print(f"Score: {analysis['overall_score']}/100")
print(f"Recommendation: {analysis['recommendation']}")

# Export
pipeline.export_report([result], 'research_report.json')
```

### Free Bulk Data (Reports API)

```python
from FBA.src.sp_api_client import SPAPIClient

client = SPAPIClient(marketplace='US')

# Get all your listings (FREE)
client.get_merchant_listings('my_listings.csv')

# Get FBA inventory (FREE)
client.get_fba_inventory('inventory.csv')

# Get sales report (FREE)
client.get_sales_traffic_report(days=30)
```

## Available Report Types (Free)

| Category | Reports |
|----------|---------|
| **Inventory** | Listings, FBA inventory, restock recommendations |
| **Sales** | Sales & traffic, orders, FBA shipments |
| **FBA** | Returns, removals, storage fees, fee estimates |
| **Analytics** | Search terms, market basket, repeat purchase |
| **B2B** | Product opportunities |

## Setup

See [FBA/SETUP.md](FBA/SETUP.md) for detailed configuration:

1. **SP-API**: Register at Seller Central → Apps & Services → Develop Apps
2. **Keepa**: Subscribe at [keepa.com/#!api](https://keepa.com/#!api)
3. **Credentials**: Copy `config/credentials.example.yml` to `~/.sp-api/credentials.yml`

## Project Structure

```
FBA/
├── src/
│   ├── sp_api_client.py        # Amazon SP-API wrapper
│   ├── keepa_client.py         # Price history client
│   ├── inventory_forecast.py   # Demand forecasting
│   └── research_pipeline.py    # Main orchestrator
├── config/
│   ├── credentials.example.yml
│   └── .env.example
├── quick_test.py               # Fast validation
├── full_demo.py                # Feature demonstration
├── SETUP.md                    # Configuration guide
└── PRODUCT_RESEARCH_TOOLS.md   # Tool reference
```

## Related Projects

### KDP Book Publishing

- **[Adult Coloring Books](coloring-books/)** - Generate professional coloring books
  - 7 themes (Mandalas, Animals, Nature, etc.)
  - Enhanced line art processing
  - 100% free generation
  - Print-ready PDFs
  - Quick start: `cd coloring-books && python complete_book_workflow.py --theme mandalas --pages 30`

- **[Puzzle Books](kdp-quality-pipeline/)** - Generate quality-validated puzzle books
  - 8 puzzle types (Sudoku, Word Search, Maze, etc.)
  - Automated quality validation
  - 24/7 daemon mode
  - KDP-compliant covers
  - Quick start: `cd kdp-quality-pipeline && python run_daemon.py --once`

### Complete Integration

See **[ECOSYSTEM_GUIDE.md](ECOSYSTEM_GUIDE.md)** for:
- How to integrate research with publishing
- Complete business workflows
- Scaling strategies
- Best practices

## License

MIT
