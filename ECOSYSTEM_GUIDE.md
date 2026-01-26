# Amazon KDP Publishing Ecosystem

Complete toolkit for Amazon FBA product research and KDP book publishing.

## 📚 Overview

This repository contains three complementary systems for Amazon-based businesses:

```
amazon/
├── FBA/                    # Product Research & Inventory Management
├── coloring-books/         # Adult Coloring Book Generator
└── kdp-quality-pipeline/   # Puzzle Book Generator
```

## 🎯 Projects

### 1. FBA Product Research Pipeline

**Purpose**: Research and manage Amazon FBA products using official APIs

**Features:**
- SP-API integration (catalog search)
- Reports API (18 free bulk data types)
- Keepa integration (price history)
- Inventory forecasting (EOQ, safety stock)
- Opportunity analysis

**Cost**: Free Reports API + Optional Keepa (~€15/mo)

**Use Case**: Finding profitable FBA products to sell

**Status**: ✅ Production Ready

**Quick Start:**
```bash
cd FBA
pip install -r requirements.txt
python quick_test.py
```

**Documentation**: [FBA/README.md](FBA/README.md)

---

### 2. Adult Coloring Book Generator

**Purpose**: Create professional coloring books for Amazon KDP

**Features:**
- 7 family-friendly themes
- Enhanced line art processing (3 quality levels)
- Free unlimited generation (Pollinations.ai)
- Print-ready PDFs (8.5x11", 300 DPI)
- Professional cover generation
- Complete KDP publishing workflow

**Cost**: 100% FREE (default backend)

**Use Case**: Publishing coloring books on Amazon KDP

**Status**: ✅ Production Ready (v2.0)

**Quick Start:**
```bash
cd coloring-books
pip install -r requirements.txt
python complete_book_workflow.py --theme mandalas --pages 30
```

**Documentation**: [coloring-books/README.md](coloring-books/README.md)

---

### 3. KDP Puzzle Book Quality Pipeline

**Purpose**: Generate puzzle books with automated quality validation

**Features:**
- 8 puzzle types (Sudoku, Word Search, Maze, Cryptogram, etc.)
- Quality validation pipeline
- Auto-iteration until all checks pass
- KDP-compliant covers with spine calculation
- 24/7 daemon mode
- Profanity scanning

**Cost**: Free

**Use Case**: Publishing puzzle books on Amazon KDP

**Status**: ✅ Production Ready

**Quick Start:**
```bash
cd kdp-quality-pipeline
pip install -r requirements.txt
python run_daemon.py --once
```

**Documentation**: [kdp-quality-pipeline/README.md](kdp-quality-pipeline/README.md)

---

## 🔄 Workflow Integration

### Complete Amazon Publishing Business

```
┌─────────────────────────────────────────────────────────────┐
│                    RESEARCH PHASE                            │
├─────────────────────────────────────────────────────────────┤
│  Use FBA/ to research profitable niches                     │
│  → Analyze demand, competition, pricing                     │
│  → Identify trending topics and keywords                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  CONTENT CREATION                            │
├─────────────────────────────────────────────────────────────┤
│  Option A: coloring-books/                                  │
│  → Generate coloring books in trending themes               │
│  → Create 30-50 page books                                  │
│  → Generate professional covers                             │
│                                                              │
│  Option B: kdp-quality-pipeline/                            │
│  → Generate puzzle books (8 types)                          │
│  → Automated quality assurance                              │
│  → Batch generation for series                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   PUBLISHING                                 │
├─────────────────────────────────────────────────────────────┤
│  → Upload to Amazon KDP (kdp.amazon.com)                    │
│  → Use research data for pricing                            │
│  → Optimize with keywords from FBA research                 │
└─────────────────────────────────────────────────────────────┘
```

### Example: Complete Publishing Workflow

**1. Research (FBA/)**
```bash
cd FBA
python full_demo.py
# Identify that "mandala coloring books" are trending
```

**2. Create Coloring Book (coloring-books/)**
```bash
cd ../coloring-books
python complete_book_workflow.py \
  --theme mandalas \
  --pages 40 \
  --title "Zen Mandalas: Stress Relief Adult Coloring Book"
```

**3. Create Puzzle Book Companion (kdp-quality-pipeline/)**
```bash
cd ../kdp-quality-pipeline
# Edit config.yaml to add "mindfulness" themed puzzles
python run_daemon.py --niche sudoku_meditation_easy
```

**4. Publish Both to KDP**
- Upload coloring book PDF + cover
- Upload puzzle book PDF + cover
- Cross-promote in descriptions
- Use FBA research data for pricing

---

## 📊 Comparison Matrix

| Feature | FBA Research | Coloring Books | Puzzle Books |
|---------|-------------|----------------|--------------|
| **Purpose** | Research | Publishing | Publishing |
| **Output** | Data/Reports | PDFs + Covers | PDFs + Covers |
| **Cost** | Free (mostly) | Free | Free |
| **API Keys** | Optional | No (default) | No |
| **Automation** | Manual | CLI + Batch | 24/7 Daemon |
| **Quality Control** | N/A | Line Art Processing | Multi-Stage Validation |
| **Book Types** | N/A | 7 Themes | 8 Puzzle Types |
| **KDP Ready** | N/A | ✅ Yes | ✅ Yes |
| **Use Case** | Find products | Creative books | Activity books |

---

## 🚀 Getting Started

### For Product Research Only
```bash
cd FBA
pip install -r requirements.txt
python quick_test.py
```

### For Publishing Coloring Books
```bash
cd coloring-books
pip install -r requirements.txt
python test_improved_lineart.py --quick
python complete_book_workflow.py --theme animals --pages 30
```

### For Publishing Puzzle Books
```bash
cd kdp-quality-pipeline
pip install -r requirements.txt
python run_daemon.py --once
```

### For Complete Business Setup
```bash
# Install all dependencies
cd FBA && pip install -r requirements.txt && cd ..
cd coloring-books && pip install -r requirements.txt && cd ..
cd kdp-quality-pipeline && pip install -r requirements.txt && cd ..

# Run research
cd FBA && python full_demo.py && cd ..

# Generate coloring book
cd coloring-books && python complete_book_workflow.py --theme mandalas --pages 30 && cd ..

# Generate puzzle book
cd kdp-quality-pipeline && python run_daemon.py --once && cd ..
```

---

## 💰 Business Models

### Model 1: KDP Publishing Only
**Use:** coloring-books/ + kdp-quality-pipeline/
- Generate multiple books per week
- Build a catalog of 50-100+ titles
- Passive income from royalties
- **Investment:** Time only (free tools)

### Model 2: FBA + KDP Hybrid
**Use:** All three systems
- Research trending niches with FBA/
- Create books matching trends
- Sell physical products AND books
- **Investment:** Time + FBA inventory

### Model 3: Research Service
**Use:** FBA/ primarily
- Research products for clients
- Provide market analysis reports
- Consult on product selection
- **Investment:** Time only

---

## 📈 Scaling Strategies

### Phase 1: Proof of Concept (Week 1-2)
```bash
# Generate 3 test books
cd coloring-books
python complete_book_workflow.py --theme mandalas --pages 20
python complete_book_workflow.py --theme animals --pages 20

cd ../kdp-quality-pipeline
python run_daemon.py --niche sudoku_easy
```

### Phase 2: Build Catalog (Month 1-3)
```bash
# Generate 2 books per week
# Target: 20-30 books total
cd coloring-books
python batch_generator.py --all-themes --pages 30

cd ../kdp-quality-pipeline
# Configure daemon for daily generation
python run_daemon.py
```

### Phase 3: Optimize (Month 4+)
- Use FBA research to identify best-selling niches
- Focus on top-performing themes
- Create book series (Vol. 1, 2, 3...)
- Cross-promote within catalog

---

## 🎓 Best Practices

### Research-Driven Publishing
1. **Research First**: Use FBA/ to identify trends
2. **Validate Demand**: Check Best Seller Rank (BSR) data
3. **Competitive Analysis**: Count competitors, check prices
4. **Create Content**: Generate books matching demand
5. **Iterate**: Focus on what sells

### Quality Over Quantity
- Always use `--force-lineart` for coloring books
- Let puzzle pipeline run full validation
- Review output before publishing
- Maintain high standards

### Catalog Building
- Publish consistently (2-4 books/month minimum)
- Create series for recurring buyers
- Use themes that complement each other
- Build brand recognition

### Pricing Strategy
- Use FBA research data for competitive pricing
- Start with mid-range prices ($6.99-9.99)
- Test price points
- Factor in printing costs (KDP calculator)

---

## 🛠️ Technical Architecture

### Technology Stack

| Component | Technologies |
|-----------|-------------|
| **FBA Research** | Python, SP-API, Keepa API, Requests |
| **Coloring Books** | Python, OpenCV, NumPy, Pillow, ReportLab, Pollinations.ai |
| **Puzzle Books** | Python, YAML, PyPDF2, PIL, Custom validators |

### Data Flow

```
Research (FBA)
    ↓ [Market data]
Content Creation (coloring-books/ OR kdp-quality-pipeline/)
    ↓ [PDF + Cover]
Quality Check (Built-in validators)
    ↓ [Approved books]
Amazon KDP
    ↓ [Published]
Sales & Royalties
```

---

## 📦 Repository Structure

```
amazon/
├── README.md                      # Main overview
├── ECOSYSTEM_GUIDE.md            # This file
│
├── FBA/                           # Product Research
│   ├── src/
│   │   ├── sp_api_client.py
│   │   ├── keepa_client.py
│   │   ├── inventory_forecast.py
│   │   └── research_pipeline.py
│   ├── SETUP.md
│   └── README.md
│
├── coloring-books/                # Coloring Book Generator
│   ├── coloring_book_generator.py
│   ├── complete_book_workflow.py
│   ├── batch_generator.py
│   ├── cover_generator.py
│   ├── test_improved_lineart.py
│   ├── QUICK_REFERENCE.md
│   ├── IMPROVEMENTS.md
│   ├── SETUP_GUIDE.md
│   ├── CHANGELOG.md
│   ├── PROJECT_STATUS.md
│   └── README.md
│
└── kdp-quality-pipeline/          # Puzzle Book Generator
    ├── src/
    │   ├── generators/
    │   ├── validators/
    │   ├── pipeline/
    │   ├── utils/
    │   └── daemon/
    ├── config.yaml
    ├── run_daemon.py
    ├── generate_test_book.py
    └── README.md
```

---

## 🎯 Quick Command Reference

### Research Commands
```bash
# FBA research
cd FBA
python quick_test.py              # Quick validation
python full_demo.py               # Full demo
```

### Publishing Commands
```bash
# Coloring books
cd coloring-books
python test_improved_lineart.py --quick
python complete_book_workflow.py --theme animals --pages 30
python batch_generator.py --all-themes --pages 30

# Puzzle books
cd kdp-quality-pipeline
python run_daemon.py --once
python run_daemon.py --niche sudoku_easy
python run_daemon.py              # Start daemon
```

---

## 📄 License

MIT License - All projects are free for commercial use on Amazon KDP/FBA

---

## 🔗 External Resources

- **Amazon KDP**: https://kdp.amazon.com
- **Amazon Seller Central**: https://sellercentral.amazon.com
- **SP-API Documentation**: https://developer-docs.amazon.com/sp-api/
- **Keepa API**: https://keepa.com/#!api

---

## 🎓 Learning Path

### Beginner (Week 1)
1. Read all README files
2. Run test commands for each system
3. Generate 1 test book from each publisher

### Intermediate (Month 1)
1. Research niche with FBA/
2. Generate 5-10 books
3. Publish to KDP
4. Monitor sales

### Advanced (Month 2+)
1. Automate with batch/daemon modes
2. Build 50+ book catalog
3. Optimize based on sales data
4. Scale to 100+ books

---

*Last Updated: 2026-01-26*
*Version: 1.0*
*Maintained by: Elliott*
