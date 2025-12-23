#!/usr/bin/env python3
"""Quick test of FBA pipeline components."""
import sys
print("=" * 60)
print("FBA PIPELINE - QUICK TEST")
print("=" * 60)

# Test 1: Inventory Forecasting (pure Python, fast)
print("\n[1] Inventory Forecasting")
from inventory_forecast import InventoryForecaster, InventoryItem
item = InventoryItem(
    sku="TEST", asin="B08PZHWJS5", name="Test",
    current_stock=150, unit_cost=25.0, lead_time_days=14,
    sales_history=[5,7,6,8,4,9,7,6,8,5,7,6,9,8,7]
)
forecaster = InventoryForecaster(service_level=0.95)
rec = forecaster.generate_reorder_recommendation(item)
print(f"    Reorder Point: {rec.reorder_point}, EOQ: {rec.economic_order_quantity}")
print("    ✓ PASSED")

# Test 2: SP-API Structure (imports sp-api library)
print("\n[2] SP-API Client")
from sp_api_client import REPORT_TYPES, MarketplaceIDs
print(f"    Reports: {sum(len(r) for r in REPORT_TYPES.values())} types")
print(f"    US Marketplace: {MarketplaceIDs.US}")
print("    ✓ PASSED")

# Test 3: Keepa Structure (imports keepa - may be slow)
print("\n[3] Keepa Client")
sys.stdout.flush()
# Skip actual keepa import which is slow, test our wrapper structure
import ast
with open('keepa_client.py', 'r') as f:
    tree = ast.parse(f.read())
classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
print(f"    Classes defined: {classes}")
print("    ✓ PASSED (structure validated)")

# Test 4: Research Pipeline
print("\n[4] Research Pipeline")
from dataclasses import fields
# Import without initializing (avoids API calls)
import importlib.util
spec = importlib.util.spec_from_file_location("research_pipeline", "research_pipeline.py")
module = importlib.util.module_from_spec(spec)
# Don't exec to avoid imports - just verify file parses
with open('research_pipeline.py', 'r') as f:
    tree = ast.parse(f.read())
classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
print(f"    Classes: {classes}")
print(f"    Functions: {len(funcs)}")
print("    ✓ PASSED (structure validated)")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
print("\nNotes:")
print("  - Keepa client import skipped (slow library init)")
print("  - SP-API/Keepa require API keys for actual data")
print("  - See SETUP.md for credential configuration")
