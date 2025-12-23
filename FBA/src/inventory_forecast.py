"""
Inventory Forecasting & Reorder Optimization

Uses statistical methods for demand forecasting and reorder point calculation.
Optional integration with supplychainpy for advanced analysis.
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import json

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class InventoryItem:
    """Represents an inventory item with sales data."""
    sku: str
    asin: str
    name: str
    current_stock: int
    unit_cost: float
    lead_time_days: int  # Time to receive new inventory
    sales_history: List[int]  # Daily sales history
    reorder_cost: float = 50.0  # Cost per order
    holding_cost_percent: float = 0.25  # Annual holding cost as % of unit cost


@dataclass
class ForecastResult:
    """Demand forecast results."""
    sku: str
    avg_daily_demand: float
    demand_std_dev: float
    forecast_30_day: float
    forecast_60_day: float
    forecast_90_day: float
    seasonality_factor: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    confidence: str  # 'high', 'medium', 'low'


@dataclass
class ReorderRecommendation:
    """Inventory reorder recommendation."""
    sku: str
    current_stock: int
    reorder_point: int
    safety_stock: int
    economic_order_quantity: int
    days_of_stock: float
    reorder_urgency: str  # 'critical', 'soon', 'adequate', 'overstocked'
    recommended_order_qty: int
    stockout_risk_percent: float


class InventoryForecaster:
    """
    Demand forecasting and inventory optimization.

    Methods:
    - Simple moving average
    - Exponential smoothing
    - Safety stock calculation
    - Economic Order Quantity (EOQ)
    - Reorder point optimization
    """

    def __init__(self, service_level: float = 0.95):
        """
        Initialize forecaster.

        Args:
            service_level: Target service level (0.95 = 95% in-stock rate)
        """
        self.service_level = service_level
        # Z-score for service level (standard normal distribution)
        self.z_score = self._get_z_score(service_level)

    def _get_z_score(self, service_level: float) -> float:
        """Get Z-score for target service level."""
        # Common service levels
        z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.97: 1.88,
            0.99: 2.33,
            0.999: 3.09,
        }
        # Find closest match
        closest = min(z_scores.keys(), key=lambda x: abs(x - service_level))
        return z_scores[closest]

    def _calculate_stats(self, data: List[float]) -> Tuple[float, float]:
        """Calculate mean and standard deviation."""
        if not data:
            return 0.0, 0.0

        n = len(data)
        mean = sum(data) / n

        if n < 2:
            return mean, 0.0

        variance = sum((x - mean) ** 2 for x in data) / (n - 1)
        std_dev = math.sqrt(variance)

        return mean, std_dev

    def _detect_trend(self, data: List[float], window: int = 7) -> str:
        """Detect trend in time series data."""
        if len(data) < window * 2:
            return 'stable'

        first_half = sum(data[:window]) / window
        second_half = sum(data[-window:]) / window

        change_percent = (second_half - first_half) / max(first_half, 0.01) * 100

        if change_percent > 10:
            return 'increasing'
        elif change_percent < -10:
            return 'decreasing'
        return 'stable'

    def simple_moving_average(
        self,
        sales_history: List[int],
        window: int = 7
    ) -> float:
        """
        Calculate simple moving average.

        Args:
            sales_history: Daily sales data
            window: Number of days for average

        Returns:
            Average daily demand
        """
        if not sales_history:
            return 0.0

        recent = sales_history[-window:] if len(sales_history) >= window else sales_history
        return sum(recent) / len(recent)

    def exponential_smoothing(
        self,
        sales_history: List[int],
        alpha: float = 0.3
    ) -> float:
        """
        Exponential smoothing forecast.

        More recent data has higher weight.

        Args:
            sales_history: Daily sales data
            alpha: Smoothing factor (0-1, higher = more weight on recent)

        Returns:
            Forecasted daily demand
        """
        if not sales_history:
            return 0.0

        forecast = sales_history[0]
        for actual in sales_history[1:]:
            forecast = alpha * actual + (1 - alpha) * forecast

        return forecast

    def forecast_demand(self, item: InventoryItem) -> ForecastResult:
        """
        Generate demand forecast for an item.

        Args:
            item: Inventory item with sales history

        Returns:
            ForecastResult with predictions and analysis
        """
        sales = item.sales_history

        if not sales:
            return ForecastResult(
                sku=item.sku,
                avg_daily_demand=0,
                demand_std_dev=0,
                forecast_30_day=0,
                forecast_60_day=0,
                forecast_90_day=0,
                seasonality_factor=1.0,
                trend='stable',
                confidence='low'
            )

        # Calculate statistics
        avg_demand, std_dev = self._calculate_stats([float(s) for s in sales])

        # Use exponential smoothing for forecast
        smoothed_demand = self.exponential_smoothing(sales)

        # Detect trend
        trend = self._detect_trend([float(s) for s in sales])

        # Apply trend adjustment
        trend_multiplier = 1.0
        if trend == 'increasing':
            trend_multiplier = 1.1
        elif trend == 'decreasing':
            trend_multiplier = 0.9

        # Confidence based on data quality
        confidence = 'high' if len(sales) >= 60 else 'medium' if len(sales) >= 30 else 'low'

        return ForecastResult(
            sku=item.sku,
            avg_daily_demand=round(avg_demand, 2),
            demand_std_dev=round(std_dev, 2),
            forecast_30_day=round(smoothed_demand * 30 * trend_multiplier, 0),
            forecast_60_day=round(smoothed_demand * 60 * trend_multiplier ** 2, 0),
            forecast_90_day=round(smoothed_demand * 90 * trend_multiplier ** 3, 0),
            seasonality_factor=1.0,  # Would need yearly data for true seasonality
            trend=trend,
            confidence=confidence
        )

    def calculate_safety_stock(
        self,
        demand_std_dev: float,
        lead_time_days: int
    ) -> int:
        """
        Calculate safety stock for target service level.

        Safety Stock = Z * σ * √L

        Where:
        - Z = service level z-score
        - σ = standard deviation of daily demand
        - L = lead time in days

        Args:
            demand_std_dev: Standard deviation of daily demand
            lead_time_days: Lead time to receive inventory

        Returns:
            Safety stock quantity
        """
        safety_stock = self.z_score * demand_std_dev * math.sqrt(lead_time_days)
        return max(0, int(math.ceil(safety_stock)))

    def calculate_reorder_point(
        self,
        avg_daily_demand: float,
        lead_time_days: int,
        safety_stock: int
    ) -> int:
        """
        Calculate reorder point.

        ROP = (Average Daily Demand × Lead Time) + Safety Stock

        Args:
            avg_daily_demand: Average units sold per day
            lead_time_days: Time to receive new inventory
            safety_stock: Safety stock buffer

        Returns:
            Reorder point (trigger level)
        """
        lead_time_demand = avg_daily_demand * lead_time_days
        return int(math.ceil(lead_time_demand + safety_stock))

    def calculate_eoq(
        self,
        annual_demand: float,
        order_cost: float,
        unit_cost: float,
        holding_cost_percent: float = 0.25
    ) -> int:
        """
        Calculate Economic Order Quantity (EOQ).

        EOQ = √((2 × D × S) / H)

        Where:
        - D = Annual demand
        - S = Order/setup cost
        - H = Annual holding cost per unit

        Args:
            annual_demand: Units demanded per year
            order_cost: Cost per order
            unit_cost: Cost per unit
            holding_cost_percent: Annual holding cost as % of unit cost

        Returns:
            Economic order quantity
        """
        if annual_demand <= 0 or order_cost <= 0 or unit_cost <= 0:
            return 0

        holding_cost = unit_cost * holding_cost_percent

        eoq = math.sqrt((2 * annual_demand * order_cost) / holding_cost)
        return max(1, int(round(eoq)))

    def generate_reorder_recommendation(
        self,
        item: InventoryItem
    ) -> ReorderRecommendation:
        """
        Generate complete reorder recommendation for an item.

        Args:
            item: Inventory item

        Returns:
            ReorderRecommendation with all calculations
        """
        # Get forecast
        forecast = self.forecast_demand(item)

        # Calculate safety stock
        safety_stock = self.calculate_safety_stock(
            forecast.demand_std_dev,
            item.lead_time_days
        )

        # Calculate reorder point
        reorder_point = self.calculate_reorder_point(
            forecast.avg_daily_demand,
            item.lead_time_days,
            safety_stock
        )

        # Calculate EOQ
        annual_demand = forecast.avg_daily_demand * 365
        eoq = self.calculate_eoq(
            annual_demand,
            item.reorder_cost,
            item.unit_cost,
            item.holding_cost_percent
        )

        # Days of stock remaining
        if forecast.avg_daily_demand > 0:
            days_of_stock = item.current_stock / forecast.avg_daily_demand
        else:
            days_of_stock = float('inf')

        # Determine urgency
        if item.current_stock <= safety_stock:
            urgency = 'critical'
        elif item.current_stock <= reorder_point:
            urgency = 'soon'
        elif item.current_stock > reorder_point * 2:
            urgency = 'overstocked'
        else:
            urgency = 'adequate'

        # Recommended order quantity
        if urgency in ('critical', 'soon'):
            # Order enough to reach target stock level
            target_stock = reorder_point + eoq
            recommended_qty = max(eoq, target_stock - item.current_stock)
        else:
            recommended_qty = 0

        # Stockout risk (simplified)
        if days_of_stock <= item.lead_time_days:
            stockout_risk = 90.0
        elif days_of_stock <= item.lead_time_days * 1.5:
            stockout_risk = 50.0
        elif days_of_stock <= item.lead_time_days * 2:
            stockout_risk = 20.0
        else:
            stockout_risk = 5.0

        return ReorderRecommendation(
            sku=item.sku,
            current_stock=item.current_stock,
            reorder_point=reorder_point,
            safety_stock=safety_stock,
            economic_order_quantity=eoq,
            days_of_stock=round(days_of_stock, 1),
            reorder_urgency=urgency,
            recommended_order_qty=int(recommended_qty),
            stockout_risk_percent=stockout_risk
        )

    def analyze_inventory(
        self,
        items: List[InventoryItem]
    ) -> Dict[str, Any]:
        """
        Analyze entire inventory and generate report.

        Args:
            items: List of inventory items

        Returns:
            Comprehensive inventory analysis
        """
        recommendations = []
        critical_items = []
        reorder_soon = []
        overstocked = []

        total_inventory_value = 0
        total_recommended_orders = 0

        for item in items:
            rec = self.generate_reorder_recommendation(item)
            recommendations.append(rec)

            total_inventory_value += item.current_stock * item.unit_cost

            if rec.reorder_urgency == 'critical':
                critical_items.append(rec)
            elif rec.reorder_urgency == 'soon':
                reorder_soon.append(rec)
            elif rec.reorder_urgency == 'overstocked':
                overstocked.append(rec)

            if rec.recommended_order_qty > 0:
                total_recommended_orders += rec.recommended_order_qty * item.unit_cost

        return {
            'summary': {
                'total_skus': len(items),
                'critical_count': len(critical_items),
                'reorder_soon_count': len(reorder_soon),
                'overstocked_count': len(overstocked),
                'adequate_count': len(items) - len(critical_items) - len(reorder_soon) - len(overstocked),
                'total_inventory_value': round(total_inventory_value, 2),
                'recommended_order_value': round(total_recommended_orders, 2),
            },
            'critical_items': [vars(r) for r in critical_items],
            'reorder_soon': [vars(r) for r in reorder_soon],
            'overstocked': [vars(r) for r in overstocked],
            'all_recommendations': [vars(r) for r in recommendations],
        }


def try_supplychainpy():
    """
    Optional: Use supplychainpy for advanced analysis.

    Install: pip install supplychainpy
    """
    try:
        from supplychainpy import model_inventory
        from supplychainpy.inventory.summarise import Inventory

        print("supplychainpy available for advanced analysis")
        return True
    except ImportError:
        print("supplychainpy not installed. Using built-in forecasting.")
        print("For advanced features: pip install supplychainpy")
        return False


if __name__ == "__main__":
    # Example usage
    print("Inventory Forecasting Module")
    print("=" * 50)

    # Sample item
    sample_item = InventoryItem(
        sku="TEST-001",
        asin="B08PZHWJS5",
        name="Sample Product",
        current_stock=150,
        unit_cost=25.00,
        lead_time_days=14,
        sales_history=[5, 7, 6, 8, 4, 9, 7, 6, 8, 5, 7, 6, 9, 8, 7,
                       6, 8, 9, 7, 5, 6, 8, 7, 9, 6, 8, 7, 6, 9, 8],
        reorder_cost=75.00
    )

    forecaster = InventoryForecaster(service_level=0.95)

    # Generate forecast
    forecast = forecaster.forecast_demand(sample_item)
    print(f"\nForecast for {sample_item.sku}:")
    print(f"  Avg Daily Demand: {forecast.avg_daily_demand}")
    print(f"  Std Dev: {forecast.demand_std_dev}")
    print(f"  30-Day Forecast: {forecast.forecast_30_day}")
    print(f"  Trend: {forecast.trend}")

    # Generate recommendation
    rec = forecaster.generate_reorder_recommendation(sample_item)
    print(f"\nReorder Recommendation:")
    print(f"  Current Stock: {rec.current_stock}")
    print(f"  Reorder Point: {rec.reorder_point}")
    print(f"  Safety Stock: {rec.safety_stock}")
    print(f"  EOQ: {rec.economic_order_quantity}")
    print(f"  Days of Stock: {rec.days_of_stock}")
    print(f"  Urgency: {rec.reorder_urgency}")
    print(f"  Stockout Risk: {rec.stockout_risk_percent}%")

    # Check supplychainpy
    print("\n" + "=" * 50)
    try_supplychainpy()
