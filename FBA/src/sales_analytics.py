#!/usr/bin/env python3
"""
Sales Analytics for Amazon FBA Research

Analyzes sales data to provide:
- Revenue and profit trends
- Unit economics breakdown
- Seasonality patterns
- Product performance metrics
- ROI calculations
- Growth projections
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json


@dataclass
class DailySales:
    """Single day of sales data"""
    date: datetime
    units_sold: int
    revenue: float
    orders: int
    sessions: int = 0
    page_views: int = 0
    ppc_spend: float = 0
    ppc_sales: float = 0
    refunds: int = 0
    refund_amount: float = 0


@dataclass
class ProductCosts:
    """Cost structure for a product"""
    unit_cost: float
    fba_fee: float
    referral_fee_pct: float = 0.15
    shipping_to_fba: float = 0
    other_costs: float = 0

    def total_cost_per_unit(self, selling_price: float) -> float:
        return (
            self.unit_cost +
            self.fba_fee +
            (selling_price * self.referral_fee_pct) +
            self.shipping_to_fba +
            self.other_costs
        )


@dataclass
class PeriodMetrics:
    """Metrics for a time period"""
    period: str  # "day", "week", "month"
    start_date: datetime
    end_date: datetime

    # Sales metrics
    total_revenue: float
    total_units: int
    total_orders: int
    avg_order_value: float

    # Profitability
    gross_profit: float
    net_profit: float  # After PPC
    profit_margin: float
    roi: float

    # Traffic metrics
    total_sessions: int
    conversion_rate: float
    avg_units_per_order: float

    # PPC metrics
    ppc_spend: float
    ppc_sales: float
    acos: float
    tacos: float

    # Trends
    revenue_change_pct: float = 0
    units_change_pct: float = 0


@dataclass
class ProductPerformance:
    """Performance metrics for a product"""
    asin: str
    name: str
    period_days: int

    # Volume
    total_units: int
    total_revenue: float
    avg_daily_units: float
    avg_daily_revenue: float

    # Profitability
    gross_profit: float
    net_profit: float
    profit_per_unit: float
    profit_margin: float
    roi: float

    # Efficiency
    inventory_turnover: float
    break_even_units: int

    # Growth
    velocity_trend: str  # "increasing", "decreasing", "stable"
    projected_monthly_units: int
    projected_monthly_revenue: float


@dataclass
class SeasonalityAnalysis:
    """Seasonality patterns"""
    peak_months: list[int]
    low_months: list[int]
    avg_monthly_index: dict[int, float]  # 1.0 = average
    best_month: int
    worst_month: int
    seasonality_strength: float  # 0-1, higher = more seasonal


@dataclass
class SalesReport:
    """Complete sales analytics report"""
    report_date: datetime
    period_start: datetime
    period_end: datetime
    total_days: int

    # Summary metrics
    total_revenue: float
    total_units: int
    total_orders: int
    total_profit: float
    overall_margin: float
    overall_roi: float

    # Period breakdowns
    daily_metrics: list[PeriodMetrics]
    weekly_metrics: list[PeriodMetrics]
    monthly_metrics: list[PeriodMetrics]

    # Analysis
    best_day: Optional[DailySales]
    worst_day: Optional[DailySales]
    trend: str
    growth_rate: float

    # Projections
    projected_monthly_revenue: float
    projected_monthly_profit: float


class SalesAnalyzer:
    """Analyzes Amazon FBA sales data"""

    def __init__(self):
        self.sales_data: list[DailySales] = []
        self.costs: Optional[ProductCosts] = None

    def load_sales_data(self, sales: list[DailySales], costs: ProductCosts):
        """Load sales data for analysis"""
        self.sales_data = sorted(sales, key=lambda x: x.date)
        self.costs = costs

    def calculate_period_metrics(self, sales: list[DailySales],
                                period: str,
                                prev_revenue: float = 0,
                                prev_units: int = 0) -> PeriodMetrics:
        """Calculate metrics for a time period"""
        if not sales or not self.costs:
            raise ValueError("No sales data loaded")

        total_revenue = sum(s.revenue for s in sales)
        total_units = sum(s.units_sold for s in sales)
        total_orders = sum(s.orders for s in sales)
        total_sessions = sum(s.sessions for s in sales)
        ppc_spend = sum(s.ppc_spend for s in sales)
        ppc_sales = sum(s.ppc_sales for s in sales)
        refunds = sum(s.refund_amount for s in sales)

        # Average order value
        avg_order = total_revenue / max(1, total_orders)

        # Selling price assumption
        avg_price = total_revenue / max(1, total_units)

        # Profitability
        cost_per_unit = self.costs.total_cost_per_unit(avg_price)
        gross_profit = total_revenue - (total_units * cost_per_unit) - refunds
        net_profit = gross_profit - ppc_spend

        profit_margin = (net_profit / max(1, total_revenue)) * 100
        total_investment = total_units * self.costs.unit_cost + ppc_spend
        roi = (net_profit / max(1, total_investment)) * 100

        # Conversion
        conversion = (total_orders / max(1, total_sessions)) * 100 if total_sessions else 0
        units_per_order = total_units / max(1, total_orders)

        # ACOS and TACOS
        acos = (ppc_spend / max(1, ppc_sales)) * 100 if ppc_sales else 0
        tacos = (ppc_spend / max(1, total_revenue)) * 100

        # Changes
        rev_change = ((total_revenue - prev_revenue) / max(1, prev_revenue)) * 100 if prev_revenue else 0
        unit_change = ((total_units - prev_units) / max(1, prev_units)) * 100 if prev_units else 0

        return PeriodMetrics(
            period=period,
            start_date=sales[0].date,
            end_date=sales[-1].date,
            total_revenue=round(total_revenue, 2),
            total_units=total_units,
            total_orders=total_orders,
            avg_order_value=round(avg_order, 2),
            gross_profit=round(gross_profit, 2),
            net_profit=round(net_profit, 2),
            profit_margin=round(profit_margin, 1),
            roi=round(roi, 1),
            total_sessions=total_sessions,
            conversion_rate=round(conversion, 2),
            avg_units_per_order=round(units_per_order, 2),
            ppc_spend=round(ppc_spend, 2),
            ppc_sales=round(ppc_sales, 2),
            acos=round(acos, 1),
            tacos=round(tacos, 1),
            revenue_change_pct=round(rev_change, 1),
            units_change_pct=round(unit_change, 1)
        )

    def analyze_product_performance(self, asin: str, name: str,
                                   current_inventory: int = 0) -> ProductPerformance:
        """Analyze product performance"""
        if not self.sales_data or not self.costs:
            raise ValueError("No sales data loaded")

        days = (self.sales_data[-1].date - self.sales_data[0].date).days + 1
        total_units = sum(s.units_sold for s in self.sales_data)
        total_revenue = sum(s.revenue for s in self.sales_data)
        ppc_spend = sum(s.ppc_spend for s in self.sales_data)
        refunds = sum(s.refund_amount for s in self.sales_data)

        avg_price = total_revenue / max(1, total_units)
        cost_per_unit = self.costs.total_cost_per_unit(avg_price)

        gross_profit = total_revenue - (total_units * cost_per_unit) - refunds
        net_profit = gross_profit - ppc_spend
        profit_per_unit = net_profit / max(1, total_units)
        profit_margin = (net_profit / max(1, total_revenue)) * 100

        investment = total_units * self.costs.unit_cost + ppc_spend
        roi = (net_profit / max(1, investment)) * 100

        # Velocity trend
        if days >= 14:
            first_half = self.sales_data[:len(self.sales_data)//2]
            second_half = self.sales_data[len(self.sales_data)//2:]
            first_avg = sum(s.units_sold for s in first_half) / len(first_half)
            second_avg = sum(s.units_sold for s in second_half) / len(second_half)

            if second_avg > first_avg * 1.1:
                velocity_trend = "increasing"
            elif second_avg < first_avg * 0.9:
                velocity_trend = "decreasing"
            else:
                velocity_trend = "stable"
        else:
            velocity_trend = "stable"

        avg_daily = total_units / days
        inventory_turnover = (avg_daily * 365) / max(1, current_inventory) if current_inventory else 0

        # Break even
        fixed_costs = ppc_spend / days * 30  # Monthly PPC
        contribution = avg_price - cost_per_unit
        break_even = int(fixed_costs / max(0.01, contribution)) if contribution > 0 else 9999

        return ProductPerformance(
            asin=asin,
            name=name,
            period_days=days,
            total_units=total_units,
            total_revenue=round(total_revenue, 2),
            avg_daily_units=round(avg_daily, 1),
            avg_daily_revenue=round(total_revenue / days, 2),
            gross_profit=round(gross_profit, 2),
            net_profit=round(net_profit, 2),
            profit_per_unit=round(profit_per_unit, 2),
            profit_margin=round(profit_margin, 1),
            roi=round(roi, 1),
            inventory_turnover=round(inventory_turnover, 1),
            break_even_units=break_even,
            velocity_trend=velocity_trend,
            projected_monthly_units=int(avg_daily * 30),
            projected_monthly_revenue=round(avg_daily * 30 * avg_price, 2)
        )

    def analyze_seasonality(self, monthly_data: dict[int, float]) -> SeasonalityAnalysis:
        """Analyze seasonal patterns"""
        if not monthly_data:
            raise ValueError("No monthly data provided")

        avg_value = sum(monthly_data.values()) / len(monthly_data)

        # Calculate index for each month
        monthly_index = {
            month: value / avg_value
            for month, value in monthly_data.items()
        }

        # Find peaks and lows
        sorted_months = sorted(monthly_data.items(), key=lambda x: x[1], reverse=True)
        peak_months = [m for m, v in sorted_months if v > avg_value * 1.15][:3]
        low_months = [m for m, v in sorted_months if v < avg_value * 0.85][-3:]

        best_month = sorted_months[0][0]
        worst_month = sorted_months[-1][0]

        # Seasonality strength (coefficient of variation)
        variance = sum((v - avg_value) ** 2 for v in monthly_data.values()) / len(monthly_data)
        std_dev = variance ** 0.5
        seasonality_strength = min(1.0, std_dev / avg_value)

        return SeasonalityAnalysis(
            peak_months=peak_months,
            low_months=low_months,
            avg_monthly_index=monthly_index,
            best_month=best_month,
            worst_month=worst_month,
            seasonality_strength=round(seasonality_strength, 2)
        )

    def generate_report(self) -> SalesReport:
        """Generate comprehensive sales report"""
        if not self.sales_data or not self.costs:
            raise ValueError("No sales data loaded")

        # Group by periods
        daily_groups = defaultdict(list)
        weekly_groups = defaultdict(list)
        monthly_groups = defaultdict(list)

        for sale in self.sales_data:
            daily_groups[sale.date.date()].append(sale)
            week_key = sale.date.isocalendar()[:2]
            weekly_groups[week_key].append(sale)
            month_key = (sale.date.year, sale.date.month)
            monthly_groups[month_key].append(sale)

        # Calculate period metrics
        daily_metrics = []
        prev_rev, prev_units = 0, 0
        for date_key in sorted(daily_groups.keys()):
            sales = daily_groups[date_key]
            metrics = self.calculate_period_metrics(sales, "day", prev_rev, prev_units)
            daily_metrics.append(metrics)
            prev_rev = metrics.total_revenue
            prev_units = metrics.total_units

        weekly_metrics = []
        prev_rev, prev_units = 0, 0
        for week_key in sorted(weekly_groups.keys()):
            sales = weekly_groups[week_key]
            metrics = self.calculate_period_metrics(sales, "week", prev_rev, prev_units)
            weekly_metrics.append(metrics)
            prev_rev = metrics.total_revenue
            prev_units = metrics.total_units

        monthly_metrics = []
        prev_rev, prev_units = 0, 0
        for month_key in sorted(monthly_groups.keys()):
            sales = monthly_groups[month_key]
            metrics = self.calculate_period_metrics(sales, "month", prev_rev, prev_units)
            monthly_metrics.append(metrics)
            prev_rev = metrics.total_revenue
            prev_units = metrics.total_units

        # Overall metrics
        overall = self.calculate_period_metrics(self.sales_data, "all")

        # Best/worst days
        best_day = max(self.sales_data, key=lambda x: x.revenue)
        worst_day = min(self.sales_data, key=lambda x: x.revenue)

        # Trend analysis
        if len(weekly_metrics) >= 2:
            recent = sum(m.total_revenue for m in weekly_metrics[-2:])
            older = sum(m.total_revenue for m in weekly_metrics[:2])
            if recent > older * 1.1:
                trend = "growing"
            elif recent < older * 0.9:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # Growth rate
        if len(monthly_metrics) >= 2:
            growth = (monthly_metrics[-1].total_revenue - monthly_metrics[0].total_revenue) / max(1, monthly_metrics[0].total_revenue) * 100
        else:
            growth = 0

        days = (self.sales_data[-1].date - self.sales_data[0].date).days + 1
        daily_avg_rev = overall.total_revenue / days
        daily_avg_profit = overall.net_profit / days

        return SalesReport(
            report_date=datetime.now(),
            period_start=self.sales_data[0].date,
            period_end=self.sales_data[-1].date,
            total_days=days,
            total_revenue=overall.total_revenue,
            total_units=overall.total_units,
            total_orders=overall.total_orders,
            total_profit=overall.net_profit,
            overall_margin=overall.profit_margin,
            overall_roi=overall.roi,
            daily_metrics=daily_metrics,
            weekly_metrics=weekly_metrics,
            monthly_metrics=monthly_metrics,
            best_day=best_day,
            worst_day=worst_day,
            trend=trend,
            growth_rate=round(growth, 1),
            projected_monthly_revenue=round(daily_avg_rev * 30, 2),
            projected_monthly_profit=round(daily_avg_profit * 30, 2)
        )

    def calculate_unit_economics(self, selling_price: float) -> dict:
        """Calculate detailed unit economics"""
        if not self.costs:
            raise ValueError("No cost data loaded")

        referral_fee = selling_price * self.costs.referral_fee_pct
        total_fees = self.costs.fba_fee + referral_fee
        total_cost = self.costs.total_cost_per_unit(selling_price)
        gross_profit = selling_price - total_cost
        margin = (gross_profit / selling_price) * 100

        return {
            "selling_price": selling_price,
            "breakdown": {
                "unit_cost": self.costs.unit_cost,
                "fba_fee": self.costs.fba_fee,
                "referral_fee": round(referral_fee, 2),
                "shipping_to_fba": self.costs.shipping_to_fba,
                "other_costs": self.costs.other_costs,
                "total_cost": round(total_cost, 2),
            },
            "profit": {
                "gross_profit": round(gross_profit, 2),
                "margin_pct": round(margin, 1),
                "roi_per_unit": round((gross_profit / self.costs.unit_cost) * 100, 1),
            },
            "targets": {
                "min_price_20_margin": round(total_cost / 0.80, 2),
                "min_price_30_margin": round(total_cost / 0.70, 2),
                "break_even_price": round(total_cost, 2),
            }
        }


def create_sample_sales_data() -> tuple[list[DailySales], ProductCosts]:
    """Create sample sales data for testing"""
    base_date = datetime(2024, 1, 1)
    sales = []

    for i in range(90):  # 90 days
        date = base_date + timedelta(days=i)

        # Simulate growing sales with weekday patterns
        base_units = 8 + (i // 30) * 2  # Growth over time
        weekday_mult = 1.2 if date.weekday() < 5 else 0.8  # Lower on weekends
        units = int(base_units * weekday_mult + (i % 7) - 3)
        units = max(1, units)

        price = 29.99
        revenue = units * price
        orders = max(1, units - (units // 5))  # Some multi-unit orders
        sessions = units * 15 + 50  # ~7% conversion

        # PPC spend (decreasing ratio over time)
        ppc_ratio = 0.30 - (i / 300)  # Starts at 30%, decreases
        ppc_spend = revenue * max(0.10, ppc_ratio)
        ppc_sales = ppc_spend / 0.25  # 25% ACOS

        sales.append(DailySales(
            date=date,
            units_sold=units,
            revenue=round(revenue, 2),
            orders=orders,
            sessions=sessions,
            page_views=sessions * 2,
            ppc_spend=round(ppc_spend, 2),
            ppc_sales=round(min(ppc_sales, revenue * 0.6), 2),
            refunds=1 if i % 15 == 0 else 0,
            refund_amount=price if i % 15 == 0 else 0
        ))

    costs = ProductCosts(
        unit_cost=8.00,
        fba_fee=5.50,
        referral_fee_pct=0.15,
        shipping_to_fba=0.50,
        other_costs=0.25
    )

    return sales, costs


if __name__ == "__main__":
    print("=" * 70)
    print("SALES ANALYTICS")
    print("=" * 70)

    sales_data, costs = create_sample_sales_data()

    analyzer = SalesAnalyzer()
    analyzer.load_sales_data(sales_data, costs)

    # Generate report
    report = analyzer.generate_report()

    print(f"\nPeriod: {report.period_start.date()} to {report.period_end.date()} ({report.total_days} days)")

    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  Total Revenue:    ${report.total_revenue:,.2f}")
    print(f"  Total Units:      {report.total_units:,}")
    print(f"  Total Orders:     {report.total_orders:,}")
    print(f"  Total Profit:     ${report.total_profit:,.2f}")
    print(f"  Profit Margin:    {report.overall_margin:.1f}%")
    print(f"  ROI:              {report.overall_roi:.1f}%")
    print(f"  Trend:            {report.trend}")
    print(f"  Growth Rate:      {report.growth_rate:+.1f}%")

    print("\n" + "-" * 70)
    print("MONTHLY BREAKDOWN")
    print("-" * 70)
    for m in report.monthly_metrics:
        print(f"  {m.start_date.strftime('%b %Y')}: "
              f"${m.total_revenue:,.0f} rev | "
              f"{m.total_units} units | "
              f"${m.net_profit:,.0f} profit | "
              f"{m.acos:.0f}% ACOS")

    print("\n" + "-" * 70)
    print("WEEKLY PERFORMANCE (Last 4 weeks)")
    print("-" * 70)
    for m in report.weekly_metrics[-4:]:
        print(f"  Week of {m.start_date.strftime('%m/%d')}: "
              f"${m.total_revenue:,.0f} | "
              f"{m.total_units} units | "
              f"{m.revenue_change_pct:+.0f}% vs prev")

    # Unit economics
    print("\n" + "-" * 70)
    print("UNIT ECONOMICS @ $29.99")
    print("-" * 70)
    econ = analyzer.calculate_unit_economics(29.99)
    print(f"  Unit Cost:        ${econ['breakdown']['unit_cost']:.2f}")
    print(f"  FBA Fee:          ${econ['breakdown']['fba_fee']:.2f}")
    print(f"  Referral Fee:     ${econ['breakdown']['referral_fee']:.2f}")
    print(f"  Total Cost:       ${econ['breakdown']['total_cost']:.2f}")
    print(f"  Gross Profit:     ${econ['profit']['gross_profit']:.2f}")
    print(f"  Margin:           {econ['profit']['margin_pct']:.1f}%")
    print(f"  Break-even:       ${econ['targets']['break_even_price']:.2f}")

    # Product performance
    print("\n" + "-" * 70)
    print("PRODUCT PERFORMANCE")
    print("-" * 70)
    perf = analyzer.analyze_product_performance("B0TEST123", "Test Product", 200)
    print(f"  Avg Daily Units:  {perf.avg_daily_units:.1f}")
    print(f"  Avg Daily Rev:    ${perf.avg_daily_revenue:.2f}")
    print(f"  Profit/Unit:      ${perf.profit_per_unit:.2f}")
    print(f"  Velocity Trend:   {perf.velocity_trend}")
    print(f"  Projected Mo Rev: ${perf.projected_monthly_revenue:,.2f}")

    # Best/worst days
    print("\n" + "-" * 70)
    print("NOTABLE DAYS")
    print("-" * 70)
    print(f"  Best Day:  {report.best_day.date.strftime('%Y-%m-%d')} - "
          f"${report.best_day.revenue:.2f} ({report.best_day.units_sold} units)")
    print(f"  Worst Day: {report.worst_day.date.strftime('%Y-%m-%d')} - "
          f"${report.worst_day.revenue:.2f} ({report.worst_day.units_sold} units)")

    # Projections
    print("\n" + "-" * 70)
    print("30-DAY PROJECTIONS")
    print("-" * 70)
    print(f"  Revenue:          ${report.projected_monthly_revenue:,.2f}")
    print(f"  Profit:           ${report.projected_monthly_profit:,.2f}")

    print("\n" + "=" * 70)
