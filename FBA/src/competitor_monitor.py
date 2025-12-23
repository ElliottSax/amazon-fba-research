#!/usr/bin/env python3
"""
Competitor Monitoring System for Amazon FBA Research

Tracks competitor products over time:
- Price changes and history
- BSR (Best Seller Rank) trends
- Review velocity and rating changes
- Listing changes (title, bullets, images)
- Stock availability / out-of-stock events
- Advertising activity detection
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import json


class AlertType(Enum):
    PRICE_DROP = "price_drop"
    PRICE_INCREASE = "price_increase"
    BSR_IMPROVEMENT = "bsr_improvement"
    BSR_DECLINE = "bsr_decline"
    REVIEW_SPIKE = "review_spike"
    RATING_CHANGE = "rating_change"
    OUT_OF_STOCK = "out_of_stock"
    BACK_IN_STOCK = "back_in_stock"
    LISTING_CHANGE = "listing_change"
    NEW_COMPETITOR = "new_competitor"
    COMPETITOR_EXIT = "competitor_exit"
    AD_DETECTED = "ad_detected"


class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProductSnapshot:
    """Point-in-time snapshot of product data"""
    asin: str
    timestamp: datetime
    price: float
    bsr: int
    rating: float
    review_count: int
    is_in_stock: bool
    title: Optional[str] = None
    bullet_points: Optional[list[str]] = None
    image_count: Optional[int] = None
    seller_count: Optional[int] = None
    is_sponsored: bool = False
    buy_box_winner: Optional[str] = None


@dataclass
class CompetitorAlert:
    """Alert for significant competitor change"""
    alert_id: str
    asin: str
    alert_type: AlertType
    priority: AlertPriority
    timestamp: datetime
    message: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    change_pct: Optional[float] = None
    recommendation: Optional[str] = None


@dataclass
class PriceHistory:
    """Price tracking over time"""
    asin: str
    current_price: float
    min_price: float
    max_price: float
    avg_price: float
    price_points: list[tuple[datetime, float]] = field(default_factory=list)

    @property
    def volatility(self) -> float:
        """Price volatility (std dev / mean)"""
        if not self.price_points or self.avg_price == 0:
            return 0
        variance = sum((p - self.avg_price) ** 2 for _, p in self.price_points) / len(self.price_points)
        return (variance ** 0.5) / self.avg_price


@dataclass
class BSRHistory:
    """BSR tracking over time"""
    asin: str
    current_bsr: int
    best_bsr: int
    worst_bsr: int
    avg_bsr: float
    bsr_points: list[tuple[datetime, int]] = field(default_factory=list)
    trend: str = "stable"  # improving, declining, stable, volatile

    @property
    def improvement_rate(self) -> float:
        """Weekly BSR improvement rate (negative = improving)"""
        if len(self.bsr_points) < 2:
            return 0
        first_bsr = self.bsr_points[0][1]
        last_bsr = self.bsr_points[-1][1]
        days = (self.bsr_points[-1][0] - self.bsr_points[0][0]).days
        if days == 0:
            return 0
        return ((last_bsr - first_bsr) / first_bsr) * (7 / days) * 100


@dataclass
class CompetitorProfile:
    """Complete competitor profile"""
    asin: str
    title: str
    brand: Optional[str]
    current_price: float
    current_bsr: int
    current_rating: float
    review_count: int

    price_history: PriceHistory
    bsr_history: BSRHistory

    first_seen: datetime
    last_updated: datetime

    listing_changes: int = 0
    out_of_stock_events: int = 0
    review_velocity: float = 0  # reviews per month

    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    threat_level: str = "medium"  # low, medium, high


@dataclass
class MarketPosition:
    """Your product's position relative to competitors"""
    your_asin: str
    price_rank: int  # 1 = cheapest
    bsr_rank: int  # 1 = best selling
    rating_rank: int  # 1 = highest rated
    review_rank: int  # 1 = most reviews
    total_competitors: int

    price_vs_avg: float  # % above/below average
    bsr_vs_avg: float
    rating_vs_avg: float
    reviews_vs_avg: float


class CompetitorMonitor:
    """Monitors competitor products and generates alerts"""

    # Alert thresholds
    PRICE_CHANGE_THRESHOLD = 0.05  # 5% price change
    BSR_CHANGE_THRESHOLD = 0.20  # 20% BSR change
    REVIEW_SPIKE_THRESHOLD = 10  # 10+ new reviews in a day

    def __init__(self):
        self.snapshots: dict[str, list[ProductSnapshot]] = defaultdict(list)
        self.alerts: list[CompetitorAlert] = []
        self.profiles: dict[str, CompetitorProfile] = {}
        self._alert_counter = 0

    def record_snapshot(self, snapshot: ProductSnapshot) -> list[CompetitorAlert]:
        """Record a product snapshot and check for alerts"""
        asin = snapshot.asin
        self.snapshots[asin].append(snapshot)

        new_alerts = []

        # Check for alerts if we have history
        if len(self.snapshots[asin]) > 1:
            prev = self.snapshots[asin][-2]
            new_alerts = self._check_for_changes(prev, snapshot)

        # Update profile
        self._update_profile(asin)

        return new_alerts

    def _check_for_changes(self, old: ProductSnapshot, new: ProductSnapshot) -> list[CompetitorAlert]:
        """Compare snapshots and generate alerts"""
        alerts = []

        # Price changes
        if old.price > 0:
            price_change = (new.price - old.price) / old.price
            if abs(price_change) >= self.PRICE_CHANGE_THRESHOLD:
                alert_type = AlertType.PRICE_DROP if price_change < 0 else AlertType.PRICE_INCREASE
                priority = AlertPriority.HIGH if abs(price_change) > 0.15 else AlertPriority.MEDIUM

                alerts.append(self._create_alert(
                    asin=new.asin,
                    alert_type=alert_type,
                    priority=priority,
                    message=f"Price {'dropped' if price_change < 0 else 'increased'} by {abs(price_change)*100:.1f}%",
                    old_value=f"${old.price:.2f}",
                    new_value=f"${new.price:.2f}",
                    change_pct=price_change * 100,
                    recommendation=self._get_price_recommendation(alert_type, price_change)
                ))

        # BSR changes
        if old.bsr > 0:
            bsr_change = (new.bsr - old.bsr) / old.bsr
            if abs(bsr_change) >= self.BSR_CHANGE_THRESHOLD:
                # Note: lower BSR is better, so improvement means decrease
                alert_type = AlertType.BSR_IMPROVEMENT if bsr_change < 0 else AlertType.BSR_DECLINE
                priority = AlertPriority.HIGH if abs(bsr_change) > 0.40 else AlertPriority.MEDIUM

                alerts.append(self._create_alert(
                    asin=new.asin,
                    alert_type=alert_type,
                    priority=priority,
                    message=f"BSR {'improved' if bsr_change < 0 else 'declined'} by {abs(bsr_change)*100:.1f}%",
                    old_value=f"#{old.bsr:,}",
                    new_value=f"#{new.bsr:,}",
                    change_pct=bsr_change * 100,
                    recommendation=self._get_bsr_recommendation(alert_type, bsr_change)
                ))

        # Review velocity
        review_diff = new.review_count - old.review_count
        if review_diff >= self.REVIEW_SPIKE_THRESHOLD:
            alerts.append(self._create_alert(
                asin=new.asin,
                alert_type=AlertType.REVIEW_SPIKE,
                priority=AlertPriority.HIGH,
                message=f"Review spike: +{review_diff} new reviews",
                old_value=str(old.review_count),
                new_value=str(new.review_count),
                recommendation="Monitor for potential review manipulation or viral product"
            ))

        # Rating changes
        rating_change = new.rating - old.rating
        if abs(rating_change) >= 0.2:
            alerts.append(self._create_alert(
                asin=new.asin,
                alert_type=AlertType.RATING_CHANGE,
                priority=AlertPriority.MEDIUM,
                message=f"Rating {'increased' if rating_change > 0 else 'decreased'} from {old.rating:.1f} to {new.rating:.1f}",
                old_value=f"{old.rating:.1f}",
                new_value=f"{new.rating:.1f}",
                change_pct=rating_change / old.rating * 100 if old.rating > 0 else 0
            ))

        # Stock status
        if old.is_in_stock and not new.is_in_stock:
            alerts.append(self._create_alert(
                asin=new.asin,
                alert_type=AlertType.OUT_OF_STOCK,
                priority=AlertPriority.HIGH,
                message="Competitor went out of stock",
                recommendation="Opportunity to capture their sales - consider increasing ad spend"
            ))
        elif not old.is_in_stock and new.is_in_stock:
            alerts.append(self._create_alert(
                asin=new.asin,
                alert_type=AlertType.BACK_IN_STOCK,
                priority=AlertPriority.MEDIUM,
                message="Competitor is back in stock",
                recommendation="Monitor for potential price wars or increased competition"
            ))

        # Sponsored/Ad detection
        if not old.is_sponsored and new.is_sponsored:
            alerts.append(self._create_alert(
                asin=new.asin,
                alert_type=AlertType.AD_DETECTED,
                priority=AlertPriority.LOW,
                message="Competitor started running sponsored ads",
                recommendation="Review your ad strategy and keyword targeting"
            ))

        # Listing changes
        if old.title and new.title and old.title != new.title:
            alerts.append(self._create_alert(
                asin=new.asin,
                alert_type=AlertType.LISTING_CHANGE,
                priority=AlertPriority.LOW,
                message="Competitor changed their listing title",
                old_value=old.title[:50] + "..." if len(old.title) > 50 else old.title,
                new_value=new.title[:50] + "..." if len(new.title) > 50 else new.title,
                recommendation="Analyze new keywords in their title"
            ))

        self.alerts.extend(alerts)
        return alerts

    def _create_alert(self, asin: str, alert_type: AlertType, priority: AlertPriority,
                     message: str, old_value: str = None, new_value: str = None,
                     change_pct: float = None, recommendation: str = None) -> CompetitorAlert:
        """Create a new alert"""
        self._alert_counter += 1
        return CompetitorAlert(
            alert_id=f"ALT-{self._alert_counter:04d}",
            asin=asin,
            alert_type=alert_type,
            priority=priority,
            timestamp=datetime.now(),
            message=message,
            old_value=old_value,
            new_value=new_value,
            change_pct=change_pct,
            recommendation=recommendation
        )

    def _get_price_recommendation(self, alert_type: AlertType, change_pct: float) -> str:
        """Get recommendation for price change"""
        if alert_type == AlertType.PRICE_DROP:
            if change_pct < -0.15:
                return "Major price drop - evaluate if you need to respond or if competitor is clearing inventory"
            else:
                return "Monitor competitor's sales velocity - may need to adjust pricing strategy"
        else:
            return "Competitor raised prices - potential opportunity to capture price-sensitive customers"

    def _get_bsr_recommendation(self, alert_type: AlertType, change_pct: float) -> str:
        """Get recommendation for BSR change"""
        if alert_type == AlertType.BSR_IMPROVEMENT:
            if change_pct < -0.40:
                return "Significant competitor improvement - analyze their recent changes (price, listing, ads)"
            else:
                return "Competitor gaining traction - monitor their strategy"
        else:
            return "Competitor losing ground - potential opportunity to gain market share"

    def _update_profile(self, asin: str):
        """Update competitor profile from snapshots"""
        if asin not in self.snapshots or not self.snapshots[asin]:
            return

        snaps = self.snapshots[asin]
        latest = snaps[-1]
        first = snaps[0]

        # Build price history
        prices = [(s.timestamp, s.price) for s in snaps if s.price > 0]
        price_values = [p for _, p in prices]
        price_history = PriceHistory(
            asin=asin,
            current_price=latest.price,
            min_price=min(price_values) if price_values else 0,
            max_price=max(price_values) if price_values else 0,
            avg_price=sum(price_values) / len(price_values) if price_values else 0,
            price_points=prices
        )

        # Build BSR history
        bsrs = [(s.timestamp, s.bsr) for s in snaps if s.bsr > 0]
        bsr_values = [b for _, b in bsrs]
        bsr_history = BSRHistory(
            asin=asin,
            current_bsr=latest.bsr,
            best_bsr=min(bsr_values) if bsr_values else 0,
            worst_bsr=max(bsr_values) if bsr_values else 0,
            avg_bsr=sum(bsr_values) / len(bsr_values) if bsr_values else 0,
            bsr_points=bsrs,
            trend=self._calculate_bsr_trend(bsr_values)
        )

        # Calculate review velocity
        days = (latest.timestamp - first.timestamp).days
        review_growth = latest.review_count - first.review_count
        review_velocity = (review_growth / max(1, days)) * 30  # reviews per month

        # Count out of stock events
        oos_events = sum(1 for i in range(1, len(snaps))
                        if snaps[i-1].is_in_stock and not snaps[i].is_in_stock)

        # Determine threat level
        threat_level = self._calculate_threat_level(latest, bsr_history, review_velocity)

        # Identify strengths and weaknesses
        strengths, weaknesses = self._analyze_competitor_sw(latest, price_history, bsr_history)

        self.profiles[asin] = CompetitorProfile(
            asin=asin,
            title=latest.title or "Unknown",
            brand=self._extract_brand(latest.title),
            current_price=latest.price,
            current_bsr=latest.bsr,
            current_rating=latest.rating,
            review_count=latest.review_count,
            price_history=price_history,
            bsr_history=bsr_history,
            first_seen=first.timestamp,
            last_updated=latest.timestamp,
            listing_changes=self._count_listing_changes(snaps),
            out_of_stock_events=oos_events,
            review_velocity=review_velocity,
            strengths=strengths,
            weaknesses=weaknesses,
            threat_level=threat_level
        )

    def _calculate_bsr_trend(self, bsr_values: list[int]) -> str:
        """Determine BSR trend"""
        if len(bsr_values) < 3:
            return "stable"

        first_third = sum(bsr_values[:len(bsr_values)//3]) / (len(bsr_values)//3)
        last_third = sum(bsr_values[-len(bsr_values)//3:]) / (len(bsr_values)//3)

        change = (last_third - first_third) / first_third if first_third > 0 else 0

        if change < -0.20:
            return "improving"
        elif change > 0.20:
            return "declining"
        else:
            return "stable"

    def _calculate_threat_level(self, snapshot: ProductSnapshot,
                               bsr_history: BSRHistory,
                               review_velocity: float) -> str:
        """Calculate competitor threat level"""
        threat_score = 0

        # Good BSR
        if snapshot.bsr < 5000:
            threat_score += 3
        elif snapshot.bsr < 20000:
            threat_score += 2
        elif snapshot.bsr < 50000:
            threat_score += 1

        # Improving BSR
        if bsr_history.trend == "improving":
            threat_score += 2

        # High review velocity
        if review_velocity > 50:
            threat_score += 2
        elif review_velocity > 20:
            threat_score += 1

        # Good rating
        if snapshot.rating >= 4.5:
            threat_score += 2
        elif snapshot.rating >= 4.0:
            threat_score += 1

        # Many reviews
        if snapshot.review_count > 1000:
            threat_score += 2
        elif snapshot.review_count > 500:
            threat_score += 1

        if threat_score >= 8:
            return "high"
        elif threat_score >= 4:
            return "medium"
        else:
            return "low"

    def _analyze_competitor_sw(self, snapshot: ProductSnapshot,
                               price_history: PriceHistory,
                               bsr_history: BSRHistory) -> tuple[list[str], list[str]]:
        """Analyze competitor strengths and weaknesses"""
        strengths = []
        weaknesses = []

        # Rating
        if snapshot.rating >= 4.5:
            strengths.append(f"High rating ({snapshot.rating:.1f})")
        elif snapshot.rating < 3.5:
            weaknesses.append(f"Low rating ({snapshot.rating:.1f})")

        # Reviews
        if snapshot.review_count > 1000:
            strengths.append(f"Social proof ({snapshot.review_count:,} reviews)")
        elif snapshot.review_count < 50:
            weaknesses.append(f"Limited reviews ({snapshot.review_count})")

        # BSR
        if snapshot.bsr < 10000:
            strengths.append(f"Strong BSR (#{snapshot.bsr:,})")
        elif snapshot.bsr > 100000:
            weaknesses.append(f"Weak BSR (#{snapshot.bsr:,})")

        # BSR trend
        if bsr_history.trend == "improving":
            strengths.append("BSR improving")
        elif bsr_history.trend == "declining":
            weaknesses.append("BSR declining")

        # Price stability
        if price_history.volatility > 0.2:
            weaknesses.append("Volatile pricing")
        elif price_history.volatility < 0.05:
            strengths.append("Stable pricing")

        return strengths, weaknesses

    def _extract_brand(self, title: str) -> Optional[str]:
        """Extract brand from title"""
        if not title:
            return None
        # Usually first word or first few words before a common separator
        parts = title.split(" - ")
        if len(parts) > 1:
            return parts[0].strip()
        parts = title.split(",")
        if len(parts) > 1:
            return parts[0].strip()
        # Return first 2 words as potential brand
        words = title.split()[:2]
        return " ".join(words) if words else None

    def _count_listing_changes(self, snapshots: list[ProductSnapshot]) -> int:
        """Count how many times listing was changed"""
        changes = 0
        for i in range(1, len(snapshots)):
            if snapshots[i].title != snapshots[i-1].title:
                changes += 1
            if snapshots[i].bullet_points != snapshots[i-1].bullet_points:
                changes += 1
        return changes

    def calculate_market_position(self, your_asin: str) -> Optional[MarketPosition]:
        """Calculate your position relative to competitors"""
        if your_asin not in self.profiles:
            return None

        your_profile = self.profiles[your_asin]
        competitors = [p for a, p in self.profiles.items() if a != your_asin]

        if not competitors:
            return None

        # Calculate rankings
        all_products = competitors + [your_profile]

        prices = sorted(all_products, key=lambda x: x.current_price)
        bsrs = sorted(all_products, key=lambda x: x.current_bsr)
        ratings = sorted(all_products, key=lambda x: x.current_rating, reverse=True)
        reviews = sorted(all_products, key=lambda x: x.review_count, reverse=True)

        price_rank = next(i+1 for i, p in enumerate(prices) if p.asin == your_asin)
        bsr_rank = next(i+1 for i, p in enumerate(bsrs) if p.asin == your_asin)
        rating_rank = next(i+1 for i, p in enumerate(ratings) if p.asin == your_asin)
        review_rank = next(i+1 for i, p in enumerate(reviews) if p.asin == your_asin)

        # Calculate vs averages
        avg_price = sum(p.current_price for p in competitors) / len(competitors)
        avg_bsr = sum(p.current_bsr for p in competitors) / len(competitors)
        avg_rating = sum(p.current_rating for p in competitors) / len(competitors)
        avg_reviews = sum(p.review_count for p in competitors) / len(competitors)

        return MarketPosition(
            your_asin=your_asin,
            price_rank=price_rank,
            bsr_rank=bsr_rank,
            rating_rank=rating_rank,
            review_rank=review_rank,
            total_competitors=len(competitors),
            price_vs_avg=(your_profile.current_price - avg_price) / avg_price * 100 if avg_price > 0 else 0,
            bsr_vs_avg=(your_profile.current_bsr - avg_bsr) / avg_bsr * 100 if avg_bsr > 0 else 0,
            rating_vs_avg=(your_profile.current_rating - avg_rating) / avg_rating * 100 if avg_rating > 0 else 0,
            reviews_vs_avg=(your_profile.review_count - avg_reviews) / avg_reviews * 100 if avg_reviews > 0 else 0
        )

    def get_high_priority_alerts(self, limit: int = 10) -> list[CompetitorAlert]:
        """Get most recent high priority alerts"""
        high_priority = [a for a in self.alerts
                        if a.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL]]
        return sorted(high_priority, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_threat_summary(self) -> dict:
        """Summarize competitor threat levels"""
        threats = {"high": [], "medium": [], "low": []}
        for asin, profile in self.profiles.items():
            threats[profile.threat_level].append({
                "asin": asin,
                "title": profile.title[:40] + "..." if len(profile.title) > 40 else profile.title,
                "bsr": profile.current_bsr,
                "rating": profile.current_rating
            })
        return threats

    def export_data(self, filepath: str):
        """Export monitoring data to JSON"""
        data = {
            "profiles": {asin: {
                "asin": p.asin,
                "title": p.title,
                "brand": p.brand,
                "price": p.current_price,
                "bsr": p.current_bsr,
                "rating": p.current_rating,
                "reviews": p.review_count,
                "threat_level": p.threat_level,
                "strengths": p.strengths,
                "weaknesses": p.weaknesses
            } for asin, p in self.profiles.items()},
            "alerts": [{
                "id": a.alert_id,
                "asin": a.asin,
                "type": a.alert_type.value,
                "priority": a.priority.value,
                "message": a.message,
                "timestamp": a.timestamp.isoformat()
            } for a in self.alerts[-100:]]  # Last 100 alerts
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def create_sample_data() -> list[ProductSnapshot]:
    """Create sample snapshots for testing"""
    base_date = datetime(2024, 1, 1)
    snapshots = []

    # Competitor 1: Stable performer
    for i in range(30):
        snapshots.append(ProductSnapshot(
            asin="B0COMP001",
            timestamp=base_date + timedelta(days=i),
            price=29.99 + (i % 5 - 2) * 0.5,  # Small price fluctuations
            bsr=15000 - i * 100,  # Slowly improving
            rating=4.3,
            review_count=500 + i * 5,
            is_in_stock=True,
            title="Premium Wireless Earbuds - Brand A",
            is_sponsored=i > 20
        ))

    # Competitor 2: Price warrior
    for i in range(30):
        price = 24.99 if i < 15 else 19.99  # Price drop at day 15
        snapshots.append(ProductSnapshot(
            asin="B0COMP002",
            timestamp=base_date + timedelta(days=i),
            price=price,
            bsr=25000 + i * 50 if i < 15 else 18000 - (i-15) * 200,  # Improves after price drop
            rating=4.0,
            review_count=300 + i * 3,
            is_in_stock=True,
            title="Budget Earbuds Value Pack - Brand B"
        ))

    # Competitor 3: Stock issues
    for i in range(30):
        in_stock = not (10 <= i <= 15)  # Out of stock days 10-15
        snapshots.append(ProductSnapshot(
            asin="B0COMP003",
            timestamp=base_date + timedelta(days=i),
            price=34.99,
            bsr=8000 if in_stock else 50000,
            rating=4.6,
            review_count=1200 + (i * 2 if in_stock else 0),
            is_in_stock=in_stock,
            title="Pro Audio Wireless Earbuds - Brand C"
        ))

    # Your product
    for i in range(30):
        snapshots.append(ProductSnapshot(
            asin="B0YOURS01",
            timestamp=base_date + timedelta(days=i),
            price=27.99,
            bsr=20000 - i * 150,  # Improving
            rating=4.4,
            review_count=150 + i * 10,
            is_in_stock=True,
            title="Your Amazing Earbuds"
        ))

    return snapshots


if __name__ == "__main__":
    print("=" * 70)
    print("COMPETITOR MONITORING SYSTEM")
    print("=" * 70)

    monitor = CompetitorMonitor()
    snapshots = create_sample_data()

    # Process all snapshots
    print("\nProcessing 30 days of competitor data...")
    all_alerts = []
    for snap in sorted(snapshots, key=lambda x: x.timestamp):
        alerts = monitor.record_snapshot(snap)
        all_alerts.extend(alerts)

    print(f"  Processed {len(snapshots)} snapshots")
    print(f"  Generated {len(all_alerts)} alerts")

    # Show profiles
    print("\n" + "-" * 70)
    print("COMPETITOR PROFILES")
    print("-" * 70)
    for asin, profile in monitor.profiles.items():
        if asin.startswith("B0COMP"):
            print(f"\n  {profile.title[:45]}...")
            print(f"    ASIN: {asin} | Threat: {profile.threat_level.upper()}")
            print(f"    Price: ${profile.current_price:.2f} | BSR: #{profile.current_bsr:,}")
            print(f"    Rating: {profile.current_rating:.1f} | Reviews: {profile.review_count:,}")
            print(f"    BSR Trend: {profile.bsr_history.trend}")
            if profile.strengths:
                print(f"    Strengths: {', '.join(profile.strengths[:2])}")
            if profile.weaknesses:
                print(f"    Weaknesses: {', '.join(profile.weaknesses[:2])}")

    # Market position
    print("\n" + "-" * 70)
    print("YOUR MARKET POSITION")
    print("-" * 70)
    position = monitor.calculate_market_position("B0YOURS01")
    if position:
        print(f"  Price Rank:  #{position.price_rank} of {position.total_competitors + 1} ({position.price_vs_avg:+.1f}% vs avg)")
        print(f"  BSR Rank:    #{position.bsr_rank} of {position.total_competitors + 1} ({position.bsr_vs_avg:+.1f}% vs avg)")
        print(f"  Rating Rank: #{position.rating_rank} of {position.total_competitors + 1} ({position.rating_vs_avg:+.1f}% vs avg)")
        print(f"  Review Rank: #{position.review_rank} of {position.total_competitors + 1} ({position.reviews_vs_avg:+.1f}% vs avg)")

    # High priority alerts
    print("\n" + "-" * 70)
    print("HIGH PRIORITY ALERTS")
    print("-" * 70)
    for alert in monitor.get_high_priority_alerts(5):
        print(f"\n  [{alert.priority.value.upper()}] {alert.message}")
        print(f"    ASIN: {alert.asin} | Type: {alert.alert_type.value}")
        if alert.old_value and alert.new_value:
            print(f"    Change: {alert.old_value} → {alert.new_value}")
        if alert.recommendation:
            print(f"    Recommendation: {alert.recommendation}")

    # Threat summary
    print("\n" + "-" * 70)
    print("THREAT SUMMARY")
    print("-" * 70)
    threats = monitor.get_threat_summary()
    print(f"  🔴 High Threat:   {len(threats['high'])} competitors")
    print(f"  🟡 Medium Threat: {len(threats['medium'])} competitors")
    print(f"  🟢 Low Threat:    {len(threats['low'])} competitors")

    print("\n" + "=" * 70)
    print("MONITORING COMPLETE")
    print("=" * 70)
