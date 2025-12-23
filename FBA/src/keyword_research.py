"""
Amazon Keyword Research

Keyword analysis for product research including:
- Search volume estimation
- Competition scoring
- Keyword relevance
- Long-tail keyword discovery
- Keyword clustering
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math


class KeywordDifficulty(Enum):
    """Keyword competition difficulty."""
    VERY_EASY = "very_easy"     # Score < 20
    EASY = "easy"               # Score 20-40
    MODERATE = "moderate"       # Score 40-60
    HARD = "hard"               # Score 60-80
    VERY_HARD = "very_hard"     # Score > 80


class KeywordIntent(Enum):
    """Search intent classification."""
    INFORMATIONAL = "informational"  # "how to", "what is"
    NAVIGATIONAL = "navigational"    # Brand searches
    COMMERCIAL = "commercial"        # "best", "review", "vs"
    TRANSACTIONAL = "transactional"  # "buy", "price", "cheap"


@dataclass
class Keyword:
    """Represents a keyword with metrics."""
    keyword: str
    search_volume: int  # Estimated monthly searches
    competition: float  # 0-1 (0=low, 1=high)
    cpc: float  # Cost per click estimate
    relevance: float  # 0-1 relevance to seed
    word_count: int
    intent: KeywordIntent

    @property
    def difficulty_score(self) -> float:
        """Calculate keyword difficulty score (0-100)."""
        # Combine competition and search volume factors
        comp_factor = self.competition * 60
        vol_factor = min(40, math.log10(max(1, self.search_volume)) * 10)
        return min(100, comp_factor + vol_factor)

    @property
    def difficulty(self) -> KeywordDifficulty:
        """Get difficulty category."""
        score = self.difficulty_score
        if score < 20:
            return KeywordDifficulty.VERY_EASY
        elif score < 40:
            return KeywordDifficulty.EASY
        elif score < 60:
            return KeywordDifficulty.MODERATE
        elif score < 80:
            return KeywordDifficulty.HARD
        else:
            return KeywordDifficulty.VERY_HARD

    @property
    def opportunity_score(self) -> float:
        """Calculate opportunity score (higher = better)."""
        # High volume + low competition + high relevance = good opportunity
        volume_score = min(50, math.log10(max(1, self.search_volume)) * 12)
        comp_score = (1 - self.competition) * 30
        relevance_score = self.relevance * 20
        return round(volume_score + comp_score + relevance_score, 1)


@dataclass
class KeywordCluster:
    """Group of related keywords."""
    name: str
    seed_keyword: str
    keywords: List[Keyword]
    total_search_volume: int
    avg_competition: float
    avg_difficulty: float

    @property
    def keyword_count(self) -> int:
        return len(self.keywords)


@dataclass
class KeywordResearchResult:
    """Complete keyword research results."""
    seed_keyword: str
    timestamp: str

    # Main keywords
    keywords: List[Keyword]
    clusters: List[KeywordCluster]

    # Summary metrics
    total_search_volume: int
    avg_competition: float
    avg_cpc: float

    # Top picks
    high_volume_keywords: List[Keyword]
    low_competition_keywords: List[Keyword]
    best_opportunities: List[Keyword]


class KeywordResearcher:
    """
    Keyword research and analysis.

    Features:
    - Long-tail keyword generation
    - Search intent classification
    - Competition analysis
    - Keyword clustering
    - Opportunity scoring
    """

    # Common modifiers for long-tail generation
    MODIFIERS = {
        'buying': ['best', 'top', 'cheap', 'affordable', 'premium', 'buy', 'price'],
        'quality': ['quality', 'durable', 'professional', 'heavy duty', 'lightweight'],
        'size': ['small', 'large', 'mini', 'portable', 'compact', 'xl', 'plus size'],
        'color': ['black', 'white', 'blue', 'red', 'pink', 'green', 'gold', 'silver'],
        'material': ['leather', 'metal', 'plastic', 'wood', 'stainless steel', 'silicone'],
        'audience': ['for men', 'for women', 'for kids', 'for seniors', 'for beginners'],
        'use_case': ['for home', 'for office', 'for travel', 'outdoor', 'indoor'],
        'features': ['with', 'wireless', 'rechargeable', 'waterproof', 'adjustable'],
        'comparison': ['vs', 'alternative', 'like', 'similar to'],
        'year': ['2024', '2025', 'new', 'latest'],
    }

    # Intent indicators
    INTENT_PATTERNS = {
        KeywordIntent.INFORMATIONAL: [
            r'\bhow to\b', r'\bwhat is\b', r'\bwhy\b', r'\bguide\b',
            r'\btutorial\b', r'\btips\b', r'\bideas\b'
        ],
        KeywordIntent.NAVIGATIONAL: [
            r'\bamazon\b', r'\bwalmart\b', r'\btarget\b', r'\bbrand\b'
        ],
        KeywordIntent.COMMERCIAL: [
            r'\bbest\b', r'\btop\b', r'\breview\b', r'\bvs\b',
            r'\bcompare\b', r'\brated\b', r'\brecommended\b'
        ],
        KeywordIntent.TRANSACTIONAL: [
            r'\bbuy\b', r'\bprice\b', r'\bcheap\b', r'\bdiscount\b',
            r'\bdeal\b', r'\bsale\b', r'\border\b', r'\bpurchase\b'
        ]
    }

    def __init__(self):
        """Initialize keyword researcher."""
        pass

    def classify_intent(self, keyword: str) -> KeywordIntent:
        """
        Classify search intent of a keyword.

        Args:
            keyword: Keyword to classify

        Returns:
            KeywordIntent classification
        """
        keyword_lower = keyword.lower()

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, keyword_lower):
                    return intent

        # Default to transactional for product keywords
        return KeywordIntent.TRANSACTIONAL

    def calculate_relevance(
        self,
        keyword: str,
        seed_keyword: str
    ) -> float:
        """
        Calculate keyword relevance to seed keyword.

        Args:
            keyword: Keyword to evaluate
            seed_keyword: Original seed keyword

        Returns:
            Relevance score (0-1)
        """
        keyword_words = set(keyword.lower().split())
        seed_words = set(seed_keyword.lower().split())

        # Calculate Jaccard similarity
        intersection = len(keyword_words & seed_words)
        union = len(keyword_words | seed_words)

        if union == 0:
            return 0.0

        base_relevance = intersection / union

        # Bonus for containing all seed words
        if seed_words <= keyword_words:
            base_relevance = min(1.0, base_relevance + 0.3)

        # Penalty for very long keywords (likely too specific)
        if len(keyword_words) > 6:
            base_relevance *= 0.8

        return round(base_relevance, 2)

    def estimate_search_volume(
        self,
        keyword: str,
        base_volume: int = 10000
    ) -> int:
        """
        Estimate search volume for a keyword.

        Note: In production, use actual data from:
        - Amazon Search Suggest API
        - Helium10/JungleScout
        - Google Keyword Planner

        Args:
            keyword: Keyword to estimate
            base_volume: Base volume for seed keyword

        Returns:
            Estimated monthly search volume
        """
        word_count = len(keyword.split())

        # Longer keywords typically have lower volume
        # Rough estimation model
        if word_count == 1:
            volume = base_volume
        elif word_count == 2:
            volume = int(base_volume * 0.4)
        elif word_count == 3:
            volume = int(base_volume * 0.15)
        elif word_count == 4:
            volume = int(base_volume * 0.05)
        else:
            volume = int(base_volume * 0.02)

        # Add some variation
        variation = hash(keyword) % 50 - 25  # -25 to +25
        volume = int(volume * (1 + variation / 100))

        return max(10, volume)

    def estimate_competition(
        self,
        keyword: str,
        search_volume: int
    ) -> float:
        """
        Estimate keyword competition.

        Args:
            keyword: Keyword to evaluate
            search_volume: Search volume

        Returns:
            Competition score (0-1)
        """
        word_count = len(keyword.split())

        # Base competition from volume
        if search_volume > 50000:
            base_comp = 0.9
        elif search_volume > 20000:
            base_comp = 0.75
        elif search_volume > 10000:
            base_comp = 0.6
        elif search_volume > 5000:
            base_comp = 0.45
        elif search_volume > 1000:
            base_comp = 0.3
        else:
            base_comp = 0.15

        # Long-tail keywords have lower competition
        if word_count >= 4:
            base_comp *= 0.6
        elif word_count >= 3:
            base_comp *= 0.8

        # Variation based on keyword content
        variation = (hash(keyword) % 20 - 10) / 100
        competition = max(0.05, min(0.95, base_comp + variation))

        return round(competition, 2)

    def estimate_cpc(
        self,
        keyword: str,
        competition: float
    ) -> float:
        """
        Estimate cost per click.

        Args:
            keyword: Keyword
            competition: Competition score

        Returns:
            Estimated CPC in dollars
        """
        # CPC correlates with competition
        base_cpc = competition * 3.0

        # Some categories have higher CPCs
        high_cpc_terms = ['insurance', 'lawyer', 'loan', 'mortgage', 'software']
        for term in high_cpc_terms:
            if term in keyword.lower():
                base_cpc *= 2

        # Add variation
        variation = (hash(keyword) % 30 - 15) / 100
        cpc = max(0.10, base_cpc * (1 + variation))

        return round(cpc, 2)

    def generate_long_tail_keywords(
        self,
        seed_keyword: str,
        max_keywords: int = 100
    ) -> List[str]:
        """
        Generate long-tail keyword variations.

        Args:
            seed_keyword: Base keyword
            max_keywords: Maximum keywords to generate

        Returns:
            List of keyword variations
        """
        keywords = set()
        keywords.add(seed_keyword)

        # Add modifier variations
        for category, modifiers in self.MODIFIERS.items():
            for modifier in modifiers:
                # Prefix
                keywords.add(f"{modifier} {seed_keyword}")
                # Suffix
                keywords.add(f"{seed_keyword} {modifier}")

        # Combine modifiers
        buying_mods = self.MODIFIERS['buying'][:3]
        size_mods = self.MODIFIERS['size'][:3]
        audience_mods = self.MODIFIERS['audience'][:3]

        for buy in buying_mods:
            for size in size_mods:
                keywords.add(f"{buy} {size} {seed_keyword}")

        for audience in audience_mods:
            keywords.add(f"{seed_keyword} {audience}")
            for buy in buying_mods:
                keywords.add(f"{buy} {seed_keyword} {audience}")

        # Question variations
        keywords.add(f"what is the best {seed_keyword}")
        keywords.add(f"how to choose {seed_keyword}")
        keywords.add(f"where to buy {seed_keyword}")

        # Truncate to max
        return list(keywords)[:max_keywords]

    def analyze_keyword(
        self,
        keyword: str,
        seed_keyword: str,
        base_volume: int = 10000
    ) -> Keyword:
        """
        Analyze a single keyword.

        Args:
            keyword: Keyword to analyze
            seed_keyword: Original seed keyword
            base_volume: Base search volume for estimation

        Returns:
            Keyword object with metrics
        """
        search_volume = self.estimate_search_volume(keyword, base_volume)
        competition = self.estimate_competition(keyword, search_volume)
        cpc = self.estimate_cpc(keyword, competition)
        relevance = self.calculate_relevance(keyword, seed_keyword)
        intent = self.classify_intent(keyword)

        return Keyword(
            keyword=keyword,
            search_volume=search_volume,
            competition=competition,
            cpc=cpc,
            relevance=relevance,
            word_count=len(keyword.split()),
            intent=intent
        )

    def cluster_keywords(
        self,
        keywords: List[Keyword],
        seed_keyword: str
    ) -> List[KeywordCluster]:
        """
        Group keywords into thematic clusters.

        Args:
            keywords: List of keywords to cluster
            seed_keyword: Original seed keyword

        Returns:
            List of keyword clusters
        """
        clusters = defaultdict(list)

        # Cluster by modifier category
        for kw in keywords:
            keyword_lower = kw.keyword.lower()
            assigned = False

            for category, modifiers in self.MODIFIERS.items():
                for modifier in modifiers:
                    if modifier in keyword_lower:
                        clusters[category].append(kw)
                        assigned = True
                        break
                if assigned:
                    break

            if not assigned:
                # Check intent
                if kw.intent == KeywordIntent.INFORMATIONAL:
                    clusters['informational'].append(kw)
                elif kw.intent == KeywordIntent.COMMERCIAL:
                    clusters['commercial'].append(kw)
                else:
                    clusters['core'].append(kw)

        # Build cluster objects
        result = []
        for name, kws in clusters.items():
            if kws:
                total_vol = sum(k.search_volume for k in kws)
                avg_comp = sum(k.competition for k in kws) / len(kws)
                avg_diff = sum(k.difficulty_score for k in kws) / len(kws)

                result.append(KeywordCluster(
                    name=name,
                    seed_keyword=seed_keyword,
                    keywords=kws,
                    total_search_volume=total_vol,
                    avg_competition=round(avg_comp, 2),
                    avg_difficulty=round(avg_diff, 1)
                ))

        # Sort by total volume
        result.sort(key=lambda c: c.total_search_volume, reverse=True)
        return result

    def research_keyword(
        self,
        seed_keyword: str,
        base_volume: int = 10000,
        max_keywords: int = 100
    ) -> KeywordResearchResult:
        """
        Perform complete keyword research.

        Args:
            seed_keyword: Starting keyword
            base_volume: Estimated search volume for seed
            max_keywords: Maximum keywords to analyze

        Returns:
            KeywordResearchResult with full analysis
        """
        from datetime import datetime

        # Generate variations
        variations = self.generate_long_tail_keywords(seed_keyword, max_keywords)

        # Analyze each keyword
        keywords = [
            self.analyze_keyword(kw, seed_keyword, base_volume)
            for kw in variations
        ]

        # Sort by opportunity score
        keywords.sort(key=lambda k: k.opportunity_score, reverse=True)

        # Cluster keywords
        clusters = self.cluster_keywords(keywords, seed_keyword)

        # Calculate summary metrics
        total_volume = sum(k.search_volume for k in keywords)
        avg_competition = sum(k.competition for k in keywords) / len(keywords)
        avg_cpc = sum(k.cpc for k in keywords) / len(keywords)

        # Get top picks
        high_volume = sorted(keywords, key=lambda k: k.search_volume, reverse=True)[:10]
        low_competition = sorted(keywords, key=lambda k: k.competition)[:10]
        best_opportunities = keywords[:10]  # Already sorted by opportunity

        return KeywordResearchResult(
            seed_keyword=seed_keyword,
            timestamp=datetime.now().isoformat(),
            keywords=keywords,
            clusters=clusters,
            total_search_volume=total_volume,
            avg_competition=round(avg_competition, 2),
            avg_cpc=round(avg_cpc, 2),
            high_volume_keywords=high_volume,
            low_competition_keywords=low_competition,
            best_opportunities=best_opportunities
        )

    def find_keyword_gaps(
        self,
        your_keywords: List[str],
        competitor_keywords: List[str]
    ) -> List[str]:
        """
        Find keywords competitors rank for that you don't.

        Args:
            your_keywords: Your targeted keywords
            competitor_keywords: Competitor keywords

        Returns:
            Gap keywords to target
        """
        your_set = set(k.lower() for k in your_keywords)
        competitor_set = set(k.lower() for k in competitor_keywords)

        gaps = competitor_set - your_set
        return list(gaps)

    def suggest_backend_keywords(
        self,
        keywords: List[Keyword],
        max_bytes: int = 250
    ) -> str:
        """
        Generate Amazon backend keyword suggestions.

        Amazon allows 250 bytes for backend keywords.

        Args:
            keywords: Analyzed keywords
            max_bytes: Maximum byte limit

        Returns:
            Optimized backend keyword string
        """
        # Sort by opportunity score
        sorted_kws = sorted(keywords, key=lambda k: k.opportunity_score, reverse=True)

        # Build keyword list without duplicates
        used_words = set()
        backend_words = []

        for kw in sorted_kws:
            words = kw.keyword.lower().split()
            for word in words:
                # Skip very short words and already used
                if len(word) > 2 and word not in used_words:
                    used_words.add(word)
                    backend_words.append(word)

        # Join with spaces and truncate to byte limit
        result = ' '.join(backend_words)
        while len(result.encode('utf-8')) > max_bytes:
            backend_words.pop()
            result = ' '.join(backend_words)

        return result


if __name__ == "__main__":
    print("=" * 70)
    print("KEYWORD RESEARCH MODULE")
    print("=" * 70)

    researcher = KeywordResearcher()

    # Research a keyword
    result = researcher.research_keyword(
        seed_keyword="wireless earbuds",
        base_volume=50000,
        max_keywords=50
    )

    print(f"\nSeed Keyword: {result.seed_keyword}")
    print(f"Generated: {result.timestamp}")
    print(f"Total Keywords: {len(result.keywords)}")

    print(f"\n{'─'*70}")
    print("SUMMARY METRICS")
    print(f"{'─'*70}")
    print(f"  Total Search Volume:  {result.total_search_volume:,}")
    print(f"  Avg Competition:      {result.avg_competition:.2f}")
    print(f"  Avg CPC:              ${result.avg_cpc:.2f}")

    print(f"\n{'─'*70}")
    print("TOP 10 OPPORTUNITIES")
    print(f"{'─'*70}")
    print(f"{'Keyword':<35} {'Volume':>8} {'Comp':>6} {'Score':>6} {'Difficulty'}")
    print(f"{'─'*70}")
    for kw in result.best_opportunities[:10]:
        print(f"{kw.keyword[:34]:<35} {kw.search_volume:>7,} {kw.competition:>5.2f} "
              f"{kw.opportunity_score:>5.1f} {kw.difficulty.value}")

    print(f"\n{'─'*70}")
    print("HIGH VOLUME KEYWORDS")
    print(f"{'─'*70}")
    for kw in result.high_volume_keywords[:5]:
        print(f"  {kw.keyword}: {kw.search_volume:,} searches/mo")

    print(f"\n{'─'*70}")
    print("LOW COMPETITION KEYWORDS")
    print(f"{'─'*70}")
    for kw in result.low_competition_keywords[:5]:
        print(f"  {kw.keyword}: {kw.competition:.2f} competition")

    print(f"\n{'─'*70}")
    print("KEYWORD CLUSTERS")
    print(f"{'─'*70}")
    for cluster in result.clusters[:5]:
        print(f"\n  [{cluster.name.upper()}] - {cluster.keyword_count} keywords")
        print(f"    Volume: {cluster.total_search_volume:,} | Avg Comp: {cluster.avg_competition:.2f}")
        for kw in cluster.keywords[:3]:
            print(f"      • {kw.keyword}")

    print(f"\n{'─'*70}")
    print("BACKEND KEYWORDS SUGGESTION (250 bytes)")
    print(f"{'─'*70}")
    backend = researcher.suggest_backend_keywords(result.keywords)
    print(f"  {backend}")
    print(f"  ({len(backend.encode('utf-8'))} bytes)")
