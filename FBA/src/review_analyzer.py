#!/usr/bin/env python3
"""
Review Sentiment Analyzer for Amazon FBA Research

Analyzes product reviews to extract:
- Overall sentiment distribution
- Common themes and topics
- Quality issues and pain points
- Feature requests
- Competitive insights
- Key phrases for marketing
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from collections import Counter
import re
from datetime import datetime


class Sentiment(Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class ReviewCategory(Enum):
    QUALITY = "quality"
    VALUE = "value"
    FEATURES = "features"
    DURABILITY = "durability"
    CUSTOMER_SERVICE = "customer_service"
    SHIPPING = "shipping"
    PACKAGING = "packaging"
    SIZE_FIT = "size_fit"
    EASE_OF_USE = "ease_of_use"
    APPEARANCE = "appearance"


@dataclass
class Review:
    """Single product review"""
    review_id: str
    title: str
    body: str
    rating: int  # 1-5
    verified_purchase: bool
    helpful_votes: int = 0
    date: Optional[datetime] = None
    variant: Optional[str] = None

    @property
    def full_text(self) -> str:
        return f"{self.title} {self.body}"


@dataclass
class SentimentResult:
    """Sentiment analysis for a single review"""
    review_id: str
    sentiment: Sentiment
    confidence: float
    positive_phrases: list[str] = field(default_factory=list)
    negative_phrases: list[str] = field(default_factory=list)
    categories: list[ReviewCategory] = field(default_factory=list)
    key_topics: list[str] = field(default_factory=list)


@dataclass
class ThemeAnalysis:
    """Analysis of common themes across reviews"""
    theme: str
    mention_count: int
    avg_sentiment: float  # -1 to 1
    example_phrases: list[str] = field(default_factory=list)
    related_ratings: list[int] = field(default_factory=list)

    @property
    def avg_rating(self) -> float:
        if not self.related_ratings:
            return 0
        return sum(self.related_ratings) / len(self.related_ratings)


@dataclass
class QualityIssue:
    """Identified quality/product issue"""
    issue: str
    frequency: int
    severity: str  # low, medium, high, critical
    affected_ratings: list[int] = field(default_factory=list)
    example_quotes: list[str] = field(default_factory=list)

    @property
    def impact_score(self) -> float:
        """Higher score = more impactful issue"""
        severity_mult = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        return self.frequency * severity_mult.get(self.severity, 1)


@dataclass
class CompetitorMention:
    """Reference to competitor in reviews"""
    competitor: str
    mention_count: int
    context: str  # favorable, unfavorable, neutral
    comparison_phrases: list[str] = field(default_factory=list)


@dataclass
class ReviewAnalysisResult:
    """Complete review analysis for a product"""
    asin: str
    total_reviews: int
    avg_rating: float
    rating_distribution: dict[int, int]
    sentiment_distribution: dict[Sentiment, int]

    top_positive_themes: list[ThemeAnalysis]
    top_negative_themes: list[ThemeAnalysis]
    quality_issues: list[QualityIssue]
    competitor_mentions: list[CompetitorMention]

    verified_purchase_pct: float
    avg_review_length: float
    review_velocity: float  # reviews per month

    key_positive_phrases: list[str] = field(default_factory=list)
    key_negative_phrases: list[str] = field(default_factory=list)
    feature_requests: list[str] = field(default_factory=list)
    marketing_angles: list[str] = field(default_factory=list)


class ReviewAnalyzer:
    """Analyzes Amazon product reviews for insights"""

    # Sentiment keywords
    POSITIVE_WORDS = {
        "love", "great", "excellent", "amazing", "perfect", "best",
        "fantastic", "wonderful", "awesome", "incredible", "outstanding",
        "superb", "brilliant", "exceptional", "quality", "recommend",
        "happy", "satisfied", "impressed", "comfortable", "durable",
        "sturdy", "reliable", "beautiful", "elegant", "premium"
    }

    NEGATIVE_WORDS = {
        "hate", "terrible", "awful", "horrible", "worst", "bad",
        "poor", "cheap", "disappointed", "disappointing", "broke",
        "broken", "defective", "flimsy", "useless", "waste",
        "regret", "return", "refund", "garbage", "junk",
        "uncomfortable", "unreliable", "overpriced", "misleading"
    }

    INTENSIFIERS = {
        "very", "extremely", "really", "absolutely", "totally",
        "completely", "incredibly", "highly", "super", "so"
    }

    NEGATORS = {
        "not", "no", "never", "don't", "doesn't", "didn't",
        "won't", "wouldn't", "couldn't", "shouldn't", "isn't",
        "wasn't", "aren't", "weren't", "haven't", "hasn't"
    }

    # Category patterns
    CATEGORY_PATTERNS = {
        ReviewCategory.QUALITY: [
            r"quality", r"build quality", r"construction", r"materials?",
            r"well.?made", r"craftsmanship"
        ],
        ReviewCategory.VALUE: [
            r"price", r"value", r"worth", r"money", r"expensive",
            r"cheap", r"affordable", r"cost", r"deal", r"bargain"
        ],
        ReviewCategory.FEATURES: [
            r"features?", r"function", r"options?", r"settings?",
            r"modes?", r"controls?"
        ],
        ReviewCategory.DURABILITY: [
            r"durable", r"durability", r"last", r"lasting", r"broke",
            r"broken", r"wear", r"tear", r"sturdy", r"flimsy"
        ],
        ReviewCategory.CUSTOMER_SERVICE: [
            r"customer service", r"support", r"seller", r"warranty",
            r"replacement", r"response"
        ],
        ReviewCategory.SHIPPING: [
            r"shipping", r"delivery", r"arrived", r"package",
            r"shipped", r"arrived"
        ],
        ReviewCategory.PACKAGING: [
            r"packaging", r"packaged", r"box", r"wrapped"
        ],
        ReviewCategory.SIZE_FIT: [
            r"size", r"fit", r"fitting", r"small", r"large",
            r"tight", r"loose", r"dimensions?"
        ],
        ReviewCategory.EASE_OF_USE: [
            r"easy", r"simple", r"intuitive", r"complicated",
            r"confusing", r"setup", r"install"
        ],
        ReviewCategory.APPEARANCE: [
            r"look", r"looks", r"appearance", r"color", r"design",
            r"style", r"aesthetic", r"beautiful", r"ugly"
        ]
    }

    # Quality issue patterns
    ISSUE_PATTERNS = {
        "stops working": (r"stop(ped|s)? working", "high"),
        "breaks easily": (r"broke|broken|break(s|ing)?", "high"),
        "poor battery": (r"battery.*(dies?|drain|short|poor)", "medium"),
        "connection issues": (r"(disconnect|connection|connect).*(issue|problem)", "medium"),
        "sizing problems": (r"(too (small|large|big)|size.*(off|wrong))", "medium"),
        "cheap materials": (r"cheap|flimsy|thin|weak material", "medium"),
        "defective": (r"defective|defect|malfunction", "critical"),
        "missing parts": (r"missing (part|piece|item)", "high"),
        "wrong item": (r"wrong (item|product|order)", "high"),
        "doesn't match description": (r"(not|doesn't) (match|as described)", "high"),
        "uncomfortable": (r"uncomfortable|hurts?|pain", "medium"),
        "noise/sound issues": (r"(noise|sound).*(issue|problem|bad)", "medium"),
        "smell/odor": (r"smell|odor|stink", "low"),
        "difficult setup": (r"(hard|difficult|impossible) (to )?(setup|install)", "low"),
    }

    # Common competitor patterns
    COMPETITOR_PATTERNS = [
        r"(better|worse) than (\w+)",
        r"compared to (\w+)",
        r"switched from (\w+)",
        r"instead of (\w+)",
        r"(\w+) (is|was) better",
        r"prefer(red)? (\w+)"
    ]

    def __init__(self):
        self.reviews: list[Review] = []

    def analyze_sentiment(self, review: Review) -> SentimentResult:
        """Analyze sentiment of a single review"""
        text = review.full_text.lower()
        words = text.split()

        positive_count = 0
        negative_count = 0
        positive_phrases = []
        negative_phrases = []

        # Check for intensifiers and negators
        for i, word in enumerate(words):
            # Look ahead for intensified sentiments
            is_negated = any(words[j] in self.NEGATORS
                          for j in range(max(0, i-3), i))
            is_intensified = any(words[j] in self.INTENSIFIERS
                               for j in range(max(0, i-2), i))

            multiplier = 1.5 if is_intensified else 1.0

            if word in self.POSITIVE_WORDS:
                if is_negated:
                    negative_count += multiplier
                    negative_phrases.append(self._extract_phrase(words, i))
                else:
                    positive_count += multiplier
                    positive_phrases.append(self._extract_phrase(words, i))

            elif word in self.NEGATIVE_WORDS:
                if is_negated:
                    positive_count += multiplier * 0.5  # Double negative is weakly positive
                else:
                    negative_count += multiplier
                    negative_phrases.append(self._extract_phrase(words, i))

        # Combine with star rating
        rating_sentiment = (review.rating - 3) / 2  # -1 to 1
        text_sentiment = (positive_count - negative_count) / max(1, positive_count + negative_count)
        combined = (text_sentiment * 0.4) + (rating_sentiment * 0.6)

        # Determine sentiment category
        if combined >= 0.6:
            sentiment = Sentiment.VERY_POSITIVE
        elif combined >= 0.2:
            sentiment = Sentiment.POSITIVE
        elif combined >= -0.2:
            sentiment = Sentiment.NEUTRAL
        elif combined >= -0.6:
            sentiment = Sentiment.NEGATIVE
        else:
            sentiment = Sentiment.VERY_NEGATIVE

        # Confidence based on signal strength
        confidence = min(1.0, (positive_count + negative_count) / 5)

        # Identify categories
        categories = self._identify_categories(text)

        return SentimentResult(
            review_id=review.review_id,
            sentiment=sentiment,
            confidence=confidence,
            positive_phrases=positive_phrases[:5],
            negative_phrases=negative_phrases[:5],
            categories=categories,
            key_topics=self._extract_topics(text)
        )

    def _extract_phrase(self, words: list[str], index: int, window: int = 3) -> str:
        """Extract phrase around a word"""
        start = max(0, index - window)
        end = min(len(words), index + window + 1)
        return " ".join(words[start:end])

    def _identify_categories(self, text: str) -> list[ReviewCategory]:
        """Identify which categories a review discusses"""
        categories = []
        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    categories.append(category)
                    break
        return categories

    def _extract_topics(self, text: str) -> list[str]:
        """Extract main topics from text"""
        # Simple noun phrase extraction
        topics = []
        # Find capitalized words (likely product names/features)
        caps = re.findall(r'\b[A-Z][a-z]+\b', text)
        topics.extend(caps[:3])

        # Find common product terms
        product_terms = re.findall(
            r'\b(battery|sound|quality|fit|size|color|design|material|'
            r'comfort|durability|price|value|feature|function)\b',
            text, re.IGNORECASE
        )
        topics.extend([t.lower() for t in product_terms[:5]])

        return list(set(topics))[:5]

    def analyze_themes(self, reviews: list[Review],
                      sentiments: list[SentimentResult]) -> tuple[list[ThemeAnalysis], list[ThemeAnalysis]]:
        """Extract positive and negative themes from reviews"""
        positive_themes = Counter()
        negative_themes = Counter()
        theme_sentiments = {}
        theme_ratings = {}
        theme_examples = {}

        for review, sentiment in zip(reviews, sentiments):
            text = review.full_text.lower()

            # Extract noun phrases as themes
            phrases = self._extract_theme_phrases(text)

            for phrase in phrases:
                if phrase not in theme_sentiments:
                    theme_sentiments[phrase] = []
                    theme_ratings[phrase] = []
                    theme_examples[phrase] = []

                theme_ratings[phrase].append(review.rating)

                if sentiment.sentiment in [Sentiment.VERY_POSITIVE, Sentiment.POSITIVE]:
                    positive_themes[phrase] += 1
                    theme_sentiments[phrase].append(1)
                elif sentiment.sentiment in [Sentiment.VERY_NEGATIVE, Sentiment.NEGATIVE]:
                    negative_themes[phrase] += 1
                    theme_sentiments[phrase].append(-1)
                else:
                    theme_sentiments[phrase].append(0)

                if len(theme_examples[phrase]) < 3:
                    theme_examples[phrase].append(self._get_context(text, phrase))

        # Build theme analysis objects
        pos_results = []
        for theme, count in positive_themes.most_common(10):
            if count >= 2:
                pos_results.append(ThemeAnalysis(
                    theme=theme,
                    mention_count=count,
                    avg_sentiment=sum(theme_sentiments[theme]) / len(theme_sentiments[theme]),
                    example_phrases=theme_examples[theme],
                    related_ratings=theme_ratings[theme]
                ))

        neg_results = []
        for theme, count in negative_themes.most_common(10):
            if count >= 2:
                neg_results.append(ThemeAnalysis(
                    theme=theme,
                    mention_count=count,
                    avg_sentiment=sum(theme_sentiments[theme]) / len(theme_sentiments[theme]),
                    example_phrases=theme_examples[theme],
                    related_ratings=theme_ratings[theme]
                ))

        return pos_results, neg_results

    def _extract_theme_phrases(self, text: str) -> list[str]:
        """Extract meaningful theme phrases"""
        themes = []

        # Common product attributes
        attributes = [
            "sound quality", "battery life", "build quality", "price point",
            "customer service", "ease of use", "bang for buck", "value for money",
            "noise cancellation", "comfort level", "fit and finish",
            "packaging", "instructions", "setup process"
        ]

        for attr in attributes:
            if attr in text:
                themes.append(attr)

        # Single important words
        important = re.findall(
            r'\b(quality|comfort|durability|design|fit|battery|sound|'
            r'value|price|shipping|packaging|material|size)\b',
            text
        )
        themes.extend(important)

        return list(set(themes))

    def _get_context(self, text: str, phrase: str, window: int = 50) -> str:
        """Get context around a phrase"""
        idx = text.find(phrase)
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(text), idx + len(phrase) + window)
        return "..." + text[start:end] + "..."

    def identify_quality_issues(self, reviews: list[Review]) -> list[QualityIssue]:
        """Identify product quality issues from negative reviews"""
        issues = {}

        for review in reviews:
            if review.rating > 3:  # Focus on negative reviews
                continue

            text = review.full_text.lower()

            for issue_name, (pattern, severity) in self.ISSUE_PATTERNS.items():
                if re.search(pattern, text, re.IGNORECASE):
                    if issue_name not in issues:
                        issues[issue_name] = {
                            "frequency": 0,
                            "severity": severity,
                            "ratings": [],
                            "quotes": []
                        }
                    issues[issue_name]["frequency"] += 1
                    issues[issue_name]["ratings"].append(review.rating)
                    if len(issues[issue_name]["quotes"]) < 3:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            issues[issue_name]["quotes"].append(
                                self._get_context(text, match.group())
                            )

        results = []
        for name, data in issues.items():
            if data["frequency"] >= 2:  # At least 2 mentions
                results.append(QualityIssue(
                    issue=name,
                    frequency=data["frequency"],
                    severity=data["severity"],
                    affected_ratings=data["ratings"],
                    example_quotes=data["quotes"]
                ))

        return sorted(results, key=lambda x: x.impact_score, reverse=True)

    def find_competitor_mentions(self, reviews: list[Review]) -> list[CompetitorMention]:
        """Find references to competitor products"""
        competitor_data = {}

        # Known brand names to look for
        known_brands = [
            "apple", "samsung", "sony", "bose", "anker", "jabra",
            "beats", "jbl", "sennheiser", "airpods", "raycon",
            "skullcandy", "soundcore", "tozo", "galaxy buds"
        ]

        for review in reviews:
            text = review.full_text.lower()

            # Look for known brands
            for brand in known_brands:
                if brand in text:
                    if brand not in competitor_data:
                        competitor_data[brand] = {
                            "count": 0,
                            "contexts": [],
                            "sentiments": []
                        }
                    competitor_data[brand]["count"] += 1

                    # Determine context (favorable, unfavorable, neutral)
                    context_text = self._get_context(text, brand, 100)
                    competitor_data[brand]["contexts"].append(context_text)

                    # Simple sentiment check for the mention
                    if any(w in context_text for w in ["better", "prefer", "switched from", "love my"]):
                        competitor_data[brand]["sentiments"].append("favorable")
                    elif any(w in context_text for w in ["worse", "returned", "broke", "didn't like"]):
                        competitor_data[brand]["sentiments"].append("unfavorable")
                    else:
                        competitor_data[brand]["sentiments"].append("neutral")

        results = []
        for brand, data in competitor_data.items():
            if data["count"] >= 2:
                # Determine overall context
                sentiment_counts = Counter(data["sentiments"])
                most_common = sentiment_counts.most_common(1)[0][0]

                results.append(CompetitorMention(
                    competitor=brand.title(),
                    mention_count=data["count"],
                    context=most_common,
                    comparison_phrases=data["contexts"][:3]
                ))

        return sorted(results, key=lambda x: x.mention_count, reverse=True)

    def extract_feature_requests(self, reviews: list[Review]) -> list[str]:
        """Extract feature requests and suggestions from reviews"""
        requests = []

        patterns = [
            r"wish (it|they|this) (had|would|could) (.+?)(?:\.|,|$)",
            r"would be (nice|great|better) if (.+?)(?:\.|,|$)",
            r"should (have|include|add) (.+?)(?:\.|,|$)",
            r"needs? (.+?)(?:\.|,|$)",
            r"missing (.+?)(?:\.|,|$)",
            r"if only (.+?)(?:\.|,|$)",
        ]

        for review in reviews:
            text = review.full_text.lower()

            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        request = match[-1].strip()
                    else:
                        request = match.strip()
                    if len(request) > 5 and len(request) < 100:
                        requests.append(request)

        # Deduplicate similar requests
        unique = []
        for r in requests:
            if not any(self._similar(r, u) for u in unique):
                unique.append(r)

        return unique[:10]

    def _similar(self, a: str, b: str, threshold: float = 0.6) -> bool:
        """Check if two strings are similar"""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return False
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union > threshold

    def generate_marketing_angles(self,
                                  positive_themes: list[ThemeAnalysis],
                                  quality_issues: list[QualityIssue]) -> list[str]:
        """Generate marketing angles based on review analysis"""
        angles = []

        # Leverage positive themes
        for theme in positive_themes[:5]:
            if theme.avg_rating >= 4:
                angles.append(f"Highlight: {theme.theme} (mentioned {theme.mention_count}x positively)")

        # Address competitor weaknesses
        for issue in quality_issues[:5]:
            angles.append(f"Differentiate: Solve '{issue.issue}' problem (competitor weakness)")

        return angles

    def analyze_reviews(self, asin: str, reviews: list[Review]) -> ReviewAnalysisResult:
        """Complete analysis of product reviews"""
        if not reviews:
            raise ValueError("No reviews to analyze")

        # Basic stats
        total = len(reviews)
        ratings = [r.rating for r in reviews]
        avg_rating = sum(ratings) / total

        rating_dist = Counter(ratings)
        rating_distribution = {i: rating_dist.get(i, 0) for i in range(1, 6)}

        verified_count = sum(1 for r in reviews if r.verified_purchase)
        verified_pct = verified_count / total * 100

        avg_length = sum(len(r.full_text) for r in reviews) / total

        # Sentiment analysis
        sentiments = [self.analyze_sentiment(r) for r in reviews]
        sentiment_dist = Counter(s.sentiment for s in sentiments)

        # Theme analysis
        pos_themes, neg_themes = self.analyze_themes(reviews, sentiments)

        # Quality issues
        quality_issues = self.identify_quality_issues(reviews)

        # Competitor mentions
        competitor_mentions = self.find_competitor_mentions(reviews)

        # Feature requests
        feature_requests = self.extract_feature_requests(reviews)

        # Key phrases
        all_pos_phrases = []
        all_neg_phrases = []
        for s in sentiments:
            all_pos_phrases.extend(s.positive_phrases)
            all_neg_phrases.extend(s.negative_phrases)

        pos_phrase_counts = Counter(all_pos_phrases)
        neg_phrase_counts = Counter(all_neg_phrases)

        # Marketing angles
        marketing = self.generate_marketing_angles(pos_themes, quality_issues)

        # Review velocity (assume reviews span 12 months for demo)
        velocity = total / 12

        return ReviewAnalysisResult(
            asin=asin,
            total_reviews=total,
            avg_rating=avg_rating,
            rating_distribution=rating_distribution,
            sentiment_distribution={s: sentiment_dist.get(s, 0) for s in Sentiment},
            top_positive_themes=pos_themes,
            top_negative_themes=neg_themes,
            quality_issues=quality_issues,
            competitor_mentions=competitor_mentions,
            verified_purchase_pct=verified_pct,
            avg_review_length=avg_length,
            review_velocity=velocity,
            key_positive_phrases=[p for p, _ in pos_phrase_counts.most_common(10)],
            key_negative_phrases=[p for p, _ in neg_phrase_counts.most_common(10)],
            feature_requests=feature_requests,
            marketing_angles=marketing
        )


def create_sample_reviews() -> list[Review]:
    """Create sample reviews for testing"""
    return [
        Review(
            review_id="R1",
            title="Absolutely love these earbuds!",
            body="The sound quality is amazing and battery life is great. Much better than my old Sony earbuds. Comfortable fit and the noise cancellation works perfectly. Worth every penny!",
            rating=5,
            verified_purchase=True,
            helpful_votes=45
        ),
        Review(
            review_id="R2",
            title="Great value for money",
            body="For the price, these are excellent. Sound quality is good, not as good as Bose but much cheaper. Easy to setup and comfortable for long use. Recommend!",
            rating=4,
            verified_purchase=True,
            helpful_votes=23
        ),
        Review(
            review_id="R3",
            title="Disappointed - broke after 2 months",
            body="Started great but stopped working after two months. The left earbud just died. Very disappointing as I really liked the sound. Customer service was unhelpful. Wish they had better durability.",
            rating=2,
            verified_purchase=True,
            helpful_votes=67
        ),
        Review(
            review_id="R4",
            title="Perfect for workouts",
            body="These stay in my ears during running and gym sessions. Sweat resistant and battery lasts all week. Sound quality is excellent for music. Love the carrying case!",
            rating=5,
            verified_purchase=True,
            helpful_votes=18
        ),
        Review(
            review_id="R5",
            title="Not worth it",
            body="Cheap materials, uncomfortable after 30 minutes. Connection issues with my phone. The sound quality is terrible compared to even basic Apple earbuds. Don't waste your money.",
            rating=1,
            verified_purchase=True,
            helpful_votes=89
        ),
        Review(
            review_id="R6",
            title="Good but not great",
            body="Sound is decent, fit is okay. Battery life could be better - only get about 3 hours. Packaging was nice though. Would be nice if they had longer battery life.",
            rating=3,
            verified_purchase=True,
            helpful_votes=12
        ),
        Review(
            review_id="R7",
            title="Best earbuds I've owned",
            body="Switched from Samsung Galaxy Buds and these are so much better! The fit is perfect, sound is incredible, and I love the touch controls. Premium quality.",
            rating=5,
            verified_purchase=True,
            helpful_votes=34
        ),
        Review(
            review_id="R8",
            title="Connection keeps dropping",
            body="Every 10 minutes the connection disconnects. Very frustrating. Sound is good when it works but the bluetooth issues make it unusable. Returning these.",
            rating=1,
            verified_purchase=True,
            helpful_votes=56
        ),
        Review(
            review_id="R9",
            title="Solid product, minor issues",
            body="Overall happy with purchase. Good sound, comfortable, nice design. My only complaint is the case feels a bit cheap and flimsy. Wish it had wireless charging.",
            rating=4,
            verified_purchase=True,
            helpful_votes=8
        ),
        Review(
            review_id="R10",
            title="Exceeded expectations",
            body="Was skeptical at this price but wow! Sound quality rivals my expensive Bose headphones. Noise cancellation is impressive. Battery lasts forever. Highly recommend!",
            rating=5,
            verified_purchase=False,
            helpful_votes=29
        ),
    ]


if __name__ == "__main__":
    print("=" * 70)
    print("REVIEW SENTIMENT ANALYZER")
    print("=" * 70)

    analyzer = ReviewAnalyzer()
    reviews = create_sample_reviews()

    # Analyze all reviews
    result = analyzer.analyze_reviews("B0TEST123", reviews)

    print(f"\nASIN: {result.asin}")
    print(f"Total Reviews: {result.total_reviews}")
    print(f"Average Rating: {result.avg_rating:.1f}/5")
    print(f"Verified Purchase: {result.verified_purchase_pct:.0f}%")

    print("\n" + "-" * 70)
    print("RATING DISTRIBUTION")
    print("-" * 70)
    for rating in range(5, 0, -1):
        count = result.rating_distribution[rating]
        bar = "█" * (count * 3)
        print(f"  {rating}★  {bar} ({count})")

    print("\n" + "-" * 70)
    print("SENTIMENT DISTRIBUTION")
    print("-" * 70)
    for sentiment, count in result.sentiment_distribution.items():
        if count > 0:
            bar = "█" * (count * 3)
            print(f"  {sentiment.value:15} {bar} ({count})")

    print("\n" + "-" * 70)
    print("TOP POSITIVE THEMES")
    print("-" * 70)
    for theme in result.top_positive_themes[:5]:
        print(f"  • {theme.theme}: {theme.mention_count}x (avg rating: {theme.avg_rating:.1f})")

    print("\n" + "-" * 70)
    print("TOP NEGATIVE THEMES")
    print("-" * 70)
    for theme in result.top_negative_themes[:5]:
        print(f"  • {theme.theme}: {theme.mention_count}x (avg rating: {theme.avg_rating:.1f})")

    print("\n" + "-" * 70)
    print("QUALITY ISSUES IDENTIFIED")
    print("-" * 70)
    for issue in result.quality_issues[:5]:
        print(f"  ⚠ {issue.issue}")
        print(f"    Frequency: {issue.frequency} | Severity: {issue.severity}")

    print("\n" + "-" * 70)
    print("COMPETITOR MENTIONS")
    print("-" * 70)
    for comp in result.competitor_mentions[:5]:
        print(f"  • {comp.competitor}: {comp.mention_count}x ({comp.context})")

    print("\n" + "-" * 70)
    print("FEATURE REQUESTS")
    print("-" * 70)
    for req in result.feature_requests[:5]:
        print(f"  → {req}")

    print("\n" + "-" * 70)
    print("MARKETING ANGLES")
    print("-" * 70)
    for angle in result.marketing_angles[:5]:
        print(f"  ✓ {angle}")

    print("\n" + "=" * 70)
    print("REVIEW ANALYSIS COMPLETE")
    print("=" * 70)
