#!/usr/bin/env python3
"""
Amazon Listing Optimizer for FBA Research

Analyzes and optimizes product listings:
- Title optimization
- Bullet point analysis
- Keyword placement scoring
- Backend search term optimization
- Image requirements check
- A+ Content recommendations
- Mobile optimization check
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import re


class OptimizationPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ListingSection(Enum):
    TITLE = "title"
    BULLETS = "bullets"
    DESCRIPTION = "description"
    BACKEND = "backend_keywords"
    IMAGES = "images"
    A_PLUS = "a_plus_content"


@dataclass
class ListingContent:
    """Amazon listing content"""
    title: str
    bullet_points: list[str]
    description: str
    backend_keywords: str = ""
    image_count: int = 0
    has_a_plus: bool = False
    has_video: bool = False
    brand: str = ""
    category: str = ""


@dataclass
class OptimizationIssue:
    """Single optimization issue"""
    section: ListingSection
    priority: OptimizationPriority
    issue: str
    recommendation: str
    current_value: Optional[str] = None
    suggested_value: Optional[str] = None
    impact: str = ""  # Expected impact of fixing


@dataclass
class KeywordPlacement:
    """Keyword placement analysis"""
    keyword: str
    in_title: bool
    in_bullets: bool
    in_description: bool
    in_backend: bool
    placement_score: int  # 0-100
    recommendation: str


@dataclass
class SectionScore:
    """Score for a listing section"""
    section: ListingSection
    score: int  # 0-100
    max_score: int
    issues: list[OptimizationIssue]
    strengths: list[str]


@dataclass
class ListingScore:
    """Complete listing optimization score"""
    overall_score: int  # 0-100
    grade: str  # A, B, C, D, F
    section_scores: dict[ListingSection, SectionScore]
    critical_issues: list[OptimizationIssue]
    all_issues: list[OptimizationIssue]
    keyword_analysis: list[KeywordPlacement]
    competitor_comparison: Optional[dict] = None
    mobile_score: int = 0
    seo_score: int = 0


class ListingOptimizer:
    """Analyzes and optimizes Amazon product listings"""

    # Amazon limits
    TITLE_MAX_LENGTH = 200
    TITLE_RECOMMENDED_LENGTH = 150
    BULLET_MAX_LENGTH = 500
    BULLET_RECOMMENDED_LENGTH = 200
    BACKEND_MAX_BYTES = 249
    DESCRIPTION_MAX_LENGTH = 2000
    MIN_IMAGES = 6
    RECOMMENDED_IMAGES = 7

    # Title patterns to avoid
    TITLE_BANNED_PATTERNS = [
        r'\b(best|top|#1|number one|leading)\b',  # Subjective claims
        r'\b(free shipping|sale|discount|cheap)\b',  # Promotional
        r'[!@#$%^&*()+=\[\]{}|\\<>]',  # Special characters
        r'\b(guaranteed|warranty)\b',  # Guarantee claims in title
    ]

    # Power words for bullets
    BULLET_POWER_WORDS = [
        "premium", "professional", "durable", "lightweight", "portable",
        "versatile", "adjustable", "ergonomic", "eco-friendly", "waterproof",
        "rechargeable", "wireless", "compact", "heavy-duty", "multi-purpose"
    ]

    # Benefit-focused words
    BENEFIT_WORDS = [
        "save", "protect", "improve", "enhance", "maximize", "simplify",
        "eliminate", "reduce", "increase", "enjoy", "experience", "discover"
    ]

    def __init__(self):
        self.issues: list[OptimizationIssue] = []

    def analyze_listing(self, listing: ListingContent,
                       target_keywords: list[str]) -> ListingScore:
        """Complete listing analysis"""
        self.issues = []

        # Analyze each section
        title_score = self._analyze_title(listing.title, target_keywords)
        bullet_score = self._analyze_bullets(listing.bullet_points, target_keywords)
        desc_score = self._analyze_description(listing.description, target_keywords)
        backend_score = self._analyze_backend(listing.backend_keywords, target_keywords, listing)
        image_score = self._analyze_images(listing.image_count, listing.has_video)
        aplus_score = self._analyze_aplus(listing.has_a_plus)

        section_scores = {
            ListingSection.TITLE: title_score,
            ListingSection.BULLETS: bullet_score,
            ListingSection.DESCRIPTION: desc_score,
            ListingSection.BACKEND: backend_score,
            ListingSection.IMAGES: image_score,
            ListingSection.A_PLUS: aplus_score,
        }

        # Keyword placement analysis
        keyword_analysis = self._analyze_keyword_placement(listing, target_keywords)

        # Calculate overall score (weighted)
        weights = {
            ListingSection.TITLE: 0.25,
            ListingSection.BULLETS: 0.25,
            ListingSection.DESCRIPTION: 0.10,
            ListingSection.BACKEND: 0.15,
            ListingSection.IMAGES: 0.15,
            ListingSection.A_PLUS: 0.10,
        }

        overall = sum(
            section_scores[section].score * weight
            for section, weight in weights.items()
        )

        # Grade
        grade = self._calculate_grade(overall)

        # Mobile and SEO scores
        mobile_score = self._calculate_mobile_score(listing)
        seo_score = self._calculate_seo_score(listing, target_keywords)

        # Critical issues
        critical = [i for i in self.issues if i.priority == OptimizationPriority.CRITICAL]

        return ListingScore(
            overall_score=int(overall),
            grade=grade,
            section_scores=section_scores,
            critical_issues=critical,
            all_issues=sorted(self.issues, key=lambda x: (
                x.priority != OptimizationPriority.CRITICAL,
                x.priority != OptimizationPriority.HIGH,
                x.priority != OptimizationPriority.MEDIUM,
            )),
            keyword_analysis=keyword_analysis,
            mobile_score=mobile_score,
            seo_score=seo_score,
        )

    def _analyze_title(self, title: str, keywords: list[str]) -> SectionScore:
        """Analyze title optimization"""
        issues = []
        strengths = []
        score = 100

        # Length check
        title_len = len(title)
        if title_len > self.TITLE_MAX_LENGTH:
            score -= 30
            issues.append(OptimizationIssue(
                section=ListingSection.TITLE,
                priority=OptimizationPriority.CRITICAL,
                issue=f"Title exceeds maximum length ({title_len}/{self.TITLE_MAX_LENGTH})",
                recommendation="Shorten title to under 200 characters",
                impact="May be truncated or rejected by Amazon"
            ))
        elif title_len > self.TITLE_RECOMMENDED_LENGTH:
            score -= 10
            issues.append(OptimizationIssue(
                section=ListingSection.TITLE,
                priority=OptimizationPriority.MEDIUM,
                issue=f"Title longer than recommended ({title_len} chars)",
                recommendation="Consider shortening to ~150 characters for mobile",
                impact="Better mobile display"
            ))
        elif title_len < 80:
            score -= 15
            issues.append(OptimizationIssue(
                section=ListingSection.TITLE,
                priority=OptimizationPriority.HIGH,
                issue="Title too short - missing keyword opportunities",
                recommendation="Add more relevant keywords to title",
                impact="Improved search visibility"
            ))
        else:
            strengths.append(f"Good title length ({title_len} chars)")

        # Keyword presence
        title_lower = title.lower()
        keywords_in_title = [kw for kw in keywords if kw.lower() in title_lower]
        keyword_ratio = len(keywords_in_title) / max(1, len(keywords))

        if keyword_ratio < 0.3:
            score -= 25
            issues.append(OptimizationIssue(
                section=ListingSection.TITLE,
                priority=OptimizationPriority.HIGH,
                issue=f"Only {len(keywords_in_title)}/{len(keywords)} target keywords in title",
                recommendation=f"Add keywords: {', '.join(keywords[:3])}",
                impact="Significant ranking improvement"
            ))
        elif keyword_ratio >= 0.6:
            strengths.append(f"Good keyword coverage ({len(keywords_in_title)} keywords)")

        # Primary keyword at start
        if keywords and keywords[0].lower() not in title_lower[:50]:
            score -= 15
            issues.append(OptimizationIssue(
                section=ListingSection.TITLE,
                priority=OptimizationPriority.HIGH,
                issue="Primary keyword not at beginning of title",
                recommendation=f"Move '{keywords[0]}' to the start of the title",
                impact="Better ranking for main keyword"
            ))
        elif keywords:
            strengths.append("Primary keyword positioned well")

        # Check for banned patterns
        for pattern in self.TITLE_BANNED_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                score -= 10
                issues.append(OptimizationIssue(
                    section=ListingSection.TITLE,
                    priority=OptimizationPriority.MEDIUM,
                    issue="Title contains discouraged terms",
                    recommendation="Remove subjective/promotional language",
                    impact="Avoid policy issues"
                ))
                break

        # Brand presence
        if not any(word.istitle() for word in title.split()[:3]):
            score -= 5
            issues.append(OptimizationIssue(
                section=ListingSection.TITLE,
                priority=OptimizationPriority.LOW,
                issue="Brand name may not be prominent",
                recommendation="Start title with brand name",
                impact="Brand recognition"
            ))

        # Capitalization check
        words = title.split()
        all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
        if all_caps_words > 2:
            score -= 10
            issues.append(OptimizationIssue(
                section=ListingSection.TITLE,
                priority=OptimizationPriority.MEDIUM,
                issue="Too many ALL CAPS words",
                recommendation="Use Title Case instead of ALL CAPS",
                impact="Better readability, avoid policy issues"
            ))

        self.issues.extend(issues)
        return SectionScore(
            section=ListingSection.TITLE,
            score=max(0, score),
            max_score=100,
            issues=issues,
            strengths=strengths
        )

    def _analyze_bullets(self, bullets: list[str], keywords: list[str]) -> SectionScore:
        """Analyze bullet points"""
        issues = []
        strengths = []
        score = 100

        # Check bullet count
        if len(bullets) < 5:
            score -= 20
            issues.append(OptimizationIssue(
                section=ListingSection.BULLETS,
                priority=OptimizationPriority.HIGH,
                issue=f"Only {len(bullets)} bullet points (5 recommended)",
                recommendation="Add more bullet points to highlight features",
                impact="More opportunities to convert and rank"
            ))
        elif len(bullets) >= 5:
            strengths.append("Using all 5 bullet points")

        # Analyze each bullet
        total_keywords_used = set()
        for i, bullet in enumerate(bullets, 1):
            bullet_len = len(bullet)

            # Length check
            if bullet_len > self.BULLET_MAX_LENGTH:
                score -= 10
                issues.append(OptimizationIssue(
                    section=ListingSection.BULLETS,
                    priority=OptimizationPriority.MEDIUM,
                    issue=f"Bullet {i} exceeds maximum length ({bullet_len} chars)",
                    recommendation="Shorten to under 500 characters",
                    current_value=bullet[:50] + "...",
                    impact="May be truncated"
                ))
            elif bullet_len < 100:
                score -= 5
                issues.append(OptimizationIssue(
                    section=ListingSection.BULLETS,
                    priority=OptimizationPriority.LOW,
                    issue=f"Bullet {i} is short ({bullet_len} chars)",
                    recommendation="Expand with more details and keywords",
                    impact="Missing keyword opportunities"
                ))

            # Keyword check
            bullet_lower = bullet.lower()
            for kw in keywords:
                if kw.lower() in bullet_lower:
                    total_keywords_used.add(kw.lower())

            # Check for benefit-focused language
            has_benefit = any(word in bullet_lower for word in self.BENEFIT_WORDS)
            has_power = any(word in bullet_lower for word in self.BULLET_POWER_WORDS)

            if not has_benefit and not has_power:
                score -= 3

            # Check for ALL CAPS start (common practice)
            if not bullet.split()[0].isupper() if bullet else True:
                pass  # Minor issue, not penalizing

        # Overall keyword coverage in bullets
        keyword_coverage = len(total_keywords_used) / max(1, len(keywords))
        if keyword_coverage < 0.5:
            score -= 15
            issues.append(OptimizationIssue(
                section=ListingSection.BULLETS,
                priority=OptimizationPriority.HIGH,
                issue=f"Low keyword coverage in bullets ({keyword_coverage*100:.0f}%)",
                recommendation="Incorporate more target keywords naturally",
                impact="Improved search ranking"
            ))
        else:
            strengths.append(f"Good keyword coverage ({keyword_coverage*100:.0f}%)")

        # Check for feature vs benefit balance
        all_bullets = " ".join(bullets).lower()
        benefit_count = sum(1 for word in self.BENEFIT_WORDS if word in all_bullets)
        if benefit_count < 2:
            score -= 10
            issues.append(OptimizationIssue(
                section=ListingSection.BULLETS,
                priority=OptimizationPriority.MEDIUM,
                issue="Bullets focus too much on features, not benefits",
                recommendation="Add customer benefits: what problem does it solve?",
                impact="Higher conversion rate"
            ))

        self.issues.extend(issues)
        return SectionScore(
            section=ListingSection.BULLETS,
            score=max(0, score),
            max_score=100,
            issues=issues,
            strengths=strengths
        )

    def _analyze_description(self, description: str, keywords: list[str]) -> SectionScore:
        """Analyze product description"""
        issues = []
        strengths = []
        score = 100

        desc_len = len(description)

        if desc_len < 200:
            score -= 30
            issues.append(OptimizationIssue(
                section=ListingSection.DESCRIPTION,
                priority=OptimizationPriority.HIGH,
                issue="Description too short",
                recommendation="Expand description to at least 1000 characters",
                impact="More indexable content for search"
            ))
        elif desc_len < 500:
            score -= 15
            issues.append(OptimizationIssue(
                section=ListingSection.DESCRIPTION,
                priority=OptimizationPriority.MEDIUM,
                issue="Description could be longer",
                recommendation="Add more details about product usage and benefits",
                impact="Better SEO and customer information"
            ))
        elif desc_len > 1000:
            strengths.append("Good description length")

        # Keyword presence
        desc_lower = description.lower()
        keywords_found = sum(1 for kw in keywords if kw.lower() in desc_lower)
        if keywords_found < len(keywords) * 0.3:
            score -= 15
            issues.append(OptimizationIssue(
                section=ListingSection.DESCRIPTION,
                priority=OptimizationPriority.MEDIUM,
                issue="Low keyword presence in description",
                recommendation="Naturally incorporate more target keywords",
                impact="Improved indexing"
            ))

        # Check for HTML formatting (basic check)
        if '<' not in description and desc_len > 500:
            score -= 5
            issues.append(OptimizationIssue(
                section=ListingSection.DESCRIPTION,
                priority=OptimizationPriority.LOW,
                issue="No HTML formatting detected",
                recommendation="Use <br>, <b>, <ul> tags for better readability",
                impact="Better customer experience"
            ))

        self.issues.extend(issues)
        return SectionScore(
            section=ListingSection.DESCRIPTION,
            score=max(0, score),
            max_score=100,
            issues=issues,
            strengths=strengths
        )

    def _analyze_backend(self, backend: str, keywords: list[str],
                        listing: ListingContent) -> SectionScore:
        """Analyze backend search terms"""
        issues = []
        strengths = []
        score = 100

        backend_bytes = len(backend.encode('utf-8'))

        if backend_bytes == 0:
            score -= 40
            issues.append(OptimizationIssue(
                section=ListingSection.BACKEND,
                priority=OptimizationPriority.CRITICAL,
                issue="No backend keywords set",
                recommendation="Add backend search terms (up to 249 bytes)",
                impact="Major ranking improvement opportunity"
            ))
        elif backend_bytes > self.BACKEND_MAX_BYTES:
            score -= 20
            issues.append(OptimizationIssue(
                section=ListingSection.BACKEND,
                priority=OptimizationPriority.HIGH,
                issue=f"Backend keywords exceed limit ({backend_bytes}/249 bytes)",
                recommendation="Reduce to 249 bytes - Amazon will ignore excess",
                impact="Ensure all keywords are indexed"
            ))
        elif backend_bytes < 200:
            score -= 10
            issues.append(OptimizationIssue(
                section=ListingSection.BACKEND,
                priority=OptimizationPriority.MEDIUM,
                issue=f"Backend keywords underutilized ({backend_bytes}/249 bytes)",
                recommendation="Add more relevant search terms",
                impact="More keyword coverage"
            ))
        else:
            strengths.append(f"Good backend keyword usage ({backend_bytes}/249 bytes)")

        # Check for duplicates from title/bullets
        frontend_text = (listing.title + " " + " ".join(listing.bullet_points)).lower()
        backend_words = set(backend.lower().split())
        frontend_words = set(frontend_text.split())
        duplicates = backend_words & frontend_words

        if len(duplicates) > 5:
            score -= 10
            issues.append(OptimizationIssue(
                section=ListingSection.BACKEND,
                priority=OptimizationPriority.MEDIUM,
                issue="Backend contains words already in title/bullets",
                recommendation="Remove duplicates, add unique synonyms instead",
                current_value=", ".join(list(duplicates)[:5]),
                impact="Maximize keyword coverage"
            ))

        # Check for commas (not needed)
        if ',' in backend:
            score -= 5
            issues.append(OptimizationIssue(
                section=ListingSection.BACKEND,
                priority=OptimizationPriority.LOW,
                issue="Backend contains commas",
                recommendation="Use spaces instead of commas to save bytes",
                impact="More keywords fit in limit"
            ))

        self.issues.extend(issues)
        return SectionScore(
            section=ListingSection.BACKEND,
            score=max(0, score),
            max_score=100,
            issues=issues,
            strengths=strengths
        )

    def _analyze_images(self, image_count: int, has_video: bool) -> SectionScore:
        """Analyze image optimization"""
        issues = []
        strengths = []
        score = 100

        if image_count < 1:
            score -= 50
            issues.append(OptimizationIssue(
                section=ListingSection.IMAGES,
                priority=OptimizationPriority.CRITICAL,
                issue="No product images",
                recommendation="Add at least 6 high-quality images",
                impact="Critical for conversions"
            ))
        elif image_count < self.MIN_IMAGES:
            score -= 25
            issues.append(OptimizationIssue(
                section=ListingSection.IMAGES,
                priority=OptimizationPriority.HIGH,
                issue=f"Only {image_count} images (6+ recommended)",
                recommendation="Add more images: lifestyle, infographic, size comparison",
                impact="Higher conversion rate"
            ))
        elif image_count >= self.RECOMMENDED_IMAGES:
            strengths.append(f"Good image count ({image_count} images)")

        if not has_video:
            score -= 15
            issues.append(OptimizationIssue(
                section=ListingSection.IMAGES,
                priority=OptimizationPriority.MEDIUM,
                issue="No product video",
                recommendation="Add a product video for better engagement",
                impact="Up to 3x higher conversion"
            ))
        else:
            strengths.append("Has product video")

        self.issues.extend(issues)
        return SectionScore(
            section=ListingSection.IMAGES,
            score=max(0, score),
            max_score=100,
            issues=issues,
            strengths=strengths
        )

    def _analyze_aplus(self, has_aplus: bool) -> SectionScore:
        """Analyze A+ Content"""
        issues = []
        strengths = []
        score = 100

        if not has_aplus:
            score -= 30
            issues.append(OptimizationIssue(
                section=ListingSection.A_PLUS,
                priority=OptimizationPriority.HIGH,
                issue="No A+ Content",
                recommendation="Add A+ Content for brand storytelling and comparison charts",
                impact="5-10% conversion increase"
            ))
        else:
            strengths.append("Has A+ Content")
            score = 100  # Full marks for having it

        self.issues.extend(issues)
        return SectionScore(
            section=ListingSection.A_PLUS,
            score=max(0, score),
            max_score=100,
            issues=issues,
            strengths=strengths
        )

    def _analyze_keyword_placement(self, listing: ListingContent,
                                   keywords: list[str]) -> list[KeywordPlacement]:
        """Analyze where keywords appear"""
        results = []
        title_lower = listing.title.lower()
        bullets_lower = " ".join(listing.bullet_points).lower()
        desc_lower = listing.description.lower()
        backend_lower = listing.backend_keywords.lower()

        for kw in keywords:
            kw_lower = kw.lower()
            in_title = kw_lower in title_lower
            in_bullets = kw_lower in bullets_lower
            in_desc = kw_lower in desc_lower
            in_backend = kw_lower in backend_lower

            # Calculate placement score
            score = 0
            if in_title:
                score += 40
            if in_bullets:
                score += 30
            if in_desc:
                score += 15
            if in_backend:
                score += 15

            # Recommendation
            if score == 100:
                rec = "Excellent coverage"
            elif not in_title:
                rec = "Add to title for maximum impact"
            elif not in_bullets:
                rec = "Add to bullet points"
            elif not in_backend:
                rec = "Consider adding to backend"
            else:
                rec = "Good placement"

            results.append(KeywordPlacement(
                keyword=kw,
                in_title=in_title,
                in_bullets=in_bullets,
                in_description=in_desc,
                in_backend=in_backend,
                placement_score=score,
                recommendation=rec
            ))

        return sorted(results, key=lambda x: x.placement_score)

    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _calculate_mobile_score(self, listing: ListingContent) -> int:
        """Calculate mobile optimization score"""
        score = 100

        # Title length for mobile
        if len(listing.title) > 80:
            score -= 20  # Truncated on mobile

        # First bullet should be compelling (most visible)
        if listing.bullet_points and len(listing.bullet_points[0]) > 200:
            score -= 10

        # Image count matters on mobile
        if listing.image_count < 3:
            score -= 30

        return max(0, score)

    def _calculate_seo_score(self, listing: ListingContent,
                            keywords: list[str]) -> int:
        """Calculate SEO score"""
        score = 100
        all_text = f"{listing.title} {' '.join(listing.bullet_points)} {listing.description} {listing.backend_keywords}".lower()

        # Keyword density
        keyword_hits = sum(1 for kw in keywords if kw.lower() in all_text)
        keyword_ratio = keyword_hits / max(1, len(keywords))

        if keyword_ratio < 0.5:
            score -= 30
        elif keyword_ratio < 0.7:
            score -= 15

        # Backend utilization
        if len(listing.backend_keywords.encode()) < 100:
            score -= 20

        return max(0, score)

    def generate_optimized_title(self, current_title: str, keywords: list[str],
                                 brand: str, max_length: int = 150) -> str:
        """Generate an optimized title suggestion"""
        # Start with brand
        parts = [brand] if brand else []

        # Add primary keyword
        if keywords:
            parts.append(keywords[0].title())

        # Add current title keywords (deduped)
        current_words = set(current_title.lower().split())
        keyword_words = set(kw.lower() for kw in keywords)

        # Extract important words from current title
        for word in current_title.split():
            if len(word) > 3 and word.lower() not in keyword_words:
                parts.append(word)

        # Add secondary keywords
        for kw in keywords[1:4]:
            if kw.lower() not in " ".join(parts).lower():
                parts.append(kw.title())

        # Build title
        title = " - ".join(parts[:2]) + " " + " ".join(parts[2:])

        # Truncate if needed
        if len(title) > max_length:
            title = title[:max_length-3] + "..."

        return title


def create_sample_listing() -> ListingContent:
    """Create sample listing for testing"""
    return ListingContent(
        title="Wireless Earbuds Bluetooth Headphones",
        bullet_points=[
            "HIGH QUALITY SOUND - Crystal clear audio with deep bass",
            "LONG BATTERY - Up to 8 hours playtime, 32 hours with case",
            "COMFORTABLE FIT - Ergonomic design with 3 ear tip sizes",
            "EASY PAIRING - One-step Bluetooth connection to any device",
        ],
        description="Experience premium sound quality with our wireless earbuds.",
        backend_keywords="earphones headset music audio",
        image_count=4,
        has_a_plus=False,
        has_video=False,
        brand="TechSound",
        category="Electronics"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("LISTING OPTIMIZER")
    print("=" * 70)

    optimizer = ListingOptimizer()
    listing = create_sample_listing()

    keywords = [
        "wireless earbuds",
        "bluetooth earbuds",
        "earbuds with microphone",
        "sport earbuds",
        "noise canceling earbuds"
    ]

    result = optimizer.analyze_listing(listing, keywords)

    print(f"\nOVERALL SCORE: {result.overall_score}/100 (Grade: {result.grade})")
    print(f"Mobile Score: {result.mobile_score}/100")
    print(f"SEO Score: {result.seo_score}/100")

    print("\n" + "-" * 70)
    print("SECTION SCORES")
    print("-" * 70)
    for section, score in result.section_scores.items():
        print(f"  {section.value:20} {score.score:3}/100")
        if score.strengths:
            for s in score.strengths:
                print(f"    ✓ {s}")
        if score.issues:
            for i in score.issues[:2]:
                print(f"    ✗ [{i.priority.value}] {i.issue}")

    print("\n" + "-" * 70)
    print("CRITICAL ISSUES")
    print("-" * 70)
    if result.critical_issues:
        for issue in result.critical_issues:
            print(f"  ⚠ {issue.issue}")
            print(f"    → {issue.recommendation}")
    else:
        print("  No critical issues found")

    print("\n" + "-" * 70)
    print("KEYWORD PLACEMENT")
    print("-" * 70)
    print(f"  {'Keyword':<25} {'Title':^6} {'Bullet':^6} {'Desc':^6} {'Back':^6} {'Score':^6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for kp in result.keyword_analysis:
        print(f"  {kp.keyword:<25} {'✓' if kp.in_title else '✗':^6} {'✓' if kp.in_bullets else '✗':^6} {'✓' if kp.in_description else '✗':^6} {'✓' if kp.in_backend else '✗':^6} {kp.placement_score:^6}")

    print("\n" + "-" * 70)
    print("TOP RECOMMENDATIONS")
    print("-" * 70)
    for i, issue in enumerate(result.all_issues[:5], 1):
        print(f"  {i}. [{issue.priority.value.upper()}] {issue.issue}")
        print(f"     → {issue.recommendation}")

    # Generate optimized title
    print("\n" + "-" * 70)
    print("SUGGESTED TITLE")
    print("-" * 70)
    optimized = optimizer.generate_optimized_title(
        listing.title, keywords, listing.brand
    )
    print(f"  Current: {listing.title}")
    print(f"  Suggested: {optimized}")

    print("\n" + "=" * 70)
