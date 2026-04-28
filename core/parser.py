import fitz
import re
import uuid
import json
import unicodedata
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Any

from models.schemas import ResearchPaper, PaperSection


# ─────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────

KNOWN_HEADINGS = [
    "abstract", "introduction", "background", "related work",
    "literature review", "methodology", "materials and methods",
    "proposed work", "method", "methods", "approach", "system design",
    "architecture", "framework", "model", "training", "preliminaries",
    "problem statement", "experiments", "experimental setup",
    "experimental results", "results", "findings", "evaluation",
    "ablation", "ablation study", "analysis", "case study",
    "discussion", "conclusion", "conclusions", "limitations",
    "future work", "acknowledgements", "acknowledgments",
    "references", "bibliography", "works cited", "data", "dataset",
    "threat model",
]
KNOWN_HEADINGS_SET = {h.lower() for h in KNOWN_HEADINGS}

REFERENCE_PATTERNS = [
    r'^\[\d+\]\s+.{10,}',
    r'^\d+\.\s+[A-Z].{10,}',
    r'^[A-Z][a-z]+,\s+[A-Z]\.?.{10,}',
    r'.{10,}\(\d{4}\)[.,].{5,}',
    r'^\d+\.\s+[A-Z][a-z]+\s+[A-Z]{1,3}[.,].{5,}',
    r'^[A-Z][a-z]+\s+[A-Z]+\s+\(\d{4}\).{5,}',
]

VENUE_TOKENS = (
    r'(Proceedings of|Journal of|Conference on|Workshop on|Transactions on|'
    r'arXiv:\d{4}\.\d{4,5}|bioRxiv|medRxiv|IEEE|ACM|ICML|NeurIPS|CVPR|ICCV|'
    r'ECCV|ACL|NAACL|EMNLP|ICLR|AAAI|KDD|SIGIR|WWW|TACL|'
    r'Findings of (?:ACL|EMNLP|NAACL)|Springer|Elsevier|Nature|Science|'
    r'PLOS|PLoS|Cell|Lancet)'
)

AFFIL_RE = re.compile(
    r'university|institute|department|college|school|laboratory|\blab\b|'
    r'center|centre|faculty|@|\.com|\.edu|\.org|\bcorp\b|\binc\b|\bltd\b',
    re.IGNORECASE,
)

# OS/default usernames that commonly leak into PDF Author metadata.
JUNK_AUTHOR_TOKENS = {
    "hp", "user", "admin", "administrator", "owner", "pc", "dell",
    "lenovo", "acer", "asus", "windows", "root", "guest", "default",
    "microsoft office user", "ms office user", "office user",
}

YEAR_MIN = 1950
YEAR_MAX = 2030

HEADING_ACCEPT_THRESHOLD = 0.50   # tunable


# ─────────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Line:
    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool
    page_width: float
    page_height: float


@dataclass
class Extracted:
    value: Any
    confidence: float
    source: str


@dataclass
class HeadingAnchor:
    line_idx: int
    text: str
    score: float


# ─────────────────────────────────────────────────────────────────────
#  TEXT UTILITIES
# ─────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[\ue000-\uf8ff]', '', text)
    return text


def _clean_text(text: str, repeating: set) -> str:
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if len(s) <= 1:
            out.append(line)
            continue
        if s in repeating:
            continue
        if re.search(r'\b\d{3,4}[–\-]\d{3,4}\b', s):
            continue
        if re.search(r'et al\.\s*\/', s, re.IGNORECASE):
            continue
        if re.fullmatch(r'(Page\s+)?\d{1,4}(\s+of\s+\d+)?', s, re.IGNORECASE):
            continue
        if re.fullmatch(r'https?://\S+', s):
            continue
        out.append(line)
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────
#  PAGE MODEL — structure-aware view over the PDF
# ─────────────────────────────────────────────────────────────────────

class PageModel:
    def __init__(self, doc):
        self.doc = doc
        self.lines: List[Line] = self._build_lines()
        self.layout = self._detect_layout()
        self.body_size = self._body_size()
        self.repeating = self._repeating_lines()

    def _build_lines(self) -> List[Line]:
        out: List[Line] = []
        for pi, page in enumerate(self.doc):
            pw, ph = page.rect.width, page.rect.height
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for ln in block.get("lines", []):
                    spans = ln.get("spans", [])
                    if not spans:
                        continue
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if not text:
                        continue
                    total = sum(len(s.get("text", "")) for s in spans) or 1
                    size = sum(
                        s.get("size", 0) * len(s.get("text", ""))
                        for s in spans
                    ) / total
                    bold = any(s.get("flags", 0) & 16 for s in spans)
                    bbox = ln.get("bbox", [0, 0, 0, 0])
                    out.append(Line(
                        text=text, page=pi,
                        x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                        size=round(size, 2), bold=bold,
                        page_width=pw, page_height=ph,
                    ))
        return out

    def _detect_layout(self) -> str:
        # Vote across first 5 pages — single page is too brittle.
        n = min(5, len(self.doc))
        if n == 0:
            return "scanned"
        votes = {"two_column": 0, "text": 0, "scanned": 0}
        for pi in range(n):
            page = self.doc[pi]
            blocks = [b for b in page.get_text("blocks")
                      if b[6] == 0 and b[4].strip()]
            if not blocks:
                votes["scanned"] += 1
                continue
            mid = page.rect.width * 0.45
            left = [b for b in blocks if b[0] < mid]
            right = [b for b in blocks if b[0] > mid]
            if len(left) >= 3 and len(right) >= 3:
                votes["two_column"] += 1
            else:
                votes["text"] += 1
        if votes["scanned"] == n:
            return "scanned"
        return "two_column" if votes["two_column"] > votes["text"] else "text"

    def _body_size(self) -> float:
        sizes = [round(l.size, 1) for l in self.lines if len(l.text) > 20]
        if not sizes:
            return 10.0
        return Counter(sizes).most_common(1)[0][0]

    def _repeating_lines(self) -> set:
        # Compute on normalized text so matching survives _clean_text().
        page_map: Dict[str, set] = defaultdict(set)
        for l in self.lines:
            t = _normalize(l.text).strip()
            if len(t) > 5:
                page_map[t].add(l.page)
        return {t for t, pages in page_map.items() if len(pages) >= 3}

    def column_aware_text(self) -> str:
        if self.layout != "two_column":
            return "\n".join(l.text for l in self.lines)
        out: List[str] = []
        for pi in range(len(self.doc)):
            page_lines = [l for l in self.lines if l.page == pi]
            if not page_lines:
                continue
            mid = page_lines[0].page_width / 2
            left = sorted([l for l in page_lines if l.x0 < mid],
                          key=lambda l: l.y0)
            right = sorted([l for l in page_lines if l.x0 >= mid],
                           key=lambda l: l.y0)
            out.extend(l.text for l in left)
            out.extend(l.text for l in right)
        return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────
#  VALIDATORS
# ─────────────────────────────────────────────────────────────────────

def _validate_title(t: Optional[str]) -> bool:
    if not t:
        return False
    t = t.strip()
    if len(t) < 10 or len(t) > 250:
        return False
    if t.lower().startswith("microsoft word"):
        return False
    if re.fullmatch(r'https?://\S+', t):
        return False
    # Reject venue/journal banners — these sit at the very top of page 1
    # in a large font and otherwise win the title-scoring contest.
    # Use match (anchored) so a real title that *mentions* a journal mid-string survives.
    if re.match(rf'^\s*{VENUE_TOKENS}', t, re.IGNORECASE):
        return False
    if re.search(r'\bISSN\b|\bDOI\b|\bvol\.?\s*\d', t, re.IGNORECASE):
        return False
    return True


def _validate_author(name: str) -> bool:
    if not name:
        return False
    name = name.strip()
    if len(name) < 3 or len(name) > 80:
        return False
    if name.lower() in JUNK_AUTHOR_TOKENS:
        return False
    if re.search(r'\d{3,}', name):
        return False
    if AFFIL_RE.search(name):
        return False
    if not re.search(r'[A-Za-z]', name):
        return False
    # A real name has a separator (space / hyphen / dot). Single-token
    # strings like "hp" or "admin" are almost always metadata junk.
    if not re.search(r'[\s\-\.]', name):
        return False
    return True


def _validate_year(y: Any) -> bool:
    try:
        y = int(y)
    except Exception:
        return False
    return YEAR_MIN <= y <= YEAR_MAX


def _validate_abstract(t: Optional[str]) -> bool:
    if not t:
        return False
    t = t.strip()
    if len(t) < 100 or len(t) > 4000:
        return False
    if t.count('.') < 2:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────
#  HEADING SCORER
# ─────────────────────────────────────────────────────────────────────

def _heading_score(line: Line, body_size: float, full_text_lower: str) -> float:
    t = line.text.strip()
    if not t or len(t) > 120 or len(t) < 2:
        return 0.0
    if re.match(r'^(Fig(ure)?\.?|Table|Algorithm|Listing)\s*\d+',
                t, re.IGNORECASE):
        return 0.0

    t_low = t.lower()
    stripped = re.sub(
        r'^(\d+(\.\d+)*\.?|[IVX]+\.|[A-Z]\.)\s+', '', t_low
    ).strip().rstrip(':')

    # Hard override — clean known heading standing alone.
    if (stripped in KNOWN_HEADINGS_SET or t_low in KNOWN_HEADINGS_SET) \
            and len(t) < 100:
        return 0.9

    size_ratio = line.size / body_size if body_size else 1.0
    size_s = max(0.0, min(1.0, (size_ratio - 1.0) / 0.5))
    bold_s = 1.0 if line.bold else 0.0
    numbered = bool(re.match(
        r'^(\d+(\.\d+)*\.?|[IVX]+\.|[A-Z]\.)\s+\S', t
    ))
    num_s = 1.0 if numbered else 0.0
    short_s = 1.0 if len(t) < 60 else 0.5 if len(t) < 100 else 0.0
    caps_s = 1.0 if (t.isupper() and 3 < len(t) < 80) else 0.0

    penalty = 0.0
    if t[-1] in '.,;?!' and not numbered:
        penalty += 0.4
    if full_text_lower.count(t_low) > 3:
        penalty += 0.3

    score = (0.30 * size_s + 0.20 * bold_s + 0.20 * num_s
             + 0.15 * short_s + 0.15 * caps_s - penalty)
    return max(0.0, min(1.0, score))


def _detect_anchors(pm: PageModel) -> List[HeadingAnchor]:
    full_lower = "\n".join(l.text for l in pm.lines).lower()
    anchors: List[HeadingAnchor] = []
    for i, l in enumerate(pm.lines):
        s = _heading_score(l, pm.body_size, full_lower)
        if s >= HEADING_ACCEPT_THRESHOLD:
            anchors.append(HeadingAnchor(i, l.text.strip(), s))
    return anchors


# ─────────────────────────────────────────────────────────────────────
#  SECTION SPLIT
# ─────────────────────────────────────────────────────────────────────

# Author-bio / contact-info markers that often appear at the end of a paper
# without a proper heading — they bleed into the last real section.
AUTHOR_BIO_MARKERS = re.compile(
    r'(?im)^\s*('
    r'mailing\s+address|e-?mail\s*:|'
    r'orcid\s*(id)?\s*:?\s*https?|orcid\.org/|'
    r'\bmobile\s*:|\btel(?:\.|ephone)?\s*:|\bphone\s*:|\bfax\s*:|'
    r'about\s+the\s+authors?|author\s+biograph(?:y|ies)|biograph(?:y|ies)\s+of\s+author|'
    r'corresponding\s+author\s*:'
    r')'
)


def _trim_author_bio(text: str) -> str:
    """Cuts a trailing author-bio block off a section's content."""
    if not text:
        return text
    m = AUTHOR_BIO_MARKERS.search(text)
    if not m:
        return text
    # Only trim if the bio block is in the latter half — a contact mention
    # mid-paper (e.g., a corresponding-author footnote on page 1) should stay.
    if m.start() < len(text) * 0.4:
        return text
    return text[:m.start()].rstrip()


def _split_sections(pm: PageModel, anchors: List[HeadingAnchor],
                    clean_text: str) -> List[PaperSection]:
    if not anchors:
        return _regex_section_fallback(clean_text)

    out: List[PaperSection] = []
    first = anchors[0].line_idx
    header_txt = "\n".join(l.text for l in pm.lines[:first]).strip()
    if header_txt:
        out.append(PaperSection(
            section_name="Header/Metadata", content=header_txt,
        ))

    for i, a in enumerate(anchors):
        start = a.line_idx + 1
        end = anchors[i + 1].line_idx if i + 1 < len(anchors) else len(pm.lines)
        body = "\n".join(l.text for l in pm.lines[start:end]).strip()
        # Trim trailing author-bio block from the LAST section only.
        if i + 1 == len(anchors):
            body = _trim_author_bio(body)
        if len(body) >= 30:
            out.append(PaperSection(section_name=a.text, content=body))
    return out


def _regex_section_fallback(clean_text: str) -> List[PaperSection]:
    pattern = '|'.join(re.escape(h) for h in KNOWN_HEADINGS)
    parts = re.split(
        rf'(?im)^\s*((?:\d+\.?\d*\s+|[IVX]+\.\s+)?(?:{pattern}))\s*:?\s*$',
        clean_text,
    )
    sections: List[PaperSection] = []
    if parts and parts[0].strip():
        sections.append(PaperSection(
            section_name="Header/Metadata", content=parts[0].strip(),
        ))
    last_idx = max((i for i in range(1, len(parts), 2)), default=-1)
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if i == last_idx:
            content = _trim_author_bio(content)
        if len(content) > 30:
            sections.append(PaperSection(section_name=name, content=content))
    return sections


# ─────────────────────────────────────────────────────────────────────
#  TITLE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────

def _score_title(text: str, size: float, body_size: float,
                 y0: float, ph: float) -> float:
    if not text:
        return 0.0
    len_s = 1.0 if 15 <= len(text) <= 200 else 0.3
    size_s = max(0.0, min(1.0,
                 (size / body_size - 1.0) / 0.8 if body_size else 0.5))
    top_s = 1.0 - min(y0 / (ph * 0.4), 1.0)
    bad = bool(re.search(
        r'(doi|vol\.|volume|issue|pp\.|\bno\b|©|copyright|\bissn\b)',
        text, re.IGNORECASE,
    )) or text.lower().startswith("microsoft word") \
       or bool(re.match(rf'^\s*{VENUE_TOKENS}', text, re.IGNORECASE))
    clean_s = 0.0 if bad else 1.0
    return max(0.0, min(1.0,
               0.35 * size_s + 0.25 * len_s + 0.20 * top_s + 0.20 * clean_s))


def extract_title(pm: PageModel, pdf_meta_title: Optional[str]) -> Extracted:
    if pdf_meta_title and _validate_title(pdf_meta_title):
        return Extracted(pdf_meta_title.strip(), 0.9, "pdf_meta")

    p0 = sorted([l for l in pm.lines if l.page == 0], key=lambda l: l.y0)
    if not p0:
        return Extracted(None, 0.0, "none")

    ph = p0[0].page_height
    max_size = max(l.size for l in p0)

    # Group adjacent lines of near-equal size in the top region.
    groups: List[List[Line]] = []
    current: List[Line] = []
    prev_size: Optional[float] = None
    for l in p0:
        if l.y0 > ph * 0.45:
            break
        if l.size >= max_size * 0.9:
            if prev_size is None or abs(l.size - prev_size) < 0.5:
                current.append(l)
            else:
                if current:
                    groups.append(current)
                current = [l]
            prev_size = l.size
        else:
            if current:
                groups.append(current)
            current = []
            prev_size = None
    if current:
        groups.append(current)

    best_text, best_score = None, 0.0
    for g in groups:
        text = " ".join(l.text for l in g).strip()
        s = _score_title(text, g[0].size, pm.body_size, g[0].y0, ph)
        if s > best_score:
            best_score, best_text = s, text

    if best_text and _validate_title(best_text):
        return Extracted(best_text, best_score, "font_score")
    return Extracted(None, 0.0, "none")


# ─────────────────────────────────────────────────────────────────────
#  AUTHOR EXTRACTOR
# ─────────────────────────────────────────────────────────────────────

def _score_author_line(text: str) -> float:
    if len(text) < 3 or len(text) > 250:
        return 0.0
    if AFFIL_RE.search(text):
        return 0.0
    if not re.search(r'[A-Z]', text):
        return 0.0
    tokens = [t.strip() for t in
              re.split(r'[,;]|\band\b|&|\s+', text, flags=re.IGNORECASE)
              if t.strip()]
    if not tokens:
        return 0.0
    name_like = sum(
        1 for t in tokens
        if re.match(r"^[A-Z][A-Za-z\.\-'’]{0,30}$", t)
        or re.match(r"^[a-z]{2,3}$", t)   # particles: de, van, der
    )
    ratio = name_like / len(tokens)
    digit_pen = 0.3 if re.search(r'\b\d{3,}\b', text) else 0.0
    return max(0.0, min(1.0, ratio - digit_pen))


def extract_authors(pm: PageModel, title_text: Optional[str],
                    pdf_meta_authors: Optional[str]) -> Extracted:
    if pdf_meta_authors:
        raw = [a.strip() for a in re.split(r'[;,]', pdf_meta_authors)
               if a.strip()]
        clean = [a for a in raw if _validate_author(a)]
        # Trust PDF metadata only when it looks like multiple real names
        # OR a single multi-word name. Single tokens like "hp" slip past
        # otherwise — they're OS usernames, not authors.
        if len(clean) >= 2 or (clean and ' ' in clean[0]):
            return Extracted(clean, 0.85, "pdf_meta")

    p0 = sorted([l for l in pm.lines if l.page == 0], key=lambda l: l.y0)
    if not p0:
        return Extracted([], 0.0, "none")

    title_end = -1
    if title_text:
        probe = title_text.lower()[:40]
        for i, l in enumerate(p0):
            lt = l.text.lower()
            if probe and (probe in lt or lt[:40] in title_text.lower()):
                title_end = i
    abstract_idx = len(p0)
    for i, l in enumerate(p0):
        if re.match(r'^\s*(abstract|summary)\b', l.text, re.IGNORECASE):
            abstract_idx = i
            break

    zone = p0[title_end + 1:abstract_idx]

    names: List[str] = []
    matched_any = False
    for l in zone:
        s = _score_author_line(l.text)
        if s < 0.4:
            continue
        matched_any = True
        parts = re.split(r',|;|\band\b|&', l.text, flags=re.IGNORECASE)
        for p in parts:
            p = re.sub(r'\d+[\*†‡§¶]*|[\*†‡§¶]', '', p).strip()
            if _validate_author(p):
                names.append(p)

    names = list(dict.fromkeys(names))[:12]
    if names:
        return Extracted(names, 0.7, "font_score")
    return Extracted([], 0.2 if matched_any else 0.0, "none")


# ─────────────────────────────────────────────────────────────────────
#  YEAR, VENUE, KEYWORDS, ABSTRACT
# ─────────────────────────────────────────────────────────────────────

def extract_year(head_text: str) -> Extracted:
    candidates = []
    for m in re.finditer(r'\b(19|20)\d{2}\b', head_text):
        y = int(m.group())
        if not _validate_year(y):
            continue

        before = head_text[max(0, m.start() - 40):m.start()].lower()
        after = head_text[m.end():m.end() + 40].lower()
        ctx = before + " " + after

        # Hard reject — manuscript / copyright / license context.
        if re.search(
            r'(received|accepted|submitted|revised|'
            r'©|\(c\)|copyright|doi\s*[:/]|'
            r'volume|vol\.|issue|pp\.|pages?\s+\d|'
            r'licen[cs]e|correspondence)', ctx,
        ):
            continue

        # Positive signals raise score: venue token nearby is strongest.
        score = 0.55
        window = head_text[max(0, m.start() - 200):m.end() + 200]
        if re.search(VENUE_TOKENS, window, re.IGNORECASE):
            score = 0.85
        if re.search(
            r'(published|proceedings|conference|workshop)', ctx,
        ):
            score = max(score, 0.80)

        candidates.append((y, score, m.start()))

    if not candidates:
        return Extracted(None, 0.0, "none")

    # Highest score wins; ties → latest year (pub ≥ received).
    candidates.sort(key=lambda c: (-c[1], -c[0]))
    y, score, _ = candidates[0]
    return Extracted(y, score, "regex")


def extract_venue(head_text: str) -> Extracted:
    m = re.search(VENUE_TOKENS + r'[^,\n]{0,80}', head_text, re.IGNORECASE)
    if m:
        v = m.group().strip().rstrip('.,:;')
        if 3 <= len(v) <= 120:
            return Extracted(v, 0.7, "regex")
    return Extracted(None, 0.0, "none")


def extract_keywords(head: str) -> List[str]:
    m = re.search(
        r'(?:[Kk]ey\s*[Ww]ords?|Index\s+Terms)\s*[:\-—]?\s*(.+)', head,
    )
    if not m:
        return []
    return [k.strip() for k in re.split(r'[;,·•–]', m.group(1))
            if k.strip()][:15]


def extract_abstract(pm: PageModel, sections: List[PaperSection],
                     clean_text: str) -> Extracted:
    for s in sections:
        if "abstract" in s.section_name.lower():
            content = s.content.strip()
            # Oversized section means the next heading anchor was missed —
            # abstract grabbed everything until the next detected heading.
            # Clip to the first paragraph boundary or first sentence block.
            if len(content) > 3500:
                m = re.search(r'^(.{200,3500}?)\n\s*\n', content, re.DOTALL)
                if m:
                    content = m.group(1).strip()
                else:
                    content = content[:2000].rsplit('.', 1)[0] + '.'
            if _validate_abstract(content):
                return Extracted(content, 0.80, "section_split")

    p0_text = _normalize("\n".join(
        l.text for l in pm.lines if l.page == 0
    ))
    m = re.search(
        r'(?is)\babstract\b[:.\s—\-]+(.{200,3500}?)'
        r'(?=\n\s*(?:keywords?\b|index\s+terms|'
        r'1\.?\s+introduction|i\.\s+introduction|\n\s*introduction\b))',
        p0_text,
    )
    if m and _validate_abstract(m.group(1)):
        return Extracted(m.group(1).strip(), 0.65, "inline_regex")

    for s in sections:
        if s.section_name.lower().startswith("header"):
            m = re.search(
                r'(?is)\babstract\b[:.\s—\-]+(.{200,3500})', s.content,
            )
            if m and _validate_abstract(m.group(1)):
                return Extracted(m.group(1).strip(), 0.55, "inline_regex")

    head = clean_text[:600].strip()
    return Extracted(head, 0.2, "truncate")


# ─────────────────────────────────────────────────────────────────────
#  REFERENCES
# ─────────────────────────────────────────────────────────────────────

def extract_references(sections: List[PaperSection]) -> Tuple[List[str], str]:
    raw = ""
    for s in sections:
        n = s.section_name.lower()
        if any(w in n for w in ("reference", "bibliography", "works cited")):
            raw = s.content
            break
    if not raw:
        return [], ""

    compiled = [re.compile(p, re.MULTILINE) for p in REFERENCE_PATTERNS]
    for pat in compiled:
        matches = pat.findall(raw)
        if len(matches) >= 2:
            return [m.strip() for m in matches if m.strip()], raw

    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    refs: List[str] = []
    current = ""
    for line in lines:
        is_new = any(p.match(line) for p in compiled) \
                 or bool(re.match(r'^(\[\d+\]|\d+[\.\)])', line))
        if is_new and current:
            if len(current) > 20:
                refs.append(current.strip())
            current = line
        elif is_new:
            current = line
        elif current and len(line) > 10:
            current += " " + line
    if current and len(current) > 20:
        refs.append(current.strip())
    return refs, raw


# ─────────────────────────────────────────────────────────────────────
#  LLM REFINER — batched, confidence-triggered
# ─────────────────────────────────────────────────────────────────────

def llm_refine(fields: Dict[str, Extracted], first_page: str, llm,
               threshold: float = 0.6) -> Dict[str, Extracted]:
    needs = [k for k, ex in fields.items() if ex.confidence < threshold]
    if not needs or llm is None:
        return fields

    try:
        keys_block = "\n".join(f'  "{k}": null,' for k in needs).rstrip(",")
        prompt = f"""Extract these fields from the paper first page.
Return ONLY valid JSON. No markdown. Use null if unknown.

{{
{keys_block}
}}

For "authors" return a list of names. For "year" return an integer.

TEXT:
{first_page[:2500]}

JSON:"""
        resp = llm.invoke(prompt)
        raw = resp.content.strip()
        raw = re.sub(r'^```json\s*|```$', '', raw,
                     flags=re.MULTILINE).strip()
        parsed = json.loads(raw)

        for k in needs:
            v = parsed.get(k)
            if v in (None, "", []):
                continue
            if k == "title" and isinstance(v, str) and _validate_title(v):
                fields[k] = Extracted(v.strip(), 0.75, "llm")
            elif k == "authors" and isinstance(v, list):
                clean = [a.strip() for a in v
                         if isinstance(a, str) and _validate_author(a)]
                if clean:
                    fields[k] = Extracted(clean, 0.75, "llm")
            elif k == "year" and _validate_year(v):
                fields[k] = Extracted(int(v), 0.75, "llm")
            elif k == "venue" and isinstance(v, str) and 3 <= len(v) <= 120:
                fields[k] = Extracted(v.strip(), 0.7, "llm")
    except Exception as e:
        # Don't crash parsing on a bad LLM response, but make the failure
        # visible so a silent rate-limit or JSON parse error can be diagnosed.
        print(f"[parser.llm_refine] LLM refinement skipped: {type(e).__name__}: {e}")
    return fields


# ─────────────────────────────────────────────────────────────────────
#  MAIN PARSER
# ─────────────────────────────────────────────────────────────────────

class PDFParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.doc = fitz.open(file_path)
        self.pm = PageModel(self.doc)

    def parse(self, paper_id: Optional[str] = None, llm=None) -> ResearchPaper:
        if paper_id is None:
            paper_id = str(uuid.uuid4())[:8]

        if self.pm.layout == "scanned":
            return ResearchPaper(
                paper_id=paper_id,
                title=paper_id,
                abstract=(
                    " This PDF appears to be scanned (image-only). "
                    "Text extraction requires OCR which is not currently "
                    "supported."
                ),
                page_count=len(self.doc),
            )

        full_text = _normalize(self.pm.column_aware_text())
        clean_text = _clean_text(full_text, self.pm.repeating)

        anchors = _detect_anchors(self.pm)
        sections = _split_sections(self.pm, anchors, clean_text)

        meta = self.doc.metadata or {}
        pdf_title = (meta.get("title") or "").strip() or None
        pdf_authors = (meta.get("author") or "").strip() or None

        # Independent extractors — each reads PageModel directly,
        # not each other's output.
        title_ex = extract_title(self.pm, pdf_title)
        authors_ex = extract_authors(self.pm, title_ex.value, pdf_authors)
        head = clean_text[:3500]
        year_ex = extract_year(head)
        venue_ex = extract_venue(head)

        fields = {
            "title": title_ex,
            "authors": authors_ex,
            "year": year_ex,
            "venue": venue_ex,
        }
        if llm is not None:
            first_page = "\n".join(
                l.text for l in self.pm.lines if l.page == 0
            )[:2500]
            fields = llm_refine(fields, first_page, llm)

        abstract_ex = extract_abstract(self.pm, sections, clean_text)
        # Keywords often appear after a long author/affiliation block — widen
        # the scan window beyond the 3500-char `head` so they aren't missed.
        keywords = extract_keywords(clean_text[:8000])
        parsed_refs, raw_refs = extract_references(sections)

        return ResearchPaper(
            paper_id=paper_id,
            title=fields["title"].value or paper_id,
            authors=fields["authors"].value or [],
            abstract=abstract_ex.value or "",
            year=fields["year"].value,
            venue=fields["venue"].value,
            keywords=keywords,
            sections=sections,
            references=parsed_refs,
            raw_references=raw_refs or None,
            full_text=clean_text,
            page_count=len(self.doc),
        )

    def close(self):
        if self.doc:
            self.doc.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
