import fitz           # PyMuPDF — reads PDF files
import re            
import uuid           # Generates unique IDs for each paper
import unicodedata    # Handles special characters like ligatures, dashes, etc.
from collections import Counter, defaultdict   # Counter counts items, defaultdict is a dict with defaults
from typing import List, Dict, Optional, Tuple
from models.schemas import ResearchPaper, PaperSection

KNOWN_HEADINGS = [
    "abstract", "introduction", "background", "related work",
    "literature review", "methodology", "proposed work", "method",
    "approach", "system design", "architecture", "framework",
    "experiments", "experimental setup", "experimental results",
    "results", "evaluation", "discussion", "conclusion",
    "future work", "acknowledgements", "acknowledgments", "references",
    "bibliography", "works cited",   # added: other names for reference sections
]

# ──────────────────────────────────────────────────────────────────────────────
#  REFERENCE PATTERNS
#  Different papers use different citation styles.
#  We try each pattern and use whichever one gets at least 2 matches.
#
#  Pattern 1: [1] Author et al. ...         ← IEEE / most CS papers
#  Pattern 2: 1. Author Name. ...           ← Numbered without brackets
#  Pattern 3: Author, F. Title ...          ← Author-last-name-first style
#  Pattern 4: ... (2020). or (2020),        ← APA / author-year style
#  Pattern 5: 1. Smith JA, Jones B. ...     ← Vancouver / medical style
#  Pattern 6: Smith J (2017) Title ...      ← Author-year without parens
# ──────────────────────────────────────────────────────────────────────────────

REFERENCE_PATTERNS = [
    r'^\[\d+\]\s+.{10,}',                          # [1] ...
    r'^\d+\.\s+[A-Z].{10,}',                       # 1. Author ...
    r'^[A-Z][a-z]+,\s+[A-Z]\.?.{10,}',             # Smith, J. ...
    r'.{10,}\(\d{4}\)[.,].{5,}',                   # ... (2020). ...
    r'^\d+\.\s+[A-Z][a-z]+\s+[A-Z]{1,3}[.,].{5,}',# 1. Smith JA, ... 
    r'^[A-Z][a-z]+\s+[A-Z]+\s+\(\d{4}\).{5,}',    # Smith J (2017) ... 
]


class PDFParser:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.doc       = fitz.open(file_path)       # Open the PDF file
        self.pdf_type  = self._detect_pdf_type()    # Figure out what kind of PDF it is

    # ================================================================== #
    #  STEP 1 — DETECT PDF TYPE  
    #  - Look at the first page's text blocks
    #  - If blocks exist on BOTH left and right halves → two_column
    #  - If NO blocks at all → scanned (image-only, can't extract text)
    #  - Otherwise → normal single-column text
    # ================================================================== #

    def _detect_pdf_type(self) -> str:
        if len(self.doc) == 0:
            return "scanned"

        page        = self.doc[0]
        # get_text("blocks") returns rectangles of text.
        # b[6] == 0 means it's a text block (not an image block)
        # b[4] is the actual text content of that block
        text_blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]

        if not text_blocks:
            return "scanned"

        # Split the page down the middle and count blocks on each side
        mid_x        = page.rect.width * 0.45
        left_blocks  = [b for b in text_blocks if b[0] < mid_x]
        right_blocks = [b for b in text_blocks if b[0] > mid_x]

        # If there are 3+ blocks on BOTH sides, it's a two-column layout
        if len(left_blocks) >= 3 and len(right_blocks) >= 3:
            return "two_column"

        return "text"

    # ================================================================== #
    #  STEP 2 — TEXT EXTRACTION
    #  IMPORTANT RULE: Always use get_text("text") for actual content.
    #  Never manually join spans — that causes garbage characters.
    #
    #  For two-column: read LEFT column top-to-bottom, then RIGHT column.
    #  This gives us the correct reading order.
    # ================================================================== #

    def _get_page_text(self, page) -> str:
        """Single-column page: PyMuPDF handles reading order automatically."""
        return page.get_text("text")

    def _get_page_text_two_column(self, page) -> str:
        """
        Two-column page: manually sort blocks into left/right,
        then read each column top-to-bottom.
        b[0] = x position (left edge of block)
        b[1] = y position (top edge of block) — used for sorting
        b[4] = text content
        """
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        mid_x  = page.rect.width / 2

        left  = sorted([b for b in blocks if b[0] < mid_x],  key=lambda b: b[1])
        right = sorted([b for b in blocks if b[0] >= mid_x], key=lambda b: b[1])

        return "\n".join(b[4].strip() for b in left + right)

    def _get_full_text(self) -> str:
        """Combine all pages into one big string."""
        if self.pdf_type == "scanned":
            return ""
        elif self.pdf_type == "two_column":
            return "\n".join(self._get_page_text_two_column(p) for p in self.doc)
        else:
            return "\n".join(self._get_page_text(p) for p in self.doc)

    # ================================================================== #
    #  Academic papers use many special characters:
    #    - Greek letters: α, β, γ (used in math/statistics)
    #    - Math symbols: ∑, ∇, ≤
    #    - Ligatures: ﬁ (fi combined), ﬂ (fl combined)
    #    - Dashes: em-dash (—), en-dash (–)
    #
    #  NFKC normalization:
    #    - Converts ligatures to normal letters  (ﬁ → fi)
    #    - Normalizes dashes and quotes to standard forms
    #    - Handles compatibility characters
    #
    #  This runs ONCE right after extraction, before anything else.
    # ================================================================== #

    def _normalize_text(self, text: str) -> str:
        """
        Normalize unicode characters so downstream processing works cleanly.
        NFKC = Normalization Form Compatibility Decomposition + Composition
        """
        # Step 1: NFKC normalization (handles ligatures, compatibility chars)
        text = unicodedata.normalize("NFKC", text)

        # Step 2: Remove control characters that sometimes sneak in from PDFs.
        # We keep \n (newline=10) and \t (tab=9) because those are useful.
        # Everything else below ASCII 32 is invisible garbage.
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Step 3: Remove Private Use Area (PUA) characters.
        # These are Unicode slots (\ue000 to \uf8ff) that fonts sometimes use
        # for custom glyphs. In PDFs they often appear as random symbols or
        # bullets that have no real meaning outside that font.
        text = re.sub(r'[\ue000-\uf8ff]', '', text)

        return text

    # ================================================================== 
    #  Problem: Journal/conference papers repeat lines on every page:
    #    "IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 34, 2023"
    #    "Authorized licensed use limited to: University of..."
    #    "SMITH ET AL.: ATTENTION IS ALL YOU NEED"
    #
    #  These lines end up inside section content and corrupt everything.
    #
    #  Solution: Count how many DIFFERENT pages each line appears on.
    #  If a line appears on 3+ different pages → it's a header/footer → remove it.
    #
    #  Why "3 or more"? A real sentence could appear twice (quote + reference),
    #  but appearing on 3+ different pages is almost certainly boilerplate.
    # ================================================================== #

    def _detect_repeating_lines(self) -> set:
        """
        Returns a set of lines that appear on 3 or more different pages.
        These are headers/footers/watermarks that should be removed.
        """
        # For each line, track which page numbers it appears on
        # defaultdict(set) means: if key doesn't exist, create an empty set
        line_page_map = defaultdict(set)

        for page_num, page in enumerate(self.doc):
            for line in page.get_text("text").split("\n"):
                stripped = line.strip()
                # Only track lines with real content (ignore blanks and single chars)
                if len(stripped) > 5:
                    line_page_map[stripped].add(page_num)

        # Return only lines that appear on 3 or more different pages
        repeating = {
            line for line, pages in line_page_map.items()
            if len(pages) >= 3
        }

        return repeating

    # ================================================================== #
    #  STEP 3 — CLEAN NOISE FROM TEXT
    #  Removes:
    #    - Page numbers ("1", "Page 3 of 10")
    #    - Repeating headers/footers 
    #    - Journal header lines with page ranges (e.g., "123-145")
    #    - "et al. /" citation noise
    #    - Standalone URLs
    # ================================================================== #

    def _clean_text(self, text: str, repeating_lines: set = None) -> str:
        """
        Clean noise from extracted text.
        repeating_lines: the set returned by _detect_repeating_lines()
        """
        if repeating_lines is None:
            repeating_lines = set()

        lines   = text.split("\n")
        cleaned = []

        for line in lines:
            s = line.strip()

            # Keep blank lines (they help preserve paragraph structure)
            if len(s) <= 1:
                cleaned.append(line)
                continue

            # skip any line that's a repeating header/footer
            if s in repeating_lines:
                continue   

            # Skip lines that look like page number ranges (e.g., "123-456")
            if re.search(r'\b\d{3,4}[–\-]\d{3,4}\b', s):
                continue

            # Skip citation noise like "Smith et al. / Journal Name"
            if re.search(r'et al\.\s*\/', s, re.IGNORECASE):
                continue

            # Skip standalone page numbers like "1", "42", "Page 3 of 10"
            if re.fullmatch(r'(Page\s+)?\d{1,4}(\s+of\s+\d+)?', s, re.IGNORECASE):
                continue

            # Skip standalone URLs
            if re.fullmatch(r'https?://\S+', s):
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    # ================================================================== #
    #  STEP 4 — FONT SIZE DETECTION
    #  We need to know the "normal" body text size so we can identify
    #  text that's LARGER than normal (which is likely a heading).
    #
    #  How: collect ALL font sizes across ALL pages, find the most common one.
    #  The most common font size = body text size (there's more body than headings).
    # ================================================================== #

    def _get_body_font_size(self) -> float:
        """Returns the most common font size in the document = body text size."""
        sizes = []
        for page in self.doc:
            # get_text("dict") gives us detailed info including font sizes
            # We use this ONLY for font size detection, not for content
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sz = round(span.get("size", 0), 1)
                        if sz > 0:
                            sizes.append(sz)

        if not sizes:
            return 10.0   # fallback if we can't detect

        # most_common(1) returns [(most_common_size, count)]
        return Counter(sizes).most_common(1)[0][0]

    # ================================================================== #
    #  STEP 4b — FONT-BASED HEADING DETECTION
    #  This finds "custom" headings that aren't in KNOWN_HEADINGS.
    #  Example: "Threat Model", "Dataset Description", "Ablation Study"
    #
    #  A line is a heading candidate if ANY of these is true:
    #    - Font size > body size * 1.15  (it's noticeably larger)
    #    - Font flags include bold (flags & 16)
    #    - ALL CAPS and short (3-60 chars)
    #    - Starts with a section number like "1." or "II."
    #
    # We now filter candidates more strictly:
    #    - Must appear ≤ 2 times (headings don't repeat; body text might)
    #    - Must not end with sentence punctuation (. , ; ?)
    #    - Must not look like a figure/table caption
    # ================================================================== #

    def _detect_headings(self, body_size: float, full_text: str) -> List[str]:
        """
        Returns a list of heading strings found via font analysis.
        full_text is passed so we can filter out lines that repeat too much.
        """
        candidates = set()

        for page in self.doc:
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    # Build the line text from spans (for identification only)
                    line_text = "".join(
                        s.get("text", "") for s in line.get("spans", [])
                    ).strip()

                    # Skip: too short, too long, or empty
                    if not line_text or len(line_text) > 100 or len(line_text) < 2:
                        continue

                    # Skip figure/table captions — these are not section headings
                    if re.match(r'^(Fig(ure)?\.?|Table|Algorithm|Listing)\s*\d+', line_text, re.IGNORECASE):
                        continue

                    # Skip lines ending with sentence punctuation — headings don't end with . , ;
                    if line_text[-1] in '.,:;?!':
                        continue

                    # Check font properties of spans in this line
                    for span in line.get("spans", []):
                        sz    = round(span.get("size", 0), 1)
                        flags = span.get("flags", 0)

                        is_larger   = sz > body_size * 1.15
                        is_bold     = bool(flags & 16)
                        is_caps     = line_text.isupper() and 3 < len(line_text) < 60
                        is_numbered = bool(re.match(r'^(\d+\.?\d*|[IVX]+\.)\s+[A-Z]', line_text))

                        if is_larger or is_bold or is_caps or is_numbered:
                            candidates.add(line_text)
                            break   # one span match per line is enough

        # Filter: remove candidates that appear 3+ times in full text
        # (real headings appear once; if it appears 3+ times it's probably body text)
        filtered = [
            h for h in candidates
            if full_text.lower().count(h.lower()) <= 2
        ]

        return filtered

    # ================================================================== #
    #  STEP 5 — SECTION EXTRACTION
    #
    #   approach: split ONLY when heading is on its OWN LINE (anchored)
    #
    #  Two-tier strategy:
    #    Tier 1 (PRIMARY): Split on KNOWN_HEADINGS with line anchoring
    #    Tier 2 (SUPPLEMENT): Add font-detected custom headings
    #
    #  We prefer known headings as primary because they're the most reliable.
    # ================================================================== #

    def _extract_structured_sections(self, clean_text: str) -> List[PaperSection]:
        """
        Main section extraction. Tries known headings first, then supplements
        with font-detected headings for papers with custom section names.
        """
        body_size = self._get_body_font_size()

        # Tier 1: Try splitting on known headings (most reliable)
        sections = self._split_by_known_headings(clean_text)

        # If we got meaningful sections, we're done
        if len(sections) > 2:
            return sections

        # Tier 2: Known headings didn't work well — try font-detected headings
        # (handles papers with unusual section names)
        font_headings = self._detect_headings(body_size, clean_text)
        if font_headings:
            sections = self._split_by_headings(clean_text, font_headings)
            if len(sections) > 1:
                return sections

        # Last resort: regex fallback on known heading names
        return self._regex_fallback_sections(clean_text)

    def _split_by_known_headings(self, clean_text: str) -> List[PaperSection]:
        """
        The key pattern is:  (?m)^\s*(optional_number + heading_name)\s*$

        Breaking that down:
          (?m)   = multiline mode — ^ and $ match start/end of each LINE
          ^      = start of a line
          \s*    = optional leading whitespace (indented headings)
          (?:...) = the heading text (with optional number prefix)
          \s*$   = optional trailing whitespace, then END of line

        The $ at the end is THE MOST IMPORTANT PART.
        Without it, "introduction" would match inside sentences like
        "This introduces a new method" → causing false splits.
        With $ it only matches when "introduction" is the ENTIRE line.
        """
        # Build a pattern that matches any known heading on its own line
        # Optional prefix: "1.", "1.1", "I.", "II." etc.
        number_prefix = r'(?:\d+\.?\d*\s+|[IVX]+\.\s+)?'

        # Join all known headings with | (OR)
        headings_pattern = '|'.join(re.escape(h) for h in KNOWN_HEADINGS)

        # Full pattern: start-of-line, optional number, heading word, end-of-line
        pattern = rf'(?m)^\s*({number_prefix}(?:{headings_pattern}))\s*$'

        parts = re.split(pattern, clean_text, flags=re.IGNORECASE)

        return self._build_sections_from_parts(parts)

    def _split_by_headings(
        self, clean_text: str, headings: List[str]
    ) -> List[PaperSection]:
        """
         for font-detected headings.
        Same line-anchoring approach — heading must be the ENTIRE line.
        """
        if not headings:
            return []

        escaped = [re.escape(h) for h in headings]
        # (?m)^ = line start,  \s*$ = end of line — heading must be alone on line
        pattern = r'(?m)^\s*(' + '|'.join(escaped) + r')\s*$'

        parts = re.split(pattern, clean_text, flags=re.IGNORECASE)

        return self._build_sections_from_parts(parts)

    def _build_sections_from_parts(self, parts: List[str]) -> List[PaperSection]:
        """
        Helper: converts the output of re.split into PaperSection objects.

        re.split with a capture group returns:
          [before_first_match, match1, content1, match2, content2, ...]

        So parts[0] = text before first heading (title/abstract area)
        Then alternating: heading text, content, heading text, content...
        """
        sections: List[PaperSection] = []

        # Text before the first heading (usually title + authors + abstract)
        if parts[0].strip():
            sections.append(PaperSection(
                section_name="Header/Metadata",
                content=parts[0].strip(),
            ))

        # Walk through heading + content pairs
        for i in range(1, len(parts), 2):
            header  = parts[i].strip()
            content = parts[i + 1].strip() if (i + 1) < len(parts) else ""

            # Skip sections with very little content (likely a false split)
            if content and len(content) > 30:
                sections.append(PaperSection(
                    section_name=header,
                    content=content,
                ))

        return sections

    def _regex_fallback_sections(self, clean_text: str) -> List[PaperSection]:
        """
        Last resort: simple regex on known heading names.
        Less strict than _split_by_known_headings but catches papers
        where headings aren't perfectly on their own line.
        """
        pattern = "|".join([f"^{re.escape(h)}" for h in KNOWN_HEADINGS])
        parts   = re.split(
            f"({pattern})", clean_text,
            flags=re.MULTILINE | re.IGNORECASE
        )
        return self._build_sections_from_parts(parts)

    # ================================================================== #
    #  STEP 6 — METADATA EXTRACTION
    #  Extracts: title, authors, year, venue, keywords
    #
    #  Four layers in order:
    #    Layer 1: PDF built-in metadata (most reliable when present)
    #    Layer 2: Font-based title detection (largest text on page 1)
    #    Layer 3: Positional author detection (between title and Abstract)
    #    Layer 4: Regex on first 3000 chars for year/venue/keywords
    # ================================================================== #

    def _extract_title_from_fonts(self) -> Optional[str]:
        """
        Find the title = the largest text on the first page.

        Improvements over original:
        - Score by font size AND length (prefer 15-150 chars)
        - Skip ALL-CAPS short lines (usually journal name banners)
        - Skip lines with volume/issue/DOI patterns
        """
        if not self.doc:
            return None

        page        = self.doc[0]
        best_text   = ""
        best_score  = 0.0

        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                line_text = "".join(
                    s.get("text", "") for s in line.get("spans", [])
                ).strip()

                # Skip too short or too long
                if len(line_text) < 10 or len(line_text) > 200:
                    continue

                # Skip journal banners like "IEEE TRANSACTIONS ON ..." (all caps, short)
                if line_text.isupper() and len(line_text) < 80:
                    continue

                # Skip lines with DOI, volume, issue numbers
                if re.search(r'(doi|vol\.|volume|issue|pp\.|pages|\bno\b)', line_text, re.IGNORECASE):
                    continue

                spans    = line.get("spans", [])
                avg_size = sum(s.get("size", 0) for s in spans) / max(len(spans), 1)

                # Score = font size (bigger is better for title)
                # We keep it simple — just use font size as the score
                if avg_size > best_score:
                    best_score = avg_size
                    best_text  = line_text

        return best_text or None

    def _extract_authors_from_fonts(self, title: Optional[str], body_size: float) -> List[str]:
        """
        Extract authors using POSITIONAL logic:
        Authors appear BETWEEN the title and the Abstract/Introduction.

        Strategy:
        1. Find where the title ends (by matching title text)
        2. Find where the abstract starts (by looking for "Abstract" heading)
        3. Scan lines in between — name-like lines = authors

        A line looks like a name if:
        - 1 to 6 words
        - Each word starts with a capital letter
        - No digits, @ symbols, or institution keywords
        """
        if not self.doc:
            return []

        page       = self.doc[0]
        page_text  = page.get_text("text")
        lines      = [l.strip() for l in page_text.split("\n") if l.strip()]

        # Find title position in lines
        title_idx    = -1
        abstract_idx = len(lines)   # default: scan until end if no abstract found

        if title:
            for i, line in enumerate(lines):
                if title.lower()[:30] in line.lower():
                    title_idx = i
                    break

        # Find where "Abstract" appears
        for i, line in enumerate(lines):
            if re.match(r'^\s*abstract\s*$', line, re.IGNORECASE):
                abstract_idx = i
                break

        # The author zone is between title and abstract
        # If we couldn't find title, scan first 15 lines
        start = max(title_idx + 1, 0)
        end   = min(abstract_idx, start + 15)   # don't scan too far
        zone  = lines[start:end]

        authors = []
        for line in zone:
            # Skip institution/affiliation lines
            if re.search(
                r'university|institute|department|college|school|laboratory|'
                r'lab\b|center|centre|faculty|@|\.com|\.edu|\.org|\d{4,5}',
                line, re.IGNORECASE
            ):
                continue

            # Skip lines that are clearly not names (too long, or look like sentences)
            if len(line) > 100 or len(line) < 3:
                continue

            # Skip if it's (part of) the title
            if title and line.lower() in title.lower():
                continue

            # A name: 1-6 words, starts with capital, no unusual punctuation
            words = line.split()
            if 1 <= len(words) <= 6:
                # Every word should start with a capital letter (names do)
                all_capitalized = all(w[0].isupper() for w in words if w.isalpha())
                if all_capitalized:
                    authors.append(line)

        if not authors:
            return []

        # If multiple lines were collected, they might be:
        #   A) One author per line → ["Alice Smith", "Bob Jones"] → return as-is
        #   B) All authors on one line → "Alice Smith, Bob Jones" → split by comma

        # Check if any single line contains commas (all-on-one-line format)
        combined = []
        for line in authors[:5]:   # limit to first 5 candidate lines
            if ',' in line or ' and ' in line.lower():
                # Split this line by comma/and
                parts = re.split(r'[,;]|\band\b', line)
                combined.extend([p.strip() for p in parts if p.strip() and len(p.strip()) > 2])
            else:
                combined.append(line)

        return combined[:10]   # return at most 10 authors

    def extract_metadata(self, clean_text: str) -> Dict:
        """
        Extract paper metadata in layers.
        Each layer fills in only what the previous layer missed.
        """
        body_size = self._get_body_font_size()

        # ── Layer 1: PDF built-in metadata ──────────────────────────────
        # PDFs sometimes store title/author in their metadata fields.
        # This is the most reliable source when available.
        meta        = self.doc.metadata
        title       = meta.get("title", "").strip() or None
        raw_authors = meta.get("author", "").strip()
        authors     = (
            [a.strip() for a in re.split(r"[;,]", raw_authors) if a.strip()]
            if raw_authors else []
        )

        # ── Layer 2: Font-based title ────────────────────────────────────
        if not title:
            title = self._extract_title_from_fonts()

        # ── Layer 3: Positional author extraction ────────────────────────
        if not authors:
            authors = self._extract_authors_from_fonts(title, body_size)

        # ── Layer 4: Regex on first 3000 characters ──────────────────────
        head = clean_text[:3000]

        # Year: find first 4-digit year like 2019, 2023
        year = None
        m    = re.search(r'\b(19|20)\d{2}\b', head)
        if m:
            year = int(m.group())

        # Venue: look for publisher/conference names
        venue = None
        m = re.search(
            r'(Proceedings of|Journal of|Conference on|Workshop on|'
            r'arXiv:\d{4}\.\d{4,5}|IEEE|ACM|ICML|NeurIPS|CVPR|ACL|'
            r'EMNLP|ICLR|Springer|Elsevier|Nature|Science|PLOS)'
            r'[^,\n]{0,60}',         # stop at comma or newline, max 60 chars
            head, re.IGNORECASE
        )
        if m:
            venue = m.group().strip().rstrip('.:;')   # clean trailing punctuation

        # Keywords: look for "Keywords: word1, word2, ..."
        keywords: List[str] = []
        m = re.search(r'[Kk]ey\s*[Ww]ords?\s*[:\-—]?\s*(.+)', head)
        if m:
            keywords = [
                k.strip() for k in re.split(r'[;,·•–]', m.group(1))
                if k.strip()
            ]

        return {
            "title":    title,
            "authors":  authors,
            "year":     year,
            "venue":    venue,
            "keywords": keywords,
        }

    # ================================================================== #
    #  STEP 7 — REFERENCE EXTRACTION
    #  References are in their own section, so we first find that section,
    #  then try to split it into individual references.
    #
    #  Two strategies:
    #    Strategy 1: Regex patterns for known citation styles
    #    Strategy 2: Line accumulator (builds multi-line references)
    # ================================================================== #

    def extract_references(self, sections: List[PaperSection]) -> Tuple[List[str], str]:
        """
        Find the references section and extract individual references from it.
        Returns (list_of_references, raw_text).
        """
        raw_text = ""

        # Find the references/bibliography section
        # We check for multiple names because papers use different terms
        for section in sections:
            name_lower = section.section_name.lower()
            if any(word in name_lower for word in ["reference", "bibliography", "works cited"]):
                raw_text = section.content
                break

        if not raw_text:
            return [], ""

        # ── Strategy 1: Try each regex pattern ──────────────────────────
        # Compile all patterns so we can also use them in Strategy 2
        compiled_patterns = [re.compile(p, re.MULTILINE) for p in REFERENCE_PATTERNS]

        for pattern in compiled_patterns:
            matches = pattern.findall(raw_text)
            if len(matches) >= 2:
                # Found a pattern that works — use it
                return [m.strip() for m in matches if m.strip()], raw_text

        # ── Strategy 2: Line accumulator ────────────────────────────────
        # For formats where each reference spans multiple lines.
        # We detect the START of a new reference, then collect lines
        # until the next reference starts.
        lines       = [l.strip() for l in raw_text.split("\n") if l.strip()]
        parsed_refs = []
        current_ref = ""

        for line in lines:
            # Check if this line looks like the START of a new reference
            # using any of our compiled patterns
            is_new_ref = any(p.match(line) for p in compiled_patterns)

            # Also catch simple patterns: [1], 1. etc.
            if not is_new_ref:
                is_new_ref = bool(re.match(r'^(\[\d+\]|\d+[\.\)])', line))

            if is_new_ref and current_ref:
                # Save the previous reference (if it has enough content)
                if len(current_ref) > 20:
                    parsed_refs.append(current_ref.strip())
                current_ref = line
            elif is_new_ref:
                # First reference found
                current_ref = line
            elif current_ref and len(line) > 10:
                # Continue building the current reference
                current_ref += " " + line

        # Don't forget the last reference
        if current_ref and len(current_ref) > 20:
            parsed_refs.append(current_ref.strip())

        return parsed_refs, raw_text

    # ================================================================== #
    #  STEP 8 — LLM METADATA FALLBACK
    #  If title or authors are still missing after all the above steps,
    #  we ask the LLM to extract them from the first page text.
    #  This is the last resort — it's slower but more flexible.
    # ================================================================== #

    def _llm_metadata_fallback(self, meta: dict, first_page_text: str, llm) -> dict:
        """Ask the LLM to extract missing metadata from first page text."""
        try:
            prompt = f"""Extract metadata from this research paper's first page.
Return ONLY valid JSON. No explanation. No markdown fences.

{{
  "title": "string or null",
  "authors": ["list", "of", "author", "names"],
  "year": 2024,
  "venue": "string or null"
}}

TEXT:
{first_page_text[:2000]}

JSON:"""
            response = llm.invoke(prompt)
            raw      = response.content.strip()

            # Remove markdown code fences if LLM adds them despite instructions
            raw = re.sub(r'^```json\s*|```$', '', raw, flags=re.MULTILINE).strip()

            import json
            parsed = json.loads(raw)

            # Only fill in fields that are still empty
            if not meta["title"]   and parsed.get("title"):
                meta["title"]   = parsed["title"]
            if not meta["authors"] and parsed.get("authors"):
                meta["authors"] = parsed["authors"]
            if not meta["year"]    and parsed.get("year"):
                meta["year"]    = int(parsed["year"])
            if not meta["venue"]   and parsed.get("venue"):
                meta["venue"]   = parsed["venue"]

        except Exception:
            pass   # If LLM fails, we just use whatever we have

        return meta

    # ================================================================== #
    #  MAIN PARSE METHOD
    #  This is the entry point — it calls everything above in order.
    #
    #  The complete flow:
    #    1. Detect PDF type
    #    2. Extract full text
    #    3. Normalize unicode 
    #    4. Detect repeating headers/footers 
    #    5. Clean text
    #    6. Extract sections 
    #    7. Extract metadata
    #    8. LLM fallback if needed
    #    9. Extract references
    #    10. Build and return ResearchPaper object
    # ================================================================== #

    def parse(self, paper_id: str = None, llm=None) -> ResearchPaper:
        """
        Parse a PDF file into a ResearchPaper object.

        paper_id: optional ID string. If not provided, a random 8-char ID is generated.
        llm: optional LangChain LLM instance for metadata fallback.
        """
        if paper_id is None:
            paper_id = str(uuid.uuid4())[:8]

        # Early exit for scanned PDFs — we can't extract text from images
        if self.pdf_type == "scanned":
            return ResearchPaper(
                paper_id = paper_id,
                title    = paper_id,
                abstract = (
                    "⚠️ This PDF appears to be scanned (image-only). "
                    "Text extraction requires OCR which is not currently supported."
                ),
            )

        # Step 1: Extract raw text
        full_text = self._get_full_text()

        # Step 2:  — Normalize unicode (ligatures, dashes, PUA chars)
        full_text = self._normalize_text(full_text)

        # Step 3: — Detect repeating header/footer lines
        repeating_lines = self._detect_repeating_lines()

        # Step 4: Clean text (removes page numbers, noise, + repeating lines)
        clean_text = self._clean_text(full_text, repeating_lines)

        # Step 5: Extract sections (FIX 3 — line-anchored splitting is inside)
        sections = self._extract_structured_sections(clean_text)

        # Step 6: Extract metadata
        meta = self.extract_metadata(clean_text)

        # Step 7: LLM fallback if title or authors still missing
        if llm and (not meta["title"] or not meta["authors"]):
            meta = self._llm_metadata_fallback(meta, clean_text[:2000], llm)

        # Step 8: Extract references
        parsed_refs, raw_refs = self.extract_references(sections)

        # Step 9: Get abstract (prefer dedicated Abstract section,
        # fallback to first 500 chars of clean text)
        abstract = next(
            (s.content for s in sections if "abstract" in s.section_name.lower()),
            clean_text[:500],
        )

        # Step 10: Build and return the ResearchPaper object
        return ResearchPaper(
            paper_id       = paper_id,
            title          = meta["title"] or paper_id,
            authors        = meta["authors"],
            abstract       = abstract,
            year           = meta["year"],
            venue          = meta["venue"],
            keywords       = meta["keywords"],
            sections       = sections,
            references     = parsed_refs,
            raw_references = raw_refs or None,
            full_text      = clean_text,
            page_count     = len(self.doc),   # ← ADD THIS
        )

    # ================================================================== #
    #  The "with" statement calls __enter__ and __exit__ automatically:
    #    with PDFParser("paper.pdf") as parser:
    #        paper = parser.parse()
    #  ← file is automatically closed here
    # ================================================================== #

    def close(self):
        if self.doc:
            self.doc.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()