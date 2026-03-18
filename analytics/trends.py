from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
from models.schemas import ResearchPaper


class TrendAnalyzer:
    def __init__(self, papers: List[ResearchPaper]):
        self.papers = papers

    # ------------------------------------------------------------------ #
    #  HELPER
    # ------------------------------------------------------------------ #

    def _papers_with_year(self) -> List[ResearchPaper]:
        """Returns only papers that have a valid year."""
        return [p for p in self.papers if p.year is not None]

    # ------------------------------------------------------------------ #
    #  TAB 1 — Library Overview
    # ------------------------------------------------------------------ #

    def get_library_stats(self) -> Dict[str, Any]:
        """
        Returns high-level library statistics for the overview tab.

        Returns:
            {
                "total":        12,
                "year_range":   "2018 – 2024",
                "unique_venues": 5,
                "authors_count": 34,
            }
        """
        valid      = self._papers_with_year()
        years      = [p.year for p in valid]
        venues     = [p.venue for p in self.papers if p.venue]
        all_authors = [a for p in self.papers for a in p.authors]

        return {
            "total":          len(self.papers),
            "year_range":     f"{min(years)} – {max(years)}" if years else "N/A",
            "unique_venues":  len(set(venues)),
            "authors_count":  len(set(all_authors)),
        }

    def get_papers_per_year(self) -> Dict[int, int]:
        """
        Returns paper count per year, sorted ascending.
        Used for the bar chart in Library Overview.
        """
        counts: Counter = Counter()
        for paper in self._papers_with_year():
            counts[paper.year] += 1
        return dict(sorted(counts.items()))

    def get_venue_table(self) -> List[Dict[str, Any]]:
        """
        Returns a simple venue breakdown as a list of dicts.
        Shown as a table (not a chart) in Library Overview.

        Returns:
            [ {"Venue": "NeurIPS", "Papers": 3, "Years": "2021, 2023"}, ... ]
        """
        venue_data: Dict[str, Dict] = {}

        for paper in self.papers:
            if not paper.venue:
                continue
            if paper.venue not in venue_data:
                venue_data[paper.venue] = {"papers": [], "years": set()}
            venue_data[paper.venue]["papers"].append(paper.title)
            if paper.year:
                venue_data[paper.venue]["years"].add(paper.year)

        result = []
        for venue, data in sorted(
            venue_data.items(),
            key=lambda x: len(x[1]["papers"]),
            reverse=True,
        ):
            result.append({
                "Venue":  venue,
                "Papers": len(data["papers"]),
                "Years":  ", ".join(str(y) for y in sorted(data["years"])),
            })

        return result

    # ------------------------------------------------------------------ #
    #  TAB 2 — Research Topics (LLM-powered, on-demand)
    # ------------------------------------------------------------------ #

    def get_category_distribution(
        self,
        categories: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Builds a category breakdown from LLM-assigned categories.

        Args:
            categories: { paper_id: "Machine Learning", ... }
                        (computed on-demand in analytics_view.py)

        Returns:
            [
              {
                "category":    "Machine Learning",
                "count":       4,
                "year_range":  "2020 – 2024",
                "papers":      ["Paper A", "Paper B", ...]
              },
              ...
            ]
        """
        cat_data: Dict[str, Dict] = {}

        for paper in self.papers:
            cat = categories.get(paper.paper_id, "Uncategorized")
            if cat not in cat_data:
                cat_data[cat] = {"papers": [], "years": set()}
            cat_data[cat]["papers"].append(paper.title)
            if paper.year:
                cat_data[cat]["years"].add(paper.year)

        result = []
        for cat, data in sorted(
            cat_data.items(),
            key=lambda x: len(x[1]["papers"]),
            reverse=True,
        ):
            years = sorted(data["years"])
            result.append({
                "category":   cat,
                "count":      len(data["papers"]),
                "year_range": f"{years[0]} – {years[-1]}" if len(years) > 1
                              else str(years[0]) if years else "N/A",
                "papers":     data["papers"],
            })

        return result

    def get_category_by_year(
        self,
        categories: Dict[str, str],
    ) -> Dict[int, Dict[str, int]]:
        """
        Returns category counts broken down by year.
        Used for the Category × Year trend table.

        Returns:
            { 2022: {"Machine Learning": 2, "NLP": 1}, 2023: {...}, ... }
        """
        result: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for paper in self._papers_with_year():
            cat = categories.get(paper.paper_id, "Uncategorized")
            result[paper.year][cat] += 1

        return {
            year: dict(cats)
            for year, cats in sorted(result.items())
        }

    # ------------------------------------------------------------------ #
    #  TAB 3 — Emerging Trends (newest papers by year)
    # ------------------------------------------------------------------ #

    def get_newest_papers(
        self,
        limit_years: Optional[int] = None,
        categories: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns papers sorted by year descending (newest first).

        Args:
            limit_years: If set, only return papers from the last N years.
                         e.g. limit_years=2 → only last 2 years.
            categories:  Optional { paper_id: category } dict to include
                         category column in the result.

        Returns:
            [
              {
                "title":    "Attention Is All You Need",
                "year":     2023,
                "venue":    "NeurIPS",
                "authors":  "Vaswani et al.",
                "category": "Machine Learning",   # if categories provided
                "abstract": "...",
              },
              ...
            ]
        """
        valid = self._papers_with_year()

        # Apply year filter
        if limit_years is not None:
            max_year = max(p.year for p in valid) if valid else 0
            cutoff   = max_year - limit_years + 1
            valid    = [p for p in valid if p.year >= cutoff]

        # Sort newest first
        valid = sorted(valid, key=lambda p: p.year, reverse=True)

        result = []
        for paper in valid:
            entry = {
                "title":   paper.title,
                "year":    paper.year,
                "venue":   paper.venue    or "N/A",
                "authors": ", ".join(paper.authors[:3]) +
                           (" et al." if len(paper.authors) > 3 else "")
                           if paper.authors else "Unknown",
                "abstract": paper.abstract[:200] + "..."
                            if paper.abstract and len(paper.abstract) > 200
                            else paper.abstract or "N/A",
            }
            if categories:
                entry["category"] = categories.get(paper.paper_id, "Uncategorized")
            result.append(entry)

        return result

    # ------------------------------------------------------------------ #
    #  TAB 4 — Citation Network
    # ------------------------------------------------------------------ #

    def get_citation_graph(self) -> Dict[str, Dict[str, Any]]:
        """
        Builds a bidirectional citation adjacency structure.

        Returns:
            {
              "Paper Title A": {
                "references": ["Paper B", "Paper C"],
                "cited_by":   ["Paper D"]
              },
              ...
            }
        """
        graph: Dict[str, Dict] = {
            paper.title: {
                "references": paper.references,
                "cited_by":   [],
            }
            for paper in self.papers
        }

        # Build reverse edges
        for paper in self.papers:
            for ref_title in paper.references:
                if ref_title in graph:
                    graph[ref_title]["cited_by"].append(paper.title)
                else:
                    graph[ref_title] = {
                        "references": [],
                        "cited_by":   [paper.title],
                    }

        return graph

    def get_most_referenced_external(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Returns external papers (not in local library) most referenced
        across all papers in the library.

        Returns:
            [ {"title": "...", "cited_by_count": 3}, ... ]
        """
        local_titles = {paper.title for paper in self.papers}
        ref_counts: Counter = Counter()

        for paper in self.papers:
            for ref in paper.references:
                if ref not in local_titles:
                    ref_counts[ref] += 1

        return [
            {"title": title, "cited_by_count": count}
            for title, count in ref_counts.most_common(top_n)
        ]