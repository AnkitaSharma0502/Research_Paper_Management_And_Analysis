from pydantic import BaseModel, Field
from typing import List, Optional, Literal

#gives structure of file
class PaperSection(BaseModel):
    """Represents a specific section of a research paper."""
    section_name: str = Field(..., description="Name of the section (e.g., Introduction, Results)")
    content: str      = Field(..., description="The raw text content of this section")


class ResearchPaper(BaseModel):
    """The unified internal representation of a research paper."""
    paper_id:  str
    title:     str
    authors:   List[str]  = Field(default_factory=list)
    abstract:  str        = ""
    full_text: Optional[str] = None
    year:      Optional[int] = None
    venue:     Optional[str] = None
    keywords:  List[str]  = Field(default_factory=list)

    sections: List[PaperSection] = Field(
        default_factory=list,
        description="The paper broken down into named sections",
    )

    references: List[str] = Field(
        default_factory=list,
        description="Parsed individual reference strings",
    )
   
    # Used as fallback display when individual parsing fails
    raw_references: Optional[str] = Field(
        None,
        description="Raw reference section text — shown when parsing into individual refs fails",
    )

    page_count: Optional[int] = None
    reading_status: Literal["to-read", "reading", "completed"] = "to-read"
