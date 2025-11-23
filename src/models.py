"""Pydantic models for input validation and output structure."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class W6H(BaseModel):
    """W6H analysis model."""
    who: str = Field(description="Target audience or customer base")
    what: str = Field(description="Product or service offering")
    where: str = Field(description="Geographic location or market")
    when: str = Field(description="Timeline or founding date")
    why: str = Field(description="Mission or purpose")
    how: str = Field(description="How they operate or deliver value")
    how_much: str = Field(description="Pricing or cost structure")


class LeadInput(BaseModel):
    """Input model for a single lead."""
    company_name: str
    website_url: str
    industry: Optional[str] = None
    location: Optional[str] = None
    social_links: Optional[List[str]] = None
    homepage_text: Optional[str] = None
    about_text: Optional[str] = None


class LeadEvaluation(BaseModel):
    """Evaluation result for a single lead."""
    company_name: str
    website_url: str
    website_quality_score: int = Field(ge=0, le=100, description="Website quality score 0-100")
    branding_need: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Level of branding need")
    online_presence_score: int = Field(ge=0, le=100, description="Online presence score 0-100")
    brand_value_match: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Brand value alignment")
    w6h: W6H = Field(description="W6H analysis")
    qualified: bool = Field(description="Whether the lead is qualified")
    reasons: List[str] = Field(description="List of reasons for the decision")
    error: Optional[str] = Field(default=None, description="Error message if evaluation failed")

    @field_validator("branding_need", "brand_value_match")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate that level is one of the allowed values."""
        if v not in ["LOW", "MEDIUM", "HIGH"]:
            raise ValueError(f"Value must be one of ['LOW', 'MEDIUM', 'HIGH'], got '{v}'")
        return v


class ActorInput(BaseModel):
    """Input model for the actor."""
    leads: List[LeadInput] = Field(description="List of leads to evaluate")
    openai_model: Optional[str] = Field(default="gpt-4o-mini", description="OpenAI model to use")

