"""LLM client for evaluating leads using OpenAI API."""

import json
import re
from typing import Optional
from openai import OpenAI
from apify import Actor

from .models import LeadInput, LeadEvaluation, W6H
from .config import get_openai_api_key


def build_evaluation_prompt(lead: LeadInput) -> str:
    """Build the prompt for LLM evaluation."""
    prompt = f"""You are an expert brand strategist for a branding and web design agency. Your task is to evaluate a potential client lead and determine if they are a good fit for your agency.

AGENCY VALUES:
- Focus on storytelling and long-term brand building
- Quality over speed/cheap solutions
- Strategic brand development
- Companies that value brand as an investment

EVALUATE THIS LEAD:

Company Name: {lead.company_name}
Website URL: {lead.website_url}
Industry: {lead.industry or "Not provided"}
Location: {lead.location or "Not provided"}
Social Links: {', '.join(lead.social_links) if lead.social_links else "Not provided"}
Homepage Text: {lead.homepage_text or "Not provided"}
About Text: {lead.about_text or "Not provided"}

EVALUATION CRITERIA:
1. **website_quality_score** (0-100): Assess the website's design quality, functionality, user experience, and overall professionalism based on the provided text content.
2. **branding_need** (LOW/MEDIUM/HIGH): Determine how much the company needs branding improvement. LOW = already strong brand, HIGH = significant branding gaps.
3. **online_presence_score** (0-100): Evaluate their social media presence and digital footprint based on provided social links and content.
4. **brand_value_match** (LOW/MEDIUM/HIGH): Assess alignment with agency values. HIGH = values storytelling, long-term brand building, quality. LOW = only wants cheap/fast solutions, no brand strategy interest.
5. **w6h**: Extract and structure the W6H information (who, what, where, when, why, how, how_much) from the provided information.
6. **qualified** (boolean): True if the lead is a good fit (decent website quality score, medium-to-high branding need, and medium-to-high brand value match). False otherwise.
7. **reasons** (array of strings): Provide 2-4 short, clear reasons explaining your decision.

IMPORTANT: You MUST respond with ONLY valid JSON in this exact schema, no additional text or markdown:

{{
  "website_quality_score": <0-100 integer>,
  "branding_need": "<LOW|MEDIUM|HIGH>",
  "online_presence_score": <0-100 integer>,
  "brand_value_match": "<LOW|MEDIUM|HIGH>",
  "w6h": {{
    "who": "<string>",
    "what": "<string>",
    "where": "<string>",
    "when": "<string>",
    "why": "<string>",
    "how": "<string>",
    "how_much": "<string>"
  }},
  "qualified": <boolean>,
  "reasons": ["<string>", "<string>", ...]
}}

Respond with ONLY the JSON object, no markdown code blocks, no explanations, no additional text."""

    return prompt


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks if present."""
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Try to find JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
    
    return json.loads(text)


def normalize_evaluation(
    lead: LeadInput,
    llm_response: dict,
    error: Optional[str] = None
) -> LeadEvaluation:
    """Normalize and validate LLM response into LeadEvaluation model."""
    # If there's an error, return unqualified evaluation
    if error:
        return LeadEvaluation(
            company_name=lead.company_name,
            website_url=lead.website_url,
            website_quality_score=0,
            branding_need="LOW",
            online_presence_score=0,
            brand_value_match="LOW",
            w6h=W6H(
                who="Unknown",
                what="Unknown",
                where=lead.location or "Unknown",
                when="Unknown",
                why="Unknown",
                how="Unknown",
                how_much="Unknown"
            ),
            qualified=False,
            reasons=[f"Evaluation failed: {error}"],
            error=error
        )
    
    # Extract and validate scores
    website_quality_score = llm_response.get("website_quality_score", 0)
    if not isinstance(website_quality_score, int):
        try:
            website_quality_score = int(float(website_quality_score))
        except (ValueError, TypeError):
            website_quality_score = 0
    website_quality_score = max(0, min(100, website_quality_score))
    
    online_presence_score = llm_response.get("online_presence_score", 0)
    if not isinstance(online_presence_score, int):
        try:
            online_presence_score = int(float(online_presence_score))
        except (ValueError, TypeError):
            online_presence_score = 0
    online_presence_score = max(0, min(100, online_presence_score))
    
    # Validate and normalize branding_need
    branding_need = llm_response.get("branding_need", "LOW")
    if isinstance(branding_need, str):
        branding_need = branding_need.upper().strip()
        if branding_need not in ["LOW", "MEDIUM", "HIGH"]:
            branding_need = "LOW"
    else:
        branding_need = "LOW"
    
    # Validate and normalize brand_value_match
    brand_value_match = llm_response.get("brand_value_match", "LOW")
    if isinstance(brand_value_match, str):
        brand_value_match = brand_value_match.upper().strip()
        if brand_value_match not in ["LOW", "MEDIUM", "HIGH"]:
            brand_value_match = "LOW"
    else:
        brand_value_match = "LOW"
    
    # Extract W6H
    w6h_data = llm_response.get("w6h", {})
    w6h = W6H(
        who=str(w6h_data.get("who", "Unknown")),
        what=str(w6h_data.get("what", "Unknown")),
        where=str(w6h_data.get("where", lead.location or "Unknown")),
        when=str(w6h_data.get("when", "Unknown")),
        why=str(w6h_data.get("why", "Unknown")),
        how=str(w6h_data.get("how", "Unknown")),
        how_much=str(w6h_data.get("how_much", "Unknown"))
    )
    
    # Extract qualified status
    qualified = llm_response.get("qualified", False)
    if not isinstance(qualified, bool):
        qualified = bool(qualified)
    
    # Apply simple rule: if scores are invalid or branding need is too low, disqualify
    if website_quality_score < 0 or website_quality_score > 100:
        qualified = False
    if branding_need == "LOW" and brand_value_match == "LOW":
        qualified = False
    
    # Extract reasons
    reasons = llm_response.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)] if reasons else []
    reasons = [str(r) for r in reasons if r]
    if not reasons:
        reasons = ["No reasons provided"]
    
    return LeadEvaluation(
        company_name=lead.company_name,
        website_url=lead.website_url,
        website_quality_score=website_quality_score,
        branding_need=branding_need,
        online_presence_score=online_presence_score,
        brand_value_match=brand_value_match,
        w6h=w6h,
        qualified=qualified,
        reasons=reasons
    )


async def evaluate_lead(
    lead: LeadInput,
    model: str = "gpt-4o-mini"
) -> LeadEvaluation:
    """Evaluate a single lead using LLM."""
    try:
        # Initialize OpenAI client
        api_key = get_openai_api_key()
        client = OpenAI(api_key=api_key)
        
        # Build prompt
        prompt = build_evaluation_prompt(lead)
        
        # Call OpenAI API
        Actor.log.info(f"Evaluating lead: {lead.company_name}")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a brand strategist. Always respond with valid JSON only, no additional text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Extract response text
        response_text = response.choices[0].message.content
        
        if not response_text:
            raise ValueError("Empty response from LLM")
        
        # Parse JSON from response
        try:
            llm_response = extract_json_from_response(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {str(e)}")
        
        # Normalize and validate
        evaluation = normalize_evaluation(lead, llm_response)
        
        Actor.log.info(f"Successfully evaluated lead: {lead.company_name} - Qualified: {evaluation.qualified}")
        
        return evaluation
        
    except Exception as e:
        error_msg = f"LLM evaluation failed: {str(e)}"
        Actor.log.error(f"Error evaluating lead {lead.company_name}: {error_msg}")
        return normalize_evaluation(lead, {}, error=error_msg)

