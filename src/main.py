"""Main entry point for the Lead Filtering Actor."""

import asyncio
from apify import Actor
from .models import ActorInput, LeadInput, LeadEvaluation
from .llm_client import evaluate_lead
from .config import get_default_model


async def main():
    """Main async function for the actor."""
    async with Actor:
        # Get input
        actor_input = await Actor.get_input()
        
        if not actor_input:
            Actor.log.error("No input provided")
            return
        
        # Parse and validate input
        try:
            input_data = ActorInput(**actor_input)
        except Exception as e:
            Actor.log.error(f"Invalid input format: {str(e)}")
            return
        
        Actor.log.info(f"Processing {len(input_data.leads)} leads")
        
        # Get model from input or use default
        model = input_data.openai_model or get_default_model()
        Actor.log.info(f"Using OpenAI model: {model}")
        
        # Process each lead
        for lead_data in input_data.leads:
            try:
                # Validate lead input
                lead = LeadInput(**lead_data)
                
                # Evaluate lead
                evaluation = await evaluate_lead(lead, model=model)
                
                # Push evaluation result
                await Actor.push_data(evaluation.model_dump(mode="json"))
                
            except Exception as e:
                # Log error and push error record
                error_msg = f"Failed to process lead: {str(e)}"
                Actor.log.error(f"Error processing lead {lead_data.get('company_name', 'Unknown')}: {error_msg}")
                
                # Push error record
                error_evaluation = LeadEvaluation(
                    company_name=lead_data.get("company_name", "Unknown"),
                    website_url=lead_data.get("website_url", ""),
                    website_quality_score=0,
                    branding_need="LOW",
                    online_presence_score=0,
                    brand_value_match="LOW",
                    w6h={
                        "who": "Unknown",
                        "what": "Unknown",
                        "where": lead_data.get("location") or "Unknown",
                        "when": "Unknown",
                        "why": "Unknown",
                        "how": "Unknown",
                        "how_much": "Unknown"
                    },
                    qualified=False,
                    reasons=[error_msg],
                    error=error_msg
                )
                await Actor.push_data(error_evaluation.model_dump(mode="json"))
        
        Actor.log.info("Finished processing all leads")


if __name__ == "__main__":
    asyncio.run(main())

