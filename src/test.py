from src.api.data_extraction import DataExtractionRequest, VariableDefinition

def test_prompt_generation():
    # Example variables for seroprevalence extraction
    variables = [
        VariableDefinition(
            name="Seroprevalence",
            description="The prevalence of individuals with SARS-CoV-2 antibodies",
            data_type="numeric",
            formatting_instructions="Report as percentage (e.g., 15.3%)"
        ),
        VariableDefinition(
            name="Study Date",
            description="When the study was conducted",
            data_type="date",
            formatting_instructions="Report as YYYY-MM-DD"
        ),
        VariableDefinition(
            name="Location",
            description="Geographic location of the study",
            data_type="string",
            formatting_instructions="Report as City, Country"
        )
    ]

    # Create example request
    request = DataExtractionRequest(
        preprompt="""We are performing data extraction for a systematic review. 
Please carefully read the provided text and extract the requested information.""",
        prompt="Please extract the requested variables from the text above.",
        variables=variables
    )

    # Generate the extraction prompt
    final_prompt = request.generate_extraction_prompt()
    print("Generated Prompt:")
    print("-" * 80)
    print(final_prompt)

if __name__ == "__main__":
    test_prompt_generation()