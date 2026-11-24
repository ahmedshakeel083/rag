from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""\
    Please summarize the research paper titled '{paper_name}' with the following explanation
    Explanation Style: {style}
    Explanation Depth: {depth}
    1. Mathematical details:
        - Include relevant equations and derivations.
        - Explain the mathematical concepts in a clear and concise manner with code snippet where applicable.
    2. Analogies:
        - Use relatable analogies to explain complex concepts in a simple way.
    If certain concepts are not applicable to the paper, respond with "Insufficient information available" instead of guessing.
    Ensure that the summary is accurate, concise, and aligned with provided style and depth.
    """,
    input_variables=["paper_name", "style", "depth"],
    validate_template=True
)

template.save("dynamic_prompt_template.json")