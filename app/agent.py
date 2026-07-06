from google import genai
from google.adk.agents import Agent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.workflow import START, Workflow, node
from google.genai import types

shipping_faq_agent = Agent(
    name="shipping_faq_agent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a helpful customer support agent for a shipping company. "
        "Answer shipping-related questions about rates, tracking, delivery, and returns. "
        "Be polite and professional. Keep your responses concise and accurate."
    ),
)


@node
def classify_query(ctx: Context, node_input: str) -> str:
    """Classifies the query as shipping-related or unrelated and routes accordingly."""
    client = genai.Client()
    prompt = (
        "Classify if the following user query is related to shipping "
        "(rates, tracking, delivery, returns) or unrelated.\n"
        "Return exactly 'shipping' if it is related to shipping, or 'unrelated' otherwise. "
        "Do not include any other text.\n\n"
        f"Query: {node_input}"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    classification = response.text.strip().lower()

    if "shipping" in classification:
        ctx.route = "shipping"
    else:
        ctx.route = "unrelated"

    return node_input


@node
def decline_query(ctx: Context, node_input: str) -> str:
    """Politely declines to answer unrelated queries."""
    return types.Content(
        role="model",
        parts=[types.Part.from_text(text=(
            "I am sorry, but I can only assist with shipping-related inquiries "
            "(such as rates, tracking, delivery, or returns). "
            "Is there anything else shipping-related I can help you with?"
        ))],
    )


root_agent = Workflow(
    name="customer_support_workflow",
    edges=[
        (START, classify_query),
        (classify_query, {
            "shipping": shipping_faq_agent,
            "unrelated": decline_query
        })
    ]
)

app = App(
    root_agent=root_agent,
    name="customer-support-agent",
)
