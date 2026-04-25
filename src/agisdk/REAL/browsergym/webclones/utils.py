import os
from openai import OpenAI
from anthropic import Anthropic

# Constants
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_PREFIX = "openrouter/"
DEFAULT_MODEL = "gpt-4o"


async def scrape_content(page) -> None:
    """
    Scrape content from a page by extracting text from a <pre> tag within #__next div.

    Args:
        page: The playwright page object to scrape from
    """
    # Evaluate JavaScript to extract the content inside the <pre> tag within the #__next div
    content = await page.evaluate('''() => {
        const nextDiv = document.querySelector('#__next');
        const preTag = nextDiv ? nextDiv.querySelector('pre') : null;
        return preTag ? preTag.innerText : "No <pre> tag found.";
    }''')

    print(content)


def call_openrouter(model: str, prompt: str) -> str:
    """
    Call OpenRouter API to generate content using the specified model.

    Args:
        model: The model identifier (should start with 'openrouter/')
        prompt: The prompt to send to the model

    Returns:
        Generated content from the model

    Raises:
        ValueError: If required environment variables are missing
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )

    # Extract model name by removing the openrouter prefix
    model_name = model.replace(OPENROUTER_PREFIX, "", 1)

    # Get optional site configuration
    site_url = os.getenv("OPENROUTER_SITE_URL", "")
    site_name = os.getenv("OPENROUTER_SITE_NAME", "")

    response = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        },
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def call_openai(model: str, prompt: str) -> str:
    """
    Call OpenAI API to generate content using the specified model.

    Args:
        model: The OpenAI model identifier (e.g., 'gpt-4', 'gpt-3.5-turbo')
        prompt: The prompt to send to the model

    Returns:
        Generated content from the model

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is missing
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def call_anthropic(model: str, prompt: str) -> str:
    """
    Call Anthropic API to generate content using the specified model.

    Args:
        model: The Anthropic model identifier (e.g., 'sonnet-3.7', 'sonnet-4')
        prompt: The prompt to send to the model

    Returns:
        Generated content from the model

    Raises:
        ValueError: If ANTHROPIC_API_KEY environment variable is missing
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


def generate_from_model(model: str = DEFAULT_MODEL, prompt: str = "") -> str:
    """
    Generate content from OpenRouter, Anthropic, or OpenAI based on model identifier.

    Args:
        model: Model identifier. If starts with 'openrouter/', uses OpenRouter API.
               If starts with 'sonnet-3.7' or 'sonnet-4', uses Anthropic API.
               Otherwise uses OpenAI API. Defaults to 'gpt-4o'.
        prompt: The text prompt to send to the model

    Returns:
        Generated content from the specified model

    Raises:
        ValueError: If required API keys are missing
    """
    if not model:
        # ToDo(ardroh): Try to get from environment variables.
        model = os.getenv("LLM_MODEL_NAME", "")
    if model.startswith(OPENROUTER_PREFIX):
        return call_openrouter(model, prompt)
    elif model.startswith("sonnet-3.7") or model.startswith("sonnet-4"):
        actual_model_name = "claude-3-7-sonnet-20250219" if model.startswith("sonnet-3.7") else "claude-sonnet-4-20250514"
        return call_anthropic(actual_model_name, prompt)
    else:
        return call_openai(model, prompt)
