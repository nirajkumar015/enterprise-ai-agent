import os

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()


# Get OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is not set. "
        "Please add it to the .env file."
    )


# OpenRouter provides an OpenAI-compatible API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


MODEL_NAME = "openrouter/free"


def generate_response(prompt: str) -> str:
    """
    Send a prompt to an OpenRouter free model
    and return the generated response.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    prompt = "Explain what Retrieval-Augmented Generation means in one sentence."

    answer = generate_response(prompt)

    print("\nLLM TEST")
    print("=" * 60)
    print("Prompt:")
    print(prompt)

    print("\nResponse:")
    print(answer)