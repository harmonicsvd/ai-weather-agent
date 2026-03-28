from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver

from apps.tools.weather_client import (
    CityNotFoundError,
    OpenMeteoClient,
    WeatherProviderError,
)

# Load local env vars (API keys/provider config) from weather-agent/.env.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

# Global behavior contract for the model.
# We keep this explicit so tool usage rules are easy to understand and edit.
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""


@dataclass
class Context:
    """
    Runtime context injected per request.
    Tools can read this through ToolRuntime for user-aware behavior.
    """

    user_id: str


@tool
def get_weather_for_location(city: str) -> str:
    """
    Tool wrapper around OpenMeteoClient.
    Returns a plain string because LLM tools should be concise and readable.
    """
    try:
        # `with` ensures HTTP resources are closed even when an exception occurs.
        with OpenMeteoClient() as client:
            data = client.get_current_weather_by_city(city)
    except CityNotFoundError:
        return f"Sorry, I couldn't find the city '{city}'. Maybe it's a hidden gem?"
    except WeatherProviderError:
        return "Sorry, the weather provider is currently unavailable. It's a bit cloudy on my end too!"

    location = data.location.name
    country = data.location.country
    current = data.current_weather
    location_label = f"{location}, {country}" if country else location

    return (
        f"Current weather in {location_label}: "
        f"{current.temperature_c}°C"
        f"(feels like {current.apparent_temperature_c}°C), "
        f"humidity {current.humidity_percent}%, "
        f"wind {current.wind_speed_kmh} km/h, "
        f"weather code {current.weather_code}."
    )


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """
    Demo location resolver.
    In production this would likely call a user profile service.
    """
    return "Florida" if runtime.context.user_id == "1" else "San Francisco"


# Keep model setup in one place so swapping providers is low friction.
model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    temperature=0,
)


@dataclass
class ResponseFormat:
    """
    Structured payload we expect back from the agent.
    Keeping this stable helps when we later add tests/evals.
    """

    punny_response: str
    weather_conditions: str | None = None


# In-memory state for local demo runs. Later we will replace this with persistent storage.
checkpointer = InMemorySaver()


agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer,
)


def run_demo_turn(user_message: str, thread_id: str, user_id: str = "1") -> None:
    """
    Helper for local manual testing.
    Separate thread IDs avoid memory from one scenario leaking into another.
    """
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        # thread_id controls conversation memory scope in the checkpointer.
        config={"configurable": {"thread_id": thread_id}},
        context=Context(user_id=user_id),
    )
    print(response["structured_response"])


# Demo cases:
# 1) Implicit location (agent should call get_user_location).
run_demo_turn("what is the weather outside?", thread_id="implicit-location-demo")

# 2) Explicit city.
run_demo_turn("what is the weather in Hamburg?", thread_id="explicit-city-demo")

# 3) Invalid city to verify safe fallback behavior.
run_demo_turn("what is the weather in asdasdasdasd?", thread_id="invalid-city-demo")
