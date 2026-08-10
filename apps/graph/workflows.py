
"""Graph assembly layer: wires nodes and routing edges into runnable apps."""

from langgraph.graph import END, START, StateGraph

from apps.graph.nodes import (
    fetch_weather,
    filter_in_person_events,
    format_response,
    load_calendar_events,
    resolve_location,
    route_intent,
    fetch_weather_for_events,
    score_event_weather_risk,
    llm_recommendation_rewrite,
    format_meeting_recommendations,
    format_all_meetings,
    apply_user_default_city,
    add_high_risk_actions,
    load_user_profile,
    retrieve_meeting_context,
    add_retrieved_context_recommendations,

)

from apps.graph.state import GraphState



def route_after_risk_scoring(state: GraphState) -> str:
    """
    Decide whether to inject fixed high-risk safety guidance before LLM rewrite.
    """
    risk_summary = state.get("risk_summary") or []
    has_high_risk = any(item.get("risk") == "high" for item in risk_summary)
    return "add_high_risk_actions" if has_high_risk else "llm_recommendation_rewrite"

def route_after_intent(state: GraphState) -> str:
    """Route key selector after intent node."""
    if state.get("intent") == "weather":
        return "resolve_location"
    return "format_response"


def route_after_fetch(state: GraphState) -> str:
    """Route key selector after weather fetch node."""
    return "format_response"


def build_weather_graph(checkpointer=None):
    """
    Build and compile the LangGraph workflow.

    Node names are string identifiers used by edges and routing maps.
    """
    # StateGraph(GraphState) tells LangGraph which shared state schema to manage.
    graph = StateGraph(GraphState)

    # Register executable nodes.
    graph.add_node("route_intent", route_intent)
    graph.add_node("resolve_location", resolve_location)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("format_response", format_response)

    # START is a special entry marker provided by LangGraph.
    graph.add_edge(START, "route_intent")

    # Conditional edges use a route function that returns a key.
    # Returned key is looked up in this mapping to choose the next node.
    graph.add_conditional_edges(
        "route_intent",
        route_after_intent,
        {
            "resolve_location": "resolve_location",
            "format_response": "format_response",
        },
    )

    # Deterministic path after location resolution.
    graph.add_edge("resolve_location", "fetch_weather")
    graph.add_conditional_edges(
        "fetch_weather",
        route_after_fetch,
        {"format_response": "format_response"},
    )

    # END is the special terminal marker.
    graph.add_edge("format_response", END)

    # compile() returns an executable graph app with invoke/stream APIs.
    return graph.compile(checkpointer=checkpointer)

def build_meeting_preview_graph(checkpointer=None):
    """
    Meeting preview flow:
    calendar -> profile -> split (in-person vs all) -> weather for in-person only
    -> scoring -> LLM rewrite -> format with weather for in-person, simple list for online
    """
    graph = StateGraph(GraphState)

    graph.add_node("load_calendar_events", load_calendar_events)
    graph.add_node("filter_in_person_events", filter_in_person_events)
    graph.add_node("fetch_weather_for_events", fetch_weather_for_events)
    graph.add_node("score_event_weather_risk", score_event_weather_risk)
    graph.add_node("format_meeting_recommendations", format_meeting_recommendations)
    graph.add_node("apply_user_default_city", apply_user_default_city)
    graph.add_node("add_high_risk_actions", add_high_risk_actions)
    graph.add_node("llm_recommendation_rewrite", llm_recommendation_rewrite)
    graph.add_node("load_user_profile", load_user_profile)
    graph.add_node("retrieve_meeting_context", retrieve_meeting_context)
    graph.add_node("add_retrieved_context_recommendations", add_retrieved_context_recommendations)
    graph.add_node("format_all_meetings", format_all_meetings)

    graph.add_edge(START, "load_calendar_events")
    graph.add_edge("load_calendar_events", "load_user_profile")
    graph.add_edge("load_user_profile", "filter_in_person_events")
    graph.add_edge("filter_in_person_events", "apply_user_default_city")
    graph.add_edge("apply_user_default_city", "retrieve_meeting_context")
    graph.add_edge("retrieve_meeting_context", "fetch_weather_for_events")

    graph.add_edge("fetch_weather_for_events", "score_event_weather_risk")
    graph.add_conditional_edges(
        "score_event_weather_risk",
        route_after_risk_scoring,
        {
            "add_high_risk_actions": "add_high_risk_actions",
            "llm_recommendation_rewrite": "llm_recommendation_rewrite",
        },
    )
    graph.add_edge("add_high_risk_actions", "add_retrieved_context_recommendations")
    graph.add_edge("add_retrieved_context_recommendations", "llm_recommendation_rewrite")
    graph.add_edge("llm_recommendation_rewrite", "format_all_meetings")
    graph.add_edge("format_all_meetings", END)

    return graph.compile(checkpointer=checkpointer)
