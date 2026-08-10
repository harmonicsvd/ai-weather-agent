"""
Tool registry for weather agent.
Central registry for all LangChain tools that can be executed remotely.
"""

from typing import Dict, List, Any
from langchain.tools import BaseTool

# Tool registry: maps tool names to tool instances
_TOOL_REGISTRY: Dict[str, BaseTool] = {}

def register_tool(tool: BaseTool):
    """Register a tool in the central registry."""
    _TOOL_REGISTRY[tool.name] = tool

def get_tool(tool_name: str) -> BaseTool | None:
    """Get a tool by name from the registry."""
    return _TOOL_REGISTRY.get(tool_name)

def list_tools() -> List[str]:
    """List all registered tool names."""
    return list(_TOOL_REGISTRY.keys())

def get_all_tools() -> List[BaseTool]:
    """Get all registered tools."""
    return list(_TOOL_REGISTRY.values())
