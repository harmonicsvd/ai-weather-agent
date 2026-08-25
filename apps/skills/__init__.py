"""
Skill registry for weather agent.
Central registry for all LangChain skills that can be executed remotely.
"""

from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
import time
import logging

logger = logging.getLogger(__name__)

# Skill registry: maps skill names to skill instances
_SKILL_REGISTRY: Dict[str, BaseTool] = {}

# Cache for user-specific skills: {user_sub: (skill_names, timestamp)}
_USER_SKILLS_CACHE: Dict[str, tuple[List[str], float]] = {}
CACHE_TTL = 300  # 5 minutes cache TTL

def register_skill(skill: BaseTool):
    """Register a skill in the central registry."""
    _SKILL_REGISTRY[skill.name] = skill

def get_skill(skill_name: str) -> BaseTool | None:
    """Get a skill by name from the registry."""
    return _SKILL_REGISTRY.get(skill_name)

def list_skills() -> List[str]:
    """List all registered skill names."""
    return list(_SKILL_REGISTRY.keys())

def get_all_skills() -> List[BaseTool]:
    """Get all registered skills."""
    return list(_SKILL_REGISTRY.values())

def get_user_skills(user_sub: str, force_reload: bool = False) -> List[str]:
    """Get skills available to a specific user with caching.
    
    Args:
        user_sub: User identifier
        force_reload: If True, bypass cache and reload from database
    
    Returns:
        List of skill names available to the user
    """
    current_time = time.time()
    
    # Check cache first
    if not force_reload and user_sub in _USER_SKILLS_CACHE:
        skill_names, timestamp = _USER_SKILLS_CACHE[user_sub]
        if current_time - timestamp < CACHE_TTL:
            logger.debug(f"Using cached skills for user {user_sub}: {skill_names}")
            return skill_names
    
    # Load from database
    try:
        from apps.graph.db import get_user_installed_skills
        skill_names = get_user_installed_skills(user_sub)
        # Update cache
        _USER_SKILLS_CACHE[user_sub] = (skill_names, current_time)
        logger.info(f"Loaded skills from database for user {user_sub}: {skill_names}")
        return skill_names
    except Exception as e:
        logger.error(f"Failed to load user skills from database: {e}")
        # Return empty list on error
        return []

def get_user_skill_instances(user_sub: str) -> List[BaseTool]:
    """Get actual skill instances available to a user.
    
    Args:
        user_sub: User identifier
    
    Returns:
        List of BaseTool instances available to the user
    """
    user_skill_names = get_user_skills(user_sub)
    return [get_skill(name) for name in user_skill_names if get_skill(name)]

def clear_user_skills_cache(user_sub: Optional[str] = None):
    """Clear skills cache for a specific user or all users.
    
    Args:
        user_sub: User identifier, or None to clear all cache
    """
    if user_sub:
        _USER_SKILLS_CACHE.pop(user_sub, None)
        logger.info(f"Cleared skills cache for user {user_sub}")
    else:
        _USER_SKILLS_CACHE.clear()
        logger.info("Cleared all skills cache")
