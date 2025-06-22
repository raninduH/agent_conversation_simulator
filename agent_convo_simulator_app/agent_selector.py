"""
Agent Selector Module
This module provides an LLM-based agent selection mechanism for conversations.
"""

import os
import re
import json
from typing import List, Dict, Optional, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from datetime import datetime

from config import MODEL_SETTINGS, AGENT_SETTINGS


class AgentSelector:
    """
    Uses LLM to determine which agent should speak next in a conversation,
    or if the conversation should be terminated based on a termination condition.
    """
    
    def __init__(self, google_api_key: Optional[str] = None):
        """Initialize the agent selector with API key."""
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")
        
        # Initialize LLM for agent selection
        self.model = ChatGoogleGenerativeAI(
            model=MODEL_SETTINGS["agent_model"],
            temperature=AGENT_SETTINGS["response_temperature"],
            max_retries=AGENT_SETTINGS["max_retries"],
            google_api_key=self.google_api_key
        )
    
    def select_next_agent(
        self, 
        messages: List[Dict[str, Any]], 
        environment: str, 
        scene: str, 
        agents: List[Dict[str, str]],
        termination_condition: Optional[str] = None,
        agent_invocation_counts: Optional[Dict[str, int]] = None
    ) -> Dict[str, str]:
        """
        Determine which agent should speak next or if the conversation should terminate.
        
        Args:
            messages: List of recent messages (up to 10)
            environment: The conversation environment description
            scene: The conversation scene description
            agents: List of agent configurations with name and role
            termination_condition: Optional condition for when to terminate the conversation
            agent_invocation_counts: Optional dict tracking how many times each agent has been invoked
            
        Returns:
            Dictionary with {"next_response": agent_name} or {"next_response": "terminate"}
        """
        # Get the most recent messages (up to 10)
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        # Format messages for the prompt
        formatted_messages = []
        for msg in recent_messages:
            formatted_messages.append(f"{msg['agent_name']}: {msg['message']}")
        messages_str = "\n".join(formatted_messages)
        
        # Format agents for the prompt
        agents_str = ", ".join([f"{agent['name']} ({agent['role']})" for agent in agents])
        
        # Add invocation count information if available
        invocation_info = ""
        if agent_invocation_counts:
            invocation_info = "\nAgent invocation counts: " + ", ".join([f"{name}: {count}" for name, count in agent_invocation_counts.items()])
        
        # Create the prompt
        prompt = f"""You are handling a role play of agents. 
This is the last 10 messages of the current conversation: {messages_str}, 
this is the current environment the agents are in: {environment} 
and this is the current starting scene: {scene}. 
These are the active agents: {agents_str}.{invocation_info}
This is the termination condition for the conversation: {termination_condition or 'None'}. 
Decide which agent should invoke next and output the following JSON: 
{{ "next_response": "agent_name" }} or output the following if the conversation is ready to terminate: 
{{ "next_response": "terminate" }}.
Don't output anything else only the JSON response. 
Note: sometimes the last response agent might need to invoke right again if that agent needs to give more to the conversation"""
        
        # Call the LLM to select the next agent
        response = self.model.invoke([HumanMessage(content=prompt)])
        response_text = response.content
        
        # Extract the JSON response
        return self._extract_json(response_text)
    
    def _extract_json(self, text: str) -> Dict[str, str]:
        """Extract JSON from the response text, handling different formats."""
        # First try direct JSON parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON using regex
            try:
                # Look for JSON-like structure with curly braces
                json_match = re.search(r'({.*?})', text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                    return json.loads(json_text)
                
                # If still no match, try to extract key-value from markdown format
                markdown_match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
                if markdown_match:
                    json_text = markdown_match.group(1).strip()
                    return json.loads(json_text)
            except Exception:
                # If all parsing attempts fail, return a default response
                pass
            
            # Default response if no valid JSON found
            return {"next_response": "error_parsing"}
