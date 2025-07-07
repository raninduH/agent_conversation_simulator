"""
Simple Multi-Agent Conversation Engine
Real LangGraph react agents with Gemini in conversation cycle.
"""

import os
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import threading
import time
import traceback

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from functools import partial

from .tools import (
    internet_search_tool,
    search_internet_strucutred_output,
    search_images_from_internet,
    search_news_from_internet,
    search_places_from_internet,
    knowledge_base_retriever,
    get_browser_tools
)
from .config import CONVERSATION_TIMING, MESSAGE_SETTINGS, AGENT_SETTINGS, MODEL_SETTINGS
from .data_manager import DataManager
from .agent_selector import AgentSelector
from .round_robin_engine import RoundRobinEngine
from .human_like_chat_engine import HumanLikeChatEngine
from .simple_conversation_engine import SimpleConversationEngine
from .agent_selector_engine import AgentSelectorEngine


AVAILABLE_TOOLS = {
    "internet_search_tool": internet_search_tool,
    "search_internet_strucutred_output": search_internet_strucutred_output,
    "search_images_from_internet": search_images_from_internet,
    "search_news_from_internet": search_news_from_internet,
    "search_places_from_internet": search_places_from_internet,
    "knowledge_base_retriever": knowledge_base_retriever,
}
# Add browser tools separately as they are loaded lazily
try:
    for tool in get_browser_tools():
        AVAILABLE_TOOLS[tool.name] = tool
except Exception as e:
    print(f"Warning: Could not load browser tools: {e}")


class ConversationSimulatorEngine:
    """
    Multi-agent conversation simulator using real LangGraph react agents with Gemini.
    Agents take turns in a cycle, maintaining conversation history with summarization.
    """
    
    def __init__(self, google_api_key: Optional[str] = None):
        """Initialize the conversation simulator engine."""
        # Store the provided API key as default for agents without specific keys
        self.default_api_key = google_api_key
        
        # Only use the environment variable for the summary model
        summary_api_key = os.getenv("GOOGLE_API_KEY")
        if not summary_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for the summary model.")
        
        print(f"DEBUG: ConversationEngine initialized with default API key: {'Yes' if self.default_api_key else 'No'}")
        print(f"DEBUG: Summary model using environment API key")
        
        # Separate model for summarization - always uses environment variable
        self.summary_model = ChatGoogleGenerativeAI(
            model=MODEL_SETTINGS["summary_model"],
            temperature=AGENT_SETTINGS["summary_temperature"],            max_retries=AGENT_SETTINGS["max_retries"],
            google_api_key=summary_api_key
        )
        
        self.memory = MemorySaver()
        self.active_conversations = {}
        self.message_callbacks = {}  # For real-time message updates

        # Initialize engines
        self.round_robin_engine = RoundRobinEngine(self)
        self.human_like_chat_engine = HumanLikeChatEngine(self)
        self.agent_selector_engine = AgentSelectorEngine(self)

    def message_list_summarization(self, messages: List[Dict[str, str]], 
                                 no_of_messages_to_trigger_summarization: int = None) -> List[Dict[str, str]]:
        """
        Summarize messages when they exceed the trigger threshold.
        
        Args:
            messages: List of message dictionaries
            no_of_messages_to_trigger_summarization: Threshold for triggering summarization (defaults to config value)
            
        Returns:
            Updated messages list with summary and last N messages (N from config)
        """
        # Use config values if not specified
        if no_of_messages_to_trigger_summarization is None:
            no_of_messages_to_trigger_summarization = MESSAGE_SETTINGS["max_messages_before_summary"]
        
        messages_to_keep = MESSAGE_SETTINGS["messages_to_keep_after_summary"]
        
        if len(messages) <= no_of_messages_to_trigger_summarization:
            return messages
        
        # Check if there's already a summary at the beginning
        has_existing_summary = (messages and 
                               len(messages) > 0 and 
                               "past_convo_summary" in messages[0])
        
        if has_existing_summary:
            # Get existing summary and messages to summarize
            existing_summary = messages[0]["past_convo_summary"]
            messages_to_summarize = messages[1:-messages_to_keep]  # Exclude summary and last N
            last_n_messages = messages[-messages_to_keep:]
        else:
            # No existing summary
            existing_summary = None
            messages_to_summarize = messages[:-messages_to_keep]  # All except last N
            last_n_messages = messages[-messages_to_keep:]
        
        # Create summarization prompt
        if existing_summary:
            summary_prompt = f"Previous conversation summary: {existing_summary}\n\nRecent conversation messages:\n"
        else:
            summary_prompt = "Conversation messages to summarize:\n"
          # Add messages to summarize
        for msg in messages_to_summarize:
            if "agent_name" in msg and "message" in msg:
                summary_prompt += f"{msg['agent_name']}: {msg['message']}\n"
        
        summary_prompt += "\nPlease provide a concise summary of the conversation above, capturing the key topics, main points discussed, and important context. Only return the summary text, nothing else."
        
        try:
            # Get summary from LLM
            response = self.summary_model.invoke([HumanMessage(content=summary_prompt)])
            new_summary = response.content.strip()
            
            # Create new messages list with summary + last N messages            
            new_messages = [{"past_convo_summary": new_summary}] + last_n_messages
            
            return new_messages
            
        except Exception as e:
            print(f"Error during summarization: {e}")
            # Fallback: just keep last N+5 messages if summarization fails
            fallback_count = messages_to_keep + 5
            return messages[-fallback_count:]
    
    def create_agent_prompt(self, agent_config: Dict[str, str], environment: str, 
                          scene_description: str, messages: List[Dict[str, str]], 
                          all_agents: List[str], termination_condition: Optional[str] = None,
                          should_remind_termination: bool = False, conversation_id: Optional[str] = None,
                          agent_name: Optional[str] = None, available_tools: List[str] = None,
                          agent_obj: Optional[Any] = None) -> str:
        """
        Create the prompt for an agent including scene, participants, and conversation history.
        
        Args:
            agent_config: Agent configuration
            environment: Current environment
            scene_description: Scene description
            messages: Current messages list
            all_agents: List of all agent names
            termination_condition: Optional termination condition for the conversation
            should_remind_termination: Whether to include termination condition reminder
            conversation_id: Conversation ID (unused, kept for compatibility)
            agent_name: Agent name to load specific context for
            available_tools: List of available tool names for this agent
            
        Returns:
            Formatted prompt string
        """
        if not agent_name:
            agent_name = agent_config["name"]
        agent_role = agent_config["role"]
        base_prompt = agent_config["base_prompt"]
        
        prompt = f"""You are {agent_name}, a {agent_role}.

{base_prompt}
Always answer based on the above characteristics. Stay in character always.
INITIAL SCENE: {environment}
SCENE DESCRIPTION: {scene_description}

PARTICIPANTS: {', '.join(all_agents)}

Tool Usage: Use your tools freely in the first instance you feel,  just like a noraml person using their mobile phone as a tool. No need to get permsission from other agents. But when it's necessary discuss with other agents how the tools should be used.

"""
        
        # Always use the current messages list as the single source of truth
        print(f"DEBUG: Using current messages for '{agent_name}' ({len(messages)} items)")
        if messages:
            if messages[0].get("past_convo_summary"):
                prompt += f"PREVIOUS CONVERSATION SUMMARY: {messages[0]['past_convo_summary']}\n\n"
                recent_messages = messages[1:]
            else:
                recent_messages = messages
            
            if recent_messages:
                prompt += "CONVERSATION SO FAR:\n"
                for msg in recent_messages:
                    if "agent_name" in msg and "message" in msg:
                        prompt += f"{msg['agent_name']}: {msg['message']}\n"
                prompt += "\n"
        
        # Add termination condition reminder if appropriate
        if should_remind_termination and termination_condition:
            prompt += f"""TERMINATION CONDITION REMINDER: The conversation should end when the following condition is met:
{termination_condition}

Keep this condition in mind while participating in the conversation. Naturally deviate the conversation into the direction where the condition will be met. and stay true to your personality traits.

"""
        
        # Add tool information if available
        if available_tools:
            prompt += f"""AVAILABLE TOOLS: You have access to the following tools: {', '.join(available_tools)}
Use these tools when they can help you respond more effectively to the conversation.
Only use tools when they are relevant to the current conversation context.
Don't mention the tools explicitly unless asked about your capabilities.

"""
        
        # Add knowledge base information if agent has documents
        if agent_obj and hasattr(agent_obj, 'knowledge_base') and agent_obj.knowledge_base:
            knowledge_descriptions = []
            for doc in agent_obj.knowledge_base:
                if hasattr(doc, 'metadata') and 'description' in doc.metadata:
                    knowledge_descriptions.append(f"- {doc.metadata['description']}")
            
            prompt += f"""PERSONAL KNOWLEDGE BASE: You have access to a personal knowledge base containing the following documents:
{chr(10).join(knowledge_descriptions)}

Use the knowledge_base_retriever tool to search through these documents when relevant to the conversation. 
This knowledge base contains specialized information that can help you stay true to your role and provide more informed responses.
Only search your knowledge base when the conversation topic relates to the content of your documents.

"""
        
        prompt += f"""Give your response to the ongoing conversation as {agent_name}. 
Keep your response natural, conversational, and true to your character. Always respons with the charateristics/personality of your character. 
Respond as if you're speaking directly in the conversation (don't say "As {agent_name}, I would say..." just respond naturally).
Respond only to the dialog parts said by the other agents.
Keep responses to 1-3 sentences to maintain good conversation flow."""
        
        return prompt
    
    def start_conversation(self, conversation_id: str, agents_config: List[Dict[str, str]],
                        environment: str, scene_description: str, initial_message: str = None,
                        invocation_method: str = "round_robin", termination_condition: Optional[str] = None,
                        agent_selector_api_key: Optional[str] = None, voices_enabled: bool = False) -> str:
        """
        Start a new conversation session.
        
        Args:
            conversation_id: Unique identifier for the conversation
            agents_config: List of agent configurations (name, role, base_prompt, api_key)
            environment: Environment description for the conversation
            scene_description: Scene description for the conversation
            initial_message: Optional initial message to start the conversation
            invocation_method: Method for selecting which agent speaks next ("round_robin" or "agent_selector")
            termination_condition: Optional condition for when to end the conversation. 
                                  When provided with "round_robin" method, an LLM will evaluate
                                  after each round if the condition is met. When provided with 
                                  "agent_selector" method, the selector LLM decides termination.
                                  If not provided, conversation continues indefinitely.
            agent_selector_api_key: Optional API key for the agent selector (required if using agent_selector method)
            
        Returns:
            Thread ID for the conversation
        """
        thread_id = f"thread_{conversation_id}"
        
        print(f"DEBUG: ===== STARTING NEW CONVERSATION {conversation_id} =====")
        print(f"DEBUG: Using invocation method: {invocation_method}")
        print(f"DEBUG: Environment: {environment}")
        print(f"DEBUG: Agents: {[config['name'] for config in agents_config]}")
        print(f"DEBUG: ===== END NEW CONVERSATION INFO =====")
          # Validate agent API keys
        agent_names = [config["name"] for config in agents_config]
        for agent_config in agents_config:
            if not agent_config.get("api_key") and not self.default_api_key:
                raise ValueError(f"API key must be provided for agent '{agent_config['name']}' or as a default.")

        # Validate agent selector API key if needed
        if invocation_method == "agent_selector":
            if not agent_selector_api_key:
                raise ValueError("agent_selector_api_key is required when using agent_selector invocation method.")

        # Assign temporary numbers to agents starting from 1 for chat bubble alignment
        agent_temp_numbers = {}
        for i, agent_config in enumerate(agents_config, 1):
            agent_temp_numbers[agent_config["name"]] = i

        # Store conversation data
        self.active_conversations[conversation_id] = {
            "agents": agents_config,
            "agents_config": {config["name"]: config for config in agents_config},
            "agent_names": [config["name"] for config in agents_config],
            "agent_invocation_counts": {config["name"]: 0 for config in agents_config},  # <-- add this line
            "environment": environment,
            "scene_description": scene_description,
            "invocation_method": invocation_method,
            "termination_condition": termination_condition,
            "agent_selector_api_key": agent_selector_api_key,
            "voices_enabled": voices_enabled,
            "messages": [],
            "message_count": 0,
            "thread_id": thread_id,
            "status": "active",
            "is_paused": False,
            "stop_conversation": False,
            "current_agent_index": 0,
            "message_callbacks": []
        }
        # Defensive: ensure stop_conversation is always present
        if "stop_conversation" not in self.active_conversations[conversation_id]:
            self.active_conversations[conversation_id]["stop_conversation"] = False

        if initial_message:
            self.active_conversations[conversation_id]["messages"].append({
                "agent_name": "Human",
                "message": initial_message,
                "timestamp": datetime.now().isoformat()
            })

        self._start_conversation_cycle(conversation_id)
        return thread_id


    def _start_conversation_cycle(self, conversation_id: str):
        """Starts the conversation cycle based on the invocation method."""
        convo = self.active_conversations.get(conversation_id)
        if not convo or convo["stop_conversation"]:
            return

        if convo["invocation_method"] == "round_robin":
            self.round_robin_engine.start_cycle(conversation_id)
        elif convo["invocation_method"] == "agent_selector":
            self.agent_selector_engine.start_cycle(conversation_id)
        elif convo["invocation_method"] == "human_like_chat":
            self.human_like_chat_engine.start_cycle(conversation_id)
        else:
            print(f"ERROR: Unknown invocation method: {convo['invocation_method']}")

    def _invoke_next_agent(self, conversation_id: str):
        """Invokes the next agent in the conversation."""
        convo = self.active_conversations.get(conversation_id)
        if not convo or convo["stop_conversation"]:
            return

        if convo["invocation_method"] == "round_robin":
            self.round_robin_engine.invoke_next_agent(conversation_id)
        elif convo["invocation_method"] == "agent_selector":
            self.agent_selector_engine.invoke_next_agent(conversation_id)
        elif convo["invocation_method"] == "human_like_chat":
            self.human_like_chat_engine.invoke_next_agent(conversation_id)
        else:
            print(f"ERROR: Unknown invocation method: {convo['invocation_method']}")

    def stop_conversation(self, conversation_id: str):
        """Stops the conversation cycle for a given conversation ID."""
        if conversation_id in self.active_conversations:
            convo = self.active_conversations[conversation_id]
            convo["stop_conversation"] = True
            convo["status"] = "stopped"
            
            # Clear any pending audio messages for human_like_chat mode
            if convo.get("invocation_method") == "human_like_chat" and "audio_queue" in convo:
                convo["audio_queue"] = []
                convo["is_audio_playing"] = False # Ensure no further audio processing
                print(f"DEBUG: Cleared pending audio queue for conversation {conversation_id}.")

            print(f"DEBUG: Conversation {conversation_id} marked to stop.")

    def get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """Generates and returns a summary of the conversation history."""
        if conversation_id not in self.active_conversations:
            print(f"Error: Conversation {conversation_id} not found for summarization.")
            return None

        convo = self.active_conversations[conversation_id]
        messages = convo["messages"]

        if not messages:
            return "The conversation is empty."

        # Check if there's already a summary at the beginning
        existing_summary = None
        messages_to_summarize = messages
        if messages and messages[0].get("past_convo_summary"):
            existing_summary = messages[0]["past_convo_summary"]
            messages_to_summarize = messages[1:]

        if not messages_to_summarize:
            return existing_summary or "No new messages to summarize."

        # Create summarization prompt
        if existing_summary:
            summary_prompt = f"Previous conversation summary: {existing_summary}\n\nRecent conversation messages:\n"
        else:
            summary_prompt = "Conversation messages to summarize:\n"
        
        for msg in messages_to_summarize:
            if "agent_name" in msg and "message" in msg:
                summary_prompt += f"{msg['agent_name']}: {msg['message']}\n"
        
        summary_prompt += "\nPlease provide a concise summary of the conversation above, capturing the key topics, main points discussed, and important context. Only return the summary text, nothing else."

        try:
            # Get summary from LLM
            response = self.summary_model.invoke([HumanMessage(content=summary_prompt)])
            summary = response.content.strip()
            return summary
        except Exception as e:
            print(f"Error during summarization: {e}")
            return f"An error occurred while generating the summary: {e}"

    def pause_conversation(self, conversation_id: str):
        """Pauses the conversation cycle for a given conversation ID."""
        if conversation_id in self.active_conversations:
            convo = self.active_conversations[conversation_id]
            convo["stop_conversation"] = True
            convo["status"] = "paused"
            print(f"DEBUG: Conversation {conversation_id} paused.")

    def resume_conversation(self, conversation_id: str):
        """Resumes the conversation cycle for a given conversation ID."""
        if conversation_id in self.active_conversations:
            convo = self.active_conversations[conversation_id]
            self._patch_conversation_state(conversation_id)
            if convo["stop_conversation"]:
                convo["stop_conversation"] = False
            convo["status"] = "active"
            print(f"DEBUG: Conversation {conversation_id} resumed.")
            # Restart the cycle by invoking the next agent in a new thread
            threading.Thread(target=self._invoke_next_agent, args=(conversation_id,), daemon=True).start()

    def register_message_callback(self, conversation_id: str, callback: Callable):
        """Registers a callback function to receive messages for a conversation."""
        self.message_callbacks[conversation_id] = callback

    def _create_agent(self, conversation_id: str, agent_name: str):
        """Creates and returns a new agent instance."""
        convo = self.active_conversations[conversation_id]
        agent_config = convo["agents_config"][agent_name]
        
        api_key = agent_config.get("api_key") or self.default_api_key
        if not api_key:
            raise ValueError(f"API key is missing for agent {agent_name}")

        model = ChatGoogleGenerativeAI(
            model=MODEL_SETTINGS["agent_model"],
            temperature=AGENT_SETTINGS["response_temperature"],
            max_retries=AGENT_SETTINGS["max_retries"],
            google_api_key=api_key
        )
        
        tools = self._load_agent_tools(conversation_id, agent_name)
        
        agent = create_react_agent(model, tools, checkpointer=self.memory)
        convo["agents"][agent_name] = agent
        return agent

    def _get_agent(self, conversation_id: str, agent_name: str):
        """Retrieves an agent instance, creating it if it doesn't exist."""
        convo = self.active_conversations[conversation_id]
        if agent_name not in convo["agents"]:
            return self._create_agent(conversation_id, agent_name)
        return convo["agents"][agent_name]

    def _load_agent_tools(self, conversation_id: str, agent_name: str) -> List[Tool]:
        """Loads tools for a specific agent based on their configuration."""
        convo = self.active_conversations[conversation_id]
        agent_config = convo["agents_config"][agent_name]
        agent_tool_names = agent_config.get("tools", [])
        
        loaded_tools = []
        for tool_name in agent_tool_names:
            if tool_name in AVAILABLE_TOOLS:
                tool_to_load = AVAILABLE_TOOLS[tool_name]
                # Special handling for knowledge_base_retriever to pass agent_id
                if tool_name == "knowledge_base_retriever":
                    agent_id = agent_config.get("id")
                    if agent_id:
                        # Use partial to pre-fill the agent_id argument
                        loaded_tools.append(Tool(
                            name="knowledge_base_retriever",
                            func=partial(knowledge_base_retriever.func, agent_id=agent_id),
                            description=tool_to_load.description
                        ))
                    else:
                        print(f"Warning: Agent '{agent_name}' has 'knowledge_base_retriever' tool but no 'id' in config.")
                else:
                    loaded_tools.append(tool_to_load)
            else:
                print(f"Warning: Tool '{tool_name}' for agent '{agent_name}' not found.")
                
        return loaded_tools

    def _get_agent_context_messages(self, conversation_id: str, agent_name: str) -> List[Dict[str, str]]:
        """Gets the message history for a specific agent, potentially summarized."""
        convo = self.active_conversations[conversation_id]
        return self.message_list_summarization(convo["messages"])

    def _update_agent_sending_messages(self, conversation_id: str, agent_name: str):
        """Placeholder for updating agent-specific message states."""
        pass

    def _save_conversation_state(self, conversation_id: str):
        """Saves the current state of the conversation."""
        # This is a placeholder. In a real implementation, you might save
        # to a database or file.
        pass

    def _check_round_robin_termination(self, conversation_id: str) -> bool:
        """
        Checks if the conversation should terminate based on the termination condition.
        This method is used by the round_robin_engine.
        """
        convo = self.active_conversations.get(conversation_id)
        if not convo or not convo.get("termination_condition"):
            return False

        # To prevent spamming the termination check prompt, we only send it once
        if convo.get("termination_check_prompt_sent"):
            return False

        prompt = f'''Analyze the following conversation and determine if the termination condition has been met.
        Termination Condition: {convo["termination_condition"]}
        Conversation History:
        '''
        for msg in convo["messages"]:
            prompt += f'{msg["agent_name"]}: {msg["message"]}\n'
        
        prompt += "Respond with only 'true' if the condition is met, and 'false' otherwise."

        try:
            response = self.summary_model.invoke([HumanMessage(content=prompt)])
            result = response.content.strip().lower()
            
            # Mark that the check has been sent to avoid repeated checks
            convo["termination_check_prompt_sent"] = True
            
            return result == 'true'
        except Exception as e:
            print(f"Error checking termination condition: {e}")
            return False

    def _patch_conversation_state(self, conversation_id: str):
        """Ensures all required fields are present and correctly typed in the conversation dict."""
        convo = self.active_conversations[conversation_id]
        # Ensure agents_config is a dict
        if isinstance(convo.get("agents_config"), list):
            convo["agents_config"] = {config["name"]: config for config in convo["agents_config"]}
        # Ensure agent_names is a list
        if "agent_names" not in convo or not isinstance(convo["agent_names"], list):
            if "agents_config" in convo:
                convo["agent_names"] = list(convo["agents_config"].keys())
            elif "agents" in convo:
                convo["agent_names"] = [a["name"] for a in convo["agents"]]
            else:
                convo["agent_names"] = []
        # Ensure agent_invocation_counts is a dict with all agent names
        if "agent_invocation_counts" not in convo or not isinstance(convo["agent_invocation_counts"], dict):
            convo["agent_invocation_counts"] = {name: 0 for name in convo["agent_names"]}
        else:
            for name in convo["agent_names"]:
                if name not in convo["agent_invocation_counts"]:
                    convo["agent_invocation_counts"][name] = 0
        # Ensure agent_temp_numbers exists
        if "agent_temp_numbers" not in convo or not isinstance(convo["agent_temp_numbers"], dict):
            convo["agent_temp_numbers"] = {name: i+1 for i, name in enumerate(convo["agent_names"])}
        # Ensure current_agent_index exists
        if "current_agent_index" not in convo:
            convo["current_agent_index"] = 0
        # Ensure message_count exists
        if "message_count" not in convo:
            convo["message_count"] = len(convo.get("messages", []))
        # Ensure status is set to 'active' when resuming
        convo["status"] = "active"
        # Ensure is_paused is False
        convo["is_paused"] = False
        # Ensure stop_conversation is False
        convo["stop_conversation"] = False
        # Ensure messages is a list
        if "messages" not in convo or not isinstance(convo["messages"], list):
            convo["messages"] = []
        # Ensure message_callbacks is a list
        if "message_callbacks" not in convo or not isinstance(convo["message_callbacks"], list):
            convo["message_callbacks"] = []
        # Defensive: add any other fields as needed for round robin
        # (e.g., agent_colors, if used elsewhere)