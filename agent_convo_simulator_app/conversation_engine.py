"""
Simple Multi-Agent Conversation Engine
Real LangGraph react agents with Gemini in conversation cycle.
"""

import os
import random
import json
import re
import traceback
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import threading
import time
from functools import partial
import concurrent.futures
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import configuration settings
from config import CONVERSATION_TIMING, MESSAGE_SETTINGS, AGENT_SETTINGS, MODEL_SETTINGS
from agent_selector import AgentSelector
from utility import extract_json_from_markdown


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
            temperature=AGENT_SETTINGS["summary_temperature"],  # Lower temperature for more consistent summaries
            max_retries=AGENT_SETTINGS["max_retries"],
            google_api_key=summary_api_key
        )
        
        self.memory = MemorySaver()
          # Storage for active conversations
        self.active_conversations = {}
        self.message_callbacks = {}  # For real-time message updates
    
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
                knowledge_descriptions.append(f"- {doc['doc_name']}: {doc['description']}")
            
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
            agent_name = agent_config["name"]
            agent_api_key = agent_config.get("api_key")
            if not agent_api_key:
                print(f"INFO: Agent '{agent_name}' will use default API key (no specific key provided)")
            else:
                print(f"INFO: Agent '{agent_name}' will use their specific API key")
        
        # Validate agent selector API key if needed
        if invocation_method == "agent_selector":
            if not agent_selector_api_key:
                print("INFO: Agent selector will use default API key (no specific key provided)")
            else:
                print("INFO: Agent selector will use the provided specific API key")
        
        # Assign temporary numbers to agents starting from 1 for chat bubble alignment
        agent_temp_numbers = {}
        for i, agent_config in enumerate(agents_config, 1):
            agent_id = agent_config.get("id")
            if agent_id:
                agent_temp_numbers[agent_id] = i
                print(f"DEBUG: Assigned temp number {i} to agent '{agent_config['name']}' (ID: {agent_id})")
        
        # Store conversation data (agents will be created individually when invoked)
        self.active_conversations[conversation_id] = {
            "agents_config": agents_config,
            "agent_names": agent_names,
            "thread_id": thread_id,
            "environment": environment,
            "scene_description": scene_description,
            "status": "active",
            "messages": [],  # Our main message list
            "current_agent_index": 0,
            "conversation_started": False,
            "invocation_method": invocation_method,
            "termination_condition": termination_condition,
            "agent_invocation_counts": {name: 0 for name in agent_names},  # Track invocations per agent
            "agent_selector_api_key": agent_selector_api_key,  # Store agent selector API key
            "agent_sending_messages": {name: [] for name in agent_names},  # Initialize per-agent message lists
            "round_counter": 0,  # Track rounds for human-like-chat mode
            "last_round_participants": [],  # Track who participated in the last round
            "current_round_responses": {},  # Track responses in current round for parallel processing
            "agent_temp_numbers": agent_temp_numbers,  # Store agent temp numbers for chat bubble alignment
            "voices_enabled": voices_enabled,  # Store voice enablement flag
            # Audio synchronization state
            "waiting_for_audio_generation": False,  # Track if we're waiting for audio generation
            "waiting_for_audio_playback": False,    # Track if we're waiting for audio playback
            "last_agent_audio_sent": None,          # Track which agent last had audio generation requested
            "next_agent_scheduled": None,           # Track the next agent to be invoked after audio completes
            "audio_callbacks_set": False            # Track if audio callbacks are set
        }
            # Start the conversation automatically after a short delay
        threading.Timer(CONVERSATION_TIMING["start_delay"], self._start_conversation_cycle, args=(conversation_id,)).start()
        
        return thread_id

    def _start_conversation_cycle(self, conversation_id: str):
        """Start the conversation cycle with the first agent."""
        if conversation_id not in self.active_conversations:
            return
        
        conv_data = self.active_conversations[conversation_id]
        if conv_data["status"] != "active":
            return
        
        # Mark conversation as started
        conv_data["conversation_started"] = True
        
        invocation_method = conv_data.get("invocation_method", "round_robin")
        
        if invocation_method == "human_like_chat":
            print(f"DEBUG: ===== STARTING HUMAN-LIKE-CHAT CONVERSATION =====")
            # Use agent selector to choose the first agent
            self._start_human_like_chat(conversation_id)
        else:
            print(f"DEBUG: ===== STARTING CONVERSATION CYCLE =====")
            # Choose random first agent for round_robin and agent_selector methods
            conv_data["current_agent_index"] = random.randint(0, len(conv_data["agent_names"]) - 1)
            first_agent = conv_data["agent_names"][conv_data["current_agent_index"]]
            print(f"DEBUG: First agent selected: {first_agent} (index {conv_data['current_agent_index']})")
            print(f"DEBUG: ===== STARTING FIRST AGENT INVOCATION =====")
            
            # Start the conversation
            self._invoke_next_agent(conversation_id)
    
    def _invoke_next_agent(self, conversation_id: str):
        """Invoke the next agent in the cycle."""
        print(f"DEBUG: _invoke_next_agent called for conversation {conversation_id}")
        
        if conversation_id not in self.active_conversations:
            print(f"ERROR: Conversation {conversation_id} not found in active conversations")
            return
        
        conv_data = self.active_conversations[conversation_id]
        if conv_data["status"] != "active":
            print(f"DEBUG: Conversation {conversation_id} status is {conv_data['status']}, not active")
            return
        
        # Check if this is human-like-chat mode
        invocation_method = conv_data.get("invocation_method", "round_robin")
        if invocation_method == "human_like_chat":
            # For human-like-chat, after the first agent speaks, switch to round-based mode
            if len(conv_data["messages"]) == 0:
                # This is the initial agent invocation, proceed normally
                self._invoke_next_agent_regular(conversation_id)
            else:
                # Switch to round-based human-like-chat mode
                print("DEBUG: Switching to human-like-chat round mode")
                self._start_human_like_chat_round(conversation_id)
            return
        
        # Handle regular round_robin and agent_selector modes
        self._invoke_next_agent_regular(conversation_id)
    
    def _invoke_next_agent_regular(self, conversation_id: str):
        """Invoke the next agent using regular (non-human-like-chat) logic."""
        conv_data = self.active_conversations[conversation_id]
        
        # Get current agent
        current_agent_name = conv_data["agent_names"][conv_data["current_agent_index"]]
        current_agent_config = next(config for config in conv_data["agents_config"] 
                                  if config["name"] == current_agent_name)
        
        print(f"DEBUG: Invoking agent '{current_agent_name}' (index {conv_data['current_agent_index']})")
        
        # Increment invocation count for current agent
        conv_data["agent_invocation_counts"][current_agent_name] += 1
        
        # Determine if we should remind about termination condition
        termination_condition = conv_data.get("termination_condition")
        should_remind_termination = False
        
        if termination_condition:
            reminder_frequency = AGENT_SETTINGS["termination_reminder_frequency"]
            current_count = conv_data["agent_invocation_counts"][current_agent_name]
            should_remind_termination = (current_count % reminder_frequency == 0)
            if should_remind_termination:
                print(f"DEBUG: Agent '{current_agent_name}' will be reminded about termination condition (invocation #{current_count})")
        
        # Apply per-agent message summarization before creating prompt
        self._update_agent_sending_messages(conversation_id, current_agent_name)
        
        # Load agent tools first to include in prompt
        agent_tools = []
        tool_names = []
        
        try:
            # Import tools module at the function level
            import tools as tools_module
            
            # Get the Agent object from our data_manager using agent name
            agent_id = current_agent_config.get("id")
            agent_obj = None
            
            # Load agent data from data manager
            if not hasattr(self, 'agents_data'):
                # We need to initialize this first
                from data_manager import DataManager
                data_manager = DataManager(os.path.dirname(__file__))
                self.agents_data = {agent.id: agent for agent in data_manager.load_agents()}
            
            # Get the agent object
            agent_obj = self.agents_data.get(agent_id)
            
            # Check if agent has tools
            if agent_obj and hasattr(agent_obj, 'tools') and agent_obj.tools:
                print(f"DEBUG: Loading {len(agent_obj.tools)} tools for agent '{current_agent_name}': {agent_obj.tools}")
                
                for tool_name in agent_obj.tools:
                    try:
                        if tool_name == "knowledge_base_retriever":
                            # Only add knowledge_base_retriever if agent has documents in knowledge_base
                            if hasattr(agent_obj, 'knowledge_base') and agent_obj.knowledge_base:
                                if hasattr(tools_module, "knowledge_base_retriever"):
                                    retriever_tool_func = getattr(tools_module, "knowledge_base_retriever").func
                                    
                                    # Create a partial function with agent_id
                                    partial_func = partial(retriever_tool_func, agent_id=agent_id)
                                    
                                    # Create a new tool that only expects a query
                                    new_tool = Tool(
                                        name="knowledge_base_retriever",
                                        func=partial_func,
                                        description="Retrieves the top 3 most relevant results from your personal knowledge base to help answer questions. Input should be a search query."
                                    )
                                    agent_tools.append(new_tool)
                                    tool_names.append(new_tool.name)
                                    print(f"DEBUG: ✓ Added knowledge_base_retriever tool for agent {current_agent_name} (has {len(agent_obj.knowledge_base)} documents)")
                                else:
                                    print(f"WARNING: ✗ Tool 'knowledge_base_retriever' not found in tools module")
                            else:
                                print(f"DEBUG: ⚠ Skipping knowledge_base_retriever for agent {current_agent_name} (no documents in knowledge base)")
                        # Get the tool object by name from tools module
                        elif tool_name == "browser_manipulation_toolkit":
                            # Special case for browser toolkit which is a list of tools
                            browser_tools = tools_module.get_browser_tools()
                            if browser_tools:  # Only add if tools were successfully loaded
                                agent_tools.extend(browser_tools)
                                # Add individual tool names from the toolkit
                                for bt in browser_tools:
                                    if hasattr(bt, 'name'):
                                        tool_names.append(bt.name)
                                    elif hasattr(bt, '__name__'):
                                        tool_names.append(bt.__name__)
                                print(f"DEBUG: ✓ Added browser toolkit ({len(browser_tools)} tools) for agent {current_agent_name}")
                            else:
                                print(f"DEBUG: ⚠ Browser toolkit could not be loaded for agent {current_agent_name}")
                        elif hasattr(tools_module, tool_name):
                            tool_obj = getattr(tools_module, tool_name)
                            agent_tools.append(tool_obj)
                            if hasattr(tool_obj, 'name'):
                                tool_names.append(tool_obj.name)
                            elif hasattr(tool_obj, '__name__'):
                                tool_names.append(tool_obj.__name__)
                            else:
                                tool_names.append(tool_name)
                            print(f"DEBUG: ✓ Added tool '{tool_name}' for agent {current_agent_name}")
                        else:
                            print(f"WARNING: ✗ Tool '{tool_name}' not found in tools module")
                    except Exception as e:
                        print(f"ERROR: ✗ Failed to add tool '{tool_name}': {e}")
            else:
                print(f"DEBUG: Agent '{current_agent_name}' has no tools assigned")
                
            print(f"DEBUG: Agent '{current_agent_name}' loaded with {len(agent_tools)} total tools")
        except Exception as e:
            print(f"ERROR: Failed to load tools for agent {current_agent_name}: {e}")
          # Create prompt for current agent (now with agent-specific message context)
        agent_messages = self._get_agent_context_messages(conversation_id, current_agent_name)
        prompt = self.create_agent_prompt(
            current_agent_config,
            conv_data["environment"],
            conv_data["scene_description"],
            agent_messages,  # Use agent-specific messages instead of global messages
            conv_data["agent_names"],
            termination_condition,
            should_remind_termination,
            conversation_id,  # Pass conversation_id to load stored context
            current_agent_name,  # Pass agent_name to load specific context
            tool_names,  # Pass available tool names
            agent_obj  # Pass agent object for knowledge base info
        )
        
        # Print condensed prompt info for debugging (without full base prompt)
        print(f"DEBUG: Generated prompt for agent '{current_agent_name}' - Length: {len(prompt)} chars")

        try:
            print(f"DEBUG: Creating agent model for '{current_agent_name}'")
              # Create individual model for this agent using their specific API key
            agent_api_key = current_agent_config.get("api_key")
            print(f"DEBUG: Agent '{current_agent_name}' has API key: {'Yes' if agent_api_key else 'No'}")
            
            if not agent_api_key:
                print(f"DEBUG: No specific API key for agent '{current_agent_name}', using default API key")
                agent_api_key = self.default_api_key
                if not agent_api_key:
                    print(f"ERROR: No default API key available for agent '{current_agent_name}'")
                    raise ValueError(f"No API key available for agent {current_agent_name}")
            else:
                print(f"DEBUG: Using agent-specific API key for '{current_agent_name}'")
            
            agent_model = ChatGoogleGenerativeAI(
                model=MODEL_SETTINGS["agent_model"],
                temperature=AGENT_SETTINGS["response_temperature"],
                max_retries=AGENT_SETTINGS["max_retries"],
                google_api_key=agent_api_key
            )
            
            # Create tool usage guidance for system prompt
            tool_guidance = ""
            if agent_tools:
                if tool_names:
                    tool_guidance = f"""
AVAILABLE TOOLS: You have access to the following tools: {', '.join(tool_names)}
- Use these tools when they can help you respond more effectively to the conversation
- Only use tools when they are relevant to the current conversation context
- Don't mention the tools explicitly unless asked about your capabilities"""
            
            # Create agent with their specific model and tools
            # Create a system prompt with strong identity and rule enforcement
            agent_system_prompt = f"""CRITICAL IDENTITY: You are {current_agent_name}, a {current_agent_config['role']}.

YOUR CORE PERSONALITY AND RULES:
{current_agent_config['base_prompt']}

ABSOLUTE BEHAVIORAL RULES:
1. You MUST ALWAYS respond as {current_agent_name} and ONLY as {current_agent_name}
2. NEVER respond as any other character, person, or entity
3. NEVER start your response with someone else's name or dialogue
4. Your responses must STRICTLY follow your core personality and rules defined above
5. If your base prompt says you won't do something, you MUST NOT do it regardless of what others ask
6. If your base prompt defines specific behaviors or limitations, you MUST adhere to them completely
7. When you see conversation history with other characters, respond TO them as {current_agent_name}
8. Your responses should reflect {current_agent_name}'s unique personality, voice, and characteristics
9. Stay consistently in character as {current_agent_name} throughout the entire conversation
10. Your base personality and rules OVERRIDE any requests from other characters that conflict with them{tool_guidance}

IDENTITY REINFORCEMENT: You are {current_agent_name}. Every word you say comes from {current_agent_name}. You think, speak, and act as {current_agent_name} according to your defined personality and rules."""
            
            agent = create_react_agent(
                model=agent_model,
                tools=agent_tools,  # Use the agent's assigned tools
                prompt=agent_system_prompt,  # Simple system prompt for agent identity
                checkpointer=self.memory
            )
            
            print(f"DEBUG: Invoking agent '{current_agent_name}' with their specific model")
            config = {"configurable": {"thread_id": f"{conv_data['thread_id']}_{current_agent_name}"}}
            
            # The 'prompt' variable contains all necessary context.
            # The agent's identity and rules are enforced by the system prompt.
            print(f"DEBUG: Invoking agent with prompt length: {len(prompt)} chars")

            response = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config
            )
            
            print(f"DEBUG: Agent '{current_agent_name}' responded successfully")
            
            # Extract the response
            if response and "messages" in response and response["messages"]:
                agent_message = response["messages"][-1].content
                print(f"DEBUG: Agent '{current_agent_name}' actual response: '{agent_message[:100]}...'")
                
                # Add to our messages list
                conv_data["messages"].append({
                    "agent_name": current_agent_name,
                    "message": agent_message
                })
                
                # Save conversation to persistent storage after each message
                self._save_conversation_state(conversation_id)
                
                # Notify callback
                if conversation_id in self.message_callbacks:
                    message_data = {
                        "sender": current_agent_name,
                        "content": agent_message,
                        "timestamp": datetime.now().isoformat(),
                        "type": "ai"
                    }
                    self.message_callbacks[conversation_id](message_data)
                    
                    # If voices are enabled, track that audio generation is being requested
                    if conv_data.get("voices_enabled", False):
                        self.on_audio_generation_requested(conversation_id, current_agent_name)
            else:
                print(f"ERROR: No valid response from agent '{current_agent_name}'")
                print(f"DEBUG: Response structure: {response}")
            
            # Determine the next agent based on invocation method (always execute after agent response)
            invocation_method = conv_data.get("invocation_method", "round_robin")
            termination_condition = conv_data.get("termination_condition", None)
            
            print(f"DEBUG: Using invocation method: {invocation_method}")
            
            if invocation_method == "agent_selector":
                print("DEBUG: Using agent_selector to determine next agent")
                # Get agent selector API key
                agent_selector_api_key = conv_data.get("agent_selector_api_key")
                print(f"DEBUG: Agent selector has API key: {'Yes' if agent_selector_api_key else 'No'}")
                
                if not agent_selector_api_key:
                    print("DEBUG: No specific API key for agent_selector, using default API key")
                    agent_selector_api_key = self.default_api_key
                    
                if not agent_selector_api_key:
                    print("ERROR: No default API key available for agent_selector")
                    # Fall back to round robin
                    print("DEBUG: Falling back to round robin due to missing agent_selector API key")
                    conv_data["current_agent_index"] = (conv_data["current_agent_index"] + 1) % len(conv_data["agent_names"])
                else:
                    print(f"DEBUG: Creating agent_selector with API key")
                    # Create agent selector with specific API key
                    temp_agent_selector = AgentSelector(google_api_key=agent_selector_api_key)
                    
                    # Use the agent selector to determine the next agent
                    selection_result = temp_agent_selector.select_next_agent(
                        messages=conv_data["messages"],
                        environment=conv_data["environment"],
                        scene=conv_data["scene_description"],
                        agents=conv_data["agents_config"],
                        termination_condition=termination_condition,
                        agent_invocation_counts=conv_data.get("agent_invocation_counts", {})
                    )
                    
                    next_response = selection_result.get("next_response", "error_parsing")
                    print(f"DEBUG: Agent_selector decision: {next_response}")
                    
                    if next_response == "terminate":
                        print("DEBUG: Agent_selector decided to terminate conversation")
                        # End the conversation
                        if conversation_id in self.message_callbacks:
                            self.message_callbacks[conversation_id]({
                                "sender": "System",
                                "content": "The conversation has reached its termination condition and has ended.",
                                "timestamp": datetime.now().isoformat(),
                                "type": "system"
                            })
                        # Stop the conversation
                        conv_data["status"] = "completed"
                        return
                    elif next_response == "error_parsing":
                        print("DEBUG: Agent_selector had parsing error, using round robin fallback")
                        # Error in parsing, use round robin as fallback
                        conv_data["current_agent_index"] = (conv_data["current_agent_index"] + 1) % len(conv_data["agent_names"])
                    else:
                        # Set the next agent
                        if next_response in conv_data["agent_names"]:
                            print(f"DEBUG: Agent_selector chose agent: {next_response}")
                            # Find the index of the selected agent
                            for i, name in enumerate(conv_data["agent_names"]):
                                if name == next_response:
                                    conv_data["current_agent_index"] = i
                                    break
                        else:
                            print(f"DEBUG: Agent_selector chose unknown agent '{next_response}', using round robin fallback")
                            # If agent name not found, use round robin as fallback
                            conv_data["current_agent_index"] = (conv_data["current_agent_index"] + 1) % len(conv_data["agent_names"])
            else:
                print("DEBUG: Using round_robin method")
                # Round robin method - move to next agent in cycle
                conv_data["current_agent_index"] = (conv_data["current_agent_index"] + 1) % len(conv_data["agent_names"])
                
                # Check if we completed a round (all agents have spoken)
                # and if there's a termination condition to check
                if (conv_data["current_agent_index"] == 0 and 
                    termination_condition and 
                    len(conv_data["messages"]) >= len(conv_data["agent_names"])):
                    
                    print("DEBUG: Checking round_robin termination condition")
                    # Check if conversation should terminate
                    if self._check_round_robin_termination(conversation_id):
                        print("DEBUG: Round_robin termination condition met")
                        # End the conversation
                        if conversation_id in self.message_callbacks:
                            self.message_callbacks[conversation_id]({
                                "sender": "System",
                                "content": "Termination condition reached.",
                                "timestamp": datetime.now().isoformat(),
                                "type": "system"
                            })
                        # Stop the conversation
                        conv_data["status"] = "completed"
                        return
                    else:
                        print("DEBUG: Round_robin termination condition not yet met")
            
            next_agent_name = conv_data['agent_names'][conv_data['current_agent_index']]
            print(f"DEBUG: Next agent will be '{next_agent_name}'")
            
            # Handle audio synchronization if voices are enabled
            if conv_data.get("voices_enabled", False):
                # Store the next agent to be scheduled
                conv_data["next_agent_scheduled"] = next_agent_name
                
                # Check if we need to wait for audio
                if self._should_wait_for_audio(conversation_id):
                    print(f"DEBUG: Voices enabled - waiting for audio before invoking next agent '{next_agent_name}'")
                    # Agent will be invoked when audio completes via callbacks
                    return
                else:
                    # No audio waiting needed, invoke immediately
                    print(f"DEBUG: Voices enabled - no audio waiting needed, invoking next agent '{next_agent_name}' immediately")
                    conv_data["next_agent_scheduled"] = None
                    self._invoke_next_agent(conversation_id)
            else:
                # Original behavior for non-voice mode
                delay = random.uniform(
                    CONVERSATION_TIMING["agent_turn_delay_min"], 
                    CONVERSATION_TIMING["agent_turn_delay_max"]
                )
                print(f"DEBUG: Scheduling next agent response in {delay:.2f} seconds")
                threading.Timer(delay, self._invoke_next_agent, args=(conversation_id,)).start()
                
        except Exception as e:
            print(f"ERROR: Error invoking agent '{current_agent_name}': {e}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            import traceback
            print(f"ERROR: Traceback: {traceback.format_exc()}")
            # Retry after error delay
            if conv_data["status"] == "active":
                print(f"DEBUG: Retrying after {CONVERSATION_TIMING['error_retry_delay']} seconds")
                threading.Timer(CONVERSATION_TIMING["error_retry_delay"], self._invoke_next_agent, args=(conversation_id,)).start()

    def send_message(self, conversation_id: str, message: str, sender: str = "user") -> Dict[str, Any]:
        """Send a user message to interrupt the conversation."""
        if conversation_id not in self.active_conversations:
            return {"success": False, "error": "Conversation not found"}
        
        conv_data = self.active_conversations[conversation_id]
        
        # Add user message to messages list
        conv_data["messages"].append({
            "agent_name": "User",
            "message": message
        })
          # If conversation is active, the next agent will respond to this message
        return {
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def pause_conversation(self, conversation_id: str):
        """Pause a conversation and clean up pending messages and audio."""
        if conversation_id in self.active_conversations:
            conv_data = self.active_conversations[conversation_id]
            conv_data["status"] = "paused"
            
            # Get currently playing audio info to preserve it
            currently_playing_info = None
            if conv_data.get("voices_enabled", False):
                # Check if audio is currently playing
                currently_playing_info = {
                    'message_id': conv_data.get("last_agent_audio_sent"),
                    'waiting_for_audio': conv_data.get("waiting_for_audio_generation", False),
                    'audio_playing': conv_data.get("waiting_for_audio_playback", False)
                }
            
            # Prepare cleanup information for the UI to handle
            cleanup_info = {
                'conversation_id': conversation_id,
                'action': 'paused',
                'voices_enabled': conv_data.get("voices_enabled", False),
                'currently_playing': currently_playing_info,
                'messages_count': len(conv_data.get("messages", []))
            }
            
            # Call cleanup callback if set (main.py will handle the actual cleanup)
            if hasattr(self, 'pause_cleanup_callback') and self.pause_cleanup_callback:
                self.pause_cleanup_callback(cleanup_info)
    
    def set_pause_cleanup_callback(self, callback):
        """Set callback for pause cleanup operations."""
        self.pause_cleanup_callback = callback
    
    def resume_conversation(self, conversation_id: str):
        """Resume a paused conversation."""
        if conversation_id in self.active_conversations:
            print(f"DEBUG: ===== RESUMING CONVERSATION {conversation_id} =====")
            conv_data = self.active_conversations[conversation_id]
            print(f"DEBUG: Conversation has {len(conv_data.get('messages', []))} messages")
            print(f"DEBUG: Current agent index: {conv_data.get('current_agent_index', 'Unknown')}")
            if conv_data.get('agent_names'):
                current_agent = conv_data['agent_names'][conv_data.get('current_agent_index', 0)]
                print(f"DEBUG: Next agent to speak: {current_agent}")
            print(f"DEBUG: ===== END RESUMING INFO =====")
            
            self.active_conversations[conversation_id]["status"] = "active"
            # Resume the conversation cycle
            threading.Timer(CONVERSATION_TIMING["resume_delay"], self._invoke_next_agent, args=(conversation_id,)).start()
        else:
            print(f"DEBUG: Cannot resume conversation {conversation_id} - not found in active conversations")
    
    def stop_conversation(self, conversation_id: str):
        """Stop and remove a conversation from active sessions."""
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]["status"] = "stopped"
            del self.active_conversations[conversation_id]
        if conversation_id in self.message_callbacks:
            del self.message_callbacks[conversation_id]
    
    def get_conversation_summary(self, conversation_id: str) -> str:
        """Generate a summary of the current conversation."""
        if conversation_id not in self.active_conversations:
            return "Conversation not found"
        
        conv_data = self.active_conversations[conversation_id]
        messages = conv_data["messages"]
        
        if not messages:
            return "No messages in conversation yet."
        
        # Use our summarization function
        if len(messages) > 5:  # Only summarize if there are enough messages
            # Create a temporary longer message list to force summarization
            temp_messages = messages + [{"agent_name": "temp", "message": "temp"}] * 25
            summarized = self.message_list_summarization(temp_messages)
            if summarized and "past_convo_summary" in summarized[0]:
                return summarized[0]["past_convo_summary"]
        
        # Fallback: create simple summary
        summary = f"Conversation between {', '.join(conv_data['agent_names'])} "
        summary += f"in {conv_data['environment']}. "
        summary += f"{len(messages)} messages exchanged so far."
        return summary
    
    def register_message_callback(self, conversation_id: str, callback: Callable):
        """Register a callback function for real-time message updates."""
        self.message_callbacks[conversation_id] = callback
    
    def change_scene(self, conversation_id: str, new_environment: str, new_scene_description: str):
        """Change the scene/environment for an active conversation."""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conv_data = self.active_conversations[conversation_id]
        
        # Update environment data
        conv_data["environment"] = new_environment
        conv_data["scene_description"] = new_scene_description
        
        # Add scene change to messages
        conv_data["messages"].append({
            "agent_name": "System",
            "message": f"Scene changed to: {new_environment}. {new_scene_description}"
        })
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_round_robin_termination(self, conversation_id: str) -> bool:
        """
        Check if the conversation should terminate for Round Robin mode.
        
        Args:
            conversation_id: The conversation identifier
            
        Returns:
            True if conversation should terminate, False otherwise
        """
        conv_data = self.active_conversations[conversation_id]
        termination_condition = conv_data.get("termination_condition")
        
        # If no termination condition provided, don't terminate
        if not termination_condition:
            return False
        
        messages = conv_data["messages"]
        environment = conv_data["environment"]
        scene_description = conv_data["scene_description"]
        
        # Get messages from the last three rounds
        # Assuming a round is when all agents have spoken once
        num_agents = len(conv_data["agent_names"])
        last_three_rounds_count = num_agents * 3
        
        if len(messages) < last_three_rounds_count:
            # Not enough messages for three rounds, don't terminate yet
            return False
            
        last_three_rounds_messages = messages[-last_three_rounds_count:]
        
        # Format messages for the prompt
        formatted_messages = []
        for msg in last_three_rounds_messages:
            if "agent_name" in msg and "message" in msg:
                formatted_messages.append(f"{msg['agent_name']}: {msg['message']}")
        messages_str = "\n".join(formatted_messages)
        
        # Create termination prompt
        prompt = f"""You are evaluating whether a conversation should terminate based on a specific condition.

MESSAGES FROM THE LAST THREE ROUNDS:
{messages_str}

CURRENT SCENE: {environment}
ENVIRONMENT: {scene_description}
TERMINATION CONDITION: {termination_condition}

Based on the conversation messages and the termination condition, determine if the conversation can terminate now.

Respond with ONLY a JSON object in this exact format:
{{ "termination_decision": true }} or {{ "termination_decision": false }}

Do not include any other text or explanation."""
        
        try:
            # Call LLM to check termination using the summary model
            response = self.summary_model.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            # Parse the JSON response
            termination_result = self._extract_termination_json(response_text)
            return termination_result.get("termination_decision", False)
            
        except Exception as e:
            print(f"Error checking termination condition: {e}")
            return False
    
    def _extract_termination_json(self, text: str) -> Dict[str, Any]:
        """Extract termination JSON from the response text, handling different formats."""
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
            return {"termination_decision": False}
    
    def _save_conversation_state(self, conversation_id: str):
        """
        Save the current conversation state to persistent storage.
        This updates the conversation in the database with current messages and agent_sending_messages.
        """
        if conversation_id not in self.active_conversations:
            return
        
        try:
            # Import DataManager if not already imported
            if not hasattr(self, 'data_manager'):
                from data_manager import DataManager
                self.data_manager = DataManager(os.path.dirname(__file__))
            
            # Get the conversation from database
            conversation = self.data_manager.get_conversation_by_id(conversation_id)
            if not conversation:
                print(f"WARNING: Conversation {conversation_id} not found in database")
                return
            
            conv_data = self.active_conversations[conversation_id]
            
            # Update conversation with current state
            conversation.messages = self._convert_messages_for_storage(conv_data["messages"])
            conversation.last_updated = datetime.now().isoformat()
            conversation.status = conv_data.get("status", "active")
            
            # Save agent_sending_messages if they exist
            if "agent_sending_messages" in conv_data:
                conversation.agent_sending_messages = conv_data["agent_sending_messages"]
                print(f"DEBUG: Saved agent_sending_messages for {len(conv_data['agent_sending_messages'])} agents")
            
            # Save agent_temp_numbers if they exist
            if "agent_temp_numbers" in conv_data:
                conversation.agent_temp_numbers = conv_data["agent_temp_numbers"]
                print(f"DEBUG: Saved agent_temp_numbers for {len(conv_data['agent_temp_numbers'])} agents")
            
            # Save to database
            self.data_manager.save_conversation(conversation)
            print(f"DEBUG: Saved conversation state for {conversation_id}")
            
        except Exception as e:
            print(f"ERROR: Failed to save conversation state for {conversation_id}: {e}")
    
    def _convert_messages_for_storage(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convert internal message format to storage format.
        """
        storage_messages = []
        for msg in messages:
            if "agent_name" in msg and "message" in msg:
                storage_messages.append({
                    "sender": msg["agent_name"],
                    "content": msg["message"],
                    "timestamp": datetime.now().isoformat(),
                    "type": "ai"
                })
        return storage_messages
    
    def restart_conversation_with_new_condition(self, conversation_id: str, new_termination_condition: str):
        """
        Restart a completed conversation with a new termination condition.
        This preserves the conversation history but allows it to continue.
        """
        print(f"DEBUG: ===== RESTARTING CONVERSATION {conversation_id} =====")
        print(f"DEBUG: New termination condition: {new_termination_condition}")
        
        # Check if conversation exists in active conversations
        if conversation_id not in self.active_conversations:
            # Try to load from database if not in active conversations
            try:
                from data_manager import DataManager
                data_manager = DataManager(os.path.dirname(__file__))
                conversation = data_manager.get_conversation_by_id(conversation_id)
                
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} not found")
                
                # Get agent objects
                all_agents = data_manager.load_agents()
                conversation_agents = []
                for agent_id in conversation.agents:
                    agent = next((a for a in all_agents if a.id == agent_id), None)
                    if agent:
                        conversation_agents.append(agent)
                
                if len(conversation_agents) < 2:
                    raise ValueError("Not enough agents found to restart conversation")
                
                # Restore conversation to active conversations
                self.active_conversations[conversation_id] = {
                    "agents_config": [
                        {
                            "id": agent.id,
                            "name": agent.name,
                            "role": agent.role,
                            "base_prompt": agent.base_prompt,
                            "api_key": agent.api_key
                        } for agent in conversation_agents
                    ],
                    "agent_names": [agent.name for agent in conversation_agents],
                    "thread_id": f"thread_{conversation_id}",
                    "environment": conversation.environment,
                    "scene_description": conversation.scene_description,
                    "status": "active",
                    "messages": self._convert_messages_from_storage(conversation.messages),
                    "current_agent_index": 0,  # Will be set properly below
                    "conversation_started": True,
                    "invocation_method": getattr(conversation, 'invocation_method', 'round_robin'),
                    "termination_condition": new_termination_condition,  # New condition
                    "agent_invocation_counts": {agent.name: 0 for agent in conversation_agents},
                    "agent_selector_api_key": getattr(conversation, 'agent_selector_api_key', None),
                }
                
                # Restore agent_sending_messages if they exist
                if hasattr(conversation, 'agent_sending_messages') and conversation.agent_sending_messages:
                    self.active_conversations[conversation_id]["agent_sending_messages"] = conversation.agent_sending_messages
                    print(f"DEBUG: Restored agent_sending_messages for {len(conversation.agent_sending_messages)} agents")
                
                # Restore agent_temp_numbers if they exist, otherwise create them
                if hasattr(conversation, 'agent_temp_numbers') and conversation.agent_temp_numbers:
                    self.active_conversations[conversation_id]["agent_temp_numbers"] = conversation.agent_temp_numbers
                    print(f"DEBUG: Restored agent_temp_numbers for {len(conversation.agent_temp_numbers)} agents")
                else:
                    # Create agent temp numbers for backward compatibility
                    agent_temp_numbers = {}
                    for i, agent in enumerate(conversation_agents, 1):
                        agent_temp_numbers[agent.id] = i
                    self.active_conversations[conversation_id]["agent_temp_numbers"] = agent_temp_numbers
                    print(f"DEBUG: Created agent_temp_numbers for backward compatibility: {len(agent_temp_numbers)} agents")
                
                print(f"DEBUG: Restored conversation from database with {len(conversation.messages)} messages")
                
            except Exception as e:
                print(f"ERROR: Failed to restore conversation {conversation_id}: {e}")
                return False
        
        conv_data = self.active_conversations[conversation_id]
        
        # Update the termination condition
        conv_data["termination_condition"] = new_termination_condition
        conv_data["status"] = "active"
        
        # Reset agent invocation counts for fair restart
        for agent_name in conv_data["agent_names"]:
            conv_data["agent_invocation_counts"][agent_name] = 0
        
        # Choose the next agent to continue (round-robin from current position or random)
        if conv_data["invocation_method"] == "round_robin":
            # Continue from where we left off, or start with first agent
            conv_data["current_agent_index"] = conv_data.get("current_agent_index", 0)
        else:
            # For agent_selector, start with a random agent
            conv_data["current_agent_index"] = random.randint(0, len(conv_data["agent_names"]) - 1)
        
        print(f"DEBUG: Conversation restarted. Next agent: {conv_data['agent_names'][conv_data['current_agent_index']]}")
        print(f"DEBUG: ===== END RESTART INFO =====")
        
        # Save the updated state
        self._save_conversation_state(conversation_id)
        
        # Start the conversation cycle with a short delay
        threading.Timer(1.0, self._invoke_next_agent, args=(conversation_id,)).start()
        
        return True

    def _convert_messages_from_storage(self, storage_messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Convert storage message format to internal format.
        """
        internal_messages = []
        for msg in storage_messages:
            if "sender" in msg and "content" in msg:
                internal_messages.append({
                    "agent_name": msg["sender"],
                    "message": msg["content"]
                })
        return internal_messages

    def _update_agent_sending_messages(self, conversation_id: str, agent_name: str):
        """
        Update the agent-specific message list with summarization when needed.
        
        This method:
        1. Maintains per-agent message lists in agent_sending_messages
        2. Adds new global messages to each agent's list
        3. Applies summarization when an agent's message count exceeds threshold
        4. Updates the agent's context with summary + recent messages
        
        Args:
            conversation_id: The conversation identifier
            agent_name: The agent whose context we're updating
        """
        if conversation_id not in self.active_conversations:
            return
        
        conv_data = self.active_conversations[conversation_id]
        
        # Initialize agent_sending_messages if not exists (for backward compatibility)
        if "agent_sending_messages" not in conv_data:
            conv_data["agent_sending_messages"] = {name: [] for name in conv_data["agent_names"]}
            print(f"DEBUG: Initialized agent_sending_messages for backward compatibility")
        
        # Initialize this agent's message list if not exists
        if agent_name not in conv_data["agent_sending_messages"]:
            conv_data["agent_sending_messages"][agent_name] = []
            print(f"DEBUG: Initialized agent_sending_messages for '{agent_name}'")
        
        agent_messages = conv_data["agent_sending_messages"][agent_name]
        global_messages = conv_data["messages"]
        
        # Find new messages that haven't been added to this agent's list yet
        # We'll compare based on message content and sender to avoid duplicates
        existing_message_signatures = set()
        for msg in agent_messages:
            if "agent_name" in msg and "message" in msg:
                signature = f"{msg['agent_name']}:{msg['message']}"
                existing_message_signatures.add(signature)
        
        new_messages = []
        for msg in global_messages:
            if "agent_name" in msg and "message" in msg:
                signature = f"{msg['agent_name']}:{msg['message']}"
                if signature not in existing_message_signatures:
                    new_messages.append(msg)
        
        # Add new messages to agent's list
        if new_messages:
            agent_messages.extend(new_messages)
            print(f"DEBUG: Added {len(new_messages)} new messages to agent '{agent_name}' (total: {len(agent_messages)})")
        
        # Check if summarization is needed for this agent
        max_messages = MESSAGE_SETTINGS["max_messages_before_summary"]
        messages_to_keep = MESSAGE_SETTINGS["messages_to_keep_after_summary"]
        
        if len(agent_messages) > max_messages:
            print(f"DEBUG: Agent '{agent_name}' message count ({len(agent_messages)}) exceeds threshold ({max_messages}), applying summarization")
            
            # Apply summarization to this agent's messages
            summarized_messages = self.message_list_summarization(agent_messages, max_messages)
            
            # Update the agent's message list
            conv_data["agent_sending_messages"][agent_name] = summarized_messages
            
            print(f"DEBUG: Agent '{agent_name}' messages summarized from {len(agent_messages)} to {len(summarized_messages)}")
        else:
            print(f"DEBUG: Agent '{agent_name}' message count ({len(agent_messages)}) below threshold ({max_messages}), no summarization needed")
    
    def _get_agent_context_messages(self, conversation_id: str, agent_name: str) -> List[Dict[str, str]]:
        """Get context messages for a specific agent."""
        conv_data = self.active_conversations[conversation_id]
        
        # Apply per-agent message summarization before getting context
        self._update_agent_sending_messages(conversation_id, agent_name)
        
        # Return the agent's specific message list
        return conv_data["agent_sending_messages"].get(agent_name, [])
    
    def _create_human_like_chat_prompt(self, conversation_id: str, agent_config: Dict[str, str], 
                                     agent_messages: List[Dict[str, str]], should_remind_termination: bool) -> str:
        """Create a prompt for human-like-chat mode."""
        conv_data = self.active_conversations[conversation_id]
        agent_name = agent_config["name"]
        
        # Load agent tools info
        agent_tools = self._load_agent_tools(agent_name, agent_config)
        tool_names = [getattr(tool, 'name', str(tool)) for tool in agent_tools]
        
        # Create the prompt using the existing create_agent_prompt method
        return self.create_agent_prompt(
            agent_config,
            conv_data["environment"],
            conv_data["scene_description"],
            agent_messages,
            conv_data["agent_names"],
            conv_data.get("termination_condition"),
            should_remind_termination,
            conversation_id,
            agent_name,
            tool_names
        )
    
    def _load_agent_tools(self, agent_name: str, agent_config: Dict[str, str]) -> List[Any]:
        """Load tools for a specific agent."""
        agent_tools = []
        
        try:
            import tools as tools_module
            
            # Get agent ID and object
            agent_id = agent_config.get("id")
            agent_obj = None
            
            if agent_id and hasattr(self, 'agents_data'):
                agent_obj = self.agents_data.get(agent_id)
            
            # Load agent-specific tools
            if agent_obj and hasattr(agent_obj, 'tools') and agent_obj.tools:
                for tool_name in agent_obj.tools:
                    try:
                        if tool_name == "knowledge_base_retriever":
                            # Handle knowledge base retriever
                            if hasattr(agent_obj, 'knowledge_base') and agent_obj.knowledge_base:
                                if hasattr(tools_module, "knowledge_base_retriever"):
                                    retriever_tool_func = getattr(tools_module, "knowledge_base_retriever").func
                                    partial_func = partial(retriever_tool_func, agent_id=agent_id)
                                    
                                    new_tool = Tool(
                                        name="knowledge_base_retriever",
                                        func=partial_func,
                                        description="Retrieves the top 3 most relevant results from your personal knowledge base to help answer questions. Input should be a search query."
                                    )
                                    agent_tools.append(new_tool)
                        elif hasattr(tools_module, tool_name):
                            tool_obj = getattr(tools_module, tool_name)
                            agent_tools.append(tool_obj)
                    except Exception as e:
                        print(f"WARNING: Could not load tool '{tool_name}' for agent {agent_name}: {e}")
        except ImportError:
            print(f"WARNING: Could not import tools module for agent {agent_name}")
        
        return agent_tools
    
    def _check_human_like_chat_termination(self, conversation_id: str) -> bool:
        """Check if human-like-chat conversation should terminate."""
        conv_data = self.active_conversations[conversation_id]
        termination_condition = conv_data.get("termination_condition")
        
        if not termination_condition:
            return False
        
        # Use the same termination logic as round_robin
        return self._check_round_robin_termination(conversation_id)
    
    def _start_human_like_chat(self, conversation_id: str):
        """Start human-like-chat mode by using agent selector to choose the first agent."""
        if conversation_id not in self.active_conversations:
            return
        
        conv_data = self.active_conversations[conversation_id]
        if conv_data["status"] != "active":
            return
        
        print("DEBUG: Using agent selector to choose initial agent for human-like-chat")
        
        # Use GOOGLE_API_KEY from environment for agent selector
        env_api_key = os.getenv("GOOGLE_API_KEY")
        if not env_api_key:
            print("ERROR: GOOGLE_API_KEY not found in environment for human-like-chat agent selector")
            # Fallback to regular round robin
            conv_data["current_agent_index"] = random.randint(0, len(conv_data["agent_names"]) - 1)
            self._invoke_next_agent_regular(conversation_id)
            return
        
        # Create agent selector
        agent_selector = AgentSelector(google_api_key=env_api_key)
        
        # Use agent selector to choose the first agent
        selection_result = agent_selector.select_next_agent(
            messages=[],  # No messages yet
            environment=conv_data["environment"],
            scene=conv_data["scene_description"],
            agents=conv_data["agents_config"],
            termination_condition=conv_data.get("termination_condition"),
            agent_invocation_counts=conv_data.get("agent_invocation_counts", {})
        )
        
        first_agent = selection_result.get("next_response", "error_parsing")
        print(f"DEBUG: Agent selector chose initial agent: {first_agent}")
        
        if first_agent == "error_parsing" or first_agent not in conv_data["agent_names"]:
            print("DEBUG: Agent selector error, choosing random first agent")
            conv_data["current_agent_index"] = random.randint(0, len(conv_data["agent_names"]) - 1)
        else:
            # Find the index of the selected agent
            for i, name in enumerate(conv_data["agent_names"]):
                if name == first_agent:
                    conv_data["current_agent_index"] = i
                    break
        
        print(f"DEBUG: Starting human-like-chat with agent: {conv_data['agent_names'][conv_data['current_agent_index']]}")
        
        # Invoke the selected first agent
        self._invoke_next_agent_regular(conversation_id)
    
    def _start_conversation_cycle(self, conversation_id: str):
        """Start the conversation cycle."""
        if conversation_id not in self.active_conversations:
            return
        
        conv_data = self.active_conversations[conversation_id]
        invocation_method = conv_data.get("invocation_method", "round_robin")
        
        print(f"DEBUG: Starting conversation cycle with method: {invocation_method}")
        
        if invocation_method == "human_like_chat":
            self._start_human_like_chat(conversation_id)
        else:
            self._invoke_next_agent(conversation_id)