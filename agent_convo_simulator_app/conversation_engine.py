"""
Simple Multi-Agent Conversation Engine
Real LangGraph react agents with Gemini in conversation cycle.
"""

import os
import random
import json
import re
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import threading
import time

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import configuration settings
from config import CONVERSATION_TIMING, MESSAGE_SETTINGS, AGENT_SETTINGS, MODEL_SETTINGS
from agent_selector import AgentSelector


class ConversationSimulatorEngine:
    """
    Multi-agent conversation simulator using real LangGraph react agents with Gemini.
    Agents take turns in a cycle, maintaining conversation history with summarization.    """
    def __init__(self, google_api_key: Optional[str] = None):
        """Initialize the conversation simulator engine."""
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")
        
        self.model = ChatGoogleGenerativeAI(
            model=MODEL_SETTINGS["agent_model"],
            temperature=AGENT_SETTINGS["response_temperature"],
            max_retries=AGENT_SETTINGS["max_retries"],
            google_api_key=self.google_api_key
        )
        
        # Separate model for summarization
        self.summary_model = ChatGoogleGenerativeAI(
            model=MODEL_SETTINGS["summary_model"],
            temperature=AGENT_SETTINGS["summary_temperature"],  # Lower temperature for more consistent summaries
            max_retries=AGENT_SETTINGS["max_retries"],
            google_api_key=self.google_api_key
        )
        
        # Initialize agent selector for dynamic agent selection
        self.agent_selector = AgentSelector(google_api_key=self.google_api_key)
        
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
            summary_prompt = f"""Previous conversation summary: {existing_summary}

Recent conversation messages:
"""
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
                          should_remind_termination: bool = False) -> str:
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
            
        Returns:
            Formatted prompt string
        """
        agent_name = agent_config["name"]
        agent_role = agent_config["role"]
        base_prompt = agent_config["base_prompt"]
        
        prompt = f"""You are {agent_name}, a {agent_role}.

{base_prompt}

INITIAL SCENE: {environment}
SCENE DESCRIPTION: {scene_description}

PARTICIPANTS: {', '.join(all_agents)}

"""
          # Add conversation history
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

Keep this condition in mind while participating in the conversation.

"""
        
        prompt += f"""Give your response to the ongoing conversation as {agent_name}. 
Keep your response natural, conversational, and true to your character. 
Respond as if you're speaking directly in the conversation (don't say "As {agent_name}, I would say..." just respond naturally).
Keep responses to 1-3 sentences to maintain good conversation flow."""
        return prompt
    
    def start_conversation(self, conversation_id: str, agents_config: List[Dict[str, str]], 
                        environment: str, scene_description: str, initial_message: str = None,
                        invocation_method: str = "round_robin", termination_condition: Optional[str] = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            conversation_id: Unique identifier for the conversation
            agents_config: List of agent configurations (name, role, base_prompt)
            environment: Environment description for the conversation
            scene_description: Scene description for the conversation
            initial_message: Optional initial message to start the conversation
            invocation_method: Method for selecting which agent speaks next ("round_robin" or "agent_selector")
            termination_condition: Optional condition for when to end the conversation. 
                                  When provided with "round_robin" method, an LLM will evaluate
                                  after each round if the condition is met. When provided with 
                                  "agent_selector" method, the selector LLM decides termination.
                                  If not provided, conversation continues indefinitely.
            
        Returns:
            Thread ID for the conversation
        """
        thread_id = f"thread_{conversation_id}"
        
        # Create individual react agents
        agents = {}
        agent_names = [config["name"] for config in agents_config]
        
        for agent_config in agents_config:
            agent_name = agent_config["name"]
            
            # Create agent with minimal prompt (detailed prompt will be provided per invocation)
            agent = create_react_agent(
                model=self.model,
                tools=[],  # No tools needed for conversation
                prompt=f"You are {agent_name}. Respond naturally to conversations.",
                checkpointer=self.memory
            )
            
            agents[agent_name] = agent
          # Store conversation data
        self.active_conversations[conversation_id] = {
            "agents": agents,
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
            "agent_invocation_counts": {name: 0 for name in agent_names}  # Track invocations per agent
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
        
        # Choose random first agent
        conv_data["current_agent_index"] = random.randint(0, len(conv_data["agent_names"]) - 1)
        
        # Start the conversation
        self._invoke_next_agent(conversation_id)
    
    def _invoke_next_agent(self, conversation_id: str):
        """Invoke the next agent in the cycle."""
        if conversation_id not in self.active_conversations:
            return
        
        conv_data = self.active_conversations[conversation_id]
        if conv_data["status"] != "active":
            return
          # Get current agent
        current_agent_name = conv_data["agent_names"][conv_data["current_agent_index"]]
        current_agent_config = next(config for config in conv_data["agents_config"] 
                                  if config["name"] == current_agent_name)
        
        # Increment invocation count for current agent
        conv_data["agent_invocation_counts"][current_agent_name] += 1
        
        # Determine if we should remind about termination condition
        termination_condition = conv_data.get("termination_condition")
        should_remind_termination = False
        
        if termination_condition:
            reminder_frequency = AGENT_SETTINGS["termination_reminder_frequency"]
            current_count = conv_data["agent_invocation_counts"][current_agent_name]
            should_remind_termination = (current_count % reminder_frequency == 0)
        
        # Apply message summarization before creating prompt
        conv_data["messages"] = self.message_list_summarization(conv_data["messages"])
        
        # Create prompt for current agent
        prompt = self.create_agent_prompt(
            current_agent_config,
            conv_data["environment"],
            conv_data["scene_description"],
            conv_data["messages"],
            conv_data["agent_names"],
            termination_condition,
            should_remind_termination
        )
        
        try:
            # Invoke the agent
            agent = conv_data["agents"][current_agent_name]
            config = {"configurable": {"thread_id": f"{conv_data['thread_id']}_{current_agent_name}"}}
            
            response = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config
            )
            
            # Extract the response
            if response and "messages" in response and response["messages"]:
                agent_message = response["messages"][-1].content
                
                # Add to our messages list
                conv_data["messages"].append({
                    "agent_name": current_agent_name,
                    "message": agent_message
                })                # Notify callback
                if conversation_id in self.message_callbacks:
                    self.message_callbacks[conversation_id]({
                        "sender": current_agent_name,
                        "content": agent_message,
                        "timestamp": datetime.now().isoformat(),
                        "type": "ai"
                    })                
                # Determine the next agent based on invocation method
                invocation_method = conv_data.get("invocation_method", "round_robin")
                termination_condition = conv_data.get("termination_condition", None)
                
                if invocation_method == "agent_selector":
                    # Use the agent selector to determine the next agent
                    selection_result = self.agent_selector.select_next_agent(
                        messages=conv_data["messages"],
                        environment=conv_data["environment"],
                        scene=conv_data["scene_description"],
                        agents=conv_data["agents_config"],
                        termination_condition=termination_condition,
                        agent_invocation_counts=conv_data.get("agent_invocation_counts", {})
                    )
                    
                    next_response = selection_result.get("next_response", "error_parsing")
                    
                    if next_response == "terminate":
                        # End the conversation
                        if conversation_id in self.message_callbacks:
                            self.message_callbacks[conversation_id]({
                                "sender": "System",
                                "content": "The conversation has reached its termination condition and has ended.",
                                "timestamp": datetime.now().isoformat(),
                                "type": "system"
                            })                        # Stop the conversation
                        conv_data["status"] = "completed"
                        return
                    elif next_response == "error_parsing":
                        # Error in parsing, use round robin as fallback
                        conv_data["current_agent_index"] = (conv_data["current_agent_index"] + 1) % len(conv_data["agent_names"])
                    else:
                        # Set the next agent
                        if next_response in conv_data["agent_names"]:
                            # Find the index of the selected agent
                            for i, name in enumerate(conv_data["agent_names"]):
                                if name == next_response:
                                    conv_data["current_agent_index"] = i
                                    break
                        else:
                            # If agent name not found, use round robin as fallback
                            conv_data["current_agent_index"] = (conv_data["current_agent_index"] + 1) % len(conv_data["agent_names"])
                else:
                    # Round robin method - move to next agent in cycle
                    conv_data["current_agent_index"] = (conv_data["current_agent_index"] + 1) % len(conv_data["agent_names"])
                    
                    # Check if we completed a round (all agents have spoken)
                    # and if there's a termination condition to check
                    if (conv_data["current_agent_index"] == 0 and 
                        termination_condition and 
                        len(conv_data["messages"]) >= len(conv_data["agent_names"])):
                        
                        # Check if conversation should terminate
                        if self._check_round_robin_termination(conversation_id):
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
                
                # Schedule next agent response with configurable delay
                delay = random.uniform(
                    CONVERSATION_TIMING["agent_turn_delay_min"], 
                    CONVERSATION_TIMING["agent_turn_delay_max"]
                )
                threading.Timer(delay, self._invoke_next_agent, args=(conversation_id,)).start()
                
        except Exception as e:
            print(f"Error invoking agent: {e}")
            # Retry after error delay
            if conv_data["status"] == "active":
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
        """Pause a conversation."""
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]["status"] = "paused"
    
    def resume_conversation(self, conversation_id: str):
        """Resume a paused conversation."""
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]["status"] = "active"
            # Resume the conversation cycle
            threading.Timer(CONVERSATION_TIMING["resume_delay"], self._invoke_next_agent, args=(conversation_id,)).start()
    
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
            # Call LLM to check termination
            response = self.model.invoke([HumanMessage(content=prompt)])
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
