"""
Simplified Multi-Agent Conversation Engine without Supervisor
Direct agent-to-agent interaction using LangGraph react agents with Gemini.
"""

import os
import sys
import random
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import threading
import time

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class SimpleConversationEngine:
    """
    Simplified multi-agent conversation simulator without supervisor.
    Agents take turns based on simple round-robin or random selection.
    """
    
    def __init__(self, google_api_key: Optional[str] = None):
        """Initialize the conversation simulator engine."""
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")
        
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.7,
            max_retries=2,
            google_api_key=self.google_api_key
        )
        
        self.memory = MemorySaver()
        self.max_messages = 10
        
        # Storage for active conversations
        self.active_conversations = {}
        self.message_callbacks = {}  # For real-time message updates
    
    def _create_trimming_hook(self):
        """Create a simple trimming hook to manage message history."""
        def trimming_hook(state):
            """Trim messages if they get too long."""
            messages = state.get("messages", [])
            
            if len(messages) > self.max_messages:
                # Keep first message (usually system) and last max_messages-1
                trimmed = [messages[0]] + messages[-(self.max_messages-1):]
                return {"messages": trimmed}
            
            return state
        
        return trimming_hook
    
    def create_agents(self, agents_config: List[Dict[str, str]], environment: str, scene_description: str):
        """
        Create individual agents for direct interaction.
        
        Args:
            agents_config: List of agent configurations with name, role, and prompt
            environment: The environment/setting for the conversation
            scene_description: Detailed scene description
        """
        
        agents = {}
        
        for agent_config in agents_config:
            agent_name = agent_config["name"]
            
            # Create context-aware system prompt
            system_prompt = f"""You are {agent_name}, a {agent_config["role"]}.

{agent_config["base_prompt"]}

CURRENT SETTING:
Environment: {environment}
Scene: {scene_description}

CONVERSATION GUIDELINES:
- Stay in character as {agent_name}
- Keep responses natural and conversational (1-3 sentences)
- Reference the environment when appropriate
- Engage authentically with other participants
- Don't repeat what others have just said
- Be responsive to the conversation flow
- Show your personality through your responses

You are participating in a group conversation. Respond naturally as {agent_name} would in this situation."""
            
            # Create agent with environment-aware prompt
            agent = create_react_agent(
                model=self.model,
                tools=[],  # No tools needed for conversation
                prompt=system_prompt,
                checkpointer=self.memory,
                pre_model_hook=self._create_trimming_hook()
            )
            
            agents[agent_name] = agent
        
        return agents
    
    def start_conversation(self, conversation_id: str, agents_config: List[Dict[str, str]], 
                          environment: str, scene_description: str, initial_message: str = None) -> str:
        """
        Start a new conversation session.
        
        Returns the thread_id for the conversation.
        """
        thread_id = f"thread_{conversation_id}"
        
        # Create agents
        agents = self.create_agents(agents_config, environment, scene_description)
        
        # Store in active conversations
        self.active_conversations[conversation_id] = {
            "agents": agents,
            "agent_names": list(agents.keys()),
            "thread_id": thread_id,
            "environment": environment,
            "scene_description": scene_description,
            "agents_config": agents_config,
            "status": "active",
            "turn_index": 0,  # Track whose turn it is
            "conversation_started": False
        }
        
        # Start with initial message if provided
        if initial_message:
            self.send_message(conversation_id, initial_message, "system")
        else:
            # Start with a scene-setting message
            default_message = f"Welcome to {environment}. {scene_description} Let's begin our conversation!"
            self.send_message(conversation_id, default_message, "system")
        
        return thread_id
    
    def send_message(self, conversation_id: str, message: str, sender: str = "user") -> Dict[str, Any]:
        """Send a message and trigger next agent response."""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found or not active")
        
        conv_data = self.active_conversations[conversation_id]
        
        if conv_data["status"] != "active":
            return {"success": False, "error": "Conversation is not active"}
        
        thread_id = conv_data["thread_id"]
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Add the message to conversation
            if sender == "user":
                # User message - add to history and then get agent response
                user_message = HumanMessage(content=f"[User]: {message}")
                self._add_message_to_history(conversation_id, user_message)
            elif sender == "system":
                # System message - add to history and start agent conversation
                system_message = SystemMessage(content=message)
                self._add_message_to_history(conversation_id, system_message)
                conv_data["conversation_started"] = True
            
            # Get next agent response
            if conv_data["conversation_started"]:
                return self._get_next_agent_response(conversation_id)
            
            return {"success": True, "message": "Message added to conversation"}
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def _add_message_to_history(self, conversation_id: str, message):
        """Add a message to the conversation history."""
        conv_data = self.active_conversations[conversation_id]
        
        # Initialize message history if not exists
        if "message_history" not in conv_data:
            conv_data["message_history"] = []
        
        conv_data["message_history"].append(message)
    
    def _get_next_agent_response(self, conversation_id: str) -> Dict[str, Any]:
        """Get response from the next agent in sequence."""
        conv_data = self.active_conversations[conversation_id]
        agents = conv_data["agents"]
        agent_names = conv_data["agent_names"]
        
        # Select next agent (round-robin with some randomness)
        if len(agent_names) > 2:
            # For 3+ agents, add some randomness but avoid the same agent twice in a row
            possible_agents = agent_names.copy()
            if conv_data["turn_index"] > 0:
                last_agent_idx = (conv_data["turn_index"] - 1) % len(agent_names)
                last_agent = agent_names[last_agent_idx]
                if last_agent in possible_agents:
                    possible_agents.remove(last_agent)
            
            next_agent_name = random.choice(possible_agents)
            conv_data["turn_index"] = agent_names.index(next_agent_name) + 1
        else:
            # For 2 agents, simple alternation
            agent_idx = conv_data["turn_index"] % len(agent_names)
            next_agent_name = agent_names[agent_idx]
            conv_data["turn_index"] += 1
        
        next_agent = agents[next_agent_name]
        
        try:
            # Prepare message history for the agent
            message_history = conv_data.get("message_history", [])
            
            # Run the agent
            thread_id = conv_data["thread_id"]
            config = {"configurable": {"thread_id": thread_id}}
            
            response = next_agent.invoke(
                {"messages": message_history},
                config
            )
            
            # Extract the agent's response
            if "messages" in response and response["messages"]:
                agent_message = response["messages"][-1]
                
                # Add agent's response to history
                ai_message = AIMessage(content=agent_message.content, name=next_agent_name)
                self._add_message_to_history(conversation_id, ai_message)
                
                # Notify callback if registered
                if conversation_id in self.message_callbacks:
                    self.message_callbacks[conversation_id]({
                        "sender": next_agent_name,
                        "content": agent_message.content,
                        "timestamp": datetime.now().isoformat(),
                        "type": "ai"
                    })
                
                # Schedule next agent response after a short delay
                threading.Timer(2.0, lambda: self._schedule_next_response(conversation_id)).start()
                
                return {
                    "sender": next_agent_name,
                    "content": agent_message.content,
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                }
            
        except Exception as e:
            return {
                "error": f"Agent {next_agent_name} error: {str(e)}",
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
        
        return {"success": False, "error": "No response generated"}
    
    def _schedule_next_response(self, conversation_id: str):
        """Schedule the next agent response if conversation is still active."""
        if (conversation_id in self.active_conversations and 
            self.active_conversations[conversation_id]["status"] == "active"):
            
            # Add small random delay for more natural conversation
            delay = random.uniform(1.0, 3.0)
            time.sleep(delay)
            
            try:
                self._get_next_agent_response(conversation_id)
            except Exception as e:
                print(f"Error in scheduled response: {e}")
    
    def pause_conversation(self, conversation_id: str):
        """Pause a conversation."""
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]["status"] = "paused"
    
    def resume_conversation(self, conversation_id: str):
        """Resume a paused conversation."""
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id]["status"] = "active"
            # Trigger next response
            threading.Timer(1.0, lambda: self._schedule_next_response(conversation_id)).start()
    
    def stop_conversation(self, conversation_id: str):
        """Stop and remove a conversation from active sessions."""
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
        if conversation_id in self.message_callbacks:
            del self.message_callbacks[conversation_id]
    
    def get_conversation_summary(self, conversation_id: str) -> str:
        """Generate a summary of the conversation."""
        if conversation_id not in self.active_conversations:
            return "Conversation not found"
        
        conv_data = self.active_conversations[conversation_id]
        message_history = conv_data.get("message_history", [])
        
        if not message_history:
            return "No messages in conversation yet."
        
        # Create a simple summary
        participants = conv_data["agent_names"]
        environment = conv_data["environment"]
        message_count = len([msg for msg in message_history if isinstance(msg, (AIMessage, HumanMessage))])
        
        return f"""Conversation Summary:
Environment: {environment}
Participants: {', '.join(participants)}
Messages exchanged: {message_count}
Status: {conv_data["status"]}

This is a group conversation between {len(participants)} agents in {environment}. The conversation has been flowing naturally with participants taking turns to share their thoughts and perspectives."""
    
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
        
        # Recreate agents with new environment
        agents_config = conv_data["agents_config"]
        new_agents = self.create_agents(agents_config, new_environment, new_scene_description)
        conv_data["agents"] = new_agents
        
        # Send scene change message to conversation
        scene_change_message = f"[SCENE CHANGE] The setting has changed to: {new_environment}. {new_scene_description}"
        return self.send_message(conversation_id, scene_change_message, "system")


# For backward compatibility, create an alias
ConversationSimulatorEngine = SimpleConversationEngine
