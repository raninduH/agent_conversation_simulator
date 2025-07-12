"""
Round Robin Conversation Engine
Handles sequential agent invocation in round-robin fashion.
"""

import os
import threading
import random
import traceback
import time
from datetime import datetime
from typing import List, Dict, Any, Callable
from functools import partial
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from .config import MODEL_SETTINGS, AGENT_SETTINGS, CONVERSATION_TIMING


class RoundRobinEngine:
    """Handles round-robin conversation logic."""
    
    def __init__(self, conversation_engine):
        """Initialize with reference to main conversation engine."""
        self.conversation_engine = conversation_engine
    
    def start_conversation_cycle(self, conversation_id: str):
        """Start the round-robin conversation cycle with the first agent."""
        if conversation_id not in self.conversation_engine.active_conversations:
            return
        
        conv_data = self.conversation_engine.active_conversations[conversation_id]
        if conv_data["status"] != "active":
            return
        
        # Mark conversation as started
        conv_data["conversation_started"] = True
        
        print(f"🚀 [ROUND-ROBIN] Starting conversation cycle for ID: {conversation_id}")
        
        # Choose random first agent
        conv_data["current_agent_index"] = random.randint(0, len(conv_data["agent_names"])) - 1
        first_agent = conv_data["agent_names"][conv_data["current_agent_index"]]
        print(f"DEBUG: First agent selected: {first_agent} (index {conv_data['current_agent_index']})")
        print(f"DEBUG: ===== STARTING FIRST AGENT INVOCATION =====")
        
        # Start the conversation
        self.invoke_next_agent(conversation_id)
    
    def start_cycle(self, conversation_id: str):
        """Alias for start_conversation_cycle to unify engine interface."""
        self.start_conversation_cycle(conversation_id)
    
    def invoke_next_agent(self, conversation_id: str):
        """Invoke the next agent in the round-robin cycle."""
        print(f"\n🟢 [ROUND-ROBIN] invoke_next_agent called for conversation {conversation_id}")
        if conversation_id not in self.conversation_engine.active_conversations:
            print(f"❌ ERROR: Conversation {conversation_id} not found in active conversations")
            return
        conv_data = self.conversation_engine.active_conversations[conversation_id]
        if conv_data["status"] != "active":
            print(f"⚠️ Conversation {conversation_id} status is {conv_data['status']}, not invoking agent")
            return
        agent_names = conv_data["agent_names"]
        current_agent_index = conv_data["current_agent_index"]
        current_agent_name = agent_names[current_agent_index]
        current_agent_config = conv_data["agents_config"][current_agent_name]
        print(f"\n🤖 [ROUND-ROBIN] Invoking agent: {current_agent_name} (Index: {conv_data['current_agent_index']})")
        conv_data["agent_invocation_counts"][current_agent_name] += 1
        termination_condition = conv_data.get("termination_condition")
        should_remind_termination = False
        if termination_condition:
            reminder_frequency = AGENT_SETTINGS["termination_reminder_frequency"]
            current_count = conv_data["agent_invocation_counts"][current_agent_name]
            should_remind_termination = (current_count % reminder_frequency == 0)
            if should_remind_termination:
                print(f"🔔 [ROUND-ROBIN] Agent '{current_agent_name}' will be reminded about termination condition (invocation #{current_count})")
        # --- CONTEXT: Use all previous messages as context for the agent ---
        all_messages = conv_data.get("messages", [])
        print(f"📚 [ROUND-ROBIN] Passing {len(all_messages)} previous messages as context to agent '{current_agent_name}'")
        print(f"DEBUG: voices_enabled={conv_data.get('voices_enabled', None)}")
        print(f"DEBUG: on_audio_generation_requested exists: {hasattr(self.conversation_engine, 'on_audio_generation_requested')}")
        # Use all previous messages as context
        agent_messages = all_messages.copy()
        # Apply per-agent message summarization before creating prompt
        self.conversation_engine._update_agent_sending_messages(conversation_id, current_agent_name)
        agent_tools = self.conversation_engine._load_agent_tools(conversation_id, current_agent_name)
        tool_names = [tool.name for tool in agent_tools]
        agent_obj = None
        prompt = self.conversation_engine.create_agent_prompt(
            current_agent_config,
            conv_data["environment"],
            conv_data["scene_description"],
            agent_messages,
            conv_data["agent_names"],
            termination_condition,
            should_remind_termination,
            conversation_id,
            current_agent_name,
            tool_names,
            agent_obj
        )
        print(f"📝 [ROUND-ROBIN] Generated prompt for agent '{current_agent_name}' - Length: {len(prompt)} chars")
        try:
            print(f"🛠️ [ROUND-ROBIN] Creating agent model for '{current_agent_name}'")
            agent_api_key = current_agent_config.get("api_key")
            print(f"🔑 [ROUND-ROBIN] Agent '{current_agent_name}' has API key: {'Yes' if agent_api_key else 'No'}")
            if not agent_api_key:
                agent_api_key = self.conversation_engine.default_api_key
                if not agent_api_key:
                    print(f"❌ [ROUND-ROBIN] No default API key available for agent '{current_agent_name}'")
                    raise ValueError(f"No API key available for agent {current_agent_name}")
            agent_model = ChatGoogleGenerativeAI(
                model=MODEL_SETTINGS["agent_model"],
                temperature=AGENT_SETTINGS["response_temperature"],
                max_retries=AGENT_SETTINGS["max_retries"],
                google_api_key=agent_api_key
            )
            agent_system_prompt = self.conversation_engine.create_agent_prompt(
                current_agent_config,
                conv_data["environment"],
                conv_data["scene_description"],
                agent_messages,
                conv_data["agent_names"],
                termination_condition,
                should_remind_termination,
                conversation_id,
                current_agent_name,
                tool_names,
                agent_obj
            )
            agent = create_react_agent(
                model=agent_model,
                tools=agent_tools,
                prompt=agent_system_prompt,
                checkpointer=self.conversation_engine.memory
            )
            print(f"🚀 [ROUND-ROBIN] Invoking agent '{current_agent_name}' with their specific model")
            config = {"configurable": {"thread_id": f"{conv_data['thread_id']}_{current_agent_name}"}}
            response = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config
            )
            print(f"✅ [ROUND-ROBIN] Agent '{current_agent_name}' responded successfully")
            if response and "messages" in response and response["messages"]:
                agent_message = response["messages"][-1].content
                print(f"💬 [ROUND-ROBIN] Agent '{current_agent_name}' response: '{agent_message[:100]}...'")
                conv_data["messages"].append({
                    "agent_name": current_agent_name,
                    "message": agent_message
                })
                if "agent_sending_messages" not in conv_data:
                    conv_data["agent_sending_messages"] = {}
                if current_agent_name not in conv_data["agent_sending_messages"]:
                    conv_data["agent_sending_messages"][current_agent_name] = []
                conv_data["agent_sending_messages"][current_agent_name].append(agent_message)
                self.conversation_engine._save_conversation_state(conversation_id)
                print(f"💾 [ROUND-ROBIN] Message saved to conversation and agent_sending_messages.")
                if conversation_id in self.conversation_engine.message_callbacks:
                    message_data = {
                        "sender": current_agent_name,
                        "content": agent_message,
                        "timestamp": datetime.now().isoformat(),
                        "type": "ai"
                    }
                    self.conversation_engine.message_callbacks[conversation_id](message_data)
                    print(f"📤 [ROUND-ROBIN] Message sent to UI callback.")
                    if conv_data.get("voices_enabled", False):
                        print(f"🔊 [ROUND-ROBIN] Sending message for audio generation...")
                        if hasattr(self.conversation_engine, 'on_audio_generation_requested'):
                            self.conversation_engine.on_audio_generation_requested(conversation_id, current_agent_name)
                        else:
                            print(f"❌ [ROUND-ROBIN] No on_audio_generation_requested callback set!")
                    else:
                        print(f"⏳ [ROUND-ROBIN] No audio, will schedule next agent after delay.")
            else:
                print(f"❌ [ROUND-ROBIN] No valid response from agent '{current_agent_name}'")
                print(f"DEBUG: [ROUND-ROBIN] Response structure: {response}")
            conv_data["current_agent_index"] = (conv_data["current_agent_index"] + 1) % len(conv_data["agent_names"])
            if (conv_data["current_agent_index"] == 0 and \
                termination_condition and \
                len(conv_data["messages"]) >= len(conv_data["agent_names"])):
                print("🔚 [ROUND-ROBIN] Checking termination condition")
                if self.conversation_engine._check_round_robin_termination(conversation_id):
                    print("🛑 [ROUND-ROBIN] Termination condition met. Ending conversation.")
                    if conversation_id in self.conversation_engine.message_callbacks:
                        self.conversation_engine.message_callbacks[conversation_id]({
                            "sender": "System",
                            "content": "Termination condition reached.",
                            "timestamp": datetime.now().isoformat(),
                            "type": "system"
                        })
                    self.conversation_engine.stop_conversation(conversation_id)
                    return
            # --- SCHEDULING NEXT AGENT ---
            if conv_data.get("voices_enabled", False):
                print(f"🟡 [ROUND-ROBIN] Waiting for audio to finish before next agent.")
                # Next agent will be invoked by audio_finished callback
            else:
                delay = random.uniform(CONVERSATION_TIMING["min_delay"], CONVERSATION_TIMING["max_delay"])
                print(f"⏳ [ROUND-ROBIN] Scheduling next agent in {delay:.2f} seconds (voices disabled)")
                threading.Timer(delay, self.invoke_next_agent, args=[conversation_id]).start()
        except Exception as e:
            print(f"❌ [ROUND-ROBIN] Error invoking agent '{current_agent_name}': {e}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            print(f"ERROR: Traceback: {traceback.format_exc()}")
            if conv_data["status"] == "active":
                print(f"🔁 [ROUND-ROBIN] Retrying after {CONVERSATION_TIMING['error_retry_delay']} seconds")
                threading.Timer(CONVERSATION_TIMING["error_retry_delay"], self.invoke_next_agent, args=(conversation_id,)).start()
    
    def handle_agent_response(self, conversation_id: str, agent_name: str, response: str):
        """Handles the response from an agent, sending it to the UI and invoking the next agent."""
        print(f"DEBUG: [ROUND-ROBIN] Handling response from agent '{agent_name}' for conversation {conversation_id}")
        
        if conversation_id not in self.conversation_engine.active_conversations:
            print(f"ERROR: Conversation {conversation_id} not found in active conversations")
            return
        
        conv_data = self.conversation_engine.active_conversations[conversation_id]
        if conv_data["status"] != "active":
            print(f"DEBUG: Conversation {conversation_id} status is {conv_data['status']}, not processing response")
            return
        
        # Add the agent's response to the conversation history
        conv_data["messages"].append({
            "agent_name": agent_name,
            "message": response
        })
        
        # Save conversation state
        self.conversation_engine._save_conversation_state(conversation_id)
        
        # Prepare message data for UI
        ui_message = {
            "sender": agent_name,
            "content": response,
            "timestamp": datetime.now().isoformat(),
            "type": "ai"
        }
        
        # Send the message to the UI
        if conversation_id in self.conversation_engine.message_callbacks:
            self.conversation_engine.message_callbacks[conversation_id](ui_message)
        
        # Immediately invoke the next agent for a more responsive feel
        if not conv_data["stop_conversation"]:
            threading.Timer(0.1, self.invoke_next_agent, args=[conversation_id]).start()
    
    def on_voice_audio_finished(self, conversation_id: str):
        """Callback for when voice audio finishes playing. No longer triggers the next agent."""
        convo = self.conversation_engine.active_conversations.get(conversation_id)
        if not convo or not convo.get("voices_enabled"):
            return
        
        print("✅ [ROUND-ROBIN] Audio finished playing. Next agent was already invoked.")
        # The logic to invoke the next agent has been moved to handle_agent_response
        # to make the conversation flow faster. This callback now does nothing.
        pass

    def invoke_next_agent(self, conversation_id: str):
        """Invokes the next agent in the round-robin cycle."""
        print(f"\n🟢 [ROUND-ROBIN] invoke_next_agent called for conversation {conversation_id}")
        
        if conversation_id not in self.conversation_engine.active_conversations:
            print(f"❌ ERROR: Conversation {conversation_id} not found in active conversations")
            return
        
        conv_data = self.conversation_engine.active_conversations[conversation_id]
        if conv_data["status"] != "active":
            print(f"⚠️ Conversation {conversation_id} status is {conv_data['status']}, not invoking agent")
            return
        
        # Get current agent
        agent_names = conv_data["agent_names"]
        current_agent_index = conv_data["current_agent_index"]
        current_agent_name = agent_names[current_agent_index]
        current_agent_config = conv_data["agents_config"][current_agent_name]
        
        print(f"\n🤖 [ROUND-ROBIN] Invoking agent: {current_agent_name} (Index: {conv_data['current_agent_index']})")
        
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
                print(f"🔔 [ROUND-ROBIN] Agent '{current_agent_name}' will be reminded about termination condition (invocation #{current_count})")
        
        # --- CONTEXT: Use all previous messages as context for the agent ---
        all_messages = conv_data.get("messages", [])
        print(f"📚 [ROUND-ROBIN] Passing {len(all_messages)} previous messages as context to agent '{current_agent_name}'")
        
        # Apply per-agent message summarization before creating prompt
        self.conversation_engine._update_agent_sending_messages(conversation_id, current_agent_name)
        
        # Load agent tools
        agent_tools = self.conversation_engine._load_agent_tools(conversation_id, current_agent_name)
        tool_names = [tool.name for tool in agent_tools]
        agent_obj = None  # If you have agent object logic, set it here
        
        # Use all previous messages as context
        agent_messages = all_messages.copy()
        prompt = self.conversation_engine.create_agent_prompt(
            current_agent_config,
            conv_data["environment"],
            conv_data["scene_description"],
            agent_messages,
            conv_data["agent_names"],
            termination_condition,
            should_remind_termination,
            conversation_id,
            current_agent_name,
            tool_names,
            agent_obj
        )
        
        print(f"📝 [ROUND-ROBIN] Generated prompt for agent '{current_agent_name}' - Length: {len(prompt)} chars")

        try:
            print(f"🛠️ [ROUND-ROBIN] Creating agent model for '{current_agent_name}'")
            
            # Create individual model for this agent using their specific API key
            agent_api_key = current_agent_config.get("api_key")
            print(f"🔑 [ROUND-ROBIN] Agent '{current_agent_name}' has API key: {'Yes' if agent_api_key else 'No'}")
            
            if not agent_api_key:
                agent_api_key = self.conversation_engine.default_api_key
                if not agent_api_key:
                    print(f"❌ [ROUND-ROBIN] No default API key available for agent '{current_agent_name}'")
                    raise ValueError(f"No API key available for agent {current_agent_name}")
            
            agent_model = ChatGoogleGenerativeAI(
                model=MODEL_SETTINGS["agent_model"],
                temperature=AGENT_SETTINGS["response_temperature"],
                max_retries=AGENT_SETTINGS["max_retries"],
                google_api_key=agent_api_key
            )
            
            # Create system prompt
            agent_system_prompt = self.conversation_engine.create_agent_prompt(
                current_agent_config,
                conv_data["environment"],
                conv_data["scene_description"],
                agent_messages,
                conv_data["agent_names"],
                termination_condition,
                should_remind_termination,
                conversation_id,
                current_agent_name,
                tool_names,
                agent_obj
            )
            
            agent = create_react_agent(
                model=agent_model,
                tools=agent_tools,
                prompt=agent_system_prompt,
                checkpointer=self.conversation_engine.memory
            )
            
            print(f"🚀 [ROUND-ROBIN] Invoking agent '{current_agent_name}' with their specific model")
            config = {"configurable": {"thread_id": f"{conv_data['thread_id']}_{current_agent_name}"}}
            
            response = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config
            )
            
            print(f"✅ [ROUND-ROBIN] Agent '{current_agent_name}' responded successfully")
            
            # Extract the response
            if response and "messages" in response and response["messages"]:
                agent_message = response["messages"][-1].content
                print(f"💬 [ROUND-ROBIN] Agent '{current_agent_name}' response: '{agent_message[:100]}...'")
                
                # Add to our messages list
                conv_data["messages"].append({
                    "agent_name": current_agent_name,
                    "message": agent_message
                })
                
                # Add to agent_sending_messages for this agent
                if "agent_sending_messages" not in conv_data:
                    conv_data["agent_sending_messages"] = {}
                if current_agent_name not in conv_data["agent_sending_messages"]:
                    conv_data["agent_sending_messages"][current_agent_name] = []
                conv_data["agent_sending_messages"][current_agent_name].append(agent_message)
                
                # Save conversation to persistent storage after each message
                self.conversation_engine._save_conversation_state(conversation_id)
                
                # Notify callback
                if conversation_id in self.conversation_engine.message_callbacks:
                    message_data = {
                        "sender": current_agent_name,
                        "content": agent_message,
                        "timestamp": datetime.now().isoformat(),
                        "type": "ai"
                    }
                    self.conversation_engine.message_callbacks[conversation_id](message_data)
                    print(f"📤 [ROUND-ROBIN] Message sent to UI callback.")
                    
                    # If voices are enabled, track that audio generation is being requested
                    if conv_data.get("voices_enabled", False):
                        print(f"🔊 [ROUND-ROBIN] Sending message for audio generation...")
                        if hasattr(self.conversation_engine, 'on_audio_generation_requested'):
                            self.conversation_engine.on_audio_generation_requested(conversation_id, current_agent_name)
                        else:
                            print(f"❌ [ROUND-ROBIN] No on_audio_generation_requested callback set!")
                    else:
                        print(f"⏳ [ROUND-ROBIN] No audio, will schedule next agent after delay.")
            else:
                print(f"❌ [ROUND-ROBIN] No valid response from agent '{current_agent_name}'")
                print(f"DEBUG: [ROUND-ROBIN] Response structure: {response}")
            
            # Move to next agent in round-robin cycle
            conv_data["current_agent_index"] = (conv_data["current_agent_index"] + 1) % len(conv_data["agent_names"])
            
            # Check if we completed a round (all agents have spoken) and if there's a termination condition to check
            if (conv_data["current_agent_index"] == 0 and 
                termination_condition and 
                len(conv_data["messages"]) >= len(conv_data["agent_names"])):
                
                print("🔚 [ROUND-ROBIN] Checking termination condition")
                # Check if conversation should terminate
                if self.conversation_engine._check_round_robin_termination(conversation_id):
                    print("🛑 [ROUND-ROBIN] Termination condition met. Ending conversation.")
                    # End the conversation
                    if conversation_id in self.conversation_engine.message_callbacks:
                        self.conversation_engine.message_callbacks[conversation_id]({
                            "sender": "System",
                            "content": "Termination condition reached.",
                            "timestamp": datetime.now().isoformat(),
                            "type": "system"
                        })
                    # Stop the conversation
                    self.conversation_engine.stop_conversation(conversation_id)
                    return

        except Exception as e:
            print(f"❌ [ROUND-ROBIN] Error invoking agent '{current_agent_name}': {e}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            print(f"ERROR: Traceback: {traceback.format_exc()}")
            # Retry after error delay
            if conv_data["status"] == "active":
                print(f"🔁 [ROUND-ROBIN] Retrying after {CONVERSATION_TIMING['error_retry_delay']} seconds")
                threading.Timer(CONVERSATION_TIMING["error_retry_delay"], self.invoke_next_agent, args=(conversation_id,)).start()
    
    def on_audio_finished(self, conversation_id: str, agent_name: str):
        """Called when audio finishes playing for round-robin mode."""
        if conversation_id not in self.conversation_engine.active_conversations:
            return
        
        conv_data = self.conversation_engine.active_conversations[conversation_id]
        
        # Check if conversation is still active before proceeding
        if conv_data.get("status") != "active":
            print(f"DEBUG: [ROUND-ROBIN] Conversation {conversation_id} status is {conv_data.get('status')} - skipping audio finished processing")
            return
        
        next_agent = conv_data.get("next_agent_scheduled")
        if next_agent:
            print(f"DEBUG: [ROUND-ROBIN] Audio finished for agent '{agent_name}' - invoking next agent")
            conv_data["next_agent_scheduled"] = None
            # Invoke the next agent immediately
            self.invoke_next_agent(conversation_id)
        else:
            print(f"DEBUG: [ROUND-ROBIN] Audio finished for agent '{agent_name}' - no next agent scheduled")
