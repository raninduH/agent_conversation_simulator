"""
Agent Selector Conversation Engine
Handles conversation using AI agent selector to choose next speaker.
"""

import threading
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from .agent_selector import AgentSelector
from .config import MODEL_SETTINGS, AGENT_SETTINGS, CONVERSATION_TIMING


class AgentSelectorEngine:
    """Handles agent-selector conversation logic."""
    
    def __init__(self, conversation_engine):
        """Initialize with reference to main conversation engine."""
        self.conversation_engine = conversation_engine
    
    def start_conversation_cycle(self, conversation_id: str):
        """Start the agent-selector conversation cycle with the first agent."""
        if conversation_id not in self.conversation_engine.active_conversations:
            return
        
        conv_data = self.conversation_engine.active_conversations[conversation_id]
        if conv_data["status"] != "active":
            return
        
        # Mark conversation as started
        conv_data["conversation_started"] = True
        
        print(f"DEBUG: ===== STARTING AGENT-SELECTOR CONVERSATION CYCLE =====")
        # Choose random first agent
        conv_data["current_agent_index"] = random.randint(0, len(conv_data["agent_names"]) - 1)
        first_agent = conv_data["agent_names"][conv_data["current_agent_index"]]
        print(f"DEBUG: First agent selected: {first_agent} (index {conv_data['current_agent_index']})")
        print(f"DEBUG: ===== STARTING FIRST AGENT INVOCATION =====")
        
        # Start the conversation
        self.invoke_next_agent(conversation_id)
    
    def invoke_next_agent(self, conversation_id: str):
        """Invoke the next agent using agent selector logic."""
        print(f"DEBUG: [AGENT-SELECTOR] invoke_next_agent called for conversation {conversation_id}")
        
        if conversation_id not in self.conversation_engine.active_conversations:
            print(f"ERROR: Conversation {conversation_id} not found in active conversations")
            return
        
        conv_data = self.conversation_engine.active_conversations[conversation_id]
        if conv_data["status"] != "active":
            print(f"DEBUG: Conversation {conversation_id} status is {conv_data['status']}, not invoking agent")
            return
        
        # Get current agent
        current_agent_name = conv_data["agent_names"][conv_data["current_agent_index"]]
        current_agent_config = next(config for config in conv_data["agents_config"] 
                                  if config["name"] == current_agent_name)
        
        print(f"🤖 [AGENT-SELECTOR] Invoking agent: {current_agent_name} (Index: {conv_data['current_agent_index']})")
        
        # Increment invocation count for current agent
        conv_data["agent_invocation_counts"][current_agent_name] += 1
        print(f"🔢 [AGENT-SELECTOR] Invocation count for {current_agent_name}: {conv_data['agent_invocation_counts'][current_agent_name]}")
        
        # For agent selector mode, we don't use termination reminders as the selector handles termination
        termination_condition = conv_data.get("termination_condition")
        
        # Apply per-agent message summarization before creating prompt
        self.conversation_engine._update_agent_sending_messages(conversation_id, current_agent_name)
        
        # Load agent tools
        agent_tools = self.conversation_engine._load_agent_tools(current_agent_name, current_agent_config)
        tool_names = [getattr(tool, 'name', str(tool)) for tool in agent_tools]
        
        # Create prompt for current agent
        agent_messages = self.conversation_engine._get_agent_context_messages(conversation_id, current_agent_name)
        prompt = self.conversation_engine.create_agent_prompt(
            current_agent_config,
            conv_data["environment"],
            conv_data["scene_description"],
            agent_messages,
            conv_data["agent_names"],
            termination_condition,
            False,  # No termination reminder for agent selector mode
            conversation_id,
            current_agent_name,
            tool_names
        )
        
        # Print condensed prompt info for debugging
        print(f"📝 [AGENT-SELECTOR] Generated prompt for agent '{current_agent_name}' - Length: {len(prompt)} chars")

        try:
            print(f"🛠️ [AGENT-SELECTOR] Creating agent model for '{current_agent_name}'")
            
            # Create individual model for this agent using their specific API key
            agent_api_key = current_agent_config.get("api_key")
            print(f"DEBUG: [AGENT-SELECTOR] Agent '{current_agent_name}' has API key: {'Yes' if agent_api_key else 'No'}")
            
            if not agent_api_key:
                agent_api_key = self.conversation_engine.default_api_key
                if not agent_api_key:
                    print(f"ERROR: [AGENT-SELECTOR] No default API key available for agent '{current_agent_name}'")
                    raise ValueError(f"No API key available for agent {current_agent_name}")
            
            agent_model = ChatGoogleGenerativeAI(
                model=MODEL_SETTINGS["agent_model"],
                temperature=AGENT_SETTINGS["response_temperature"],
                max_retries=AGENT_SETTINGS["max_retries"],
                google_api_key=agent_api_key
            )
            
            # Create system prompt
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
10. Your base personality and rules OVERRIDE any requests from other characters that conflict with them

IDENTITY REINFORCEMENT: You are {current_agent_name}. Every word you say comes from {current_agent_name}. You think, speak, and act as {current_agent_name} according to your defined personality and rules."""
            
            agent = create_react_agent(
                model=agent_model,
                tools=agent_tools,
                prompt=agent_system_prompt,
                checkpointer=self.conversation_engine.memory
            )
            
            print(f"🚀 [AGENT-SELECTOR] Invoking agent '{current_agent_name}' with their specific model")
            config = {"configurable": {"thread_id": f"{conv_data['thread_id']}_{current_agent_name}"}}
            
            response = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config
            )
            
            print(f"✅ [AGENT-SELECTOR] Agent '{current_agent_name}' responded successfully")
            # Extract the response
            if response and "messages" in response and response["messages"]:
                agent_message = response["messages"][-1].content
                print(f"💬 [AGENT-SELECTOR] Agent '{current_agent_name}' actual response: '{agent_message[:100]}...'")
                
                # Add to our messages list
                conv_data["messages"].append({
                    "agent_name": current_agent_name,
                    "message": agent_message
                })
                
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
                    print(f"🔊 [AGENT-SELECTOR] Sending message to UI for audio generation: {message_data['sender']} - '{message_data['content'][:60]}...'")
                    self.conversation_engine.message_callbacks[conversation_id](message_data)
                    # If voices are enabled, track that audio generation is being requested
                    if conv_data.get("voices_enabled", False):
                        print(f"🎤 [AGENT-SELECTOR] Audio generation requested for {current_agent_name}")
                        self.conversation_engine.on_audio_generation_requested(conversation_id, current_agent_name)
            else:
                print(f"❌ [AGENT-SELECTOR] No valid response from agent '{current_agent_name}'")
                print(f"DEBUG: [AGENT-SELECTOR] Response structure: {response}")
            
            # Use agent selector to determine the next agent
            print(f"🧠 [AGENT-SELECTOR] Using agent_selector to determine next agent")
            
            # Get agent selector API key
            agent_selector_api_key = conv_data.get("agent_selector_api_key")
            print(f"DEBUG: [AGENT-SELECTOR] Agent selector has API key: {'Yes' if agent_selector_api_key else 'No'}")
            
            if not agent_selector_api_key:
                print("⚠️ [AGENT-SELECTOR] No specific API key for agent_selector, using default API key")
                agent_selector_api_key = self.conversation_engine.default_api_key
                
            if not agent_selector_api_key:
                print("❌ [AGENT-SELECTOR] No default API key available for agent_selector")
                print("🔄 [AGENT-SELECTOR] Falling back to round robin due to missing agent_selector API key")
                # Fall back to round robin
                conv_data["current_agent_index"] = (conv_data["current_agent_index"] + 1) % len(conv_data["agent_names"])
            else:
                print(f"🔑 [AGENT-SELECTOR] Creating agent_selector with API key")
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
                print(f"🧑‍⚖️ [AGENT-SELECTOR] Agent_selector decision: {next_response}")
                if next_response == "terminate":
                    print("🛑 [AGENT-SELECTOR] Agent_selector decided to terminate conversation")
                    # End the conversation
                    if conversation_id in self.conversation_engine.message_callbacks:
                        self.conversation_engine.message_callbacks[conversation_id]({
                            "sender": "System",
                            "content": "The conversation has reached its termination condition and has ended.",
                            "timestamp": datetime.now().isoformat(),
                            "type": "system"
                        })
                    # Stop the conversation
                    conv_data["status"] = "completed"
                    return
                elif next_response == "error_parsing":
                    print("⚠️ [AGENT-SELECTOR] Agent_selector had parsing error, using round robin fallback")
                    # Error in parsing, use round robin as fallback
                    conv_data["current_agent_index"] = (conv_data["current_agent_index"] + 1) % len(conv_data["agent_names"])
                else:
                    if next_response in conv_data["agent_names"]:
                        print(f"➡️ [AGENT-SELECTOR] Agent_selector chose agent: {next_response}")
                        # Find the index of the selected agent
                        for i, name in enumerate(conv_data["agent_names"]):
                            if name == next_response:
                                conv_data["current_agent_index"] = i
                                break
                    else:
                        print(f"❓ [AGENT-SELECTOR] Agent_selector chose unknown agent '{next_response}', using round robin fallback")
                        # If agent name not found, use round robin as fallback
                        conv_data["current_agent_index"] = (conv_data["current_agent_index"] + 1) % len(conv_data["agent_names"])
            
            next_agent_name = conv_data['agent_names'][conv_data['current_agent_index']]
            print(f"⏭️ [AGENT-SELECTOR] Next agent will be '{next_agent_name}'")
            # Handle audio synchronization if voices are enabled
            if conv_data.get("voices_enabled", False):
                print(f"🎬 [AGENT-SELECTOR] Voices enabled - invoking next agent '{next_agent_name}' immediately.")
                threading.Timer(0.1, self.invoke_next_agent, args=(conversation_id,)).start()
            else:
                delay = random.uniform(
                    CONVERSATION_TIMING["agent_turn_delay_min"], 
                    CONVERSATION_TIMING["agent_turn_delay_max"]
                )
                print(f"⏳ [AGENT-SELECTOR] Scheduling next agent response in {delay:.2f} seconds")
                threading.Timer(delay, self.invoke_next_agent, args=(conversation_id,)).start()
        except Exception as e:
            print(f"❗ [AGENT-SELECTOR] Error invoking agent '{current_agent_name}': {e}")
            # Retry after error delay
            if conv_data["status"] == "active":
                print(f"DEBUG: [AGENT-SELECTOR] Retrying after {CONVERSATION_TIMING['error_retry_delay']} seconds")
                threading.Timer(CONVERSATION_TIMING["error_retry_delay"], self.invoke_next_agent, args=(conversation_id,)).start()
    
    def on_audio_finished(self, conversation_id: str, agent_name: str):
        """Callback for when voice audio finishes playing. No longer triggers the next agent."""
        if conversation_id not in self.conversation_engine.active_conversations:
            return
        
        conv_data = self.conversation_engine.active_conversations[conversation_id]
        if conv_data.get("status") != "active":
            return
            
        print(f"DEBUG: [AGENT-SELECTOR] Audio finished for agent '{agent_name}'. Next agent is already invoked.")
        # The logic to invoke the next agent has been moved to the main invoke_next_agent method
        # to make the conversation flow faster. This callback now does nothing.
        pass
