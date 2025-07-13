"""
Conversation Engine
Handles starting, resuming, pausing, and updating multi-agent conversations.
Delegates conversation flow to the appropriate engine based on invocation_method.
"""

import json
import os
import random
from dataclasses import asdict

from .round_robin_engine import RoundRobinEngine
from .agent_selector_engine import AgentSelectorEngine
from .human_like_chat_engine import HumanLikeChatEngine
from .config import UI_COLORS
from .data_manager import DataManager, Agent

class ConversationEngineFactory:
    """Factory to create the correct conversation engine based on invocation method."""
    def __init__(self, parent):
        self.parent = parent
        self.engines = {
            "round_robin": parent.round_robin_engine,
            "agent_selector": parent.agent_selector_engine,
            "human_like_chat": parent.human_like_chat_engine
        }

    def get_engine(self, invocation_method):
        return self.engines.get(invocation_method, self.engines["round_robin"])

class ConversationEngine:
    def on_user_message(self, conversation_id, message_data):
        """Pass user message to the correct engine."""
        print(f"[ConversationEngine] on_user_message called for {conversation_id}")
        engine = self.current_engines.get(conversation_id)
        if engine and hasattr(engine, "on_user_message"):
            engine.on_user_message(message_data)
        else:
            print(f"[ConversationEngine] No engine found for on_user_message on {conversation_id}")
    def __init__(self):
        self.active_conversations = {}
        self.round_robin_engine = RoundRobinEngine(self)
        self.agent_selector_engine = AgentSelectorEngine(self)
        self.human_like_chat_engine = HumanLikeChatEngine(self)
        self.current_engines = {}  # conversation_id -> engine instance
        self.engine_factory = ConversationEngineFactory(self)
        self.message_callbacks = {}  # <-- Ensure this is always initialized

    def _assign_agent_numbers_and_colors(self, agents_config):
        print("🎨 [ConversationEngine] Assigning agent numbers and colors...")
        agent_temp_numbers = {}
        agent_colors = {}
        color_palette = UI_COLORS["agent_colors"]
        for i, agent_config in enumerate(agents_config, 1):
            agent_temp_numbers[agent_config["name"]] = i
            agent_colors[agent_config["name"]] = color_palette[(i-1) % len(color_palette)]
        print(f"✅ [ConversationEngine] Assigned numbers: {agent_temp_numbers}, colors: {agent_colors}")
        return agent_temp_numbers, agent_colors

    def _load_conversation_details(self, conversation_id):
        print(f"📂 [ConversationEngine] Loading conversation details for ID: {conversation_id}")
        data_manager = self.data_manager if hasattr(self, 'data_manager') else DataManager(os.path.dirname(__file__))
        conversation = data_manager.get_conversation_by_id(conversation_id)
        if not conversation:
            print(f"❌ [ConversationEngine] Conversation ID '{conversation_id}' not found!")
            raise FileNotFoundError(f"Conversation ID '{conversation_id}' not found in conversations.json.")
        print(f"✅ [ConversationEngine] Conversation loaded.")
        return conversation

    def start_conversation(self, conversation_id, agents_config, environment, scene_description, title=None, invocation_method="round_robin", termination_condition=None, agent_selector_api_key=None, voices_enabled=False):
        print(f"🚀 [ConversationEngine] Starting conversation '{conversation_id}' with method '{invocation_method}'...")
        agent_temp_numbers, agent_colors = self._assign_agent_numbers_and_colors(agents_config)
        now = None
        try:
            from datetime import datetime
            now = datetime.now().isoformat()
        except Exception:
            now = ""
        convo_details = {
            "id": conversation_id,
            "title": title if title else conversation_id,
            "agents": agents_config,
            "agent_temp_numbers": agent_temp_numbers,
            "agent_colors": agent_colors,
            "environment": environment,
            "scene_description": scene_description,
            "invocation_method": invocation_method,
            "termination_condition": termination_condition,
            "agent_selector_api_key": agent_selector_api_key,
            "voices_enabled": voices_enabled,
            "messages": [],
            "LLM_sending_messages": [],
            "status": "active",
            "created_at": now,
            "last_updated": now,
            "thread_id": f"thread_{random.getrandbits(32):08x}"
        }
        conversations_path = os.path.join(os.path.dirname(__file__), "conversations.json")
        if os.path.exists(conversations_path):
            with open(conversations_path, "r", encoding="utf-8") as f:
                conversations = json.load(f)
        else:
            conversations = {"conversations": []}
        conversations["conversations"].append(convo_details)
        with open(conversations_path, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2)
        self.active_conversations[conversation_id] = convo_details
        engine = self.engine_factory.get_engine(invocation_method)
        self.current_engines[conversation_id] = engine
        print(f"🤝 [ConversationEngine] Handing over to engine: {engine.__class__.__name__}")
        engine.start_cycle(
                conversation_id,
                agents_config,
                voices_enabled,
                termination_condition,
                agent_selector_api_key
        )
        print(f"✅ [ConversationEngine] Conversation '{conversation_id}' started.")

    def pause_conversation(self, conversation_id):
        print(f"⏸️ [ConversationEngine] Pausing conversation '{conversation_id}'...")
        engine = self.current_engines.get(conversation_id)
        convo = self.active_conversations.get(conversation_id)
        invocation_method = convo.get('invocation_method') if convo else None
        if invocation_method == 'round_robin' and engine and hasattr(engine, "pause_cycle"):
            print(f"[ConversationEngine] Calling pause_cycle for round_robin engine...")
            engine.pause_cycle(conversation_id)
            print(f"✅ [ConversationEngine] Conversation '{conversation_id}' paused.")
        else:
            print(f"⚠️ [ConversationEngine] No engine found to pause conversation '{conversation_id}' or invocation_method not round_robin.")

    def update_scene_environment(self, conversation_id, environment=None, scene_description=None):
        print(f"🌄 [ConversationEngine] Updating scene/environment for conversation '{conversation_id}'...")
        engine = self.current_engines.get(conversation_id)
        if engine and hasattr(engine, "update_scene_environment"):
            engine.update_scene_environment(conversation_id, environment, scene_description)
            print(f"✅ [ConversationEngine] Scene/environment updated for '{conversation_id}'.")
        else:
            print(f"⚠️ [ConversationEngine] No engine found to update scene/environment for '{conversation_id}'.")

    def _save_conversation_state(self, conversation_id):
        print(f"💾 [ConversationEngine] Saving conversation state for '{conversation_id}'...")
        data_manager = self.data_manager if hasattr(self, 'data_manager') else DataManager(os.path.dirname(__file__))
        convo = self.active_conversations.get(conversation_id)
        if convo is not None:
            from dataclasses import asdict
            if hasattr(convo, '__dataclass_fields__'):
                data_manager.save_conversation(convo)
            else:
                from .data_manager import Conversation
                conversation_obj = Conversation(**convo)
                data_manager.save_conversation(conversation_obj)
        print(f"✅ [ConversationEngine] Conversation state saved for '{conversation_id}'.")

    def register_message_callback(self, conversation_id, callback):
        print(f"🔔 [ConversationEngine] Registering message callback for '{conversation_id}'")
        if not hasattr(self, 'message_callbacks'):
            self.message_callbacks = {}
        self.message_callbacks[conversation_id] = callback
        print(f"✅ [ConversationEngine] Callback registered for '{conversation_id}'.")

    def resume_conversation(self, conversation_id):
        print(f"🔄 [ConversationEngine] Resuming past conversation '{conversation_id}'...")
        data_manager = self.data_manager if hasattr(self, 'data_manager') else DataManager(os.path.dirname(__file__))
        print(f"📖 [ConversationEngine] Loading conversation from JSON...")
        conversation = data_manager.get_conversation_by_id(conversation_id)
        if not conversation:
            print(f"❌ [ConversationEngine] Conversation ID '{conversation_id}' not found!")
            raise FileNotFoundError(f"Conversation ID '{conversation_id}' not found in conversations.json.")
        print(f"🟢 [ConversationEngine] Conversation found! Setting status to active...")
        conversation.status = "active"
        # Ensure LLM_sending_messages exists and is a list
        if not hasattr(conversation, "LLM_sending_messages") or conversation.LLM_sending_messages is None:
            print(f"📝 [ConversationEngine] Initializing LLM_sending_messages list...")
            conversation.LLM_sending_messages = []
        print(f"💾 [ConversationEngine] Saving updated conversation status...")
        data_manager.save_conversation(conversation)
        # Store in active_conversations
        self.active_conversations[conversation_id] = asdict(conversation)
        print(f"📦 [ConversationEngine] Loaded conversation info from JSON: {conversation}")
        engine = self.engine_factory.get_engine(conversation.invocation_method)
        self.current_engines[conversation_id] = engine
        print(f"🤝 [ConversationEngine] Handing over to engine: {engine.__class__.__name__}")
        # If round robin, call start_cycle for new, resume_cycle for existing
        if hasattr(engine, 'start_cycle') and hasattr(engine, 'resume_cycle'):
            if not conversation.messages or len(conversation.messages) == 0:
                print(f"� [ConversationEngine] No messages found, starting new round robin cycle...")
                engine.start_cycle(
                    conversation_id,
                    conversation.agents,
                    conversation.voices_enabled,
                    conversation.termination_condition,
                    conversation.agent_selector_api_key
                )
            else:
                print(f"🔁 [ConversationEngine] Messages found, resuming round robin cycle...")
                engine.resume_cycle(conversation_id)
        print(f"✅ [ConversationEngine] Conversation '{conversation_id}' resumed and set to active.")

