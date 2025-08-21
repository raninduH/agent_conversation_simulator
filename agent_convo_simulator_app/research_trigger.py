"""
Research Trigger Engine
Handles starting and managing research group conversations, similar to ConversationEngine but dedicated to research chat.
"""

import uuid
from datetime import datetime
import json
import os
import random
from dataclasses import asdict

from .config import UI_COLORS
from .data_manager import DataManager

class ResearchTriggerEngine:
    def __init__(self, app):
        self.app = app
        self.active_research = {}  # research_id -> config
        self.message_callback = None
       
        self.data_manager = DataManager()
        self.agent_selector_api_key = os.getenv("GOOGLE_API_KEY2")




    def register_message_callback(self, callback):
        self.message_callback = callback
        print("[ResearchTriggerEngine] Message callback registered.")

    def on_user_message(self, research_id, message_data):
        print(f"[ResearchTriggerEngine] on_user_message called for {research_id}")
        engine = self.current_engines.get(research_id)
        if engine and hasattr(engine, "on_user_message"):
            engine.on_user_message(message_data)
        else:
            print(f"[ResearchTriggerEngine] No engine found for on_user_message on {research_id}")

    def _assign_agent_numbers_and_colors(self, agents_config):
        print("🎨 [ResearchTriggerEngine] Assigning agent numbers and colors...")
        agent_temp_numbers = {}
        agent_colors = {}
        color_palette = UI_COLORS["agent_colors"]
        for i, agent_config in enumerate(agents_config, 1):
            agent_temp_numbers[agent_config["name"]] = i
            agent_colors[agent_config["name"]] = color_palette[(i-1) % len(color_palette)]
        print(f"✅ [ResearchTriggerEngine] Assigned numbers: {agent_temp_numbers}, colors: {agent_colors}")
        return agent_temp_numbers, agent_colors

    def _load_research_details(self, research_id):
        print(f"📂 [ResearchTriggerEngine] Loading research details for ID: {research_id}")
        data_manager = self.data_manager if hasattr(self, 'data_manager') else DataManager(os.path.dirname(__file__))
        research = data_manager.get_conversation_by_id(research_id)
        if not research:
            print(f"❌ [ResearchTriggerEngine] Research ID '{research_id}' not found!")
            raise FileNotFoundError(f"Research ID '{research_id}' not found in conversations.json.")
        print(f"✅ [ResearchTriggerEngine] Research loaded.")
        return research

    def start_research(self, research_id, agents_config, research_name, research_problem, extra_consider, research_goal, voices_enabled=False):
        # Only generate a new research_id if not provided (for compatibility)
        if not research_id or not research_id.startswith("research_"):
            research_id = f"research_{uuid.uuid4().hex[:8]}"
        now = datetime.now().isoformat()

        # Assign agent colors (like normal conversations)
        agent_numbers, agent_colors = self._assign_agent_numbers_and_colors([
            {"name": agent_id} for agent_id in agents_config
        ])

        # Save research details to file and memory
        research_details = {
            "id": research_id,
            "research_name": research_name,
            "research_problem": research_problem,
            "extra_consider": extra_consider,
            "research_goal": research_goal,
            "agents": agents_config,
            "voices_enabled": voices_enabled,
            "messages": [],
            "LLM_sending_messages": [],
            "status": "active",
            "created_at": now,
            "last_updated": now,
            "thread_id": f"thread_{random.getrandbits(32):08x}",
            "agent_colors": agent_colors,
            "agent_numbers": agent_numbers
        }

        
        from .data_manager import ResearchConversation

        research_conv_obj = ResearchConversation(
            id=research_details["id"],
            research_name=research_details["research_name"],
            research_problem=research_details["research_problem"],
            extra_consider=research_details["extra_consider"],
            research_goal=research_details["research_goal"],
            agents=research_details["agents"],
            messages=research_details["messages"],
            created_at=research_details["created_at"],
            last_updated=research_details["last_updated"],
            thread_id=research_details["thread_id"],
            status=research_details["status"],
            voices_enabled=research_details["voices_enabled"],
            agent_colors=research_details["agent_colors"],
            agent_numbers=research_details["agent_numbers"]
        )

        self.data_manager.save_research_conversation(research_conv_obj)
        # Use active_researches for consistency with rest of class
        if not hasattr(self, 'active_researches'):
            self.active_researches = {}
        self.active_researches[research_id] = research_details

        # Optionally, send a system message to the UI
        if self.message_callback:
            self.message_callback({
                "sender": "System",
                "content": f"Research '{research_name}' started.",
                "timestamp": now,
                "type": "system"
            })

        print(f"🚀 [ResearchTriggerEngine] Starting research '{research_id}'...")
        engine = self.research_chat_engine
        if not hasattr(self, 'current_engines'):
            self.current_engines = {}
        self.current_engines[research_id] = engine
        print(f"🤝 [ResearchTriggerEngine] Handing over to engine: {engine.__class__.__name__}")
        engine.start_cycle(
            research_id,
            agents_config,
            voices_enabled,
            research_goal,  # Pass as 'termination_condition' for now
            self.agent_selector_api_key  # No API key needed
        )
        print(f"✅ [ResearchTriggerEngine] Research '{research_id}' started.")

    
    
    
    
    
    def pause_research(self, research_id):
        print(f"⏸️ [ResearchTriggerEngine] Pausing research '{research_id}'...")
        engine = self.current_engines.get(research_id)
        if engine and hasattr(engine, "pause_cycle"):
            engine.pause_cycle(research_id)
            print(f"✅ [ResearchTriggerEngine] Research '{research_id}' paused.")
        else:
            print(f"⚠️ [ResearchTriggerEngine] No engine found to pause research '{research_id}'.")

    
    
 
    
    
    def update_research_goal(self, research_id, research_goal=None):
        print(f"🎯 [ResearchTriggerEngine] Updating research goal for '{research_id}'...")
        engine = self.current_engines.get(research_id)
        if engine and hasattr(engine, "update_scene_environment"):
            engine.update_scene_environment(research_id, scene_description=research_goal)
            print(f"✅ [ResearchTriggerEngine] Research goal updated for '{research_id}'.")
        else:
            print(f"⚠️ [ResearchTriggerEngine] No engine found to update research goal for '{research_id}'.")

    def _save_research_state(self, research_id):
        print(f"💾 [ResearchTriggerEngine] Saving research state for '{research_id}'...")
        data_manager = self.data_manager if hasattr(self, 'data_manager') else DataManager(os.path.dirname(__file__))
        research = self.active_researches.get(research_id)
        if research is not None:
            from dataclasses import asdict
            if hasattr(research, '__dataclass_fields__'):
                data_manager.save_conversation(research)
            else:
                from .data_manager import Conversation
                research_obj = Conversation(**research)
                data_manager.save_conversation(research_obj)
        print(f"✅ [ResearchTriggerEngine] Research state saved for '{research_id}'.")

    def register_message_callback(self, research_id, callback):
        print(f"🔔 [ResearchTriggerEngine] Registering message callback for '{research_id}'")
        if not hasattr(self, 'message_callbacks'):
            self.message_callbacks = {}
        self.message_callbacks[research_id] = callback
        print(f"✅ [ResearchTriggerEngine] Callback registered for '{research_id}'.")

    def resume_research(self, research_id):
        print(f"🔄 [ResearchTriggerEngine] Resuming past research '{research_id}'...")
        data_manager = self.data_manager if hasattr(self, 'data_manager') else DataManager(os.path.dirname(__file__))
        print(f"📖 [ResearchTriggerEngine] Loading research from JSON...")
        research = data_manager.get_conversation_by_id(research_id)
        if not research:
            print(f"❌ [ResearchTriggerEngine] Research ID '{research_id}' not found!")
            raise FileNotFoundError(f"Research ID '{research_id}' not found in conversations.json.")
        print(f"🟢 [ResearchTriggerEngine] Research found! Setting status to active...")
        research.status = "active"
        if not hasattr(research, "LLM_sending_messages") or research.LLM_sending_messages is None:
            print(f"📝 [ResearchTriggerEngine] Initializing LLM_sending_messages list...")
            research.LLM_sending_messages = []
        print(f"💾 [ResearchTriggerEngine] Saving updated research status...")
        data_manager.save_conversation(research)
        self.active_researches[research_id] = asdict(research)
        print(f"📦 [ResearchTriggerEngine] Loaded research info from JSON: {research}")
        engine = self.research_chat_engine
        self.current_engines[research_id] = engine
        print(f"🤝 [ResearchTriggerEngine] Handing over to engine: {engine.__class__.__name__}")
        if not research.messages or len(research.messages) == 0:
            print(f"� [ResearchTriggerEngine] No messages found, starting new research chat cycle...")
            engine.start_cycle(
                research_id,
                research.agents,
                research.voices_enabled,
                research.research_problem,  # Pass as 'termination_condition' for now
                None
            )
        else:
            print(f"🔁 [ResearchTriggerEngine] Messages found, resuming research chat cycle...")
            engine.resume_cycle(research_id)
        print(f"✅ [ResearchTriggerEngine] Research '{research_id}' resumed and set to active.")











        # def start_research(self, research_config, callback=None):
        # # Generate a unique research ID
        # research_id = f"research_{uuid.uuid4().hex[:8]}"
        # now = datetime.now().isoformat()
        # research_data = {
        #     "id": research_id,
        #     "config": research_config,
        #     "messages": [],
        #     "created_at": now,
        #     "status": "active"
        # }
        # self.active_research[research_id] = research_data
        # print(f"[ResearchTriggerEngine] Started research: {research_id}")
        # # Optionally, send a system message to the UI
        # if self.message_callback:
        #     self.message_callback({
        #         "sender": "System",
        #         "content": f"Research '{research_config.get('research_name', '')}' started.",
        #         "timestamp": now,
        #         "type": "system"
        #     })
        # if callback:
        #     callback(research_id)
