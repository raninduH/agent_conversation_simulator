"""
Human-Like Chat Conversation Engine
Handles parallel agent invocation with audio generation fallback mechanism.
"""

import os
import threading
import random
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from .agent_selector import AgentSelector
from .config import MODEL_SETTINGS, AGENT_SETTINGS, CONVERSATION_TIMING


class HumanLikeChatEngine:
    """Handles human-like-chat conversation logic with parallel/sequential audio fallback."""
    
    def __init__(self, conversation_engine):
        """Initialize with reference to main conversation engine."""
        self.conversation_engine = conversation_engine
    
    def start_cycle(self, conversation_id: str):
        """Start the human-like-chat conversation cycle."""
        convo = self.conversation_engine.active_conversations.get(conversation_id)
        if not convo or convo["stop_conversation"]:
            return

        print(f"DEBUG: ===== STARTING HUMAN-LIKE-CHAT CONVERSATION ======")
        self.start_human_like_chat_round(conversation_id)

    def on_voice_audio_finished(self, conversation_id: str):
        """Callback for when voice audio finishes playing."""
        convo = self.conversation_engine.active_conversations.get(conversation_id)
        if not convo or not convo.get("voices_enabled"):
            return

        convo["is_audio_playing"] = False
        self._process_audio_queue(conversation_id)

    def invoke_next_agent(self, conversation_id: str):
        """In human-like chat, the round is managed by `start_human_like_chat_round`, not individual agent invocations."""
        print("DEBUG: [HUMAN-LIKE-CHAT] `invoke_next_agent` called, but logic is handled by rounds. Ignoring.")
        pass
    
    def start_human_like_chat_round(self, conversation_id: str):
        """Start a new round in human-like-chat mode."""
        convo = self.conversation_engine.active_conversations.get(conversation_id)
        if not convo or convo["stop_conversation"]:
            return
        
        # If audio is still playing from the previous round, wait before starting a new one.
        if convo.get("is_audio_playing") or (convo.get("voices_enabled") and convo.get("audio_queue")):
            print("DEBUG: [HUMAN-LIKE-CHAT] Waiting for audio queue to finish before starting new round.")
            threading.Timer(1.0, self.start_human_like_chat_round, args=[conversation_id]).start()
            return

        print("DEBUG: [HUMAN-LIKE-CHAT] Starting human-like-chat round")
        
        convo["round_count"] += 1
        current_round = convo["round_count"]
        
        if current_round > 1 and self.conversation_engine._check_round_robin_termination(conversation_id):
            print("DEBUG: [HUMAN-LIKE-CHAT] Termination condition met")
            if conversation_id in self.conversation_engine.message_callbacks:
                self.conversation_engine.message_callbacks[conversation_id]({
                    "sender": "System",
                    "content": "The conversation has reached its termination condition and has ended.",
                    "timestamp": datetime.now().isoformat(),
                    "type": "system"
                })
            self.conversation_engine.stop_conversation(conversation_id)
            return

        agent_selector = AgentSelector(google_api_key=convo["agent_selector_api_key"])
        
        num_participants = random.randint(1, min(3, len(convo["all_agents"])))
        print(f"DEBUG: [HUMAN-LIKE-CHAT] Round {current_round} will have {num_participants} participants")
        
        round_participants = []
        for _ in range(num_participants):
            selection_result = agent_selector.select_next_agent(
                messages=convo["messages"],
                environment=convo["environment"],
                scene=convo["scene_description"],
                agents=[convo["agents_config"][name] for name in convo["all_agents"]],
                termination_condition=convo.get("termination_condition"),
                agent_invocation_counts={name: 0 for name in convo["all_agents"]} # Simplified for this selection
            )
            
            selected_agent = selection_result.get("next_response", "error_parsing")
            if (selected_agent != "error_parsing" and 
                selected_agent in convo["all_agents"] and 
                selected_agent not in round_participants):
                round_participants.append(selected_agent)
            else:
                available_agents = [name for name in convo["all_agents"] if name not in round_participants]
                if available_agents:
                    round_participants.append(random.choice(available_agents))
        
        print(f"DEBUG: [HUMAN-LIKE-CHAT] Round {current_round} participants: {round_participants}")
        
        print(f"🟢 [HUMAN-LIKE-CHAT] Starting round {current_round} with participants: {round_participants}")
        self.invoke_agents_parallel_for_round(conversation_id, round_participants)

    def invoke_agents_parallel_for_round(self, conversation_id: str, participants: List[str]):
        """Invoke multiple agents in parallel for a human-like-chat round."""
        print(f"🤖 [HUMAN-LIKE-CHAT] Invoking {len(participants)} agents in parallel: {participants}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(participants)) as executor:
            future_to_agent = {executor.submit(self.invoke_single_agent_for_round, conversation_id, name): name for name in participants}
            completed_responses = []

            for future in concurrent.futures.as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    response = future.result()
                    if response:
                        print(f"💬 [HUMAN-LIKE-CHAT] Received response from agent '{agent_name}': '{response[:80]}...'")
                        completed_responses.append((agent_name, response))
                except Exception as exc:
                    print(f"❌ [HUMAN-LIKE-CHAT] Agent '{agent_name}' generated an exception: {exc}")

            print(f"📦 [HUMAN-LIKE-CHAT] All agent responses received for this round. Handling responses...")
            for agent_name, response in completed_responses:
                self.handle_parallel_agent_response(conversation_id, agent_name, response)

        convo = self.conversation_engine.active_conversations.get(conversation_id)
        if convo and not convo["stop_conversation"]:
            if not convo.get("voices_enabled"):
                delay = random.uniform(CONVERSATION_TIMING['min_pause_seconds'], CONVERSATION_TIMING['max_pause_seconds'])
                print(f"⏳ [HUMAN-LIKE-CHAT] Round complete. Waiting {delay:.2f} seconds before next round.")
                threading.Timer(delay, self.start_human_like_chat_round, args=[conversation_id]).start()
            else:
                print(f"🔊 [HUMAN-LIKE-CHAT] Voices enabled. Waiting for audio queue to finish before next round.")
                self._process_audio_queue(conversation_id)

    def invoke_single_agent_for_round(self, conversation_id: str, agent_name: str) -> Optional[str]:
        print(f"🧑‍💻 [HUMAN-LIKE-CHAT] Invoking agent: {agent_name}")
        convo = self.conversation_engine.active_conversations.get(conversation_id)
        if not convo or convo["stop_conversation"]:
            return None

        agent = self.conversation_engine._get_agent(conversation_id, agent_name)
        agent_config = convo["agents_config"][agent_name]
        
        prompt = self.conversation_engine.create_agent_prompt(
            agent_config=agent_config,
            environment=convo["environment"],
            scene_description=convo["scene_description"],
            messages=convo["messages"],
            all_agents=convo["all_agents"],
            termination_condition=convo["termination_condition"],
            agent_name=agent_name,
            available_tools=[t.name for t in agent.tools]
        )

        try:
            response = agent.invoke({"messages": [HumanMessage(content=prompt)]})
            if response and "messages" in response and response["messages"]:
                print(f"✅ [HUMAN-LIKE-CHAT] Agent '{agent_name}' responded with message.")
                return response["messages"][-1].content
        except Exception as e:
            print(f"❌ [HUMAN-LIKE-CHAT] Error invoking agent '{agent_name}': {e}")
        return None

    def handle_parallel_agent_response(self, conversation_id: str, agent_name: str, response: str):
        print(f"📥 [HUMAN-LIKE-CHAT] Handling response from agent '{agent_name}'")
        convo = self.conversation_engine.active_conversations.get(conversation_id)
        if not convo or convo["stop_conversation"]:
            return

        message_data = {
            "agent_name": agent_name,
            "message": response,
            "timestamp": datetime.now().isoformat()
        }
        convo["messages"].append(message_data)

        ui_message = {
            "sender": agent_name,
            "content": response,
            "timestamp": message_data["timestamp"],
            "type": "ai"
        }

        if convo.get("voices_enabled"):
            if "audio_queue" not in convo:
                convo["audio_queue"] = []
            convo["audio_queue"].append(ui_message)
            print(f"🔈 [HUMAN-LIKE-CHAT] Queued message from {agent_name} for audio playback. Queue size: {len(convo['audio_queue'])}")
            print(f"🎤 [HUMAN-LIKE-CHAT] Audio generation requested for {agent_name}")
            self._process_audio_queue(conversation_id)
        else:
            if conversation_id in self.conversation_engine.message_callbacks:
                print(f"📤 [HUMAN-LIKE-CHAT] Sending message from {agent_name} to UI (no audio mode).")
                self.conversation_engine.message_callbacks[conversation_id](ui_message)
        self.conversation_engine._save_conversation_state(conversation_id)

    def _process_audio_queue(self, conversation_id: str):
        convo = self.conversation_engine.active_conversations.get(conversation_id)
        if not convo or not convo.get("voices_enabled"):
            return
        if convo.get("is_audio_playing"):
            print(f"⏸️ [HUMAN-LIKE-CHAT] Audio is currently playing. Waiting... (Current: {convo.get('current_audio_agent', 'None')})")
            return
        if convo.get("audio_queue"):
            convo["is_audio_playing"] = True
            message_to_play = convo["audio_queue"].pop(0)
            convo["current_audio_agent"] = message_to_play['sender']
            print(f"🔊 [HUMAN-LIKE-CHAT] Now playing audio for agent: {message_to_play['sender']} (Queue size: {len(convo['audio_queue'])})")
            print(f"🎧 [HUMAN-LIKE-CHAT] Audio playback started for {message_to_play['sender']}")
            if conversation_id in self.conversation_engine.message_callbacks:
                self.conversation_engine.message_callbacks[conversation_id](message_to_play)
        else:
            print(f"🏁 [HUMAN-LIKE-CHAT] Audio queue is empty. Round ended. Starting next round.")
            convo["current_audio_agent"] = None
            delay = random.uniform(CONVERSATION_TIMING['min_pause_seconds'], CONVERSATION_TIMING['max_pause_seconds'])
            threading.Timer(delay, self.start_human_like_chat_round, args=[conversation_id]).start()
