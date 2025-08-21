"""
Agent Selector Conversation Engine
Handles agent invocation using LLM-based agent selection, with or without voice.
"""
import threading
import time
import random
import traceback
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from .config import CONVERSATION_TIMING, AGENT_SETTINGS, MODEL_SETTINGS
from .audio_manager import AudioManager
from .data_manager import DataManager
from .backend_utils import _load_agent_tools, create_agent_base_prompt, create_agent_prompt, message_list_summarization
from langgraph.checkpoint.memory import InMemorySaver
import os
from .agent_selector import AgentSelector

class AgentSelectorEngine:
    def on_user_message(self, message_data):
        print(f"[AgentSelectorEngine] on_user_message called with: {message_data}")
        self.active = True
        self.paused = True
        self._add_message_to_conversation(message_data)
        print("[AgentSelectorEngine] Restarting agent selector cycle after user message...")
        self.active = True
        self.paused = False

    def __init__(self, parent_engine):
        self.parent_engine = parent_engine
        self.audio_manager = AudioManager()
        self.lock = threading.Lock()
        self.data_manager = DataManager()
        self.active = True
        self.paused = False
        self.convo_id = None
        self.convo = None
        self.agents = []
        self.agent_numbers = {}
        self.agent_order = []
        self.voices_enabled = False
        self.termination_condition = None
        self.termination_reminder_frequency = AGENT_SETTINGS["termination_reminder_frequency"]
        self.round_count = 0
        self.agent_selector_api_key = None
        self.audio_manager.set_audio_ready_callback(self._on_audio_ready)
        self.audio_manager.set_audio_finished_callback(self._on_audio_finished)
        self.waiting_for_audio = threading.Event()
        self.waiting_for_audio.clear()
        self.last_message = None
        self.ui_callback = None
        self.selector = None

    def start_cycle(self, conversation_id, agents, voices_enabled, termination_condition, agent_selector_api_key):
        print(f"🚦 [AgentSelectorEngine] Agent selector engine STARTED for conversation: {conversation_id}")
        import threading as _threading
        print(f"🚦 [AgentSelectorEngine] Thread ID: {_threading.current_thread().ident}")
        self.convo_id = conversation_id
        self.convo = self.parent_engine.active_conversations[conversation_id]
        self.agents = []
        missing_agents = []
        for agent_id in agents:
            agent_obj = self.data_manager.get_agent_by_id(agent_id)
            if agent_obj:
                self.agents.append(agent_obj if isinstance(agent_obj, dict) else agent_obj.__dict__)
            else:
                missing_agents.append(agent_id)
        if missing_agents:
            print(f"❌ [AgentSelector] Missing agent(s) in DataManager: {missing_agents}")
            raise ValueError(f"Missing agent(s) in DataManager: {missing_agents}")
        self.agent_numbers = self.convo.get("agent_numbers", {})
        self.agent_order = sorted(self.agent_numbers, key=lambda k: self.agent_numbers[k])
        self.voices_enabled = voices_enabled
        self.termination_condition = termination_condition
        self.agent_selector_api_key = agent_selector_api_key
        self.round_count = 0
        self.active = True
        self.paused = False
        self.last_message = None
        self.ui_callback = self.parent_engine.message_callbacks.get(conversation_id)
        if "LLM_sending_messages" not in self.convo or not isinstance(self.convo["LLM_sending_messages"], list):
            self.convo["LLM_sending_messages"] = []
        self.selector = AgentSelector(google_api_key=agent_selector_api_key)
        self.agent_instances = []
        for agent_id in self.agent_order:
            agent_config = next(a for a in self.agents if a["id"] == agent_id)
            agent_name = agent_config["name"]
            print(f"🤖 [AgentSelector] Initializing agent: {agent_name}")
            agent_tools = _load_agent_tools(agent_name)
            base_prompt = create_agent_base_prompt(agent_config)
            agent_api_key = agent_config.get("api_key") or getattr(self.parent_engine, "default_api_key", None)
            from langchain_google_genai import ChatGoogleGenerativeAI
            agent_model = ChatGoogleGenerativeAI(
                model=MODEL_SETTINGS["agent_model"],
                temperature=AGENT_SETTINGS["response_temperature"],
                max_retries=AGENT_SETTINGS["max_retries"],
                google_api_key=agent_api_key
            )
            agent_variable = create_react_agent(
                model=agent_model,
                tools=agent_tools,
                prompt=base_prompt,
                checkpointer=InMemorySaver()
            )
            self.agent_instances.append({
                "agent_name": agent_name,
                "agent_no": self.agent_numbers[agent_id],
                "agent_variable": agent_variable,
                "config": agent_config
            })
        print(f"✅ [AgentSelectorEngine] All agents initialized. Starting agent selector thread.")
        self._thread = threading.Thread(target=self._run_agent_selector, daemon=True)
        self._thread.start()

    def _run_agent_selector(self):
        print(f"[AgentSelectorEngine] Agent selector main loop started.")
        while self.active:
            if self.paused:
                print("⏸️ [AgentSelectorEngine] Paused. Waiting...")
                time.sleep(0.2)
                return
            print(f"[AgentSelectorEngine] Selecting next agent using LLM...")
            llm_messages = self.convo.get("LLM_sending_messages", [])
            environment = self.convo.get("environment", "")
            scene = self.convo.get("scene_description", "")
            agents_for_selector = [{"name": a["name"], "role": a["role"]} for a in self.agents]
            termination_condition = self.termination_condition
            agent_invocation_counts = None  # Optional: can be tracked if needed
            selector_response = self.selector.select_next_agent(
                llm_messages,
                environment,
                scene,
                agents_for_selector,
                termination_condition,
                agent_invocation_counts
            )
            next_agent_name = selector_response.get("next_response")
            print(f"[AgentSelectorEngine] LLM selected next agent: {next_agent_name}")
            if next_agent_name == "terminate":
                print("[AgentSelectorEngine] Termination condition met. Ending conversation.")
                self.active = False
                break
            # Find agent config and instance
            agent_config = next((a for a in self.agents if a["name"] == next_agent_name), None)
            agent_instance = next((a for a in self.agent_instances if a["agent_name"] == next_agent_name), None)
            if not agent_config or not agent_instance:
                print(f"❌ [AgentSelectorEngine] Agent '{next_agent_name}' not found. Skipping.")
                time.sleep(1)
                continue
            print(f"[AgentSelectorEngine] Invoking agent: {next_agent_name}")
            should_remind = self._should_remind_termination()
            message = self._invoke_agent(agent_config, agent_instance, should_remind)
            if not message:
                print(f"⚠️ [AgentSelectorEngine] No message from agent: {next_agent_name}. Skipping.")
                time.sleep(1)
                continue
            print(f"[AgentSelectorEngine] Message received from {next_agent_name}: {message['message'][:60]}...")
            if self.voices_enabled and agent_config.get("voice"):
                print(f"🔊 [AgentSelectorEngine] Requesting audio for {next_agent_name}...")
                self.last_message = message
                self.waiting_for_audio.clear()
                loading_message_id = len(self.convo["messages"]) + 1
                loading_message = {
                    "agent_no": agent_config.get('agent_no'),
                    "agent_id": agent_config.get('id'),
                    "agent_name": next_agent_name,
                    "message_id": loading_message_id,
                    "sender": next_agent_name,
                    "type": "ai",
                    "timestamp": time.strftime("%H:%M:%S"),
                    "loading": True
                }
                self._display_message(agent_config, loading_message)
                audio_data = self.audio_manager._generate_audio_sync(message["message"], agent_config["voice"])
                if self.paused:
                    loading_message["loading"] = False
                    self._display_message(agent_config, loading_message)
                    time.sleep(0.2)
                    continue
                print(f"[AgentSelectorEngine] Audio received for agent: {next_agent_name}")
                actual_message = {
                    "agent_no": agent_config.get('agent_no'),
                    "agent_id": agent_config.get('id'),
                    "agent_name": next_agent_name,
                    "message_id": loading_message_id,
                    "sender": next_agent_name,
                    "type": "ai",
                    "timestamp": time.strftime("%H:%M:%S"),
                    "message": message["message"],
                    "loading": False
                }
                self._display_message(agent_config, actual_message, blinking=True)
                if audio_data:
                    self.audio_manager._play_audio(audio_data, {
                        'conversation_id': self.convo_id,
                        'agent_id': next_agent_name,
                        'message_id': loading_message_id,
                        'text': message["message"],
                        'voice': agent_config["voice"]
                    })
                print(f"[AgentSelectorEngine] Audio finished for {next_agent_name}.")
            else:
                self._add_message_to_conversation(message)
                self._display_message(agent_config, message)
                delay = self._get_turn_delay()
                print(f"[AgentSelectorEngine] Waiting {delay:.2f} seconds before next agent.")
                time.sleep(delay)
            self.round_count += 1
            self._maybe_remind_termination()

    def _invoke_agent(self, agent_config, agent_instance, should_remind=None):
        try:
            agent_name = agent_config["name"]
            print(f"🧠 [AgentSelector] Preparing to invoke agent: {agent_name}")
            agent_variable = agent_instance["agent_variable"]
            llm_messages = self.convo.get("LLM_sending_messages", [])
            self.convo["LLM_sending_messages"] = message_list_summarization(llm_messages)
            tool_names = agent_config["tools"]
            thread_id = self.convo.get("thread_id")
            if not thread_id:
                import uuid
                thread_id = f"thread_{uuid.uuid4().hex[:8]}"
                self.convo["thread_id"] = thread_id
            prompt = create_agent_prompt(
                agent_config,
                self.convo["environment"],
                self.convo["scene_description"],
                self.convo["LLM_sending_messages"],
                self.agent_order,
                self.termination_condition,
                should_remind_termination=should_remind,
                conversation_id=self.convo_id,
                agent_name=agent_name,
                available_tools=tool_names,
                agent_obj=agent_config            )


            print(f"📝 [AgentSelector] Sending prompt to {agent_name} (length: {len(prompt)} chars)")
            config = {"configurable": {"thread_id": f"{thread_id}_{agent_name}"}}
            response = agent_variable.invoke({"messages": [HumanMessage(content=prompt)]}, config)
            if response and "messages" in response and response["messages"]:
                agent_response = response["messages"][-1].content
            else:
                agent_response = f"(No response from {agent_name})"
            print(f"💬 [AgentSelector] {agent_name} responded: {agent_response[:60]}...")
            message = {
                "agent_name": agent_name,
                "message": agent_response,
            }
            return message
        except Exception as e:
            print(f"❌ [AgentSelector] Error invoking agent {agent_config['name']}: {e}")
            print(traceback.format_exc())
            return None

    def _add_message_to_conversation(self, message):
        msg_to_store = dict(message)
        msg_to_store.pop('blinking', None)
        with self.lock:
            if msg_to_store not in self.convo["messages"]:
                self.convo["messages"].append(msg_to_store)
            if msg_to_store not in self.convo["LLM_sending_messages"]:
                self.convo["LLM_sending_messages"].append(msg_to_store)
        self.parent_engine._save_conversation_state(self.convo_id)

    def _display_message(self, agent_config, message, blinking=False):
        ui_callback = self.ui_callback
        message.pop('timestamp', None)
        message.pop('message_id', None)
        agent_no = agent_config.get('agent_no')
        agent_id = agent_config.get('id')
        agent_name = agent_config.get('name')
        if agent_no is not None:
            message['agent_no'] = agent_no
        if agent_id:
            message['agent_id'] = agent_id
        if agent_name:
            message['agent_name'] = agent_name
        message['blinking'] = blinking
        if 'message' in message and 'content' not in message:
            message['content'] = message['message']
        print(f"[AgentSelectorEngine] Sending message to UI: {message}")
        if ui_callback:
            ui_callback(message)
        self._add_message_to_conversation(message)

    def register_message_callback(self, conversation_id, callback):
        if not hasattr(self.parent_engine, 'message_callbacks'):
            self.parent_engine.message_callbacks = {}
        self.parent_engine.message_callbacks[conversation_id] = callback
        self.ui_callback = callback

    def _get_turn_delay(self):
        return random.uniform(CONVERSATION_TIMING["agent_turn_delay_min"], CONVERSATION_TIMING["agent_turn_delay_max"])

    def _should_remind_termination(self):
        return self.termination_condition and (self.round_count % self.termination_reminder_frequency == 0)

    def _maybe_remind_termination(self):
        if self.termination_condition and self._should_remind_termination():
            print(f"[AgentSelectorEngine] Sending termination condition reminder: {self.termination_condition}")
        else:
            print(f"[AgentSelectorEngine] No termination reminder needed this round.")

    def pause_cycle(self, conversation_id):
        print(f"[AgentSelectorEngine] pause_cycle called for conversation_id={conversation_id}")
        self.active = False
        self.paused = True
        if self.convo and "messages" in self.convo:
            print(f"[AgentSelectorEngine] Saving displayed messages to conversations.json")
            self.parent_engine._save_conversation_state(conversation_id)
        if hasattr(self, 'audio_manager') and hasattr(self.audio_manager, 'pending_audio'):
            print(f"[AgentSelectorEngine] Removing pending audio messages")
            self.audio_manager.pending_audio.clear()
        if hasattr(self, 'waiting_for_audio'):
            self.waiting_for_audio.clear()
        self.last_message = None
        print(f"[AgentSelectorEngine] pause_cycle complete")

    def resume_cycle(self, conversation_id):
        import time as _time
        import threading as _threading
        self.ui_callback = self.parent_engine.message_callbacks.get(conversation_id)
        print(f"[AgentSelectorEngine] resume_cycle called for conversation_id={conversation_id} (thread: {_threading.current_thread().ident})")
        self.active = False
        self.paused = True
        if hasattr(self, '_thread') and self._thread is not None:
            if self._thread.is_alive():
                print("[AgentSelectorEngine] Waiting for previous agent selector thread to finish...")
                self._thread.join(timeout=5)
                if self._thread.is_alive():
                    print("[AgentSelectorEngine] Warning: Previous thread did not finish in time.")
        self.convo_id = conversation_id
        print(f"[AgentSelectorEngine] _run_agent_selector started (thread: {_threading.current_thread().ident})")
        self.convo = self.parent_engine.active_conversations.get(conversation_id)
        if not self.convo:
            print(f"[AgentSelectorEngine] No conversation found for id {conversation_id}")
            return
        messages = self.convo.get("messages", [])
        print(f"[AgentSelectorEngine] Loaded {len(messages)} messages from conversations.json")
        self.agents = []
        missing_agents = []
        self.agent_numbers = self.convo.get("agent_numbers", {})
        self.agent_order = sorted(self.agent_numbers, key=lambda k: self.agent_numbers[k])
        for agent_id in self.convo.get("agents", []):
            agent_obj = self.data_manager.get_agent_by_id(agent_id)
            if agent_obj:
                agent_dict = agent_obj if isinstance(agent_obj, dict) else agent_obj.__dict__
                agent_dict["agent_no"] = self.agent_numbers.get(agent_id)
                self.agents.append(agent_dict)
            else:
                missing_agents.append(agent_id)
        if missing_agents:
            print(f"❌ [AgentSelectorEngine] Missing agent(s) in DataManager: {missing_agents}")
        self.selector = AgentSelector(google_api_key=self.agent_selector_api_key)
        self.agent_instances = []
        for agent_id in self.agent_order:
            agent_config = next(a for a in self.agents if a["id"] == agent_id)
            agent_name = agent_config["name"]
            print(f"🤖 [AgentSelector] Initializing agent: {agent_name}")
            agent_tools = _load_agent_tools(agent_name)
            base_prompt = create_agent_base_prompt(agent_config)
            agent_api_key = agent_config.get("api_key") or getattr(self.parent_engine, "default_api_key", None)
            from langchain_google_genai import ChatGoogleGenerativeAI
            agent_model = ChatGoogleGenerativeAI(
                model=MODEL_SETTINGS["agent_model"],
                temperature=AGENT_SETTINGS["response_temperature"],
                max_retries=AGENT_SETTINGS["max_retries"],
                google_api_key=agent_api_key
            )
            agent_variable = create_react_agent(
                model=agent_model,
                tools=agent_tools,
                prompt=base_prompt,
                checkpointer=InMemorySaver()
            )
            self.agent_instances.append({
                "agent_name": agent_name,
                "agent_no": self.agent_numbers[agent_id],
                "agent_variable": agent_variable,
                "config": agent_config
            })
            print({
                "agent_name": agent_name,
                "agent_no": self.agent_numbers[agent_id],
                "agent_variable": agent_variable,
                "config": agent_config
            })
        _time.sleep(20)
        self.active = True
        self.paused = False
        self.voices_enabled = self.convo.get("voices_enabled", False)
        print(f"✅ [AgentSelector] Resuming convo: All agents initialized. Starting agent selector thread.")
        self._thread = threading.Thread(target=self._run_agent_selector, daemon=True)
        self._thread.start()

    def update_scene_environment(self, conversation_id, environment=None, scene_description=None):
        if environment:
            self.convo["environment"] = environment
        if scene_description:
            self.convo["scene_description"] = scene_description

    def _on_audio_ready(self, conversation_id, agent_id, message_id):
        print(f"[AUDIO READY] Audio received for agent: {agent_id}, message_id: {message_id}")
        convo = self.parent_engine.active_conversations.get(conversation_id)
        if not convo or not convo.get("messages"):
            return
        for msg in reversed(convo["messages"]):
            if msg.get("agent_name") == agent_id or msg.get("sender") == agent_id:
                agent_config = next((a for a in self.agents if a["name"] == agent_id), None)
                if agent_config:
                    self._display_message(agent_config, msg, blinking=True)
                break

    def _on_audio_finished(self, conversation_id, agent_id, message_id):
        print(f"[AUDIO FINISHED] Audio finished for agent: {agent_id}, message_id: {message_id}")
        if hasattr(self.parent_engine, "message_callbacks"):
            callback = self.parent_engine.message_callbacks.get(conversation_id)
            if callback:
                callback({
                    "action": "stop_blinking",
                    "agent_id": agent_id,
                    "message_id": message_id
                })
        if hasattr(self.parent_engine, "chat_canvas"):
            try:
                self.parent_engine.chat_canvas.stop_bubble_blink(message_id)
            except Exception:
                pass
        if hasattr(self, 'waiting_for_audio'):
            self.waiting_for_audio.set()