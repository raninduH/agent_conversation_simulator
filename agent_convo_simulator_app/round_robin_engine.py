"""
Round Robin Conversation Engine
Handles agent invocation in a round-robin fashion, with or without voice.
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



class RoundRobinEngine:
    def on_user_message(self, message_data):
        """Handle user message: terminate current thread, add message, and resume cycle."""
        print(f"[RoundRobinEngine] on_user_message called with: {message_data}")
        # Terminate current thread
        self.active = True
        self.paused = True
        # Add user message to conversation
        self._add_message_to_conversation(message_data)
        # Resume cycle in a new thread
        print("[RoundRobinEngine] Restarting round robin cycle after user message...")
        self.active = True
        self.paused = False
        

    def __init__(self, parent_engine):
        self.parent_engine = parent_engine
        self.audio_manager = AudioManager()
        self.lock = threading.Lock()
        self.data_manager = DataManager(os.path.dirname(__file__))        
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
        self.current_agent_index = 0
        self.audio_manager.set_audio_ready_callback(self._on_audio_ready)
        self.audio_manager.set_audio_finished_callback(self._on_audio_finished)
        self.waiting_for_audio = threading.Event()
        self.waiting_for_audio.clear()
        self.last_message = None
        self.ui_callback = None

    def start_cycle(self, conversation_id, agents, voices_enabled, termination_condition, agent_selector_api_key):
        print(f"🚦 [RoundRobin] Starting round robin cycle for conversation: {conversation_id}")
        self.convo_id = conversation_id
        self.convo = self.parent_engine.active_conversations[conversation_id]
        # agents is now a list of agent IDs, so fetch full agent dicts
        self.agents = []
        missing_agents = []
        for agent_id in agents:
            agent_obj = self.data_manager.get_agent_by_id(agent_id)
            if agent_obj:
                self.agents.append(agent_obj if isinstance(agent_obj, dict) else agent_obj.__dict__)
            else:
                missing_agents.append(agent_id)
        if missing_agents:
            print(f"❌ [RoundRobin] Missing agent(s) in DataManager: {missing_agents}")
            raise ValueError(f"Missing agent(s) in DataManager: {missing_agents}")
        self.agent_numbers = self.convo.get("agent_numbers", {})
        self.agent_order = sorted(self.agent_numbers, key=lambda k: self.agent_numbers[k])
        self.voices_enabled = voices_enabled
        self.termination_condition = termination_condition
        self.agent_selector_api_key = agent_selector_api_key
        self.round_count = 0
        self.active = True
        self.paused = False
        self.current_agent_index = 0
        self.last_message = None
        self.ui_callback = self.parent_engine.message_callbacks.get(conversation_id)
        # Ensure LLM_sending_messages exists and is a list
        if "LLM_sending_messages" not in self.convo or not isinstance(self.convo["LLM_sending_messages"], list):
            self.convo["LLM_sending_messages"] = []
        # --- Create all LangGraph agents at the start ---
        self.agent_instances = []
        for agent_id in self.agent_order:
            agent_config = next(a for a in self.agents if a["id"] == agent_id)
            agent_name = agent_config["name"]
            print(f"🤖 [RoundRobin] Initializing agent: {agent_name}")
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
        print(f"✅ [RoundRobin] All agents initialized. Starting round robin thread.")
        threading.Thread(target=self._run_round_robin, daemon=True).start()

    def _run_round_robin(self):
        while self.active:
            if self.paused:
                print("⏸️ [RoundRobin] Paused. Waiting...")
                time.sleep(0.2)
                continue
            agent_id = self.agent_order[self.current_agent_index]
            agent_config = next(a for a in self.agents if a["id"] == agent_id)
            agent_name = agent_config["name"]
            print(f"➡️ [RoundRobin] Invoking agent: {agent_name}")
            should_remind = self._should_remind_termination()
            message = self._invoke_agent(agent_config, should_remind)
            if not message:
                print(f"⚠️ [RoundRobin] No message from agent: {agent_name}. Skipping to next agent.")
                self._next_agent()
                continue
            print(f"📩 [RoundRobin] Message received from {agent_name}: {message['message'][:60]}...")
            if self.voices_enabled and agent_config.get("voice"):
                print(f"🔊 [RoundRobin] Requesting audio for {agent_name}...")
                self.last_message = message
                self.waiting_for_audio.clear()
                # Show loading bubble using _display_message
                loading_message_id = len(self.convo["messages"]) + 1
                loading_message = {
                    "agent_no": agent_config.get('agent_no'),
                    "agent_id": agent_config.get('id'),
                    "agent_name": agent_name,
                    "message_id": loading_message_id,
                    "sender": agent_name,
                    "type": "ai",
                    "timestamp": time.strftime("%H:%M:%S"),
                    "loading": True
                }
                self._display_message(agent_config, loading_message)
                # Request audio and wait for it to be ready
                audio_data = self.audio_manager._generate_audio_sync(message["message"], agent_config["voice"])
                print(f"[AUDIO READY] Audio received for agent: {agent_name}")
                # Remove loading bubble and display actual message
                actual_message = {
                    "agent_no": agent_config.get('agent_no'),
                    "agent_id": agent_config.get('id'),
                    "agent_name": agent_name,
                    "message_id": loading_message_id,
                    "sender": agent_name,
                    "type": "ai",
                    "timestamp": time.strftime("%H:%M:%S"),
                    "message": message["message"],
                    "loading": False
                }
                self._display_message(agent_config, actual_message, blinking=True)
                # Play audio
                if audio_data:
                    self.audio_manager._play_audio(audio_data, {
                        'conversation_id': self.convo_id,
                        'agent_id': agent_name,
                        'message_id': loading_message_id,
                        'text': message["message"],
                        'voice': agent_config["voice"]
                    })
                print(f"✅ [RoundRobin] Audio finished for {agent_name}.")
            else:
                self._add_message_to_conversation(message)
                self._display_message(agent_config, message)
                delay = self._get_turn_delay()
                print(f"⏲️ [RoundRobin] Waiting {delay:.2f} seconds before next agent.")
                time.sleep(delay)
            self._next_agent()
            if self.current_agent_index == 0:
                self.round_count += 1
                print(f"🔄 [RoundRobin] Completed a round. Total rounds: {self.round_count}")
            self._maybe_remind_termination()


    def _invoke_agent(self, agent_config, should_remind=None):
        try:
            agent_name = agent_config["name"]
            print(f"🧠 [RoundRobin] Preparing to invoke agent: {agent_name}")
            agent_entry = next(a for a in self.agent_instances if a["agent_name"] == agent_name)
            agent_variable = agent_entry["agent_variable"]
            # Use LLM_sending_messages for summarization
            llm_messages = self.convo.get("LLM_sending_messages", [])
            self.convo["LLM_sending_messages"] = message_list_summarization(llm_messages)
            # Update LLM_sending_messages with the summarized result
             
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
                agent_obj=agent_config
            )
            print(f"📝 [RoundRobin] Sending prompt to {agent_name} (length: {len(prompt)} chars)")
            config = {"configurable": {"thread_id": f"{thread_id}_{agent_name}"}}
            response = agent_variable.invoke({"messages": [HumanMessage(content=prompt)]}, config)
            if response and "messages" in response and response["messages"]:
                agent_response = response["messages"][-1].content
            else:
                agent_response = f"(No response from {agent_name})"
            print(f"💬 [RoundRobin] {agent_name} responded: {agent_response[:60]}...")
            message = {
                "agent_name": agent_name,
                "message": agent_response,
            }
          
            return message
        except Exception as e:
            print(f"❌ [RoundRobin] Error invoking agent {agent_config['name']}: {e}")
            import traceback
            print(traceback.format_exc())
            return None


    def _add_message_to_conversation(self, message):
        # Remove 'blinking' key before storing
        msg_to_store = dict(message)
        msg_to_store.pop('blinking', None)
        with self.lock:
            if msg_to_store not in self.convo["messages"]:
                self.convo["messages"].append(msg_to_store)
            if msg_to_store not in self.convo["LLM_sending_messages"]:
                self.convo["LLM_sending_messages"].append(msg_to_store)
        self.parent_engine._save_conversation_state(self.convo_id)

    def _handle_voice_for_message(self, agent_config, message):
        voice = agent_config.get("voice")
        if not voice:
            self._add_message_to_conversation(message)
            self._display_message(agent_config, message)
            return
        # Request audio
        self.last_message = message
        if not hasattr(self, 'waiting_for_audio'):
            self.waiting_for_audio = threading.Event()
        self.waiting_for_audio.clear()
        messages = self.convo.get("messages", [])
        message_id = len(messages) + 1  # Use next message ID
        self.audio_manager.request_audio(
            self.convo_id,
            agent_config["id"],
            message_id,
            message["message"],
            voice
        )
        self._display_message(agent_config, message, blinking=True)
        self.waiting_for_audio.wait()  # Wait for audio to finish

    def _display_message(self, agent_config, message, blinking=False):
        ui_callback = self.ui_callback
        # Remove timestamp and message_id if present
        message.pop('timestamp', None)
        message.pop('message_id', None)
        # Add agent_no, agent_id, agent_name to message
        agent_no = agent_config.get('agent_no')
        agent_id = agent_config.get('id')
        agent_name = agent_config.get('name')
        if agent_no is not None:
            message['agent_no'] = agent_no
        if agent_id:
            message['agent_id'] = agent_id
        if agent_name:
            message['agent_name'] = agent_name
        # Add blinking info to message
        message['blinking'] = blinking
        print(f"[RoundRobinEngine] Sending message to UI: {message}")
        if ui_callback:
            ui_callback(message)
        # Only add to conversation once per agent turn
        self._add_message_to_conversation(message)

    def _update_conversation_json_messages(self):
        # Use DataManager to update the messages for the current conversation
        self.data_manager.add_message_to_conversation(self.convo_id, self.convo["messages"][-1])

    def register_message_callback(self, conversation_id, callback):
        """Allow UI or parent engine to register a callback for new messages."""
        if not hasattr(self.parent_engine, 'message_callbacks'):
            self.parent_engine.message_callbacks = {}
        self.parent_engine.message_callbacks[conversation_id] = callback
        self.ui_callback = callback

    def _get_turn_delay(self):
        return random.uniform(CONVERSATION_TIMING["agent_turn_delay_min"], CONVERSATION_TIMING["agent_turn_delay_max"])

    def _should_remind_termination(self):
        return self.termination_condition and (self.round_count % self.termination_reminder_frequency == 0)

    def _maybe_remind_termination(self):
        # Optionally send a reminder message to all agents
        if self.termination_condition and self._should_remind_termination():
            print(f"[RoundRobinEngine] Sending termination condition reminder: {self.termination_condition}")
            # You can implement actual reminder logic here if needed
        else:
            print(f"[RoundRobinEngine] No termination reminder needed this round.")

    def _next_agent(self):
        self.current_agent_index = (self.current_agent_index + 1) % len(self.agent_order)

    def pause_cycle(self, conversation_id):
        print(f"[RoundRobinEngine] pause_cycle called for conversation_id={conversation_id}")
        # Terminate any ongoing round robin thread
        self.active = False
        self.paused = True
        # Save all displayed messages to conversations.json
        if self.convo and "messages" in self.convo:
            print(f"[RoundRobinEngine] Saving displayed messages to conversations.json")
            self.parent_engine._save_conversation_state(conversation_id)
        # Remove all pending audio messages and their audio
        if hasattr(self, 'audio_manager') and hasattr(self.audio_manager, 'pending_audio'):
            print(f"[RoundRobinEngine] Removing pending audio messages")
            self.audio_manager.pending_audio.clear()
        # Remove messages that were waiting for audio and not displayed
        if hasattr(self, 'waiting_for_audio'):
            self.waiting_for_audio.clear()
        # Optionally, remove any messages in convo that were not displayed due to waiting for audio
        self.last_message = None
        print(f"[RoundRobinEngine] pause_cycle complete")

    def resume_cycle(self, conversation_id):
        print(f"[RoundRobinEngine] resume_cycle called for conversation_id={conversation_id}")
        # Ensure previous thread is stopped/paused before rebuilding agents
        self.active = False
        self.paused = True
        # Reload messages from conversations.json
        self.convo = self.parent_engine.active_conversations.get(conversation_id)
        if not self.convo:
            print(f"[RoundRobinEngine] No conversation found for id {conversation_id}")
            return
        messages = self.convo.get("messages", [])
        print(f"[RoundRobinEngine] Loaded {len(messages)} messages from conversations.json")
        # Rebuild agents and agent_order as in start_cycle
        self.agents = []
        missing_agents = []
        for agent_id in self.convo.get("agents", []):
            agent_obj = self.data_manager.get_agent_by_id(agent_id)
            if agent_obj:
                self.agents.append(agent_obj if isinstance(agent_obj, dict) else agent_obj.__dict__)
            else:
                missing_agents.append(agent_id)
        if missing_agents:
            print(f"❌ [RoundRobinEngine] Missing agent(s) in DataManager: {missing_agents}")
        self.agent_numbers = self.convo.get("agent_numbers", {})
        self.agent_order = sorted(self.agent_numbers, key=lambda k: self.agent_numbers[k])
        # Map agent_id to agent_name
        agent_id_to_name = {a["id"]: a["name"] for a in self.agents}
        agent_name_to_id = {a["name"]: a["id"] for a in self.agents}
        # Find last agent who responded
        last_agent_name = None
        for msg in reversed(messages):
            if msg.get("agent_name"):
                last_agent_name = msg["agent_name"]
                break
        print(f"[RoundRobinEngine] Last agent to respond: {last_agent_name}")
        # Find agent_id of last agent
        last_agent_id = agent_name_to_id.get(last_agent_name) if last_agent_name else None
        print(f"[RoundRobinEngine] Last agent id: {last_agent_id}")
        # Find next agent in round robin order
        if last_agent_id and self.agent_order:
            try:
                last_index = self.agent_order.index(last_agent_id)
                next_agent_index = (last_index + 1) % len(self.agent_order)
                self.current_agent_index = next_agent_index
                print(f"[RoundRobinEngine] Next agent index: {self.current_agent_index} ({self.agent_order[self.current_agent_index]})")
            except ValueError:
                print(f"[RoundRobinEngine] Last agent id not found in agent_order, defaulting to 0")
                self.current_agent_index = 0
        else:
            self.current_agent_index = 0
        print(f"[RoundRobinEngine] Ready to invoke next agent: {self.agent_order[self.current_agent_index] if self.agent_order else 'None'}")

        # Rebuild agent_instances
        self.agent_instances = []
        for agent_id in self.agent_order:
            agent_config = next(a for a in self.agents if a["id"] == agent_id)
            agent_name = agent_config["name"]
            print(f"🤖 [RoundRobin] Initializing agent: {agent_name}")
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

        # Now safe to start the thread
        self.active = True
        self.paused = False
        print(f"✅ [RoundRobin] Resuming convo: All agents initialized. Starting round robin thread.")
        threading.Thread(target=self._run_round_robin, daemon=True).start()

    def update_scene_environment(self, conversation_id, environment=None, scene_description=None):
        if environment:
            self.convo["environment"] = environment
        if scene_description:
            self.convo["scene_description"] = scene_description

    def _on_audio_ready(self, conversation_id, agent_id, message_id):
        print(f"[AUDIO READY] Audio received for agent: {agent_id}, message_id: {message_id}")
        # Display chat bubble when audio is ready
        # Find the last message for this agent in the active conversation
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

        # Notify UI to stop blinking
        if hasattr(self.parent_engine, "message_callbacks"):
            callback = self.parent_engine.message_callbacks.get(conversation_id)
            if callback:
                # Send a special signal to UI to stop blinking for this message_id
                callback({
                    "action": "stop_blinking",
                    "agent_id": agent_id,
                    "message_id": message_id
                })
        # Stop blinking in chat_canvas if present
        if hasattr(self.parent_engine, "chat_canvas"):
            try:
                self.parent_engine.chat_canvas.stop_bubble_blink(message_id)
            except Exception:
                pass
        if hasattr(self, 'waiting_for_audio'):
            self.waiting_for_audio.set()