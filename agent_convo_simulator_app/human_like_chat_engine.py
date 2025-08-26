"""
Human-Like Chat Conversation Engine
Handles agent invocation in a human-like chat fashion, with or without voice.
"""
import threading
import time
import random
import traceback
import json
import re
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from .config import CONVERSATION_TIMING, AGENT_SETTINGS, MODEL_SETTINGS
from .audio_manager import AudioManager
from .data_manager import DataManager
from .backend_utils import _load_agent_tools, create_agent_base_prompt, create_agent_prompt, message_list_summarization
from langgraph.checkpoint.memory import InMemorySaver
import os
from .agent_selector import AgentSelector

class HumanLikeChatEngine:
    def on_user_message(self, message_data):
        print(f"[HumanLikeChatEngine] on_user_message called with: {message_data}")
        self.active = True
        self.paused = True
        self._add_message_to_conversation(message_data)
        print("[HumanLikeChatEngine] Restarting chat cycle after user message...")
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
        self.agent_instances = []
        self.agents_last_seen_messages = {}
        self.is_txt_n_audio_playing = False

    def _extract_json(self, text: str):
        """Extract JSON from the response text, handling different formats."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                json_match = re.search(r'({.*?})', text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                    return json.loads(json_text)
                markdown_match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
                if markdown_match:
                    json_text = markdown_match.group(1).strip()
                    return json.loads(json_text)
            except Exception:
                pass
            return {"next_response": "error_parsing"}

    def start_cycle(self, conversation_id, agents, voices_enabled, termination_condition, agent_selector_api_key):
        print(f"🚦 [HumanLikeChatEngine] Chat engine STARTED for conversation: {conversation_id}")
        import threading as _threading
        print(f"🚦 [HumanLikeChatEngine] Thread ID: {_threading.current_thread().ident}")
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
            print(f"❌ [HumanLikeChatEngine] Missing agent(s) in DataManager: {missing_agents}")
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
            self.agents_last_seen_messages[agent_name] = None
            print(f"🤖 [HumanLikeChatEngine] Initializing agent: {agent_name}")
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
                prompt=base_prompt
            )
            self.agent_instances.append({
                "agent_name": agent_name,
                "agent_no": self.agent_numbers[agent_id],
                "agent_variable": agent_variable,
                "config": agent_config
            })
        print(f"✅ [HumanLikeChatEngine] All agents initialized. Starting chat thread.")
        self._thread = threading.Thread(target=self._run_human_like_chat, daemon=True)
        self._thread.start()

    def _run_human_like_chat(self):
        print(f"[HumanLikeChatEngine] Main loop started.")
        # Initial agent selection using LLM
        llm_messages = self.convo.get("LLM_sending_messages", [])
        environment = self.convo.get("environment", "")
        scene = self.convo.get("scene_description", "")
        agents_for_selector = [{"name": a["name"], "role": a["role"]} for a in self.agents]
        termination_condition = self.termination_condition
        agent_invocation_counts = None
        selector_response = self.selector.select_next_agent(
            llm_messages,
            environment,
            scene,
            agents_for_selector,
            termination_condition,
            agent_invocation_counts
        )
        next_agent_name = selector_response.get("next_response")
        print(f"[HumanLikeChatEngine] LLM selected initial agent: {next_agent_name}")
        agent_config = next((a for a in self.agents if a["name"] == next_agent_name), None)
        agent_instance = next((a for a in self.agent_instances if a["agent_name"] == next_agent_name), None)
        if not agent_config or not agent_instance:
            print(f"❌ [HumanLikeChatEngine] Initial agent '{next_agent_name}' not found. Aborting.")
            return
        self._invoke_and_handle_agent(agent_config, agent_instance)
        while self.active:
            if self.paused:
                print("⏸️ [HumanLikeChatEngine] Paused. Waiting...")
                time.sleep(0.2)
                return
            # Wait for the last agent's message to finish (voice or not)
            # After a message is received, invoke all other agents in parallel
            last_message = self.last_message
        
            last_agent_name = last_message.get("agent_name")
            print(f"[HumanLikeChatEngine] Last agent to respond: {last_agent_name}")
            threads = []
            for agent_instance in self.agent_instances:
                agent_name = agent_instance["agent_name"]
                if agent_name == last_agent_name:
                    continue
                agent_config = agent_instance["config"]
                t = threading.Thread(target=self._invoke_and_handle_agent, args=(agent_config, agent_instance))
                threads.append(t)

            # If voice is not enabled, delay before parallel execution
            if not self.voices_enabled:
                delay = self._get_turn_delay()
                print(f"[HumanLikeChatEngine] Waiting {delay:.2f} seconds before parallel agent invocation.")
                time.sleep(delay)
            # Start all threads
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.round_count += 1
            self._maybe_remind_termination()
            print(f"[HumanLikeChatEngine]: ended round {self.round_count-1}")

    def _invoke_and_handle_agent(self, agent_config, agent_instance):
        try:
            agent_name = agent_config["name"]
            print(f"🧠 [HumanLikeChatEngine] Invoking agent: {agent_name}")
            prompt = self._build_human_like_prompt(
                agent_config, 
                self.convo["environment"],
                self.convo["scene_description"],
                self.agent_order,
                self.termination_condition,
                self._should_remind_termination())
            
            agent_variable = agent_instance["agent_variable"]
            config = {"configurable": {"thread_id": f"{self.convo_id}_{agent_name}"}}
            response = agent_variable.invoke({"messages": [HumanMessage(content=prompt)]}, config)
            if response and "messages" in response and response["messages"]:
                agent_response = response["messages"][-1].content
            else:
                agent_response = f"(No response from {agent_name})"
            print(f"💬 [HumanLikeChatEngine] {agent_name} raw response: {agent_response[:60]}...")
            json_result = self._extract_json(agent_response)
            is_responding = json_result.get("is_responding", "no")
            response_text = json_result.get("response", None)
            if is_responding == "yes" and response_text:
                print(f"[HumanLikeChatEngine] {agent_name} is responding: {response_text[:60]}...")
                message = {
                    "agent_name": agent_name,
                    "message": response_text,
                }
                self.last_message = message
                if self.voices_enabled and agent_config.get("voice"):
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
                    audio_data = self.audio_manager._generate_audio_sync(response_text, agent_config["voice"])
                    if self.paused:
                        loading_message["loading"] = False
                        self._display_message(agent_config, loading_message)
                        time.sleep(0.2)
                        return
                    print(f"[HumanLikeChatEngine] Audio received for agent: {agent_name}")
                    actual_message = {
                        "agent_no": agent_config.get('agent_no'),
                        "agent_id": agent_config.get('id'),
                        "agent_name": agent_name,
                        "message_id": loading_message_id,
                        "sender": agent_name,
                        "type": "ai",
                        "timestamp": time.strftime("%H:%M:%S"),
                        "message": response_text,
                        "loading": False
                    }
                    while self.is_txt_n_audio_playing:
                        time.sleep(0.5)  # Wait until audio is ready to play

                    if self.paused:
                        return
                    
                    # lock other threads from doing this
                    self.is_txt_n_audio_playing = True

                    self._display_message(agent_config, actual_message, blinking=True)
                    if audio_data:
                        self.audio_manager._play_audio(audio_data, {
                            'conversation_id': self.convo_id,
                            'agent_id': agent_name,
                            'message_id': loading_message_id,
                            'text': response_text,
                            'voice': agent_config["voice"]
                        })
                    print(f"✅ [HumanLikeChatEngine] Audio finished for {agent_name}.")

                    # Unlock the chat and audio for other threads
                    self.is_txt_n_audio_playing = False

                else:
                    while self.is_txt_n_audio_playing:
                        time.sleep(1) 
                    
                    # delay a bit between parallel message displays
                    self.is_txt_n_audio_playing = True

                    time.sleep(5)

                    self._display_message(agent_config, message)
             
                    self.is_txt_n_audio_playing = False

                    

            else:
                print(f"[HumanLikeChatEngine] {agent_name} chose not to respond.")
        except Exception as e:
            print(f"❌ [HumanLikeChatEngine] Error invoking agent {agent_config['name']}: {e}")
            print(traceback.format_exc())

    def _build_human_like_prompt(self, agent_config, environment, scene_description, all_agents, termination_condition=None, should_remind_termination=False):
        """
        Create the prompt for an agent including scene, participants, and conversation history.
        """

        self.convo["LLM_sending_messages"] = message_list_summarization(self.convo.get("LLM_sending_messages", []))
        messages = self.convo.get("LLM_sending_messages", [])

        agent_name = agent_config["name"]
    
        prompt = f"""
                Always answer based on the given characteristics of yourself. Stay in character always.
                INITIAL environment: {environment}
                SCENE DESCRIPTION: {scene_description}
                \nPARTICIPANTS: {', '.join(all_agents)}\n\nTool Usage: Use your tools freely in the first instance you feel,  just like a noraml person using their mobile phone as a tool. No need to get permsission from other agents. But when it's necessary discuss with other agents how the tools should be used.\n\n"""
        
        # Always use the current messages list as the single source of truth
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
        if should_remind_termination and termination_condition:
            prompt += f"""TERMINATION CONDITION REMINDER: The conversation should end when the following condition is met:\n{termination_condition}\n\nKeep this condition in mind while participating in the conversation. Naturally deviate the conversation into the direction where the condition will be met. and stay true to your personality traits.\n\n"""
        if agent_config["tools"]:
            prompt += f"""AVAILABLE TOOLS: You have access to the following tools: {', '.join(agent_config["tools"])}\nUse these tools when they can help you respond more effectively to the conversation.\nOnly use tools when they are relevant to the current conversation context.\nDon't mention the tools explicitly unless asked about your capabilities.\n\n"""
        if agent_config and hasattr(agent_config, 'knowledge_base') and agent_config.knowledge_base:
            knowledge_descriptions = []
            for doc in agent_config.knowledge_base:
                if hasattr(doc, 'metadata') and 'description' in doc.metadata:
                    knowledge_descriptions.append(doc.metadata['description'])
            prompt += f"""PERSONAL KNOWLEDGE BASE: You have access to a personal knowledge base containing the following documents:\n{chr(10).join(knowledge_descriptions)}\n\nUse the knowledge_base_retriever tool to search through these documents when relevant to the conversation. \nThis knowledge base contains specialized information that can help you stay true to your role and provide more informed responses.\nOnly search your knowledge base when the conversation topic relates to the content of your documents.\n\n"""
       
        if self.agents_last_seen_messages[agent_name]: 
            last_seen_message = self.agents_last_seen_messages[agent_name]
            truncated_last_seen_message_text = ' '.join(last_seen_message['message'].split()[:10])
            prompt += f"""The last message you saw was: '{truncated_last_seen_message_text}...' by {last_seen_message['agent_name']}
                        You might've sent a message after the above message. But you haven't seen before any of the messages sent by other agents after the above one.
                        So when responding/replying, focus on all the new messages you just saw.
                    \n"""
            
        prompt += f"""if you want or feel like it or you are needed to or if you can valuably contribute to the conversation Give your response to the ongoing conversation as {agent_name} , otherwise no need to send a response.
                    But if they were no previous messages based on the scene start the conversation pls. But all you responses should come under the necessary key in the JSON output.
                    Only output a JSON of the following format. do not output anything else.
                    {{
                        is_responding: "yes"/"no", : "no" if you decide not to respond
                        response: None/string : None if you decide not to respond
                    }}

        """
        prompt += f""" \nKeep your response natural, conversational, and true to your character. Always respons with the charateristics/personality of your character. \nRespond as if you're speaking directly in the conversation (don't say \"As {agent_name}, I would say...\" just respond naturally).\nRespond only to the dialog parts said by the other agents.\nKeep responses to 1-3 sentences to maintain good conversation flow. And don't mention the names of tools because all the other agents might not have that tool. Only suggest the act of the tool and NOT the name."""
        

        if messages:
            if len(recent_messages) > 0:
                self.agents_last_seen_messages[agent_name] = recent_messages[-1]
                
        return prompt


    def _add_message_to_conversation(self, message):
        print(f"[HumanLikeChatEngine] Adding message to conversation: {message}")
        if message.get('message') or message.get('content'):
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
        print(f"[HumanLikeChatEngine] Sending message to UI: {message}")
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
            print(f"[HumanLikeChatEngine] Sending termination condition reminder: {self.termination_condition}")
        else:
            print(f"[HumanLikeChatEngine] No termination reminder needed this round.")

    def pause_cycle(self, conversation_id):
        print(f"[HumanLikeChatEngine] pause_cycle called for conversation_id={conversation_id}")
        self.active = False
        self.paused = True
        if self.convo and "messages" in self.convo:
            print(f"[HumanLikeChatEngine] Saving displayed messages to conversations.json")
            self.parent_engine._save_conversation_state(conversation_id)
        if hasattr(self, 'audio_manager') and hasattr(self.audio_manager, 'pending_audio'):
            print(f"[HumanLikeChatEngine] Removing pending audio messages")
            self.audio_manager.pending_audio.clear()
        if hasattr(self, 'waiting_for_audio'):
            self.waiting_for_audio.clear()
        self.last_message = None
        print(f"[HumanLikeChatEngine] pause_cycle complete")

    def resume_cycle(self, conversation_id):
        import time as _time
        import threading as _threading
        self.ui_callback = self.parent_engine.message_callbacks.get(conversation_id)
        print(f"[HumanLikeChatEngine] resume_cycle called for conversation_id={conversation_id} (thread: {_threading.current_thread().ident})")
        self.active = False
        self.paused = True
        if hasattr(self, '_thread') and self._thread is not None:
            if self._thread.is_alive():
                print("[HumanLikeChatEngine] Waiting for previous chat thread to finish...")
                self._thread.join(timeout=5)
                if self._thread.is_alive():
                    print("[HumanLikeChatEngine] Warning: Previous thread did not finish in time.")
        self.convo_id = conversation_id
        print(f"[HumanLikeChatEngine] _run_human_like_chat started (thread: {_threading.current_thread().ident})")
        self.convo = self.parent_engine.active_conversations.get(conversation_id)
        if not self.convo:
            print(f"[HumanLikeChatEngine] No conversation found for id {conversation_id}")
            return
        messages = self.convo.get("messages", [])
        print(f"[HumanLikeChatEngine] Loaded {len(messages)} messages from conversations.json")
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
            print(f"❌ [HumanLikeChatEngine] Missing agent(s) in DataManager: {missing_agents}")
        self.selector = AgentSelector(google_api_key=self.agent_selector_api_key)
        self.agent_instances = []
        for agent_id in self.agent_order:
            agent_config = next(a for a in self.agents if a["id"] == agent_id)
            agent_name = agent_config["name"]
            self.agents_last_seen_messages[agent_name] = None
            print(f"🤖 [HumanLikeChatEngine] Initializing agent: {agent_name}")
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
        print(f"✅ [HumanLikeChatEngine] Resuming convo: All agents initialized. Starting chat thread.")
        self._thread = threading.Thread(target=self._run_human_like_chat, daemon=True)
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
