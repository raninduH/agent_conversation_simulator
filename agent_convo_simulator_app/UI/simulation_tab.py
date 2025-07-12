import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime

from .chat_widgets import ChatCanvas
from ..config import UI_COLORS
from ..conversation_engine import ConversationEngine
from ..data_manager import Conversation, Agent

class SimulationTab(ttk.Frame):
    def __init__(self, parent, app, data_manager):
        super().__init__(parent)
        self.app = app
        self.data_manager = data_manager
        # Use new ConversationEngine
        self.app.conversation_engine = ConversationEngine()
        self.create_widgets()
        self.blinking_messages = {}

    def create_widgets(self):
        """Create the conversation simulation tab."""
        sim_frame = self
        
        # Configure grid
        sim_frame.grid_rowconfigure(1, weight=1)
        sim_frame.grid_columnconfigure(0, weight=1)
        
        # Title and controls
        header_frame = ttk.Frame(sim_frame)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        header_frame.grid_columnconfigure(1, weight=1)
        
        title_label = ttk.Label(header_frame, text="Conversation Simulation", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, sticky="w")
        
        # Control buttons
        control_frame = ttk.Frame(header_frame)
        control_frame.grid(row=0, column=2, sticky="e")
        
        self.app.pause_btn = ttk.Button(control_frame, text="Pause", command=self.app.pause_conversation, state="disabled")
        self.app.pause_btn.pack(side=tk.LEFT, padx=2)
        
        self.app.resume_btn = ttk.Button(control_frame, text="Resume", command=self.app.resume_conversation, state="disabled")
        self.app.resume_btn.pack(side=tk.LEFT, padx=2)
        
        self.app.summarize_btn = ttk.Button(control_frame, text="Summarize", command=self.app.summarize_conversation, state="disabled")
        self.app.summarize_btn.pack(side=tk.LEFT, padx=2)
        
        self.app.stop_btn = ttk.Button(control_frame, text="Stop", command=self.app.stop_conversation_ui, state="disabled")
        self.app.stop_btn.pack(side=tk.LEFT, padx=2)
        
        # Main content area
        content_frame = ttk.Frame(sim_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        
        # Chat display with bubbles
        chat_frame = ttk.LabelFrame(content_frame, text="Conversation", padding="10")
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)
        
        # Create chat bubble canvas
        chat_container = ttk.Frame(chat_frame)
        chat_container.grid(row=0, column=0, sticky="nsew")
        chat_container.grid_rowconfigure(0, weight=1)
        chat_container.grid_columnconfigure(0, weight=1)
        
        self.app.chat_canvas = ChatCanvas(chat_container, bg=UI_COLORS["chat_background"])
        self.app.chat_canvas.pack(side="left", fill="both", expand=True)
        
        # Also create a reference for direct access from this tab
        self.chat_canvas = self.app.chat_canvas
        
        # Dictionary to store agent colors
        self.app.agent_colors = {}
        
        # Input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.app.message_var = tk.StringVar()
        self.app.message_entry = ttk.Entry(input_frame, textvariable=self.app.message_var)
        self.app.message_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.app.message_entry.bind('<Return>', self.app.send_user_message)
        
        self.app.send_btn = ttk.Button(input_frame, text="Send", command=self.app.send_user_message, state="disabled")
        self.app.send_btn.grid(row=0, column=1)
        
        # Scene control panel
        scene_frame = ttk.LabelFrame(content_frame, text="Scene Control", padding="10")
        scene_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        scene_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Label(scene_frame, text="Current Environment:").grid(row=0, column=0, sticky="w", pady=2)
        self.app.current_env_label = ttk.Label(scene_frame, text="None", font=("Arial", 10, "italic"))
        self.app.current_env_label.grid(row=1, column=0, sticky="w", pady=(0, 10))
        
        # Also create a reference for direct access from this tab
        self.current_env_label = self.app.current_env_label
        
        # Dictionary to store agent colors
        self.agent_colors = self.app.agent_colors
        
        ttk.Label(scene_frame, text="New Environment:").grid(row=2, column=0, sticky="w", pady=2)
        self.app.new_env_var = tk.StringVar()
        self.app.new_env_entry = ttk.Entry(scene_frame, textvariable=self.app.new_env_var, width=25)
        self.app.new_env_entry.grid(row=3, column=0, sticky="ew", pady=2)
        
        ttk.Label(scene_frame, text="New Scene:").grid(row=4, column=0, sticky="w", pady=(10, 2))
        self.app.new_scene_text = scrolledtext.ScrolledText(scene_frame, width=25, height=8)
        self.app.new_scene_text.grid(row=5, column=0, sticky="nsew", pady=2)
        scene_frame.grid_rowconfigure(5, weight=1)
        
        self.app.change_scene_btn = ttk.Button(scene_frame, text="Change Scene", command=self.app.change_scene, state="disabled")
        self.app.change_scene_btn.grid(row=6, column=0, pady=(10, 0))

    def reset_conversation_state(self):
        self._message_counter = 0
        self.app.removed_messages.clear()
        self.app.pending_audio_messages.clear()
        self.app.displayed_messages.clear()
        self.app.message_bubbles.clear()
        self.agent_colors.clear()  # Clear the agent colors reference

    def load_conversation(self, conversation):
        """Loads a selected conversation's history into the chat canvas."""
        self.chat_canvas.clear()
        self.reset_conversation_state()

        print(f"[SimulationTab] Loading conversation: {getattr(conversation, 'agent_numbers', None)} ")
        # Store agent colors and sync with app
        self.agent_colors = conversation.agent_colors if hasattr(conversation, 'agent_colors') else {}
        self.app.agent_colors = self.agent_colors
        
        # Store agent numbers for proper bubble alignment
        # Convert from agent ID mapping to agent name mapping for display
        self._loaded_conversation_agent_numbers = {}
        if hasattr(conversation, 'agent_numbers') and conversation.agent_numbers:
            for agent_id, num in conversation.agent_numbers.items():
                agent = self.app.data_manager.get_agent_by_id(agent_id)
                if agent:
                    self._loaded_conversation_agent_numbers[agent_id] = num

        # Display header
        header_text = f"""Conversation: {conversation.title}\nEnvironment: {conversation.environment}\nScene: {conversation.scene_description}"""
        self.chat_canvas.add_bubble("System", header_text, conversation.created_at[:10], "system", UI_COLORS["system_bubble"])

        # Display messages
        for message in conversation.messages:
            self.display_message(message)

        self.update_simulation_controls(False) # Not active until started
        self.current_env_label.config(text=conversation.environment)
        self.app.update_status(f"Loaded conversation: {conversation.title}")
        
        # Resume the conversation
        self.resume_loaded_conversation(conversation)

    def update_simulation_controls(self, conversation_active: bool, paused: bool = False):
        """Update the state of simulation control buttons based on conversation status."""
        if conversation_active:
            self.app.pause_btn.config(state="normal")
            self.app.summarize_btn.config(state="normal")
            self.app.stop_btn.config(state="normal")
            self.app.send_btn.config(state="normal")
            self.app.resume_btn.config(state="disabled")
            self.app.change_scene_btn.config(state="normal")
        elif paused:
            self.app.pause_btn.config(state="disabled")
            self.app.resume_btn.config(state="normal")
            self.app.summarize_btn.config(state="normal")
            self.app.stop_btn.config(state="normal")
            self.app.send_btn.config(state="disabled")
            self.app.change_scene_btn.config(state="disabled")
        else:
            self.app.pause_btn.config(state="disabled")
            self.app.resume_btn.config(state="disabled")
            self.app.summarize_btn.config(state="disabled")
            self.app.stop_btn.config(state="disabled")
            self.app.send_btn.config(state="disabled")
            self.app.change_scene_btn.config(state="disabled")

    def pause_conversation(self):
        """Pause the current conversation."""
        print("[SimulationTab] pause_conversation called")
        if self.app.conversation_active and self.app.conversation_engine:
            try:
                print("[SimulationTab] Calling engine.pause_conversation...")
                self.app.conversation_engine.pause_conversation(self.app.current_conversation_id)
                print("[SimulationTab] Updating simulation controls for pause...")
                self.update_simulation_controls(False, paused=True)
                print("[SimulationTab] Updating status for pause...")
                self.app.update_status("Conversation paused.")
                print("[SimulationTab] Stopping all blinking...")
                self.chat_canvas.stop_all_blinking()
                # Show system message in chat canvas
                pause_message = "⏸️ System: Conversation is paused."
                print(f"[SimulationTab] Adding system pause message: {pause_message}")
                self.chat_canvas.add_bubble("System", pause_message, datetime.now().strftime("%H:%M:%S"), "system", UI_COLORS["system_bubble"])
            except Exception as e:
                print(f"[SimulationTab] Error in pause_conversation: {e}")
                messagebox.showerror("Error", f"Failed to pause conversation: {str(e)}")
        else:
            print("[SimulationTab] pause_conversation called but not active or engine missing")

    def resume_conversation(self):
        """Resume the current conversation (UI only, backend resume is handled by main_app)."""
        print("[SimulationTab] resume_conversation called (UI only)")
        if self.app.conversation_active:
            try:
                print("[SimulationTab] Updating simulation controls for resume...")
                self.update_simulation_controls(True)
                print("[SimulationTab] Updating status for resume...")
                self.app.update_status("Conversation resumed.")
                # Show system message in chat canvas
                resume_message = "▶️ System: Conversation has been resumed."
                print(f"[SimulationTab] Adding system resume message: {resume_message}")
                self.chat_canvas.add_bubble("System", resume_message, datetime.now().strftime("%H:%M:%S"), "system", UI_COLORS["system_bubble"])
            except Exception as e:
                print(f"[SimulationTab] Error in resume_conversation: {e}")
                messagebox.showerror("Error", f"Failed to resume conversation: {str(e)}")
        else:
            print("[SimulationTab] resume_conversation called but not active")

    def stop_conversation(self):
        """Stop the current conversation."""
        if self.app.conversation_active and self.app.conversation_engine:
            if messagebox.askyesno("Confirm Stop", "Are you sure you want to stop the conversation?"):
                try:
                    self.app.conversation_engine.stop_conversation(self.app.current_conversation_id)
                    self.app.conversation_active = False
                    self.update_simulation_controls(False)
                    self.app.update_status("Conversation stopped.")
                    self.chat_canvas.stop_all_blinking()
                    
                    conversation = self.app.data_manager.get_conversation_by_id(self.app.current_conversation_id)
                    if conversation:
                        conversation.status = "stopped"
                        self.app.data_manager.save_conversation(conversation)
                    
                    self.app.past_conversations_tab.refresh_past_conversations()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to stop conversation: {str(e)}")

    def summarize_conversation(self):
        """Summarize the conversation so far."""
        if self.app.conversation_active and self.app.conversation_engine:
            try:
                summary = self.app.conversation_engine.summarize_conversation(self.app.current_conversation_id)
                summary_window = tk.Toplevel(self.app.root)
                summary_window.title("Conversation Summary")
                summary_window.geometry("500x400")
                text_area = scrolledtext.ScrolledText(summary_window, wrap=tk.WORD)
                text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
                text_area.insert(tk.END, summary)
                text_area.config(state="disabled")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to summarize conversation: {str(e)}")

    def change_scene(self):
        """Change the environment and scene during a conversation."""
        if not self.app.conversation_active or not self.app.conversation_engine:
            messagebox.showwarning("Not Active", "No active conversation to change.")
            return
        
        new_env = self.app.new_env_var.get().strip()
        new_scene = self.app.new_scene_text.get(1.0, tk.END).strip()
        
        if not new_env and not new_scene:
            messagebox.showwarning("No Change", "Please provide a new environment or scene.")
            return
        
        try:
            self.app.conversation_engine.change_scene(self.app.current_conversation_id, new_env, new_scene)
            if new_env:
                self.current_env_label.config(text=new_env)
                self.app.new_env_var.set("")
            if new_scene:
                self.app.new_scene_text.delete(1.0, tk.END)
            self.app.update_status("Scene updated successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to change scene: {str(e)}")

    def send_user_message(self, event=None):
        """Send a user message to the conversation."""
        message = self.app.message_var.get().strip()
        if not message or not self.app.conversation_active:
            return
        
        self.app.message_var.set("")
        self.display_message({"sender": "You", "content": message, "type": "user"})
        self.app.send_user_message(message)

    def display_message(self, message_data, blinking=False):
        """Display a message in the chat canvas."""
        # Debug print to check incoming message_data
        print(f"[SimulationTab] display_message called with: {message_data}")
        agent_id = message_data.get("agent_id")
        sender = message_data.get("sender")
        agent_name = message_data.get("agent_name")
        
        # Get agent_no from agent_numbers dict of current conversation
        agent_no = None
        if hasattr(self.app, 'current_conversation_id') and self.app.current_conversation_id:
            conversation = self.app.data_manager.get_conversation_by_id(self.app.current_conversation_id)
            if conversation and hasattr(conversation, 'agent_numbers'):
                agent_no = conversation.agent_numbers.get(agent_id)

        
        # Prefer agent_id, then agent_name, then sender
        color = None
        if agent_id and agent_id in self.agent_colors:
            color = self.agent_colors[agent_id]
        elif agent_name and agent_name in self.agent_colors:
            color = self.agent_colors[agent_name]
        elif sender and sender in self.agent_colors:
            color = self.agent_colors[sender]
        elif sender == "You":
            color = UI_COLORS["user_bubble"]
        else:
            color = UI_COLORS["agent_colors"][0]
        print(f"[SimulationTab] Resolved color: {color} for agent_id={agent_id}, agent_name={agent_name}, sender={sender}")
        message = message_data.get("content") or message_data.get("message", "")
        msg_type = message_data.get("type", "ai")
        timestamp = message_data.get("timestamp", datetime.now().strftime("%H:%M:%S"))
        message_id = message_data.get("message_id")
        # Skip summary messages
        if "past_convo_summary" in message_data:
            return
        # Alignment logic: odd agent_no = right, even agent_no = left
        print(f"[SimulationTab] agent_no={agent_no}, color={color}")
        if agent_no is not None:
            align_right = (agent_no % 2 == 1)
        else:
            align_right = (msg_type == "user") or (sender == "You")
        print(f"[SimulationTab] align_right={align_right}")
        bubble = self.chat_canvas.add_bubble(sender or agent_name or agent_id, message, timestamp, msg_type, color, align_right, message_id)
        blinking_flag = message_data.get("blinking", blinking)
        print(f"[SimulationTab] blinking_flag={blinking_flag}, bubble={bubble}")
        if blinking_flag and bubble:
            print(f"[SimulationTab] Triggering blinking for bubble={bubble}")
            bubble.start_blink()
            self.blinking_messages[bubble] = True
        elif blinking_flag:
            print(f"[SimulationTab] Blinking requested but bubble missing: bubble={bubble}")

    def handle_message_callback(self, message_data):
        """Handle messages from round robin engine, including stop_blinking action."""
        if isinstance(message_data, dict) and message_data.get("action") == "stop_blinking":
            print(f"[SimulationTab] Received stop_blinking callback from backend. Stopping all blinking bubbles.")
            for bubble in list(self.blinking_messages.keys()):
                print(f"[SimulationTab] Stopping blinking for bubble={bubble}")
                bubble.stop_blink()
            self.blinking_messages.clear()
        else:
            self.display_message(message_data)

    def on_audio_ready(self, conv_id, agent_id, message_id):
        """Start blinking the chat bubble when audio starts playing."""
        self.chat_canvas.start_bubble_blink(message_id)

    def on_audio_finished(self, conv_id, agent_id, message_id):
        """Stop blinking and display the chat bubble for the agent after audio finishes."""
        self.chat_canvas.stop_bubble_blink(message_id)
        # Find the last message for this agent in the active conversation
        convo = self.app.conversation_engine.active_conversations.get(conv_id)
        if not convo or not convo.get("messages"):
            return
        # Find the last message from this agent
        for msg in reversed(convo["messages"]):
            if msg.get("agent_name") == agent_id or msg.get("sender") == agent_id:
                self.display_message(msg)
                break

    def resume_loaded_conversation(self, conversation):
        """Resume a loaded conversation by using ConversationEngine's resume logic."""
        try:
            # Use the main app's conversation engine to resume
            if not self.app.conversation_engine:
                from ..conversation_engine import ConversationEngine
                self.app.conversation_engine = ConversationEngine()
            self.app.conversation_engine.resume_conversation(conversation.id)
            self.app.current_conversation_id = conversation.id
            self.app.conversation_active = True
            self.update_simulation_controls(True)
            if len(conversation.messages) > 0:
                resume_message = f"📍 Conversation '{conversation.title}' has been resumed."
                self.chat_canvas.add_bubble("System", resume_message, datetime.now().strftime("%H:%M:%S"), "system", UI_COLORS["system_bubble"])
                self.app.update_status(f"Conversation '{conversation.title}' resumed successfully!")
            else:
                resume_message = f"📍 Conversation '{conversation.title}' has Started."
                self.chat_canvas.add_bubble("System", resume_message, datetime.now().strftime("%H:%M:%S"), "system", UI_COLORS["system_bubble"])
                self.app.update_status(f"Conversation '{conversation.title}' started successfully!")
        except Exception as e:
            self.app.update_status(f"Error resuming conversation: {str(e)}")
            import traceback
            traceback.print_exc()
