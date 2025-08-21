import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime

from .chat_widgets import ChatCanvas
from ..config import UI_COLORS
from ..data_manager import Conversation, Agent

class ResearchConversationTab(ttk.Frame):
    def __init__(self, parent, app, data_manager):
        super().__init__(parent)
        self.app = app
        self.data_manager = data_manager
        self.create_widgets()
        self.blinking_messages = {}
        # Register callback for research messages
        if hasattr(self.app, "research_trigger_engine"):
            self.app.research_trigger_engine.register_message_callback(self.display_message)

    def display_message(self, message_data):
        # Display a message in the research chat canvas, using agent colors and alignment if available
        print(f"[ResearchConversationTab] display_message: {message_data}")
        if hasattr(self, "chat_canvas"):
            sender = message_data.get("sender", "Agent")
            content = message_data.get("content", "")
            timestamp = message_data.get("timestamp", "")
            color = None
            align_right = False
            agent_no = None
            # Try to get agent_colors and agent_numbers from the current research conversation
            if hasattr(self.app, "current_research_id") and self.app.current_research_id:
                research_conv = None
                if hasattr(self.data_manager, "get_research_conversation_by_id"):
                    research_conv = self.data_manager.get_research_conversation_by_id(self.app.current_research_id)
                if research_conv:
                    # Color logic
                    if hasattr(research_conv, "agent_colors"):
                        agent_colors = research_conv.agent_colors
                        agent_id = message_data.get("agent_id")
                        agent_name = message_data.get("agent_name")
                        if agent_id and agent_id in agent_colors:
                            color = agent_colors[agent_id]
                        elif agent_name and agent_name in agent_colors:
                            color = agent_colors[agent_name]
                        elif sender and sender in agent_colors:
                            color = agent_colors[sender]
                    # Alignment logic
                    if hasattr(research_conv, "agent_numbers"):
                        agent_numbers = research_conv.agent_numbers
                        agent_id = message_data.get("agent_id")
                        agent_name = message_data.get("agent_name")
                        # Try to resolve agent_no by id or name
                        if agent_id and agent_id in agent_numbers:
                            agent_no = agent_numbers[agent_id]
                        elif agent_name and agent_name in agent_numbers:
                            agent_no = agent_numbers[agent_name]
                        elif sender and sender in agent_numbers:
                            agent_no = agent_numbers[sender]
            # Alignment: odd agent_no = right, even = left; fallback for user
            msg_type = message_data.get("type", "ai")
            if agent_no is not None:
                align_right = (agent_no % 2 == 1)
            else:
                align_right = (msg_type == "user") or (sender == "You")
            self.chat_canvas.add_bubble(sender, content, timestamp, msg_type, color, align_right)

    def create_widgets(self):
        sim_frame = self
        sim_frame.grid_rowconfigure(1, weight=1)
        sim_frame.grid_columnconfigure(0, weight=1)

        # Title and controls
        header_frame = ttk.Frame(sim_frame)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        header_frame.grid_columnconfigure(1, weight=1)

        title_label = ttk.Label(header_frame, text="Research Conversation", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, sticky="w")

        # Control buttons
        control_frame = ttk.Frame(header_frame)
        control_frame.grid(row=0, column=2, sticky="e")

        self.app.research_pause_btn = ttk.Button(control_frame, text="Pause", state="disabled")
        self.app.research_pause_btn.pack(side=tk.LEFT, padx=2)

        self.app.research_resume_btn = ttk.Button(control_frame, text="Resume", state="disabled")
        self.app.research_resume_btn.pack(side=tk.LEFT, padx=2)

        self.app.research_summarize_btn = ttk.Button(control_frame, text="Summarize", state="disabled")
        self.app.research_summarize_btn.pack(side=tk.LEFT, padx=2)

        self.app.research_stop_btn = ttk.Button(control_frame, text="Stop", state="disabled")
        self.app.research_stop_btn.pack(side=tk.LEFT, padx=2)

        # Main content area
        content_frame = ttk.Frame(sim_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)

        # Chat display with bubbles
        chat_frame = ttk.LabelFrame(content_frame, text="Research Conversation", padding="10")
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)

        chat_container = ttk.Frame(chat_frame)
        chat_container.grid(row=0, column=0, sticky="nsew")
        chat_container.grid_rowconfigure(0, weight=1)
        chat_container.grid_columnconfigure(0, weight=1)

        self.app.research_chat_canvas = ChatCanvas(chat_container, bg=UI_COLORS["chat_background"])
        self.app.research_chat_canvas.pack(side="left", fill="both", expand=True)
        self.chat_canvas = self.app.research_chat_canvas
        self.app.research_agent_colors = {}
        # Input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        input_frame.grid_columnconfigure(0, weight=1)

        self.app.research_message_var = tk.StringVar()
        self.app.research_message_entry = ttk.Entry(input_frame, textvariable=self.app.research_message_var)
        self.app.research_message_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        # No send button for now, can add if needed
        self.app.research_send_btn = ttk.Button(input_frame, text="Send", state="disabled")
        self.app.research_send_btn.grid(row=0, column=1)

        # Right panel - keep empty for now
        right_frame = ttk.Frame(content_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(0, weight=1)
        # (Intentionally left empty)
