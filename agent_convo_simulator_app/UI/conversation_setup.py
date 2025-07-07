import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime
import threading

from ..data_manager import Conversation
from ..conversation_engine import ConversationSimulatorEngine
from ..config import UI_COLORS
from .main_utils import _toggle_termination_condition

class ConversationSetupTab(ttk.Frame):
    def __init__(self, parent, app, data_manager):
        super().__init__(parent)
        self.app = app
        self.data_manager = data_manager

        self.create_widgets()

    def create_widgets(self):
        """Create the conversation setup tab."""
        conv_frame = self
        
        # Configure grid
        conv_frame.grid_rowconfigure(1, weight=1)
        conv_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(conv_frame, text="Conversation Setup", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(10, 20))
        
        # Main content frame
        content_frame = ttk.Frame(conv_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        
        # Left panel - Conversation settings
        settings_frame = ttk.LabelFrame(content_frame, text="Conversation Settings", padding="10")
        settings_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        settings_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(settings_frame, text="Title:").grid(row=0, column=0, sticky="w", pady=2)
        self.app.conv_title_var = tk.StringVar()
        self.app.conv_title_entry = ttk.Entry(settings_frame, textvariable=self.app.conv_title_var, width=30)
        self.app.conv_title_entry.grid(row=0, column=1, sticky="ew", pady=2, padx=(10, 0))
        
        ttk.Label(settings_frame, text="Environment:").grid(row=1, column=0, sticky="w", pady=2)
        self.app.conv_env_var = tk.StringVar()
        self.app.conv_env_entry = ttk.Entry(settings_frame, textvariable=self.app.conv_env_var, width=30)
        self.app.conv_env_entry.grid(row=1, column=1, sticky="ew", pady=2, padx=(10, 0))
        ttk.Label(settings_frame, text="Scene Description:").grid(row=2, column=0, sticky="nw", pady=2)
        self.app.conv_scene_text = scrolledtext.ScrolledText(settings_frame, width=40, height=8)
        self.app.conv_scene_text.grid(row=2, column=1, sticky="nsew", pady=2, padx=(10, 0))
        
        # Agent Invocation Method
        ttk.Label(settings_frame, text="Invocation Method:").grid(row=3, column=0, sticky="w", pady=(10, 2))
        self.app.invocation_method_var = tk.StringVar(value="round_robin")
        method_frame = ttk.Frame(settings_frame)
        method_frame.grid(row=3, column=1, sticky="ew", pady=(10, 2), padx=(10, 0))
        
        rb1 = ttk.Radiobutton(
            method_frame, 
            text="Round Robin", 
            variable=self.app.invocation_method_var, 
            value="round_robin",
            command=lambda: _toggle_termination_condition(self.app)
        )
        rb1.pack(side=tk.LEFT, padx=(0, 10))
        
        rb2 = ttk.Radiobutton(
            method_frame, 
            text="Agent Selector (LLM)", 
            variable=self.app.invocation_method_var, 
            value="agent_selector",
            command=lambda: _toggle_termination_condition(self.app)
        )
        rb2.pack(side=tk.LEFT, padx=(0, 10))
        
        rb3 = ttk.Radiobutton(
            method_frame, 
            text="Human-like Chat", 
            variable=self.app.invocation_method_var, 
            value="human_like_chat",
            command=lambda: _toggle_termination_condition(self.app)
        )
        rb3.pack(side=tk.LEFT)
        
        # Add tooltips to radio buttons
        rb1.bind("<Enter>", lambda e: self.app.show_tooltip(e, "Agents take turns speaking in a fixed order"))
        rb1.bind("<Leave>", self.app.hide_tooltip)
        
        rb2.bind("<Enter>", lambda e: self.app.show_tooltip(e, "An LLM intelligently chooses which agent should speak next"))
        rb2.bind("<Leave>", self.app.hide_tooltip)
        
        rb3.bind("<Enter>", lambda e: self.app.show_tooltip(e, "Natural conversation flow where agents respond in parallel and decide whether to participate"))
        rb3.bind("<Leave>", self.app.hide_tooltip)
        
        # Termination condition (available for all modes)
        ttk.Label(settings_frame, text="Termination Condition:").grid(row=4, column=0, sticky="nw", pady=2)
        self.app.termination_condition_text = scrolledtext.ScrolledText(settings_frame, width=40, height=4)
        self.app.termination_condition_text.grid(row=4, column=1, sticky="ew", pady=2, padx=(10, 0))
        # Always enable termination condition input for both modes
        self.app.termination_condition_text.config(state=tk.NORMAL)
        
        # Agent Selector API Key (only visible with agent selector)
        ttk.Label(settings_frame, text="Agent Selector API Key:").grid(row=5, column=0, sticky="w", pady=2)
        self.app.agent_selector_api_key_var = tk.StringVar()
        self.app.agent_selector_api_key_entry = ttk.Entry(settings_frame, textvariable=self.app.agent_selector_api_key_var, width=40, show="*")
        self.app.agent_selector_api_key_entry.grid(row=5, column=1, sticky="ew", pady=2, padx=(10, 0))
        self.app.agent_selector_api_key_entry.config(state=tk.DISABLED)  # Disabled by default for round-robin
        
        # Voice Settings
        ttk.Label(settings_frame, text="Voice Synthesis:").grid(row=6, column=0, sticky="w", pady=2)
        self.app.voices_enabled_var = tk.BooleanVar()
        voices_frame = ttk.Frame(settings_frame)
        voices_frame.grid(row=6, column=1, sticky="ew", pady=2, padx=(10, 0))
        
        voices_checkbox = ttk.Checkbutton(
            voices_frame,
            text="Enable voice synthesis for agents",
            variable=self.app.voices_enabled_var
        )
        voices_checkbox.pack(side=tk.LEFT)
        
        settings_frame.grid_rowconfigure(2, weight=1)
        
        # Right panel - Agent selection
        agents_frame = ttk.LabelFrame(content_frame, text="Select Agents (2-4)", padding="10")
        agents_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        agents_frame.grid_rowconfigure(0, weight=1)
        agents_frame.grid_columnconfigure(0, weight=1)
        
        # Agent selection with checkboxes
        self.app.selected_agents = []
        self.app.agent_checkboxes = []
        self.app.agents_checkbox_frame = ttk.Frame(agents_frame)
        self.app.agents_checkbox_frame.grid(row=0, column=0, sticky="nsew")
        
        # Control buttons
        btn_frame = ttk.Frame(content_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=(20, 0))
        
        ttk.Button(btn_frame, text="Start Conversation", command=self.app.start_conversation).pack(side=tk.LEFT, padx=(0, 10))

    def refresh_agent_checkboxes(self):
        """Refresh the agent checkboxes in the conversation setup."""
        # Clear existing checkboxes
        for widget in self.app.agents_checkbox_frame.winfo_children():
            widget.destroy()
        self.app.agent_checkboxes.clear()
        
        # Get agents from data manager
        agents = self.data_manager.load_agents()
        
        # Create new checkboxes
        for agent in agents:
            var = tk.BooleanVar()
            checkbox = ttk.Checkbutton(
                self.app.agents_checkbox_frame,
                text=f"{agent.name} ({agent.role})",
                variable=var
            )
            checkbox.pack(anchor="w", pady=2)
            self.app.agent_checkboxes.append((agent, var, checkbox))

    def update_selected_agents(self):
        """Update and return the list of selected agents."""
        self.app.selected_agents = []
        for agent, var, checkbox in self.app.agent_checkboxes:
            if var.get():
                self.app.selected_agents.append(agent)
        return self.app.selected_agents
