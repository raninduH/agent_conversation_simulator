"""
Multi-Agent Conversation Simulator GUI
A desktop application for simulating group conversations between AI agents using LangGraph and Gemini.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import json
import os
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import our custom modules
try:
    from data_manager import DataManager, Agent, Conversation
    from conversation_engine import ConversationSimulatorEngine
    from config import UI_COLORS
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"LangGraph dependencies not available: {e}")
    print("Please ensure all dependencies are installed.")
    print("Run: pip install -r requirements.txt")
    exit(1)


class ChatBubble(tk.Frame):
    """Represents a chat message bubble in the conversation UI."""
    
    def __init__(self, parent, sender, message, timestamp, msg_type="ai", color=None, **kwargs):
        """Initialize the chat bubble.
        
        Args:
            parent: Parent widget
            sender: Name of the message sender
            message: Content of the message
            timestamp: Time the message was sent
            msg_type: Type of message (user, system, ai)
            color: Background color for the bubble
        """
        # Choose appropriate color
        if color is None:
            if msg_type == "user":
                color = UI_COLORS["user_bubble"]
            elif msg_type == "system":
                color = UI_COLORS["system_bubble"]
            else:
                # Default agent color if none specified
                color = UI_COLORS["agent_colors"][0]
        
        # Initialize frame with appropriate background
        super().__init__(parent, bg=UI_COLORS["chat_background"], **kwargs)
          # Create bubble layout
        self.bubble_frame = tk.Frame(self, bg=color, padx=10, pady=5)
        self.bubble_frame.pack(fill="x", padx=10, pady=6, anchor="w" if msg_type != "user" else "e")
        
        # Add rounded corners by using themed frame
        self.bubble_frame.config(highlightbackground=color, highlightthickness=1, bd=0)
        
        # Add sender name with timestamp
        header_frame = tk.Frame(self.bubble_frame, bg=color)
        header_frame.pack(fill="x", expand=True)
        
        # Icon based on message type
        if msg_type == "user":
            icon = "👤"
        elif msg_type == "system":
            icon = "🤖"
        else:
            icon = "🎭"
            
        sender_label = tk.Label(
            header_frame, 
            text=f"{icon} {sender}", 
            font=("Arial", 9, "bold"),
            bg=color,
            anchor="w"
        )
        sender_label.pack(side="left")
        
        time_label = tk.Label(
            header_frame, 
            text=timestamp, 
            font=("Arial", 8),
            bg=color,
            fg="gray",
            anchor="e"
        )
        time_label.pack(side="right")
        
        # Add message content with word wrapping
        message_label = tk.Label(
            self.bubble_frame, 
            text=message, 
            font=("Arial", 10),
            bg=color,
            justify="left",
            anchor="w",
            wraplength=400
        )
        message_label.pack(fill="x", pady=(5, 0))
        
    @staticmethod
    def get_message_height(message, width=400):
        """Estimate the height needed for a message (for canvas sizing)."""
        # Simple estimation based on message length and width
        # Each character is roughly 7 pixels wide in common fonts
        chars_per_line = width // 7
        lines = len(message) // chars_per_line + message.count("\n") + 1
        
        # Each line is about 20px, plus padding
        return max(50, lines * 20 + 40)  # Minimum height of 50px


class ChatCanvas(tk.Canvas):
    """A scrollable canvas for displaying chat bubbles."""
    
    def __init__(self, parent, **kwargs):
        """Initialize the chat canvas."""
        super().__init__(parent, **kwargs)
        
        # Create a frame inside the canvas to hold chat bubbles
        self.bubble_frame = tk.Frame(self, bg=UI_COLORS["chat_background"])
        self.bubble_frame.pack(fill="both", expand=True)
        
        # Create scrollbar
        self.scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.configure(yscrollcommand=self.scrollbar.set)
        
        # Configure canvas
        self.config(bg=UI_COLORS["chat_background"])
        
        # Track the previous sender to add spacing between different agents
        self.previous_sender = None
        
        # Create window for the frame
        self.bubble_window = self.create_window((0, 0), window=self.bubble_frame, anchor="nw", width=self.winfo_width())
        
        # Bind events
        self.bind("<Configure>", self.on_configure)
        self.bubble_frame.bind("<Configure>", self.on_frame_configure)
        
    def on_configure(self, event):
        """Handle canvas resize events."""
        # Update the width of the window to the canvas width
        self.itemconfig(self.bubble_window, width=event.width)
        
    def on_frame_configure(self, event):
        """Update scroll region when the inner frame changes size."""
        self.configure(scrollregion=self.bbox("all"))
        
    def add_bubble(self, sender, message, timestamp, msg_type="ai", color=None):
        """Add a new chat bubble to the canvas."""
        
        # Add extra spacing between messages from different senders
        if self.previous_sender is not None and self.previous_sender != sender:
            # Add a small spacer frame between different agents' messages
            spacer = tk.Frame(self.bubble_frame, bg=UI_COLORS["chat_background"], height=10)
            spacer.pack(fill="x", expand=True)
        
        # Update the previous sender
        self.previous_sender = sender
        
        bubble = ChatBubble(
            self.bubble_frame,
            sender, 
            message, 
            timestamp,
            msg_type,
            color
        )
        bubble.pack(fill="x", expand=True)
        
        # Auto-scroll to the bottom
        self.yview_moveto(1.0)
        
        return bubble
        
    def clear(self):
        """Clear all chat bubbles."""
        for widget in self.bubble_frame.winfo_children():
            widget.destroy()
        # Reset previous sender tracking
        self.previous_sender = None


class AgentConversationSimulatorGUI:
    """Main GUI application for the multi-agent conversation simulator."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Multi-Agent Conversation Simulator")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg=UI_COLORS["app_background"])
        
        # Initialize data manager
        self.data_manager = DataManager(os.path.dirname(__file__))
        
        # Initialize conversation engine (will be created when needed)
        self.conversation_engine = None
        self.current_conversation_id = None
        self.conversation_active = False
        
        # Create the main interface
        self.create_widgets()
        self.load_data()
        
        # Configure grid weights for responsive design
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def create_widgets(self):
        """Create the main GUI widgets."""        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Create tabs
        self.create_agents_tab()
        self.create_conversation_tab()
        self.create_simulation_tab()
          # Status bar
        status_text = "Ready"
        if not LANGGRAPH_AVAILABLE:
            status_text += " (Using Mock Engine - Install LangGraph for full functionality)"
        self.status_bar = ttk.Label(self.root, text=status_text, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
    
    def create_agents_tab(self):
        """Create the agents management tab."""
        agents_frame = ttk.Frame(self.notebook)
        self.notebook.add(agents_frame, text="Agents")
        
        # Configure grid
        agents_frame.grid_rowconfigure(1, weight=1)
        agents_frame.grid_columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(agents_frame, text="Agent Management", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(10, 20))
        
        # Left panel - Agent list
        list_frame = ttk.LabelFrame(agents_frame, text="Agents", padding="10")
        list_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        # Agent listbox with scrollbar
        list_container = tk.Frame(list_frame)
        list_container.grid(row=0, column=0, sticky="nsew")
        list_container.grid_rowconfigure(0, weight=1)
        list_container.grid_columnconfigure(0, weight=1)
        
        self.agents_listbox = tk.Listbox(list_container)
        self.agents_listbox.grid(row=0, column=0, sticky="nsew")
        self.agents_listbox.bind('<<ListboxSelect>>', self.on_agent_select)
        
        agents_scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.agents_listbox.yview)
        agents_scrollbar.grid(row=0, column=1, sticky="ns")
        self.agents_listbox.configure(yscrollcommand=agents_scrollbar.set)
        
        # Buttons for agent management
        btn_frame = ttk.Frame(list_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(btn_frame, text="New Agent", command=self.new_agent).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Edit Agent", command=self.edit_agent).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete Agent", command=self.delete_agent).pack(side=tk.LEFT, padx=5)
        
        # Right panel - Agent details
        details_frame = ttk.LabelFrame(agents_frame, text="Agent Details", padding="10")
        details_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        details_frame.grid_columnconfigure(1, weight=1)
        
        # Agent details form
        ttk.Label(details_frame, text="Name:").grid(row=0, column=0, sticky="w", pady=2)
        self.agent_name_var = tk.StringVar()
        self.agent_name_entry = ttk.Entry(details_frame, textvariable=self.agent_name_var, width=30)
        self.agent_name_entry.grid(row=0, column=1, sticky="ew", pady=2, padx=(10, 0))
        
        ttk.Label(details_frame, text="Role:").grid(row=1, column=0, sticky="w", pady=2)
        self.agent_role_var = tk.StringVar()
        self.agent_role_entry = ttk.Entry(details_frame, textvariable=self.agent_role_var, width=30)
        self.agent_role_entry.grid(row=1, column=1, sticky="ew", pady=2, padx=(10, 0))
        
        ttk.Label(details_frame, text="Personality Traits:").grid(row=2, column=0, sticky="nw", pady=2)
        self.agent_traits_var = tk.StringVar()
        self.agent_traits_entry = ttk.Entry(details_frame, textvariable=self.agent_traits_var, width=30)
        self.agent_traits_entry.grid(row=2, column=1, sticky="ew", pady=2, padx=(10, 0))
        
        ttk.Label(details_frame, text="Base Prompt:").grid(row=3, column=0, sticky="nw", pady=2)
        self.agent_prompt_text = scrolledtext.ScrolledText(details_frame, width=40, height=10)
        self.agent_prompt_text.grid(row=3, column=1, sticky="nsew", pady=2, padx=(10, 0))
        details_frame.grid_rowconfigure(3, weight=1)
          # Save button
        ttk.Button(details_frame, text="Save Agent", command=self.save_agent).grid(row=4, column=1, sticky="e", pady=(10, 0))
    
    def create_conversation_tab(self):
        """Create the conversation setup tab."""
        conv_frame = ttk.Frame(self.notebook)
        self.notebook.add(conv_frame, text="Conversation Setup")
        
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
        self.conv_title_var = tk.StringVar()
        self.conv_title_entry = ttk.Entry(settings_frame, textvariable=self.conv_title_var, width=30)
        self.conv_title_entry.grid(row=0, column=1, sticky="ew", pady=2, padx=(10, 0))
        
        ttk.Label(settings_frame, text="Environment:").grid(row=1, column=0, sticky="w", pady=2)
        self.conv_env_var = tk.StringVar()
        self.conv_env_entry = ttk.Entry(settings_frame, textvariable=self.conv_env_var, width=30)
        self.conv_env_entry.grid(row=1, column=1, sticky="ew", pady=2, padx=(10, 0))
        ttk.Label(settings_frame, text="Scene Description:").grid(row=2, column=0, sticky="nw", pady=2)
        self.conv_scene_text = scrolledtext.ScrolledText(settings_frame, width=40, height=8)
        self.conv_scene_text.grid(row=2, column=1, sticky="nsew", pady=2, padx=(10, 0))
        
        # Agent Invocation Method
        ttk.Label(settings_frame, text="Invocation Method:").grid(row=3, column=0, sticky="w", pady=(10, 2))
        self.invocation_method_var = tk.StringVar(value="round_robin")
        method_frame = ttk.Frame(settings_frame)
        method_frame.grid(row=3, column=1, sticky="ew", pady=(10, 2), padx=(10, 0))
        
        ttk.Radiobutton(
            method_frame, 
            text="Round Robin", 
            variable=self.invocation_method_var, 
            value="round_robin",
            command=self._toggle_termination_condition
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            method_frame, 
            text="Agent Selector (LLM)", 
            variable=self.invocation_method_var, 
            value="agent_selector",
            command=self._toggle_termination_condition
        ).pack(side=tk.LEFT)
        
        # Termination condition (only visible with agent selector)
        ttk.Label(settings_frame, text="Termination Condition:").grid(row=4, column=0, sticky="nw", pady=2)
        self.termination_condition_text = scrolledtext.ScrolledText(settings_frame, width=40, height=4)
        self.termination_condition_text.grid(row=4, column=1, sticky="ew", pady=2, padx=(10, 0))
        self.termination_condition_text.config(state=tk.DISABLED)  # Disabled by default for round-robin
        
        settings_frame.grid_rowconfigure(2, weight=1)
        
        # Right panel - Agent selection
        agents_frame = ttk.LabelFrame(content_frame, text="Select Agents (2-4)", padding="10")
        agents_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        agents_frame.grid_rowconfigure(0, weight=1)
        agents_frame.grid_columnconfigure(0, weight=1)
        
        # Agent selection with checkboxes
        self.selected_agents = []
        self.agent_checkboxes = []
        self.agents_checkbox_frame = ttk.Frame(agents_frame)
        self.agents_checkbox_frame.grid(row=0, column=0, sticky="nsew")
        
        # API Key input
        api_frame = ttk.LabelFrame(content_frame, text="API Configuration", padding="10")
        api_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        api_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(api_frame, text="Google API Key:").grid(row=0, column=0, sticky="w", pady=2)
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show="*")
        self.api_key_entry.grid(row=0, column=1, sticky="ew", pady=2, padx=(10, 0))
        
        # Control buttons
        btn_frame = ttk.Frame(content_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(20, 0))
        
        ttk.Button(btn_frame, text="Start Conversation", command=self.start_conversation).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Load Conversation", command=self.load_conversation).pack(side=tk.LEFT, padx=10)
        
    def _toggle_termination_condition(self):
        """Toggle the termination condition text field based on invocation method."""
        # If Agent Selector is chosen, enable the termination condition field
        if self.invocation_method_var.get() == "agent_selector":
            self.termination_condition_text.config(state=tk.NORMAL)
        else:
            # Clear and disable the field for round robin
            self.termination_condition_text.delete(1.0, tk.END)
            self.termination_condition_text.config(state=tk.DISABLED)
    
    def create_simulation_tab(self):
        """Create the conversation simulation tab."""
        sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(sim_frame, text="Simulation")
        
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
        
        self.pause_btn = ttk.Button(control_frame, text="Pause", command=self.pause_conversation, state="disabled")
        self.pause_btn.pack(side=tk.LEFT, padx=2)
        
        self.resume_btn = ttk.Button(control_frame, text="Resume", command=self.resume_conversation, state="disabled")
        self.resume_btn.pack(side=tk.LEFT, padx=2)
        
        self.summarize_btn = ttk.Button(control_frame, text="Summarize", command=self.summarize_conversation, state="disabled")
        self.summarize_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_conversation, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
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
        
        self.chat_canvas = ChatCanvas(chat_container, bg=UI_COLORS["chat_background"])
        self.chat_canvas.pack(side="left", fill="both", expand=True)
        
        # Dictionary to store agent colors
        self.agent_colors = {}
        
        # Input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.message_var = tk.StringVar()
        self.message_entry = ttk.Entry(input_frame, textvariable=self.message_var)
        self.message_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.message_entry.bind('<Return>', self.send_user_message)
        
        self.send_btn = ttk.Button(input_frame, text="Send", command=self.send_user_message, state="disabled")
        self.send_btn.grid(row=0, column=1)
        
        # Scene control panel
        scene_frame = ttk.LabelFrame(content_frame, text="Scene Control", padding="10")
        scene_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        scene_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Label(scene_frame, text="Current Environment:").grid(row=0, column=0, sticky="w", pady=2)
        self.current_env_label = ttk.Label(scene_frame, text="None", font=("Arial", 10, "italic"))
        self.current_env_label.grid(row=1, column=0, sticky="w", pady=(0, 10))
        
        ttk.Label(scene_frame, text="New Environment:").grid(row=2, column=0, sticky="w", pady=2)
        self.new_env_var = tk.StringVar()
        self.new_env_entry = ttk.Entry(scene_frame, textvariable=self.new_env_var, width=25)
        self.new_env_entry.grid(row=3, column=0, sticky="ew", pady=2)
        
        ttk.Label(scene_frame, text="New Scene:").grid(row=4, column=0, sticky="w", pady=(10, 2))
        self.new_scene_text = scrolledtext.ScrolledText(scene_frame, width=25, height=8)
        self.new_scene_text.grid(row=5, column=0, sticky="nsew", pady=2)
        scene_frame.grid_rowconfigure(5, weight=1)
        
        self.change_scene_btn = ttk.Button(scene_frame, text="Change Scene", command=self.change_scene, state="disabled")
        self.change_scene_btn.grid(row=6, column=0, pady=(10, 0))
    
    def load_data(self):
        """Load agents and conversations from JSON files."""
        self.refresh_agents_list()
        self.refresh_agent_checkboxes()
    
    def refresh_agents_list(self):
        """Refresh the agents listbox."""
        self.agents_listbox.delete(0, tk.END)
        agents = self.data_manager.load_agents()
        for agent in agents:
            self.agents_listbox.insert(tk.END, f"{agent.name} ({agent.role})")
    
    def refresh_agent_checkboxes(self):
        """Refresh the agent selection checkboxes."""
        # Clear existing checkboxes
        for widget in self.agents_checkbox_frame.winfo_children():
            widget.destroy()
        
        self.agent_checkboxes = []
        self.selected_agents = []
        
        agents = self.data_manager.load_agents()
        for i, agent in enumerate(agents):
            var = tk.BooleanVar()
            checkbox = ttk.Checkbutton(
                self.agents_checkbox_frame,
                text=f"{agent.name} ({agent.role})",
                variable=var
            )
            checkbox.grid(row=i, column=0, sticky="w", pady=2)            
            self.agent_checkboxes.append((agent, var))
            
    def update_selected_agents(self):
        """Update the selected_agents list based on checkbox selections."""
        self.selected_agents = []
        for agent, var in self.agent_checkboxes:
            if var.get():
                self.selected_agents.append(agent)
        return self.selected_agents
    
    def on_agent_select(self, event):
        """Handle agent selection in the listbox."""
        selection = self.agents_listbox.curselection()
        if selection:
            agents = self.data_manager.load_agents()
            if selection[0] < len(agents):
                agent = agents[selection[0]]
                self.load_agent_details(agent)
    
    def load_agent_details(self, agent: Agent):
        """Load agent details into the form."""
        self.agent_name_var.set(agent.name)
        self.agent_role_var.set(agent.role)
        self.agent_traits_var.set(", ".join(agent.personality_traits))
        
        self.agent_prompt_text.delete(1.0, tk.END)
        self.agent_prompt_text.insert(1.0, agent.base_prompt)
    
    def new_agent(self):
        """Create a new agent."""
        self.clear_agent_form()
        self.agent_name_entry.focus()
    
    def clear_agent_form(self):
        """Clear the agent form."""
        self.agent_name_var.set("")
        self.agent_role_var.set("")
        self.agent_traits_var.set("")
        self.agent_prompt_text.delete(1.0, tk.END)
    
    def edit_agent(self):
        """Edit the selected agent."""
        selection = self.agents_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an agent to edit.")
            return
        # Agent details are already loaded in the form
    
    def delete_agent(self):
        """Delete the selected agent."""
        selection = self.agents_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an agent to delete.")
            return
        
        agents = self.data_manager.load_agents()
        if selection[0] < len(agents):
            agent = agents[selection[0]]
            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete agent '{agent.name}'?"):
                self.data_manager.delete_agent(agent.id)
                self.refresh_agents_list()
                self.refresh_agent_checkboxes()
                self.clear_agent_form()
                self.update_status(f"Agent '{agent.name}' deleted.")
    
    def save_agent(self):
        """Save the current agent."""
        name = self.agent_name_var.get().strip()
        role = self.agent_role_var.get().strip()
        traits_str = self.agent_traits_var.get().strip()
        prompt = self.agent_prompt_text.get(1.0, tk.END).strip()
        
        if not all([name, role, prompt]):
            messagebox.showerror("Missing Information", "Please fill in all required fields (Name, Role, and Base Prompt).")
            return
        
        # Parse personality traits
        traits = [t.strip() for t in traits_str.split(",") if t.strip()] if traits_str else []
        
        # Check if editing existing agent
        selection = self.agents_listbox.curselection()
        if selection:
            agents = self.data_manager.load_agents()
            if selection[0] < len(agents):
                # Edit existing agent
                agent = agents[selection[0]]
                agent.name = name
                agent.role = role
                agent.base_prompt = prompt
                agent.personality_traits = traits
                self.data_manager.save_agent(agent)
                self.update_status(f"Agent '{name}' updated.")
            else:
                # Create new agent
                agent = Agent.create_new(name, role, prompt, traits)
                self.data_manager.save_agent(agent)
                self.update_status(f"Agent '{name}' created.")
        else:
            # Create new agent
            agent = Agent.create_new(name, role, prompt, traits)
            self.data_manager.save_agent(agent)
            self.update_status(f"Agent '{name}' created.")
        
        self.refresh_agents_list()
        self.refresh_agent_checkboxes()
    def start_conversation(self):
        """Start a new conversation."""
        title = self.conv_title_var.get().strip()
        environment = self.conv_env_var.get().strip()
        scene = self.conv_scene_text.get(1.0, tk.END).strip()
        api_key = self.api_key_var.get().strip()        
        if not all([title, environment, scene]):
            messagebox.showerror("Missing Information", "Please fill in title, environment and scene description.")
            return
        
        if not api_key:
            messagebox.showwarning("API Key Required", "Please enter your Google API key to start the conversation. You can get one from https://console.cloud.google.com/")
            return
        
        # Get selected agents
        selected_agents = []
        for agent, var in self.agent_checkboxes:
            if var.get():
                selected_agents.append(agent)
        
        if len(selected_agents) < 2 or len(selected_agents) > 4:
            messagebox.showerror("Invalid Selection", "Please select 2-4 agents for the conversation.")
            return
        
        try:
            # Show loading message
            self.update_status("Starting conversation...")
            
            # Initialize conversation engine
            self.conversation_engine = ConversationSimulatorEngine(api_key)            # Initialize agent colors dictionary
            self.agent_colors = {}
              # Assign unique colors to agents from the predefined palette
            available_colors = list(UI_COLORS["agent_colors"])
            # No random shuffle needed - we'll assign colors in order to ensure consistency
              # Get invocation method and termination condition
            invocation_method = self.invocation_method_var.get()
            termination_condition = None
            if invocation_method == "agent_selector":
                termination_condition = self.termination_condition_text.get(1.0, tk.END).strip()
                if not termination_condition:
                    termination_condition = None  # Use None instead of empty string
            
            # Create conversation record with agent colors
            conversation = Conversation.create_new(
                title, environment, scene, [agent.id for agent in selected_agents],
                invocation_method=invocation_method, 
                termination_condition=termination_condition
            )
            
            # Prepare agent configs
            agents_config = []
            for i, agent in enumerate(selected_agents):                # Assign a color to each agent
                color = available_colors[i % len(available_colors)]
                self.agent_colors[agent.name] = color
                
                agents_config.append({
                    "name": agent.name,
                    "role": agent.role,
                    "base_prompt": agent.base_prompt,
                    "color": color  # Add color to agent config
                })
            
            # Add agent colors to the conversation
            conversation.agent_colors = self.agent_colors
            self.data_manager.save_conversation(conversation)
            self.current_conversation_id = conversation.id
              # Start the conversation
            thread_id = self.conversation_engine.start_conversation(
                conversation.id, agents_config, environment, scene,
                invocation_method=invocation_method,
                termination_condition=termination_condition
            )
            
            # Register callback for message updates
            self.conversation_engine.register_message_callback(
                conversation.id, self.on_message_received
            )
              # Update UI
            self.conversation_active = True
            self.update_simulation_controls(True)
            self.current_env_label.config(text=environment)
            
            # Clear existing messages
            self.chat_canvas.clear()
            
            # Add header information as a system message
            header_text = f"🎬 Starting conversation: {title}\n"
            header_text += f"📍 Environment: {environment}\n"
            header_text += f"🎭 Scene: {scene}\n"
            header_text += f"👥 Participants: {', '.join([a.name for a in selected_agents])}"
            
            # Display header as system message
            self.chat_canvas.add_bubble("System", header_text, datetime.now().strftime("%H:%M:%S"), "system", UI_COLORS["system_bubble"])
            
            # Switch to simulation tab
            self.notebook.select(2)
            
            self.update_status(f"Conversation '{title}' started successfully!")
            
            # Send initial message to start the conversation
            self.send_initial_message()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start conversation: {str(e)}")
            self.update_status("Failed to start conversation.")

    def send_initial_message(self):
        """Send an initial message to start the conversation."""
        if self.conversation_engine and self.current_conversation_id:
            threading.Thread(
                target=self._send_message_thread,
                args=("Hello everyone! Let's start our conversation.", "system"),
                daemon=True
            ).start()
    
    def _send_message_thread(self, message: str, sender: str = "user"):
        """Send message in a separate thread."""
        try:
            response = self.conversation_engine.send_message(
                self.current_conversation_id, message, sender
            )
            
            if response.get("success"):
                # Message will be displayed via callback
                pass
            else:
                self.root.after(0, lambda: self.update_status(f"Error: {response.get('error', 'Unknown error')}"))
                
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"Error sending message: {str(e)}"))
    
    def send_user_message(self, event=None):
        """Send a user message to the conversation."""
        message = self.message_var.get().strip()
        if not message or not self.conversation_active:
            return
        
        self.message_var.set("")
        
        # Display user message immediately
        self.display_message("You", message, "user")
        
        # Send message in background thread
        threading.Thread(
            target=self._send_message_thread,
            args=(message, "user"),
            daemon=True
        ).start()
    def on_message_received(self, message_data: Dict[str, Any]):
        """Callback for when a new message is received."""
        # Update UI in main thread
        self.root.after(0, lambda: self.display_message(
            message_data.get("sender", "Agent"),
            message_data.get("content", ""),
            message_data.get("type", "ai")
        ))
        
    def display_message(self, sender: str, content: str, msg_type: str, color: str = None):
        """Display a message in the chat window using chat bubbles."""
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Determine message color based on type and agent
        if msg_type == "user":
            bubble_color = UI_COLORS["user_bubble"]
        elif msg_type == "system":
            bubble_color = UI_COLORS["system_bubble"]
        else:
            # For agent messages, use the agent's color or assign one
            if sender not in self.agent_colors:                # Choose a color for this agent from the palette if not already assigned
                if not color:
                    # Get all assigned colors in this conversation
                    assigned_colors = set(self.agent_colors.values())
                    # Find an available color that hasn't been used yet
                    for c in UI_COLORS["agent_colors"]:
                        if c not in assigned_colors:
                            color = c
                            break
                    # If all colors are used, fall back to the first color
                    if not color:
                        color = UI_COLORS["agent_colors"][0]
                self.agent_colors[sender] = color
                
                # Store the color association in the conversation metadata
                if self.current_conversation_id:
                    self.data_manager.update_agent_color(self.current_conversation_id, sender, color)
            
            bubble_color = self.agent_colors.get(sender, color)
        
        # Add the bubble to the chat canvas
        self.chat_canvas.add_bubble(sender, content, timestamp, msg_type, bubble_color)
        
        # Save message to conversation
        if self.current_conversation_id:
            message_data = {
                "sender": sender,
                "content": content,
                "type": msg_type,
                "timestamp": datetime.now().isoformat(),
                "color": bubble_color  # Save the color with the message
            }
            self.data_manager.add_message_to_conversation(self.current_conversation_id, message_data)
    
    def pause_conversation(self):
        """Pause the active conversation."""
        if self.conversation_engine and self.current_conversation_id:
            self.conversation_engine.pause_conversation(self.current_conversation_id)
            self.conversation_active = False
            self.update_simulation_controls(False)
            self.update_status("Conversation paused.")
    
    def resume_conversation(self):
        """Resume the paused conversation."""
        if self.conversation_engine and self.current_conversation_id:
            self.conversation_engine.resume_conversation(self.current_conversation_id)
            self.conversation_active = True
            self.update_simulation_controls(True)
            self.update_status("Conversation resumed.")
    
    def summarize_conversation(self):
        """Generate and display a conversation summary."""
        if self.conversation_engine and self.current_conversation_id:
            try:
                summary = self.conversation_engine.get_conversation_summary(self.current_conversation_id)
                messagebox.showinfo("Conversation Summary", summary)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate summary: {str(e)}")
    
    def stop_conversation(self):
        """Stop the active conversation."""
        if self.conversation_engine and self.current_conversation_id:
            self.conversation_engine.stop_conversation(self.current_conversation_id)
            self.conversation_active = False
            self.current_conversation_id = None
            self.update_simulation_controls(False)
            self.current_env_label.config(text="None")
            self.update_status("Conversation stopped.")
    
    def change_scene(self):
        """Change the scene/environment of the active conversation."""
        new_env = self.new_env_var.get().strip()
        new_scene = self.new_scene_text.get(1.0, tk.END).strip()
        
        if not new_env or not new_scene:
            messagebox.showerror("Missing Information", "Please provide both environment and scene description.")
            return
        
        if self.conversation_engine and self.current_conversation_id:
            try:
                self.conversation_engine.change_scene(self.current_conversation_id, new_env, new_scene)
                self.current_env_label.config(text=new_env)
                self.new_env_var.set("")
                self.new_scene_text.delete(1.0, tk.END)
                self.update_status(f"Scene changed to: {new_env}")
                  # Display scene change in chat
                self.display_message("System", f"Scene changed to: {new_env}. {new_scene}", "system")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to change scene: {str(e)}")
    
    def load_conversation(self):
        """Load a previously saved conversation."""
        try:
            # Get all available conversations
            conversations = self.data_manager.get_conversations()
            
            if not conversations:
                messagebox.showinfo("No Conversations", "No previous conversations found.")
                return
            
            # Create conversation selection dialog
            self.show_conversation_selection_dialog(conversations)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load conversations: {str(e)}")
    
    def show_conversation_selection_dialog(self, conversations: List[Conversation]):
        """Show dialog to select a conversation to load."""
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Load Conversation")
        dialog.geometry("800x600")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Configure grid
        dialog.grid_rowconfigure(1, weight=1)
        dialog.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(dialog, text="Select a Conversation to Load", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, pady=10)
        
        # Main frame
        main_frame = ttk.Frame(dialog)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Conversation list frame
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        # Treeview for conversations
        columns = ("Title", "Environment", "Agents", "Messages", "Created", "Status")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        tree.heading("Title", text="Title")
        tree.heading("Environment", text="Environment")
        tree.heading("Agents", text="Agents")
        tree.heading("Messages", text="Messages")
        tree.heading("Created", text="Created")
        tree.heading("Status", text="Status")
        
        tree.column("Title", width=200)
        tree.column("Environment", width=150)
        tree.column("Agents", width=100)
        tree.column("Messages", width=80)
        tree.column("Created", width=120)
        tree.column("Status", width=80)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        h_scrollbar = ttk.Scrollbar(list_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Populate treeview
        conversation_data = {}
        for conv in conversations:
            # Format creation date
            try:
                created_date = datetime.fromisoformat(conv.created_at.replace('Z', '+00:00'))
                created_str = created_date.strftime("%Y-%m-%d %H:%M")
            except:
                created_str = conv.created_at[:16] if conv.created_at else "Unknown"
            
            # Count messages
            message_count = len(conv.messages)
            
            # Get agent names
            agent_names = ", ".join(conv.agents) if len(conv.agents) <= 2 else f"{len(conv.agents)} agents"
            
            # Insert into tree
            item_id = tree.insert("", "end", values=(
                conv.title,
                conv.environment,
                agent_names,
                message_count,
                created_str,
                conv.status
            ))
            conversation_data[item_id] = conv
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Conversation Preview", padding="10")
        preview_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        preview_frame.grid_rowconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        
        # Preview labels
        self.preview_title = ttk.Label(preview_frame, text="", font=("Arial", 12, "bold"))
        self.preview_title.grid(row=0, column=0, sticky="w")
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, width=50, height=20, state="disabled")
        self.preview_text.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        
        # Selection handler
        def on_conversation_select(event):
            selection = tree.selection()
            if selection:
                selected_conv = conversation_data[selection[0]]
                self.show_conversation_preview(selected_conv)
        
        tree.bind("<<TreeviewSelect>>", on_conversation_select)
        
        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=2, column=0, pady=20)
        
        def load_selected():
            selection = tree.selection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a conversation to load.")
                return
            
            selected_conv = conversation_data[selection[0]]
            self.load_selected_conversation(selected_conv)
            dialog.destroy()
        
        def cancel_load():
            dialog.destroy()
        
        ttk.Button(button_frame, text="Load Conversation", command=load_selected).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=cancel_load).pack(side=tk.LEFT)
    
    def show_conversation_preview(self, conversation: Conversation):
        """Show preview of the selected conversation."""
        self.preview_title.config(text=conversation.title)
        
        self.preview_text.config(state="normal")
        self.preview_text.delete(1.0, tk.END)
        
        # Add conversation details
        preview_content = f"Environment: {conversation.environment}\n"
        preview_content += f"Scene: {conversation.scene_description}\n"
        preview_content += f"Agents: {', '.join(conversation.agents)}\n"
        preview_content += f"Messages: {len(conversation.messages)}\n"
        preview_content += f"Status: {conversation.status}\n"
        preview_content += f"Created: {conversation.created_at}\n\n"
        
        if conversation.messages:
            preview_content += "Recent Messages:\n"
            preview_content += "-" * 40 + "\n"
            
            # Show last 5 messages
            recent_messages = conversation.messages[-5:]
            for msg in recent_messages:
                if msg.get('type') == 'system':
                    preview_content += f"[SYSTEM] {msg.get('content', '')}\n"
                else:
                    sender = msg.get('sender', 'Unknown')
                    content = msg.get('content', '')
                    # Truncate long messages
                    if len(content) > 200:
                        content = content[:200] + "..."
                    preview_content += f"{sender}: {content}\n"
                preview_content += "\n"
        else:
            preview_content += "No messages yet."        
        self.preview_text.insert(1.0, preview_content)
        self.preview_text.config(state="disabled")
    
    def load_selected_conversation(self, conversation: Conversation):
        """Load the selected conversation into the simulation."""
        try:
            # Switch to simulation tab
            self.notebook.select(2)  # Index 2 is simulation tab
            
            # Clear existing conversation
            if self.conversation_active:
                self.stop_conversation()
            
            # Load conversation data into the interface
            self.conv_title_entry.delete(0, tk.END)
            self.conv_title_entry.insert(0, conversation.title)
            
            self.conv_env_entry.delete(0, tk.END)
            self.conv_env_entry.insert(0, conversation.environment)
            
            self.conv_scene_text.delete(1.0, tk.END)
            self.conv_scene_text.insert(1.0, conversation.scene_description)
            
            # Load invocation method and termination condition
            invocation_method = getattr(conversation, 'invocation_method', 'round_robin')
            self.invocation_method_var.set(invocation_method)
            
            # Set termination condition if any
            self.termination_condition_text.config(state=tk.NORMAL)
            self.termination_condition_text.delete(1.0, tk.END)
            if invocation_method == "agent_selector" and hasattr(conversation, 'termination_condition') and conversation.termination_condition:
                self.termination_condition_text.insert(1.0, conversation.termination_condition)
            
            # Toggle termination condition field state
            self._toggle_termination_condition()
            
            # Update agent selection
            for agent_obj, var in self.agent_checkboxes:
                var.set(False)  # Uncheck all first
            
            # Check the agents that are in this conversation
            for agent_obj, var in self.agent_checkboxes:
                if agent_obj.id in conversation.agents:
                    var.set(True)
            
            # Update selected agents list
            self.update_selected_agents()            # Clear the chat display
            self.chat_canvas.clear()
              # Load agent colors from conversation if available
            self.agent_colors = {}
            if hasattr(conversation, 'agent_colors') and conversation.agent_colors:
                self.agent_colors = conversation.agent_colors
            else:
                # Try to get colors from agent_colors field in conversation data
                agent_colors = self.data_manager.get_agent_colors(conversation.id)
                if agent_colors:
                    self.agent_colors = agent_colors
            
            # Add conversation header as a system message
            header = f"=== LOADED CONVERSATION: {conversation.title} ===\n"
            header += f"Environment: {conversation.environment}\n"
            header += f"Scene: {conversation.scene_description}\n"
            header += f"Participants: {', '.join(conversation.agents)}\n"
            header += f"Original Messages: {len(conversation.messages)}"
            
            self.chat_canvas.add_bubble("System", header, datetime.now().strftime("%H:%M:%S"), "system", UI_COLORS["system_bubble"])
            
            # Add existing messages
            for msg in conversation.messages:
                sender = msg.get('sender', 'Unknown')
                content = msg.get('content', '')
                msg_type = msg.get('type', 'ai')
                color = msg.get('color')  # Get the color from the message if available
                
                # Format timestamp
                timestamp = msg.get('timestamp', '')
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    time_str = timestamp[:8] if timestamp else datetime.now().strftime("%H:%M:%S")
                
                # If color not available in message, check agent_colors
                if not color and sender in self.agent_colors:
                    color = self.agent_colors[sender]
                      # If still no color and it's an agent message, assign a unique one from the palette
                if not color and msg_type not in ["user", "system"]:
                    # Get all assigned colors in this conversation
                    assigned_colors = set(self.agent_colors.values())
                    # Find an available color that hasn't been used yet
                    for c in UI_COLORS["agent_colors"]:
                        if c not in assigned_colors:
                            color = c
                            break
                    # If all colors are used, fall back to the first color
                    if not color:
                        color = UI_COLORS["agent_colors"][0]
                    self.agent_colors[sender] = color
                  # Display the message using chat bubble
                self.chat_canvas.add_bubble(sender, content, time_str, msg_type, color)
            
            # Store loaded conversation info
            self.loaded_conversation = conversation
            
            # Show resumption option
            result = messagebox.askyesno(
                "Resume Conversation", 
                f"Conversation '{conversation.title}' has been loaded.\n\n"
                f"Would you like to resume the conversation where it left off?\n\n"
                f"Click 'Yes' to continue with AI agents, or 'No' to just view the conversation."
            )
            
            if result:
                self.resume_loaded_conversation()
            
            messagebox.showinfo("Success", f"Conversation '{conversation.title}' loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load conversation: {str(e)}")
    
    def resume_loaded_conversation(self):
        """Resume the loaded conversation with AI agents."""
        try:
            if not hasattr(self, 'loaded_conversation'):
                messagebox.showerror("Error", "No conversation loaded to resume.")
                return
              # Initialize conversation engine if needed
            api_key = self.api_key_var.get().strip()
            if not api_key:
                messagebox.showerror("Error", "Please enter your Google API key to resume the conversation.")
                return
            
            if not self.conversation_engine:
                self.conversation_engine = ConversationSimulatorEngine(api_key)
                  # Get selected agents
            selected_agent_ids = []
            for agent, var in self.agent_checkboxes:
                if var.get():
                    # Use agent ID directly from agent object
                    selected_agent_ids.append(agent.id)
            
            if len(selected_agent_ids) < 2:
                messagebox.showerror("Error", "Please select at least 2 agents to resume the conversation.")
                return
            
            # Get agent configs
            all_agents = self.data_manager.load_agents()
            agent_configs = {agent.id: agent for agent in all_agents if agent.id in selected_agent_ids}
            
            if len(agent_configs) != len(selected_agent_ids):
                messagebox.showerror("Error", "Some selected agents were not found in the database.")
                return
                
            # Convert agent objects to config dictionaries
            agents_config = []
            for agent_id in selected_agent_ids:
                agent = agent_configs.get(agent_id)
                if agent:                    # Get color from agent attribute or from conversation agent_colors
                    color = agent.color
                    if not color and agent.name in self.agent_colors:
                        color = self.agent_colors[agent.name]
                    
                    # If still no color, assign a unique one from the palette
                    if not color:
                        # Get all assigned colors in this conversation
                        assigned_colors = set(self.agent_colors.values())
                        # Find an available color that hasn't been used yet
                        for c in UI_COLORS["agent_colors"]:
                            if c not in assigned_colors:
                                color = c
                                break
                        # If all colors are used, fall back to the first color
                        if not color:
                            color = UI_COLORS["agent_colors"][0]
                        self.agent_colors[agent.name] = color
                        
                    agents_config.append({
                        "name": agent.name,
                        "role": agent.role,
                        "base_prompt": agent.base_prompt,
                        "color": color
                    })
              # Start conversation with existing messages
            conversation_id = self.loaded_conversation.id
            
            # Get invocation method and termination condition from loaded conversation
            invocation_method = getattr(self.loaded_conversation, 'invocation_method', 'round_robin')
            termination_condition = None
            if invocation_method == "agent_selector":
                termination_condition = getattr(self.loaded_conversation, 'termination_condition', None)
            
            thread_id = self.conversation_engine.start_conversation(
                conversation_id=conversation_id,
                agents_config=agents_config,
                environment=self.loaded_conversation.environment,
                scene_description=self.loaded_conversation.scene_description,
                invocation_method=invocation_method,
                termination_condition=termination_condition
            )            
            if thread_id:
                self.current_conversation_id = conversation_id
                self.conversation_active = True
                self.update_simulation_controls(True)
                
                # Set up message callback to update display
                self.conversation_engine.register_message_callback(conversation_id, self.on_message_received)
                
                # Add a system message indicating resumption
                resume_msg = f"CONVERSATION RESUMED at {datetime.now().strftime('%H:%M:%S')}"
                self.chat_canvas.add_bubble("System", resume_msg, datetime.now().strftime("%H:%M:%S"), "system", UI_COLORS["system_bubble"])
                
                messagebox.showinfo("Success", "Conversation resumed! The AI agents will continue from where they left off.")
            else:
                messagebox.showerror("Error", "Failed to resume conversation.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to resume conversation: {str(e)}")
    
    def update_simulation_controls(self, active: bool):
        """Update the state of simulation control buttons."""
        if active:
            self.pause_btn.config(state="normal")
            self.resume_btn.config(state="disabled")
            self.summarize_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            self.send_btn.config(state="normal")
            self.change_scene_btn.config(state="normal")
        else:
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="normal" if self.current_conversation_id else "disabled")
            self.summarize_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.send_btn.config(state="disabled")
            self.change_scene_btn.config(state="disabled")
    
    def update_status(self, message: str):
        """Update the status bar message."""
        self.status_bar.config(text=message)
        self.root.after(5000, lambda: self.status_bar.config(text="Ready"))
    
    def run(self):
        """Start the GUI application."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Application interrupted by user")
        except Exception as e:
            print(f"Application error: {e}")
            messagebox.showerror("Application Error", f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    print("Starting Multi-Agent Conversation Simulator...")
    try:
        app = AgentConversationSimulatorGUI()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {e}")
