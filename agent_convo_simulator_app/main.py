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
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import importlib

# Import our custom modules
try:
    from data_manager import DataManager, Agent, Conversation
    from conversation_engine import ConversationSimulatorEngine
    from config import UI_COLORS, AGENT_SETTINGS
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"LangGraph dependencies not available: {e}")
    print("Please ensure all dependencies are installed.")
    print("Run: pip install -r requirements.txt")
    exit(1)

# Tab Navigation and Button State Management
# Tab indices:
# 0: Agent Management
# 1: Conversation Setup  
# 2: Past Conversations
# 3: Simulation (Main conversation UI)
#
# Recent fixes:
# - Fixed all tab navigation to use correct Simulation tab index (3)
# - Fixed Resume button to become active when conversation is paused
# - Updated update_simulation_controls to handle paused state properly

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
        
        # Force update to ensure proper sizing and positioning
        self.bubble_frame.update_idletasks()
        self.update_idletasks()
        
        # Update scroll region
        self.configure(scrollregion=self.bbox("all"))
        
        # Auto-scroll to the bottom with a small delay to ensure it works
        self.after_idle(lambda: self.yview_moveto(1.0))
        
        return bubble
        
    def clear(self):
        """Clear all chat bubbles."""
        for widget in self.bubble_frame.winfo_children():
            widget.destroy()
        # Reset previous sender tracking
        self.previous_sender = None
    
    def auto_scroll(self):
        """Automatically scroll to the bottom of the chat."""
        try:
            print("DEBUG: auto_scroll called")
            self.update_idletasks()  # Ensure the canvas is updated
            self.yview_moveto(1.0)  # Scroll to the bottom
            print("DEBUG: auto_scroll completed")
        except Exception as e:
            print(f"ERROR in auto_scroll: {e}")


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
        
        # Initialize tool-related variables
        self.tool_vars = {}  # For tool checkboxes
        self.tools_checkboxes_frame = None  # Will be set in create_agents_tab
        self.tooltip = None  # For displaying tool descriptions
        
        # Register window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create the main interface
        self.create_widgets()
        self.load_data()
          # Configure grid weights for responsive design
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def on_closing(self):
        """Handle application closing - stop any active conversations and clean up resources."""
        try:
            print("DEBUG: Application closing, cleaning up resources...")
            
            # Stop any active conversation
            if self.conversation_active and self.conversation_engine and self.current_conversation_id:
                print("DEBUG: Stopping active conversation before closing...")
                try:
                    # Try to pause the conversation first
                    self.conversation_engine.pause_conversation(self.current_conversation_id)
                    print("DEBUG: Conversation paused successfully")
                except Exception as e:
                    print(f"WARNING: Error pausing conversation: {e}")
                
                try:
                    # Try to gracefully stop the conversation engine
                    self.conversation_engine.stop_conversation(self.current_conversation_id)
                    print("DEBUG: Conversation stopped successfully")
                except Exception as e:
                    print(f"WARNING: Error stopping conversation: {e}")
                    
                # Update conversation status in database to reflect it was stopped
                try:
                    conversation = self.data_manager.get_conversation_by_id(self.current_conversation_id)
                    if conversation:
                        conversation.status = "stopped"
                        self.data_manager.save_conversation(conversation)
                        print(f"DEBUG: Updated conversation {self.current_conversation_id} status to 'stopped'")
                except Exception as e:
                    print(f"WARNING: Error updating conversation status: {e}")
            
            # Clean up any other resources here
            
            # Signal all threads to stop if possible
            if hasattr(self, 'conversation_engine') and self.conversation_engine:
                # Set a flag to stop background threads if the engine has such a mechanism
                if hasattr(self.conversation_engine, 'running'):
                    self.conversation_engine.running = False
                    print("DEBUG: Set conversation engine running flag to False")
                
                # Wait a moment for threads to clean up
                time.sleep(0.5)
            
            print("DEBUG: Application shutdown complete")
            # Destroy the application
            self.root.destroy()
            
        except Exception as e:
            print(f"ERROR during application shutdown: {e}")
            # Make sure the application closes even if there's an error
            self.root.destroy()
    
    def create_widgets(self):
        """Create the main GUI widgets."""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Create tabs
        self.create_agents_tab()
        self.create_conversation_tab()
        self.create_past_conversations_tab()
        self.create_simulation_tab()
        
        # Initialize termination condition field based on current invocation method
        self._toggle_termination_condition()
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
        
        ttk.Label(details_frame, text="API Key:").grid(row=4, column=0, sticky="w", pady=2)
        self.agent_api_key_var = tk.StringVar()
        self.agent_api_key_entry = ttk.Entry(details_frame, textvariable=self.agent_api_key_var, width=30, show="*")
        self.agent_api_key_entry.grid(row=4, column=1, sticky="ew", pady=2, padx=(10, 0))
        
        # Tools selection
        ttk.Label(details_frame, text="Tools:").grid(row=5, column=0, sticky="nw", pady=2)
        
        # Create a frame for tools with a scrollbar
        tools_frame = ttk.Frame(details_frame)
        tools_frame.grid(row=5, column=1, sticky="nsew", pady=2, padx=(10, 0))
        details_frame.grid_rowconfigure(5, weight=1)
        
        # Create a canvas for scrolling
        tools_canvas = tk.Canvas(tools_frame, height=100)
        tools_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add a scrollbar
        tools_scrollbar = ttk.Scrollbar(tools_frame, orient="vertical", command=tools_canvas.yview)
        tools_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tools_canvas.configure(yscrollcommand=tools_scrollbar.set)
        
        # Create a frame inside the canvas to hold the checkboxes
        self.tools_checkboxes_frame = ttk.Frame(tools_canvas)
        tools_canvas.create_window((0, 0), window=self.tools_checkboxes_frame, anchor="nw")
        
        # Configure the canvas to scroll with the content
        self.tools_checkboxes_frame.bind("<Configure>", 
            lambda e: tools_canvas.configure(scrollregion=tools_canvas.bbox("all")))
        
        # Dictionary to hold the checkbox variables
        self.tool_vars = {}
        
        # Load tools and create checkboxes
        self.load_tool_checkboxes()
        
        # Save button
        ttk.Button(details_frame, text="Save Agent", command=self.save_agent).grid(row=6, column=1, sticky="e", pady=(10, 0))
    
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
            command=self._toggle_termination_condition        ).pack(side=tk.LEFT)
        
        # Termination condition (available for both modes)
        ttk.Label(settings_frame, text="Termination Condition:").grid(row=4, column=0, sticky="nw", pady=2)
        self.termination_condition_text = scrolledtext.ScrolledText(settings_frame, width=40, height=4)
        self.termination_condition_text.grid(row=4, column=1, sticky="ew", pady=2, padx=(10, 0))
        # Always enable termination condition input for both modes
        self.termination_condition_text.config(state=tk.NORMAL)
        
        # Agent Selector API Key (only visible with agent selector)
        ttk.Label(settings_frame, text="Agent Selector API Key:").grid(row=5, column=0, sticky="w", pady=2)
        self.agent_selector_api_key_var = tk.StringVar()
        self.agent_selector_api_key_entry = ttk.Entry(settings_frame, textvariable=self.agent_selector_api_key_var, width=40, show="*")
        self.agent_selector_api_key_entry.grid(row=5, column=1, sticky="ew", pady=2, padx=(10, 0))
        self.agent_selector_api_key_entry.config(state=tk.DISABLED)  # Disabled by default for round-robin
        
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
        
        # Control buttons
        btn_frame = ttk.Frame(content_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=(20, 0))
        
        ttk.Button(btn_frame, text="Start Conversation", command=self.start_conversation).pack(side=tk.LEFT, padx=(0, 10))
    
    def _toggle_termination_condition(self):
        """Toggle the termination condition text field and agent selector API key based on invocation method."""
        # Always ensure termination condition is enabled for both modes
        self.termination_condition_text.config(state=tk.NORMAL)
        
        # If Agent Selector is chosen, enable the API key field
        if self.invocation_method_var.get() == "agent_selector":
            self.agent_selector_api_key_entry.config(state=tk.NORMAL)
        else:
            # For Round Robin, disable agent selector API key
            # Clear and disable agent selector API key for round robin
            self.agent_selector_api_key_var.set("")
            self.agent_selector_api_key_entry.config(state=tk.DISABLED)
    
    def load_conversation(self):
        """Load a conversation from the setup tab - opens dialog to select conversation."""
        try:
            # Get all conversations
            conversations = self.data_manager.load_conversations()
            
            if not conversations:
                messagebox.showinfo("No Conversations", "No saved conversations found.")
                return
            
            # Create selection dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Load Conversation")
            dialog.geometry("600x400")
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Center the dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")
            
            # Create listbox for conversations
            frame = ttk.Frame(dialog)
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            ttk.Label(frame, text="Select a conversation to load:").pack(anchor=tk.W, pady=(0, 5))
            
            # Listbox with scrollbar
            listbox_frame = ttk.Frame(frame)
            listbox_frame.pack(fill=tk.BOTH, expand=True)
            
            scrollbar = ttk.Scrollbar(listbox_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=listbox.yview)
            
            # Populate listbox
            for conv in conversations:
                agents_count = len(conv.agents)
                display_text = f"{conv.title} - {conv.environment} ({agents_count} agents) - {conv.created_at[:10]}"
                listbox.insert(tk.END, display_text)
            
            # Selected conversation variable
            selected_conversation = [None]
            
            def on_select():
                selection = listbox.curselection()
                if selection:
                    selected_conversation[0] = conversations[selection[0]]
                    dialog.destroy()
            
            def on_cancel():
                dialog.destroy()
            
            # Buttons
            btn_frame = ttk.Frame(frame)
            btn_frame.pack(fill=tk.X, pady=(10, 0))
            
            ttk.Button(btn_frame, text="Load", command=on_select).pack(side=tk.RIGHT, padx=(5, 0))
            ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT)
            
            # Handle double-click
            listbox.bind('<Double-1>', lambda e: on_select())
            
            # Wait for dialog to close
            dialog.wait_window()
            
            # Load the selected conversation if one was chosen
            if selected_conversation[0]:
                # Make sure the simulation tab is created and chat canvas is initialized
                if not hasattr(self, 'chat_canvas'):
                    print("DEBUG: Chat canvas not initialized yet, creating simulation tab")
                    try:
                        # Try to create the simulation tab if it doesn't exist
                        self.create_simulation_tab()
                    except Exception as e:
                        print(f"ERROR: Could not create simulation tab: {e}")
                        messagebox.showerror("Error", "Could not initialize the simulation tab. Please restart the application.")
                        return
                
                # Switch to simulation tab and ensure it's visible
                self.notebook.select(3)  # Simulation tab
                self.root.update_idletasks()
                self.root.update()
                
                # Load the selected conversation
                self.load_selected_conversation(selected_conversation[0])
                
                # Ensure the tab remains visible after loading
                if hasattr(self, 'chat_canvas'):
                    self.chat_canvas.focus_set()
                
        except Exception as e:
            print(f"Error in load_conversation: {e}")
            messagebox.showerror("Error", f"Failed to load conversation: {str(e)}")
    
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
        # Load past conversations for the Past Conversations tab
        self.refresh_past_conversations()
    
    def update_simulation_controls(self, conversation_active: bool, paused: bool = False):
        """Update the state of simulation control buttons based on conversation status."""
        if conversation_active:
            # Enable controls when conversation is active
            self.pause_btn.config(state="normal")
            self.summarize_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            self.send_btn.config(state="normal")
            # Resume button starts disabled (only enabled when paused)
            self.resume_btn.config(state="disabled")
            # Enable scene change when conversation is active
            if hasattr(self, 'change_scene_btn'):
                self.change_scene_btn.config(state="normal")
        elif paused:
            # When conversation is paused, enable resume and keep some controls available
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="normal")  # Enable resume when paused
            self.summarize_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            self.send_btn.config(state="disabled")
            # Disable scene change when paused
            if hasattr(self, 'change_scene_btn'):
                self.change_scene_btn.config(state="disabled")
        else:
            # Disable all controls when no conversation is active
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="disabled")
            self.summarize_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.send_btn.config(state="disabled")
            # Disable scene change when no conversation
            if hasattr(self, 'change_scene_btn'):
                self.change_scene_btn.config(state="disabled")
    
    def update_status(self, message: str):
        """Update the status message (placeholder for now)."""
        print(f"STATUS: {message}")
    
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
        self.agent_api_key_var.set(agent.api_key or "")  # Load API key
        
        self.agent_prompt_text.delete(1.0, tk.END)
        self.agent_prompt_text.insert(1.0, agent.base_prompt)
        
        # Set tool checkboxes
        if hasattr(agent, 'tools') and self.tool_vars:
            for tool_name, var in self.tool_vars.items():
                var.set(tool_name in getattr(agent, 'tools', []))
    
    def new_agent(self):
        """Create a new agent."""
        self.clear_agent_form()
        self.agent_name_entry.focus()
    def clear_agent_form(self):
        """Clear the agent form."""
        self.agent_name_var.set("")
        self.agent_role_var.set("")
        self.agent_traits_var.set("")
        self.agent_api_key_var.set("")  # Clear API key
        self.agent_prompt_text.delete(1.0, tk.END)
        
        # Clear tool checkboxes
        for var in self.tool_vars.values():
            var.set(False)
    
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
        api_key = self.agent_api_key_var.get().strip()  # Get API key
        prompt = self.agent_prompt_text.get(1.0, tk.END).strip()
        
        if not all([name, role, prompt]):
            messagebox.showerror("Missing Information", "Please fill in all required fields (Name, Role, and Base Prompt).")
            return
        
        # Parse personality traits
        traits = [t.strip() for t in traits_str.split(",") if t.strip()] if traits_str else []
        
        # Get selected tools
        selected_tools = [name for name, var in self.tool_vars.items() if var.get()]
        
        # Check if editing existing agent
        selection = self.agents_listbox.curselection()
        if selection:
            agents = self.data_manager.load_agents()
            if selection[0] < len(agents):
                # Edit existing agent
                agent = agents[selection[0]]
                agent.name = name
                agent.role = role
                agent.base_prompt = prompt  # Fix: Set base_prompt
                agent.personality_traits = traits
                agent.api_key = api_key if api_key else None  # Set API key
                agent.tools = selected_tools  # Save selected tools
                self.data_manager.save_agent(agent)
                self.update_status(f"Agent '{name}' updated.")
            else:
                # Create new agent
                agent = Agent.create_new(name, role, prompt, traits, api_key=api_key if api_key else None, tools=selected_tools)
                self.data_manager.save_agent(agent)
                self.update_status(f"Agent '{name}' created.")
        else:
            # Create new agent
            agent = Agent.create_new(name, role, prompt, traits, api_key=api_key if api_key else None, tools=selected_tools)
            self.data_manager.save_agent(agent)
            self.update_status(f"Agent '{name}' created.")
        
        self.refresh_agents_list()
        self.refresh_agent_checkboxes()
    
    def start_conversation(self):
        """Start a new conversation."""
        title = self.conv_title_var.get().strip()
        environment = self.conv_env_var.get().strip()
        scene = self.conv_scene_text.get(1.0, tk.END).strip()
        
        if not all([title, environment, scene]):
            messagebox.showerror("Missing Information", "Please fill in title, environment and scene description.")
            return
        
        # Get selected agents
        selected_agents = []
        for agent, var in self.agent_checkboxes:
            if var.get():
                print(f"DEBUG: Selected agent '{agent.name}' with API key: '{agent.api_key}' (type: {type(agent.api_key)})")
                selected_agents.append(agent)
        
        if len(selected_agents) < 2 or len(selected_agents) > 4:
            messagebox.showerror("Invalid Selection", "Please select 2-4 agents for the conversation.")
            return
        
        try:
            # Show loading message
            self.update_status("Starting conversation...")
            
            # Initialize conversation engine (no default API key - agents will use their own)
            self.conversation_engine = ConversationSimulatorEngine()            # Initialize agent colors dictionary
            self.agent_colors = {}
              # Assign unique colors to agents from the predefined palette
            available_colors = list(UI_COLORS["agent_colors"])
            # No random shuffle needed - we'll assign colors in order to ensure consistency            # Get invocation method and termination condition
            invocation_method = self.invocation_method_var.get()
            
            # Get termination condition for both modes (now supported for both round robin and agent selector)
            termination_condition = self.termination_condition_text.get(1.0, tk.END).strip()
            # Remove any hint text if it's still there
            if "Agents will be reminded every" in termination_condition or "The agent selector will check this condition" in termination_condition:
                termination_condition = None
            elif not termination_condition:
                termination_condition = None  # Use None instead of empty string
            
            print(f"DEBUG: Using termination condition: {termination_condition} with mode: {invocation_method}")
              # Get agent selector API key if using agent_selector method
            agent_selector_api_key = None
            if invocation_method == "agent_selector":
                agent_selector_api_key = self.agent_selector_api_key_var.get().strip()
                if not agent_selector_api_key:
                    agent_selector_api_key = None  # Will use default in engine
            
            # Create conversation record with agent colors
            conversation = Conversation.create_new(
                title, environment, scene, [agent.id for agent in selected_agents],
                invocation_method=invocation_method, 
                termination_condition=termination_condition,
                agent_selector_api_key=agent_selector_api_key
            )
            
            # Prepare agent configs
            agents_config = []
            for i, agent in enumerate(selected_agents):
                # Assign a color to each agent
                color = available_colors[i % len(available_colors)]
                self.agent_colors[agent.name] = color
                
                print(f"DEBUG: Agent '{agent.name}' has API key: {'Yes' if agent.api_key else 'No'}")
                if agent.api_key:
                    print(f"DEBUG: Agent '{agent.name}' API key starts with: {agent.api_key[:10]}...")
                
                agents_config.append({
                    "id": agent.id,  # Add agent ID for tool loading
                    "name": agent.name,
                    "role": agent.role,
                    "base_prompt": agent.base_prompt,
                    "color": color,  # Add color to agent config
                    "api_key": agent.api_key  # Add agent's API key
                })
            
            print(f"DEBUG: Agent selector API key provided: {'Yes' if agent_selector_api_key else 'No'}")
            if agent_selector_api_key:
                print(f"DEBUG: Agent selector API key starts with: {agent_selector_api_key[:10]}...")
            
            # Add agent colors to the conversation
            conversation.agent_colors = self.agent_colors
            self.data_manager.save_conversation(conversation)
            self.current_conversation_id = conversation.id
            
            # Start the conversation
            thread_id = self.conversation_engine.start_conversation(
                conversation.id, agents_config, environment, scene,
                invocation_method=invocation_method,
                termination_condition=termination_condition,
                agent_selector_api_key=agent_selector_api_key
            )
            
            # Register callback for message updates
            self.conversation_engine.register_message_callback(
                conversation.id, self.on_message_received
            )            # Update UI
            self.conversation_active = True
            self.update_simulation_controls(True)
            self.current_env_label.config(text=environment)
            
            # Make sure the simulation tab is created and chat canvas is initialized
            if not hasattr(self, 'chat_canvas'):
                print("DEBUG: Chat canvas not initialized yet, creating simulation tab")
                try:
                    # Try to create the simulation tab if it doesn't exist
                    self.create_simulation_tab()
                except Exception as e:
                    print(f"ERROR: Could not create simulation tab: {e}")
                    messagebox.showerror("Error", "Could not initialize the simulation tab. Please restart the application.")
                    return
                    
            # Switch to the simulation tab and ensure it's visible
            self.notebook.select(3)  # Simulation tab is index 3
            self.root.update_idletasks()
            self.root.update()
            
            # Verify the chat canvas is initialized
            if not hasattr(self, 'chat_canvas') or not self.chat_canvas:
                print("ERROR: Chat canvas not initialized")
                messagebox.showerror("Error", "Chat canvas not initialized. Please restart the application.")
                return
                
            # Force focus to ensure the tab switch is visible
            self.chat_canvas.focus_set()
            
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
            self.notebook.select(3)
            
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
            try:
                self.conversation_engine.pause_conversation(self.current_conversation_id)
                self.conversation_active = False
                self.update_simulation_controls(False, paused=True)  # Pass paused=True
                self.update_status("Conversation paused.")
                
                # Add system message about pause
                self.chat_canvas.add_bubble(
                    "System", 
                    "Conversation paused.", 
                    datetime.now().strftime("%H:%M:%S"), 
                    "system", 
                    UI_COLORS["system_bubble"]
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to pause conversation: {str(e)}")

    def resume_conversation(self):
        """Resume a paused conversation or restart a terminated one."""
        if not self.current_conversation_id:
            messagebox.showwarning("Warning", "No conversation to resume.")
            return
        
        try:
            # Check if conversation was terminated due to termination condition
            if hasattr(self, 'loaded_conversation'):
                conversation = self.loaded_conversation
            else:
                conversation = self.data_manager.get_conversation_by_id(self.current_conversation_id)
            
            if conversation and conversation.status == "completed":
                # This was a terminated conversation, ask for new termination condition
                self.show_resume_terminated_dialog(conversation)
            else:
                # Regular resume of paused conversation
                if self.conversation_engine:
                    self.conversation_engine.resume_conversation(self.current_conversation_id)
                    self.conversation_active = True
                    self.update_simulation_controls(True)
                    self.update_status("Conversation resumed.")
                    
                    # Add system message about resume
                    self.chat_canvas.add_bubble(
                        "System", 
                        "Conversation resumed.", 
                        datetime.now().strftime("%H:%M:%S"), 
                        "system", 
                        UI_COLORS["system_bubble"]
                    )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to resume conversation: {str(e)}")

    def show_resume_terminated_dialog(self, conversation: Conversation):
        """Show dialog to get new termination condition for terminated conversation."""
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Resume Terminated Conversation")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Configure grid
        dialog.grid_rowconfigure(2, weight=1)
        dialog.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(dialog, text="Resume Terminated Conversation", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, pady=10)
        
        # Information
        info_text = f"The conversation '{conversation.title}' was terminated because its termination condition was met.\n\n"
        info_text += "To resume the conversation, please provide a new termination condition:"
        
        info_label = ttk.Label(dialog, text=info_text, wraplength=550, justify="left")
        info_label.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        # Termination condition input
        input_frame = ttk.LabelFrame(dialog, text="New Termination Condition", padding="10")
        input_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)
        input_frame.grid_rowconfigure(0, weight=1)
        input_frame.grid_columnconfigure(0, weight=1)
        
        termination_text = scrolledtext.ScrolledText(input_frame, width=60, height=8)
        termination_text.grid(row=0, column=0, sticky="nsew")
        
        # Current termination condition (if any)
        if hasattr(conversation, 'termination_condition') and conversation.termination_condition:
            termination_text.insert(1.0, f"Previous condition: {conversation.termination_condition}\n\n")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=3, column=0, pady=20)
        
        def on_resume():
            new_condition = termination_text.get(1.0, tk.END).strip()
            if not new_condition:
                messagebox.showwarning("Warning", "Please enter a termination condition.")
                return
            
            # Update conversation with new termination condition
            conversation.termination_condition = new_condition
            conversation.status = "active"
            self.data_manager.save_conversation(conversation)
            
            dialog.destroy()
            self.restart_conversation_with_new_condition(conversation, new_condition)
        
        def on_cancel():
            dialog.destroy()
        
        ttk.Button(button_frame, text="Resume with New Condition", command=on_resume).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)

    def restart_conversation_with_new_condition(self, conversation: Conversation, new_termination_condition: str):
        """Restart a terminated conversation with a new termination condition."""
        try:
            # Initialize conversation engine if needed
            if not self.conversation_engine:
                self.conversation_engine = ConversationSimulatorEngine()
            
            # Get agent configs for the conversation
            all_agents = self.data_manager.load_agents()
            agents_config = []
            
            for agent_id in conversation.agents:
                agent = next((a for a in all_agents if a.id == agent_id), None)
                if agent:
                    # Get color from agent or conversation agent_colors
                    color = agent.color
                    if not color and hasattr(conversation, 'agent_colors') and agent.name in conversation.agent_colors:
                        color = conversation.agent_colors[agent.name]
                    
                    agents_config.append({
                        "name": agent.name,
                        "role": agent.role,
                        "base_prompt": agent.base_prompt,
                        "color": color
                    })
            
            if len(agents_config) < 2:
                messagebox.showerror("Error", "Not enough agents found to resume conversation.")
                return
            
            # Get invocation method
            invocation_method = getattr(conversation, 'invocation_method', 'round_robin')
            
            # Start conversation with new termination condition
            thread_id = self.conversation_engine.start_conversation(
                conversation_id=conversation.id,
                agents_config=agents_config,
                environment=conversation.environment,
                scene_description=conversation.scene_description,
                invocation_method=invocation_method,
                termination_condition=new_termination_condition
            )
            
            if thread_id:
                self.current_conversation_id = conversation.id
                self.conversation_active = True
                self.update_simulation_controls(True)
                
                # Set up message callback
                self.conversation_engine.register_message_callback(conversation.id, self.on_message_received)
                
                # Add system message about restart
                restart_msg = f"CONVERSATION RESTARTED with new termination condition at {datetime.now().strftime('%H:%M:%S')}"
                self.chat_canvas.add_bubble("System", restart_msg, datetime.now().strftime("%H:%M:%S"), "system", UI_COLORS["system_bubble"])
                
                # Switch to simulation tab
                self.notebook.select(3)
                
                self.update_status(f"Conversation '{conversation.title}' restarted with new termination condition!")
                
            else:
                messagebox.showerror("Error", "Failed to restart conversation.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to restart conversation: {str(e)}")

    def stop_conversation(self):
        """Stop the active conversation."""
        if self.conversation_engine and self.current_conversation_id:
            try:
                result = messagebox.askyesno(
                    "Stop Conversation", 
                    "Are you sure you want to stop the conversation?\n\nThis will end the current session."
                )
                
                if result:
                    self.conversation_engine.stop_conversation(self.current_conversation_id)
                    self.conversation_active = False
                    self.current_conversation_id = None
                    self.update_simulation_controls(False)
                    self.update_status("Conversation stopped.")
                    
                    # Add system message about stopping
                    self.chat_canvas.add_bubble(
                        "System", 
                        "Conversation stopped by user.", 
                        datetime.now().strftime("%H:%M:%S"), 
                        "system", 
                        UI_COLORS["system_bubble"]
                    )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to stop conversation: {str(e)}")

    def summarize_conversation(self):
        """Generate and display a summary of the current conversation."""
        if self.conversation_engine and self.current_conversation_id:
            try:
                summary = self.conversation_engine.get_conversation_summary(self.current_conversation_id)
                
                # Create summary dialog
                dialog = tk.Toplevel(self.root)
                dialog.title("Conversation Summary")
                dialog.geometry("600x400")
                dialog.transient(self.root)
                
                # Center the dialog
                dialog.update_idletasks()
                x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
                y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
                dialog.geometry(f"+{x}+{y}")
                
                # Configure grid
                dialog.grid_rowconfigure(1, weight=1)
                dialog.grid_columnconfigure(0, weight=1)
                
                # Title
                title_label = ttk.Label(dialog, text="Conversation Summary", font=("Arial", 14, "bold"))
                title_label.grid(row=0, column=0, pady=10)
                
                # Summary text
                summary_frame = ttk.Frame(dialog)
                summary_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
                summary_frame.grid_rowconfigure(0, weight=1)
                summary_frame.grid_columnconfigure(0, weight=1)
                
                summary_text = scrolledtext.ScrolledText(summary_frame, width=70, height=20, wrap=tk.WORD)
                summary_text.grid(row=0, column=0, sticky="nsew")
                summary_text.insert(1.0, summary)
                summary_text.config(state=tk.DISABLED)
                
                # Close button
                ttk.Button(dialog, text="Close", command=dialog.destroy).grid(row=2, column=0, pady=10)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate summary: {str(e)}")
    
    def change_scene(self):
        """Change the scene/environment for the active conversation."""
        if not (self.conversation_engine and self.current_conversation_id):
            messagebox.showwarning("Warning", "No conversation to change scene for.")
            return
        
        new_env = self.new_env_var.get().strip()
        new_scene = self.new_scene_text.get(1.0, tk.END).strip()
        
        if not new_env or not new_scene:
            messagebox.showwarning("Warning", "Please enter both new environment and scene description.")
            return
        
        try:
            result = self.conversation_engine.change_scene(
                self.current_conversation_id, 
                new_env, 
                new_scene
            )
            
            if result.get("success"):
                self.current_env_label.config(text=new_env)
                self.new_env_var.set("")
                self.new_scene_text.delete(1.0, tk.END)
                
                # Add system message about scene change
                scene_msg = f"Scene changed to: {new_env}. {new_scene}"
                self.chat_canvas.add_bubble(
                    "System", 
                    scene_msg, 
                    datetime.now().strftime("%H:%M:%S"), 
                    "system", 
                    UI_COLORS["system_bubble"]
                )
                
                self.update_status("Scene changed successfully!")
            else:
                messagebox.showerror("Error", "Failed to change scene.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to change scene: {str(e)}")
    
    def create_past_conversations_tab(self):
        """Create the past conversations management tab."""
        past_conv_frame = ttk.Frame(self.notebook)
        self.notebook.add(past_conv_frame, text="Past Conversations")
        
        # Configure grid
        past_conv_frame.grid_rowconfigure(1, weight=1)
        past_conv_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(past_conv_frame, text="Past Conversations", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(10, 20))
        
        # Main container for switching between list and edit views
        self.past_conv_container = ttk.Frame(past_conv_frame)
        self.past_conv_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.past_conv_container.grid_rowconfigure(0, weight=1)
        self.past_conv_container.grid_columnconfigure(0, weight=1)
        
        # Create list view
        self.create_past_conversations_list_view()
        
        # Create edit view (initially hidden)
        self.create_past_conversations_edit_view()
        
        # Show list view by default
        self.show_past_conversations_list()
    
    def create_past_conversations_list_view(self):
        """Create the list view for past conversations."""
        self.past_conv_list_frame = ttk.Frame(self.past_conv_container)
        self.past_conv_list_frame.grid(row=0, column=0, sticky="nsew")
        self.past_conv_list_frame.grid_rowconfigure(1, weight=1)
        self.past_conv_list_frame.grid_columnconfigure(0, weight=1)
        
        # Instructions
        instructions = ttk.Label(
            self.past_conv_list_frame, 
            text="Select a conversation below to edit its settings or load it into the simulation.",
            font=("Arial", 10)
        )
        instructions.grid(row=0, column=0, pady=(0, 10))
        
        # Conversations list with scrollbar
        list_container = ttk.Frame(self.past_conv_list_frame)
        list_container.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        list_container.grid_rowconfigure(0, weight=1)
        list_container.grid_columnconfigure(0, weight=1)
        
        # Listbox for conversations
        self.past_conversations_listbox = tk.Listbox(list_container, font=("Arial", 10))
        self.past_conversations_listbox.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbar
        past_conv_scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.past_conversations_listbox.yview)
        past_conv_scrollbar.grid(row=0, column=1, sticky="ns")
        self.past_conversations_listbox.configure(yscrollcommand=past_conv_scrollbar.set)
        
        # Bind selection event
        self.past_conversations_listbox.bind('<<ListboxSelect>>', self.on_past_conversation_select)
        
        # Action buttons
        btn_frame = ttk.Frame(self.past_conv_list_frame)
        btn_frame.grid(row=2, column=0, pady=(10, 0))
        
        self.edit_conv_btn = ttk.Button(btn_frame, text="Edit", command=self.edit_selected_conversation, state="disabled")
        self.edit_conv_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.load_conv_btn = ttk.Button(btn_frame, text="Load", command=self.load_selected_conversation_from_list, state="disabled")
        self.load_conv_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Refresh button
        ttk.Button(btn_frame, text="Refresh", command=self.refresh_past_conversations).pack(side=tk.LEFT, padx=(0, 10))
    
    def create_past_conversations_edit_view(self):
        """Create the edit view for conversation settings."""
        self.past_conv_edit_frame = ttk.Frame(self.past_conv_container)
        self.past_conv_edit_frame.grid(row=0, column=0, sticky="nsew")
        self.past_conv_edit_frame.grid_rowconfigure(1, weight=1)
        self.past_conv_edit_frame.grid_columnconfigure(0, weight=1)
        
        # Title and back button
        header_frame = ttk.Frame(self.past_conv_edit_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        header_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Button(header_frame, text="← Back", command=self.show_past_conversations_list).grid(row=0, column=0, sticky="w")
        
        self.edit_conv_title_label = ttk.Label(header_frame, text="Edit Conversation", font=("Arial", 14, "bold"))
        self.edit_conv_title_label.grid(row=0, column=1)
        
        # Edit form
        edit_form_frame = ttk.LabelFrame(self.past_conv_edit_frame, text="Conversation Settings", padding="20")
        edit_form_frame.grid(row=1, column=0, sticky="nsew", padx=20)
        edit_form_frame.grid_columnconfigure(1, weight=1)
        
        # Conversation title (read-only)
        ttk.Label(edit_form_frame, text="Title:").grid(row=0, column=0, sticky="w", pady=5)
        self.edit_conv_title_var = tk.StringVar()
        title_entry = ttk.Entry(edit_form_frame, textvariable=self.edit_conv_title_var, state="readonly", width=40)
        title_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=(10, 0))
        
        # Environment (read-only)
        ttk.Label(edit_form_frame, text="Environment:").grid(row=1, column=0, sticky="w", pady=5)
        self.edit_conv_env_var = tk.StringVar()
        env_entry = ttk.Entry(edit_form_frame, textvariable=self.edit_conv_env_var, state="readonly", width=40)
        env_entry.grid(row=1, column=1, sticky="ew", pady=5, padx=(10, 0))
        
        # Invocation method
        ttk.Label(edit_form_frame, text="Invocation Method:").grid(row=2, column=0, sticky="w", pady=5)
        self.edit_invocation_method_var = tk.StringVar(value="round_robin")
        
        method_frame = ttk.Frame(edit_form_frame)
        method_frame.grid(row=2, column=1, sticky="ew", pady=5, padx=(10, 0))
        
        ttk.Radiobutton(
            method_frame, 
            text="Round Robin", 
            variable=self.edit_invocation_method_var, 
            value="round_robin",
            command=self._toggle_edit_termination_condition
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            method_frame, 
            text="Agent Selector (LLM)", 
            variable=self.edit_invocation_method_var, 
            value="agent_selector",
            command=self._toggle_edit_termination_condition
        ).pack(side=tk.LEFT)
        
        # Termination condition
        ttk.Label(edit_form_frame, text="Termination Condition:").grid(row=3, column=0, sticky="nw", pady=5)
        self.edit_termination_condition_text = scrolledtext.ScrolledText(edit_form_frame, width=40, height=4)
        self.edit_termination_condition_text.grid(row=3, column=1, sticky="ew", pady=5, padx=(10, 0))
        
        # Agent selector API key
        ttk.Label(edit_form_frame, text="Agent Selector API Key:").grid(row=4, column=0, sticky="w", pady=5)
        self.edit_agent_selector_api_key_var = tk.StringVar()
        self.edit_agent_selector_api_key_entry = ttk.Entry(
            edit_form_frame, 
            textvariable=self.edit_agent_selector_api_key_var, 
            width=40, 
            show="*"
        )
        self.edit_agent_selector_api_key_entry.grid(row=4, column=1, sticky="ew", pady=5, padx=(10, 0))
        
        # Save and cancel buttons
        button_frame = ttk.Frame(self.past_conv_edit_frame)
        button_frame.grid(row=2, column=0, pady=(20, 0))
        
        ttk.Button(button_frame, text="Save Changes", command=self.save_conversation_edits).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=self.show_past_conversations_list).pack(side=tk.LEFT)
        
        # Initially hide the edit frame
        self.past_conv_edit_frame.grid_remove()
    def _toggle_edit_termination_condition(self):
        """Toggle the termination condition and API key fields in edit mode."""
        if self.edit_invocation_method_var.get() == "agent_selector":
            # For agent selector, enable both termination condition and API key fields
            self.edit_termination_condition_text.config(state=tk.NORMAL)
            self.edit_agent_selector_api_key_entry.config(state=tk.NORMAL)
        else:
            # For Round Robin, enable termination condition but disable agent selector API key
            self.edit_termination_condition_text.config(state=tk.NORMAL)
            
            # Clear and disable agent selector API key for round robin
            self.edit_agent_selector_api_key_var.set("")
            self.edit_agent_selector_api_key_entry.config(state=tk.DISABLED)
    
    def show_past_conversations_list(self):
        """Show the past conversations list view."""
        self.past_conv_edit_frame.grid_remove()
        self.past_conv_list_frame.grid()
        self.refresh_past_conversations()
    
    def show_past_conversations_edit(self):
        """Show the past conversations edit view."""
        self.past_conv_list_frame.grid_remove()
        self.past_conv_edit_frame.grid()
    
    def refresh_past_conversations(self):
        """Refresh the past conversations list."""
        print("DEBUG: refresh_past_conversations() called")
        print(f"DEBUG: DataManager methods: {[method for method in dir(self.data_manager) if not method.startswith('_')]}")
        self.past_conversations_listbox.delete(0, tk.END)
        try:
            conversations = self.data_manager.load_conversations()
            print(f"DEBUG: Loaded {len(conversations)} conversations")
            self.past_conversations_data = conversations  # Store for later use
            
            for conv in conversations:
                # Format display text
                status_indicator = "🟢" if conv.status == "active" else "🔴" if conv.status == "completed" else "⏸️"
                method_indicator = "🤖" if getattr(conv, 'invocation_method', 'round_robin') == "agent_selector" else "🔄"
                display_text = f"{status_indicator} {method_indicator} {conv.title} - {conv.environment}"
                print(f"DEBUG: Adding conversation: {display_text}")
                self.past_conversations_listbox.insert(tk.END, display_text)
        except Exception as e:
            print(f"Error loading conversations: {e}")
            self.past_conversations_data = []
    
    def on_past_conversation_select(self, event):
        """Handle conversation selection in the past conversations list."""
        selection = self.past_conversations_listbox.curselection()
        if selection:
            self.edit_conv_btn.config(state="normal")
            self.load_conv_btn.config(state="normal")
        else:
            self.edit_conv_btn.config(state="disabled")
            self.load_conv_btn.config(state="disabled")
    
    def edit_selected_conversation(self):
        """Edit the selected conversation."""
        selection = self.past_conversations_listbox.curselection()
        if not selection:
            return
        
        if not hasattr(self, 'past_conversations_data') or not self.past_conversations_data:
            messagebox.showerror("Error", "No conversation data available.")
            return
        
        conversation = self.past_conversations_data[selection[0]]
        self.current_editing_conversation = conversation
        
        # Load conversation data into edit form
        self.edit_conv_title_var.set(conversation.title)
        self.edit_conv_env_var.set(conversation.environment)
        
        # Load invocation method
        invocation_method = getattr(conversation, 'invocation_method', 'round_robin')
        self.edit_invocation_method_var.set(invocation_method)
        
        # Load termination condition
        termination_condition = getattr(conversation, 'termination_condition', '')
        self.edit_termination_condition_text.delete(1.0, tk.END)
        if termination_condition:
            self.edit_termination_condition_text.insert(1.0, termination_condition)
        
        # Load agent selector API key
        agent_selector_api_key = getattr(conversation, 'agent_selector_api_key', '')
        self.edit_agent_selector_api_key_var.set(agent_selector_api_key or '')
          # Toggle fields based on invocation method
        self._toggle_edit_termination_condition()          # Show edit view
        self.show_past_conversations_edit()
        
    def load_selected_conversation_from_list(self):
        """Load the selected conversation from the past conversations list."""
        print("DEBUG: load_selected_conversation_from_list called")
        selection = self.past_conversations_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a conversation to load.")
            print("DEBUG: No conversation selected")
            return
        
        if not hasattr(self, 'past_conversations_data') or not self.past_conversations_data:
            messagebox.showerror("Error", "No conversation data available.")
            print("DEBUG: No conversation data available")
            return
            
        # Make sure the simulation tab is created and chat canvas is initialized
        if not hasattr(self, 'chat_canvas'):
            print("DEBUG: Chat canvas not initialized yet, creating simulation tab")
            try:
                # Try to create the simulation tab if it doesn't exist
                self.create_simulation_tab()
                # Force UI update
                self.root.update_idletasks()
                self.root.update()
            except Exception as e:
                print(f"ERROR: Could not create simulation tab: {e}")
                messagebox.showerror("Error", "Could not initialize the simulation tab. Please restart the application.")
                return
        
        try:
            conversation = self.past_conversations_data[selection[0]]
            print(f"DEBUG: Selected conversation: {conversation.title}")
            
            print("DEBUG: Proceeding with conversation loading")
            print("DEBUG: Switching to simulation tab")
            # First switch to the simulation tab so UI updates are visible
            self.notebook.select(3)  # Simulation tab is index 3
            # Force UI update in the correct order
            self.root.update_idletasks()
            self.root.update()
            
            # Give focus to the chat canvas to ensure the tab is visible
            if hasattr(self, 'chat_canvas'):
                self.chat_canvas.focus_set()
            
            # Add some delay to ensure UI is updated before loading conversation
            # This can help with UI responsiveness
            self.root.after(100, lambda: self._load_conversation_after_delay(conversation))
            
        except Exception as e:
            print(f"ERROR in load_selected_conversation_from_list: {e}")
            messagebox.showerror("Error", f"Failed to load conversation: {str(e)}")
    
    def _load_conversation_after_delay(self, conversation):
        """Helper method to load a conversation after a short delay."""
        print("DEBUG: _load_conversation_after_delay called")
        
        # Verify the chat canvas is initialized
        if not hasattr(self, 'chat_canvas') or not self.chat_canvas:
            print("ERROR: Chat canvas still not initialized after delay")
            messagebox.showerror("Error", "Chat canvas not initialized. Please restart the application.")
            return
            
        # Now load the conversation
        self.load_selected_conversation(conversation)
        
        # After loading the conversation, ensure the tab is visible again
        self.notebook.select(3)  # Ensure simulation tab is still selected
        self.root.update_idletasks()
        self.root.update()
        # Give focus to a UI element in the simulation tab
        if hasattr(self, 'chat_canvas'):
            self.chat_canvas.focus_set()
    
    def save_conversation_edits(self):
        """Save the edited conversation settings."""
        if not hasattr(self, 'current_editing_conversation'):
            messagebox.showerror("Error", "No conversation selected for editing.")
            return
        
        conversation = self.current_editing_conversation
        
        # Update conversation settings
        conversation.invocation_method = self.edit_invocation_method_var.get()
          # Update termination condition (for both modes)
        termination_condition = self.edit_termination_condition_text.get(1.0, tk.END).strip()
        
        # Remove any hint text if it's still there
        if "Agents will be reminded every" in termination_condition or "The agent selector will check this condition" in termination_condition:
            termination_condition = None
        
        conversation.termination_condition = termination_condition if termination_condition else None
        print(f"DEBUG: Updated termination condition to: {conversation.termination_condition} for mode: {conversation.invocation_method}")
        
        # Update agent selector API key
        agent_selector_api_key = self.edit_agent_selector_api_key_var.get().strip()
        conversation.agent_selector_api_key = agent_selector_api_key if agent_selector_api_key else None
        
        # Save to file
        try:
            self.data_manager.save_conversation(conversation)
            messagebox.showinfo("Success", "Conversation settings saved successfully!")
            self.show_past_conversations_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save conversation: {str(e)}")
    
    def run(self):
        """Start the GUI application main loop."""
        print("Starting Multi-Agent Conversation Simulator...")
        self.root.mainloop()
    def load_selected_conversation(self, conversation):
        """Load a selected conversation into the simulation tab."""
        try:
            print(f"DEBUG: ===== LOADING EXISTING CONVERSATION =====")
            print(f"DEBUG: Conversation: {conversation.title}")
            print(f"DEBUG: Conversation ID: {conversation.id}")
            print(f"DEBUG: Number of existing messages: {len(conversation.messages)}")
            print(f"DEBUG: Conversation status: {conversation.status}")
            print(f"DEBUG: ===== END LOADING INFO =====")
            
            # Store the current conversation ID
            self.current_conversation_id = conversation.id
            
            # Get the agents for this conversation
            all_agents = self.data_manager.load_agents()
            conversation_agents = []
            
            for agent_id in conversation.agents:
                agent = next((a for a in all_agents if a.id == agent_id), None)
                if agent:
                    conversation_agents.append(agent)
                else:
                    print(f"Warning: Agent with ID {agent_id} not found")
            
            if not conversation_agents:
                messagebox.showerror("Error", "No valid agents found for this conversation.")
                print("DEBUG: No valid agents found")
                return
            
            print("DEBUG: Initializing conversation engine")
                
            self.conversation_engine = ConversationSimulatorEngine()
            
            # Prepare agent configs with their individual API keys
            agents_config = []
            self.agent_colors = conversation.agent_colors if hasattr(conversation, 'agent_colors') and conversation.agent_colors else {}
            
            # If no colors are stored, assign them
            if not self.agent_colors:
                available_colors = list(UI_COLORS["agent_colors"])
                for i, agent in enumerate(conversation_agents):
                    color = available_colors[i % len(available_colors)]
                    self.agent_colors[agent.name] = color
            
            for agent in conversation_agents:
                print(f"DEBUG: Loading agent '{agent.name}' with API key: {'Yes' if agent.api_key else 'No'}")
                if agent.api_key:
                    print(f"DEBUG: Agent '{agent.name}' API key starts with: {agent.api_key[:10]}...")
                
                agents_config.append({
                    "id": agent.id,  # Add agent ID for tool loading
                    "name": agent.name,
                    "role": agent.role,
                    "base_prompt": agent.base_prompt,
                    "color": self.agent_colors.get(agent.name, UI_COLORS["agent_colors"][0]),
                    "api_key": agent.api_key
                })
            
            # Get agent selector API key if available
            agent_selector_api_key = getattr(conversation, 'agent_selector_api_key', None)
            print(f"DEBUG: Loading conversation with agent selector API key: {'Yes' if agent_selector_api_key else 'No'}")
            if agent_selector_api_key:
                print(f"DEBUG: Agent selector API key starts with: {agent_selector_api_key[:10]}...")
            
            # Start the conversation with existing messages
            thread_id = self.conversation_engine.start_conversation(
                conversation.id, 
                agents_config, 
                conversation.environment, 
                conversation.scene_description,
                invocation_method=getattr(conversation, 'invocation_method', 'round_robin'),
                termination_condition=getattr(conversation, 'termination_condition', None),
                agent_selector_api_key=agent_selector_api_key
            )
            
            # Restore agent_sending_messages and conversation state if they exist
            if hasattr(conversation, 'agent_sending_messages') and conversation.agent_sending_messages:
                print(f"DEBUG: Restoring agent_sending_messages for {len(conversation.agent_sending_messages)} agents")
                if conversation.id in self.conversation_engine.active_conversations:
                    conv_data = self.conversation_engine.active_conversations[conversation.id]
                    conv_data["agent_sending_messages"] = conversation.agent_sending_messages
                    print("DEBUG: Agent context restored successfully")
            else:
                print("DEBUG: No stored agent_sending_messages found, will use conversation messages")
            
            # Restore the conversation messages to the engine's internal state
            if hasattr(conversation, 'messages') and conversation.messages:
                print(f"DEBUG: Restoring {len(conversation.messages)} messages to engine state")
                if conversation.id in self.conversation_engine.active_conversations:
                    conv_data = self.conversation_engine.active_conversations[conversation.id]
                    # Convert from storage format to internal format
                    internal_messages = []
                    for msg in conversation.messages:
                        if isinstance(msg, dict) and 'sender' in msg and 'content' in msg:
                            internal_messages.append({
                                "agent_name": msg['sender'],
                                "message": msg['content']
                            })
                    conv_data["messages"] = internal_messages
                    print(f"DEBUG: Restored {len(internal_messages)} messages to engine")
                    
                    # If no agent_sending_messages exist, update them now
                    if not hasattr(conversation, 'agent_sending_messages') or not conversation.agent_sending_messages:
                        print("DEBUG: Generating agent_sending_messages from conversation history")
                        for agent_name in conv_data["agent_names"]:
                            self.conversation_engine._update_agent_sending_messages(conversation.id, agent_name)
                        print("DEBUG: Generated agent context from existing messages")
                  # Register callback for message updates
            self.conversation_engine.register_message_callback(
                conversation.id, self.on_message_received
            )
            
            # Update UI state
            self.conversation_active = True
            self.update_simulation_controls(True)
            self.current_env_label.config(text=conversation.environment)
              # Double-check we've initialized everything properly
            if not hasattr(self, 'chat_canvas'):
                print("ERROR: chat_canvas not initialized!")
                messagebox.showerror("Error", "Chat canvas not initialized. Please try restarting the application.")
                return
            else:
                # Force the notebook to stay on the simulation tab and give it focus
                self.notebook.select(3)  # Simulation tab
                self.root.update_idletasks()
                self.root.update()
                self.chat_canvas.focus_set()
            
            # Make sure we're on the simulation tab and it's visible
            print("DEBUG: Selecting simulation tab")
            self.notebook.select(3)  # Simulation tab is index 3
            
            # Force UI updates
            self.root.update()
            
            # Clear existing messages and reload from conversation
            print("DEBUG: Clearing chat canvas before loading conversation messages")
            try:
                # Double-check that chat_canvas is initialized
                if hasattr(self, 'chat_canvas') and self.chat_canvas:
                    self.chat_canvas.clear()
                    self.root.update()
                    self.root.update_idletasks()  # Force UI update after clearing
                else:
                    print("WARNING: Cannot clear chat canvas - not initialized")
                    messagebox.showerror("Error", "Chat canvas not initialized. Please restart the application.")
                    return
            except Exception as e:
                print(f"ERROR during UI update: {e}")
            
            # Add header information
            header_text = f"🎬 Loaded conversation: {conversation.title}\n"
            header_text += f"📍 Environment: {conversation.environment}\n"
            header_text += f"🎭 Scene: {conversation.scene_description}\n"
            header_text += f"👥 Participants: {', '.join([a.name for a in conversation_agents])}"
            
            self.chat_canvas.add_bubble("System", header_text, datetime.now().strftime("%H:%M:%S"), "system", UI_COLORS["system_bubble"])            # Load existing messages
            print(f"DEBUG: Checking for messages in loaded conversation...")
            try:
                if hasattr(conversation, 'messages') and conversation.messages:
                    print(f"DEBUG: Found {len(conversation.messages)} messages to load")
                    # Load messages in batches for better UI performance with large conversations
                    message_count = len(conversation.messages)
                    batch_size = 5
                    batches = (message_count + batch_size - 1) // batch_size  # Ceiling division
                    
                    for batch in range(batches):
                        start_idx = batch * batch_size
                        end_idx = min((batch + 1) * batch_size, message_count)
                        print(f"DEBUG: Loading message batch {batch+1}/{batches} (messages {start_idx+1}-{end_idx})")
                        
                        for i in range(start_idx, end_idx):
                            message = conversation.messages[i]
                            sender = message.get('sender', 'Unknown')
                            content = message.get('content', '')
                            timestamp = message.get('timestamp', datetime.now().strftime("%H:%M:%S"))
                            msg_type = message.get('type', 'ai')
                            
                            # Get agent color if available
                            color = self.agent_colors.get(sender, UI_COLORS["agent_colors"][0])
                            
                            short_content = content[:30] + "..." if len(content) > 30 else content
                            print(f"DEBUG: Loading message {i+1}/{message_count} from '{sender}': {short_content}")
                            self.chat_canvas.add_bubble(sender, content, timestamp, msg_type, color)
                        
                        # Update UI after each batch
                        self.root.update()
                else:
                    print("DEBUG: No previous messages found in conversation")
            except Exception as e:
                print(f"ERROR loading messages: {e}")
                
            try:
                # Auto-scroll to bottom
                print("DEBUG: Auto-scrolling chat")
                self.chat_canvas.auto_scroll()
                
                # Try an additional yview_moveto to ensure we scroll to the bottom
                self.chat_canvas.yview_moveto(1.0)
                
                # Force updates to make sure UI reflects changes
                self.root.update()
                self.root.update_idletasks()
                
                # Schedule another auto-scroll after a short delay (in case first one didn't work)
                self.root.after(200, self.chat_canvas.auto_scroll)
                
                # Save the loaded conversation for reference
                self.loaded_conversation = conversation
                
                # Update status
                msg = f"Conversation '{conversation.title}' loaded successfully with {len(conversation.messages) if hasattr(conversation, 'messages') else 0} messages!"
                print(f"DEBUG: {msg}")
                self.update_status(msg)
                
                # Alert user that the conversation has been loaded
                messagebox.showinfo("Conversation Loaded", f"Conversation '{conversation.title}' has been loaded successfully. New messages will appear in the chat.")
            
            except Exception as e:
                print(f"ERROR in final conversation loading steps: {e}")
            
        except Exception as e:
            print(f"Error in load_selected_conversation: {e}")
            messagebox.showerror("Error", f"Failed to load conversation: {str(e)}")

    def load_tool_checkboxes(self):
        """Load tools from tools.json and create checkboxes."""
        try:
            # Clear existing checkboxes
            for widget in self.tools_checkboxes_frame.winfo_children():
                widget.destroy()
            
            # Reset tool vars
            self.tool_vars = {}
            
            # Load tools from tools.json
            tools_file = os.path.join(os.path.dirname(__file__), "tools.json")
            with open(tools_file, 'r') as f:
                tools_data = json.load(f)
                
            # Create a checkbox for each tool
            for i, tool in enumerate(tools_data.get('tools', [])):
                var = tk.BooleanVar(value=False)
                self.tool_vars[tool['name']] = var
                
                # Create frame for each tool
                tool_frame = ttk.Frame(self.tools_checkboxes_frame)
                tool_frame.pack(fill="x", expand=True, padx=5, pady=2)
                
                # Add checkbox with tool name
                cb = ttk.Checkbutton(tool_frame, text=tool['name'], variable=var)
                cb.pack(side="left", anchor="w")
                
                # Add tooltip/help icon
                help_button = ttk.Label(tool_frame, text="ℹ️")
                help_button.pack(side="left", padx=5)
                
                # Bind tooltip to show description
                tool_tip_text = tool['description']
                help_button.bind("<Enter>", lambda event, text=tool_tip_text: self.show_tooltip(event, text))
                help_button.bind("<Leave>", self.hide_tooltip)
                
        except Exception as e:
            print(f"Error loading tools: {e}")
            messagebox.showerror("Error", f"Failed to load tools: {e}")
    
    def show_tooltip(self, event, text):
        """Display tooltip near the widget."""
        x, y, _, _ = event.widget.bbox("insert")
        x += event.widget.winfo_rootx() + 25
        y += event.widget.winfo_rooty() + 25
        
        # Create a toplevel window
        self.tooltip = tk.Toplevel(event.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = ttk.Label(self.tooltip, text=text, wraplength=300, 
                         background="#ffffe0", relief="solid", borderwidth=1,
                         padding=5)
        label.pack()
    
    def hide_tooltip(self, event=None):
        """Hide the tooltip."""
        if hasattr(self, "tooltip") and self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

if __name__ == "__main__":
    """Main entry point for the application."""
    try:
        # Create and run the application
        app = AgentConversationSimulatorGUI()
        app.root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
