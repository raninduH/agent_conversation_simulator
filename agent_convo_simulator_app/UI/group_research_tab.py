import tkinter as tk
from tkinter import ttk, scrolledtext

class GroupResearchTab(ttk.Frame):
    def __init__(self, parent, app, data_manager):
        super().__init__(parent)
        self.app = app
        self.data_manager = data_manager
        self.create_widgets()

    def create_widgets(self):
        research_frame = self
        research_frame.grid_rowconfigure(1, weight=1)
        research_frame.grid_columnconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(research_frame, text="Group Research Setup", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(10, 20))

        # Main content frame
        content_frame = ttk.Frame(research_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)

        # Left panel - Research settings
        settings_frame = ttk.LabelFrame(content_frame, text="Research Settings", padding="10")
        settings_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        settings_frame.grid_columnconfigure(1, weight=1)

        # Research Name
        ttk.Label(settings_frame, text="Research Name:").grid(row=0, column=0, sticky="w", pady=2)
        self.app.research_name_var = tk.StringVar()
        self.app.research_name_entry = ttk.Entry(settings_frame, textvariable=self.app.research_name_var, width=30)
        self.app.research_name_entry.grid(row=0, column=1, sticky="ew", pady=2, padx=(10, 0))

        # Research Problem
        ttk.Label(settings_frame, text="Research Problem:").grid(row=1, column=0, sticky="nw", pady=2)
        self.app.research_problem_text = scrolledtext.ScrolledText(settings_frame, width=40, height=4)
        self.app.research_problem_text.grid(row=1, column=1, sticky="ew", pady=2, padx=(10, 0))

        # Extra Things to Consider
        ttk.Label(settings_frame, text="Extra Things to Consider:").grid(row=2, column=0, sticky="nw", pady=2)
        self.app.extra_consider_text = scrolledtext.ScrolledText(settings_frame, width=40, height=3)
        self.app.extra_consider_text.grid(row=2, column=1, sticky="ew", pady=2, padx=(10, 0))

        # Goal of Research
        ttk.Label(settings_frame, text="Goal of Research:").grid(row=3, column=0, sticky="nw", pady=2)
        self.app.research_goal_text = scrolledtext.ScrolledText(settings_frame, width=40, height=3)
        self.app.research_goal_text.grid(row=3, column=1, sticky="ew", pady=2, padx=(10, 0))

        # Voice Synthesis Enable/Disable
        self.app.voice_synthesis_var = tk.BooleanVar()
        self.app.voice_synthesis_check = ttk.Checkbutton(settings_frame, text="Enable Voice Synthesis", variable=self.app.voice_synthesis_var)
        self.app.voice_synthesis_check.grid(row=4, column=1, sticky="w", pady=6)

        # Right panel - Agent selection (reuse from conversation setup)
        agents_frame = ttk.LabelFrame(content_frame, text="Select Agents", padding="10")
        agents_frame.grid(row=0, column=1, sticky="nsew")
        agents_frame.grid_columnconfigure(0, weight=1)
        agents_frame.grid_rowconfigure(0, weight=1)

        self.app.research_agents_checkbox_frame = ttk.Frame(agents_frame)
        self.app.research_agents_checkbox_frame.grid(row=0, column=0, sticky="nsew")
        self.refresh_agent_checkboxes()

        # Bottom buttons
        button_frame = ttk.Frame(research_frame)
        button_frame.grid(row=2, column=0, pady=(10, 10))
        clear_btn = ttk.Button(button_frame, text="Clear Inputs", command=self.clear_inputs)
        clear_btn.pack(side=tk.LEFT, padx=10)
        start_btn = ttk.Button(button_frame, text="Start Research", command=self.start_research)
        start_btn.pack(side=tk.LEFT, padx=10)

    def refresh_agent_checkboxes(self):
        # Remove old checkboxes
        for widget in self.app.research_agents_checkbox_frame.winfo_children():
            widget.destroy()
        self.app.research_selected_agents = []
        self.app.research_agent_checkboxes = {}
        agents = self.data_manager.load_agents()  # Use load_agents() as in conversation_setup.py
        for i, agent in enumerate(agents):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(self.app.research_agents_checkbox_frame, text=f"{agent.name} ({getattr(agent, 'role', '')})", variable=var)
            cb.grid(row=i, column=0, sticky="w", pady=2)
            self.app.research_agent_checkboxes[getattr(agent, 'id', i)] = var

    def clear_inputs(self):
        self.app.research_name_var.set("")
        self.app.research_problem_text.delete(1.0, tk.END)
        self.app.extra_consider_text.delete(1.0, tk.END)
        self.app.research_goal_text.delete(1.0, tk.END)
        self.app.voice_synthesis_var.set(False)
        for var in self.app.research_agent_checkboxes.values():
            var.set(False)

    def start_research(self):
        # Gather research setup data from UI
        research_name = self.app.research_name_var.get().strip()
        research_problem = self.app.research_problem_text.get(1.0, "end").strip()
        extra_consider = self.app.extra_consider_text.get(1.0, "end").strip()
        research_goal = self.app.research_goal_text.get(1.0, "end").strip()
        voice_synthesis = self.app.voice_synthesis_var.get()
        # Collect selected agents
        selected_agents = []
        for agent_id, var in self.app.research_agent_checkboxes.items():
            if var.get():
                selected_agents.append(agent_id)
        # Prepare research config dict
        research_config = {
            "research_name": research_name,
            "research_problem": research_problem,
            "extra_consider": extra_consider,
            "research_goal": research_goal,
            "voice_synthesis": voice_synthesis,
            "selected_agents": selected_agents
        }
        # Start research conversation via backend
        if hasattr(self.app, "research_trigger_engine"):
            self.app.research_trigger_engine.start_research(research_config, self.on_research_started)
        else:
            print("[GroupResearchTab] No research_trigger_engine found on app!")
        # Switch to Research Conversation tab after starting
        if hasattr(self.app, 'notebook') and hasattr(self.app, 'research_conversation_tab'):
            self.app.notebook.select(self.app.research_conversation_tab)

    def on_research_started(self, research_id=None):
        # Called by backend when research conversation is started
        print(f"[GroupResearchTab] Research started with ID: {research_id}")
        # Optionally, could trigger UI updates here
