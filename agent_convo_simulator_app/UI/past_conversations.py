import tkinter as tk
from tkinter import ttk, messagebox

class PastConversationsTab(ttk.Frame):
    def __init__(self, parent, app, data_manager):
        super().__init__(parent)
        self.app = app
        self.data_manager = data_manager

        self.create_widgets()

    def create_widgets(self):
        """Create the past conversations tab with card-like conversation display."""
        past_conv_frame = ttk.Frame(self)
        past_conv_frame.pack(fill="both", expand=True)

        # Configure grid
        past_conv_frame.grid_rowconfigure(1, weight=1)
        past_conv_frame.grid_columnconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(past_conv_frame, text="Past Conversations", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(10, 20), sticky="w", padx=10)

        # Main content frame
        content_frame = ttk.Frame(past_conv_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        # Scrollable canvas for cards
        canvas = tk.Canvas(content_frame, borderwidth=0, highlightthickness=0, bg="#f7f7fa")
        self.cards_frame = ttk.Frame(canvas)
        vsb = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        canvas.create_window((0, 0), window=self.cards_frame, anchor="nw")
        self.cards_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Buttons
        btn_frame = ttk.Frame(content_frame)
        btn_frame.grid(row=1, column=0, pady=(10, 0), sticky="e")
        ttk.Button(btn_frame, text="Refresh List", command=self.refresh_past_conversations).pack(side=tk.LEFT, padx=5)

        self.card_widgets = []
        self.selected_card_idx = None
        self.refresh_past_conversations()

    def _clear_cards(self):
        for widget in self.card_widgets:
            widget.destroy()
        self.card_widgets = []
        self.selected_card_idx = None

    def _on_card_click(self, idx):
        # Highlight selected card
        for i, card in enumerate(self.card_widgets):
            card.config(style="Card.TFrame" if i != idx else "CardSelected.TFrame")
        self.selected_card_idx = idx

    def _on_load_card(self):
        if self.selected_card_idx is None:
            messagebox.showwarning("No Selection", "Please select a conversation to load.")
            return
        conversations = self.data_manager.load_conversations()
        if self.selected_card_idx < len(conversations):
            conversation = conversations[self.selected_card_idx]
            self.app.load_selected_conversation(conversation)
            self.app.notebook.select(self.app.simulation_tab)

    def _on_delete_card(self):
        if self.selected_card_idx is None:
            messagebox.showwarning("No Selection", "Please select a conversation to delete.")
            return
        conversations = self.data_manager.load_conversations()
        if self.selected_card_idx < len(conversations):
            conversation = conversations[self.selected_card_idx]
            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete conversation '{conversation.title}'?"):
                self.data_manager.delete_conversation(conversation.id)
                self.refresh_past_conversations()
                self.app.update_status(f"Conversation '{conversation.title}' deleted.")

    def refresh_past_conversations(self):
        """Refresh the list of past conversations as cards."""
        self._clear_cards()
        conversations = self.data_manager.load_conversations()
        style = ttk.Style()
        style.configure("Card.TFrame", background="#ffffff", relief="raised", borderwidth=1)
        style.configure("CardSelected.TFrame", background="#e0eaff", relief="solid", borderwidth=2)
        style.configure("CardTitle.TLabel", font=("Arial", 13, "bold"), background="#ffffff")
        style.configure("CardMeta.TLabel", font=("Arial", 10, "italic"), background="#ffffff", foreground="#666")
        style.configure("CardStatus.TLabel", font=("Arial", 10, "bold"), background="#ffffff", foreground="#2a7")
        style.configure("CardAgent.TLabel", font=("Arial", 10), background="#ffffff")
        for idx, conv in enumerate(conversations):
            card = ttk.Frame(self.cards_frame, style="Card.TFrame", padding=(12, 8, 12, 8))
            card.grid(row=idx, column=0, sticky="ew", pady=8, padx=4)
            card.grid_columnconfigure(0, weight=1)
            card.grid_columnconfigure(1, weight=0)
            # Make card stretch to fill parent width
            self.cards_frame.grid_columnconfigure(0, weight=1)
            card.bind("<Button-1>", lambda e, i=idx: self._on_card_click(i))
            # Title
            title = ttk.Label(card, text=conv.title, style="CardTitle.TLabel")
            title.grid(row=0, column=0, sticky="w")
            # Date & status
            meta = f"{conv.environment} | {conv.created_at[:10]}"
            status = f" ({conv.status})" if hasattr(conv, 'status') and conv.status else ""
            meta_label = ttk.Label(card, text=meta+status, style="CardMeta.TLabel")
            meta_label.grid(row=1, column=0, sticky="w", pady=(2, 0))
            # Agents
            agent_names = []
            for agent_id in getattr(conv, 'agents', []):
                agent_obj = self.data_manager.get_agent_by_id(agent_id)
                if agent_obj and hasattr(agent_obj, 'name'):
                    agent_names.append(agent_obj.name)
                else:
                    agent_names.append(str(agent_id))
            agents = ', '.join(agent_names)
            agent_label = ttk.Label(card, text=f"Agents: {agents}", style="CardAgent.TLabel")
            agent_label.grid(row=2, column=0, sticky="w", pady=(2, 0))
            # Buttons for each card (now act directly on this conversation)
            btns = ttk.Frame(card, style="Card.TFrame")
            btns.grid(row=0, column=1, rowspan=3, sticky="e", padx=(10, 0))
            def make_load(conv=conv):
                return lambda: (self.app.load_selected_conversation(conv), self.app.notebook.select(self.app.simulation_tab))
            def make_delete(conv=conv):
                return lambda: self._delete_conversation_card(conv)
            load_btn = ttk.Button(btns, text="Load", command=make_load())
            load_btn.pack(side=tk.TOP, fill="x", pady=2)
            del_btn = ttk.Button(btns, text="Delete", command=make_delete())
            del_btn.pack(side=tk.TOP, fill="x", pady=2)
            card.bind("<Enter>", lambda e, c=card: c.config(style="CardSelected.TFrame"))
            card.bind("<Leave>", lambda e, c=card, i=idx: c.config(style="CardSelected.TFrame" if self.selected_card_idx == i else "Card.TFrame"))
            # Make widgets clickable
            for w in (title, meta_label, agent_label, btns):
                w.bind("<Button-1>", lambda e, i=idx: self._on_card_click(i))
            self.card_widgets.append(card)

    def _delete_conversation_card(self, conv):
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete conversation '{conv.title}'?"):
            self.data_manager.delete_conversation(conv.id)
            self.refresh_past_conversations()
            self.app.update_status(f"Conversation '{conv.title}' deleted.")

    # Removed old Listbox-based methods. Only card-based UI is now used.
