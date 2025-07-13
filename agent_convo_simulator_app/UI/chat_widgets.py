import tkinter as tk
from tkinter import ttk
from ..config import UI_COLORS

class ChatBubble(tk.Frame):
    """Represents a chat message bubble in the conversation UI."""
    
    def __init__(self, parent, sender, message, timestamp, msg_type="ai", color=None, align_right=False, loading=False, **kwargs):
        """Initialize the chat bubble.
        
        Args:
            parent: Parent widget
            sender: Name of the message sender
            message: Content of the message
            timestamp: Time the message was sent
            msg_type: Type of message (user, system, ai)
            color: Background color for the bubble
            align_right: If True, align bubble to the right with right-aligned text
            loading: If True, show animated loading dots instead of message
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
        
        # Create container for precise width control
        container_frame = tk.Frame(self, bg=UI_COLORS["chat_background"])
        container_frame.pack(fill="x")
        
        # Configure the container with proper weights for bubble width
        if loading:
            # Loading bubble: 10% width
            container_frame.grid_columnconfigure(1, weight=1)  # 10% for bubble
            if align_right:
                container_frame.grid_columnconfigure(0, weight=9)  # 90% left spacer
                container_frame.grid_columnconfigure(2, weight=0)
                
                # Left spacer for right alignment
                left_spacer = tk.Frame(container_frame, bg=UI_COLORS["chat_background"])
                left_spacer.grid(row=0, column=0, sticky="ew")
                
                # Bubble frame takes 10% width, positioned on the right
                self.bubble_frame = tk.Frame(container_frame, bg=color, padx=15, pady=8)
                self.bubble_frame.grid(row=0, column=1, sticky="ew", pady=6)
            else:
                container_frame.grid_columnconfigure(0, weight=0)
                container_frame.grid_columnconfigure(2, weight=9)  # 90% right spacer
                
                # Bubble frame takes 10% width, positioned on the left
                self.bubble_frame = tk.Frame(container_frame, bg=color, padx=15, pady=8)
                self.bubble_frame.grid(row=0, column=1, sticky="ew", pady=6)
                
                # Right spacer for left alignment
                right_spacer = tk.Frame(container_frame, bg=UI_COLORS["chat_background"])
                right_spacer.grid(row=0, column=2, sticky="ew")
        else:
            # Actual message bubble: 75% width
            container_frame.grid_columnconfigure(1, weight=3)  # 75% for bubble
            if align_right:
                container_frame.grid_columnconfigure(0, weight=1)  # 25% left spacer
                container_frame.grid_columnconfigure(2, weight=0)
                
                # Left spacer for right alignment
                left_spacer = tk.Frame(container_frame, bg=UI_COLORS["chat_background"])
                left_spacer.grid(row=0, column=0, sticky="ew")
                
                # Bubble frame takes 75% width, positioned on the right
                self.bubble_frame = tk.Frame(container_frame, bg=color, padx=15, pady=8)
                self.bubble_frame.grid(row=0, column=1, sticky="ew", pady=6)
            else:
                container_frame.grid_columnconfigure(0, weight=0)
                container_frame.grid_columnconfigure(2, weight=1)  # 25% right spacer
                
                # Bubble frame takes 75% width, positioned on the left
                self.bubble_frame = tk.Frame(container_frame, bg=color, padx=15, pady=8)
                self.bubble_frame.grid(row=0, column=1, sticky="ew", pady=6)
                
                # Right spacer for left alignment
                right_spacer = tk.Frame(container_frame, bg=UI_COLORS["chat_background"])
                right_spacer.grid(row=0, column=2, sticky="ew")
        
        # Add rounded corners by using themed frame
        self.bubble_frame.config(highlightbackground=color, highlightthickness=1, bd=0)
        
        # Add sender name with timestamp (different layout for right-aligned)
        header_frame = tk.Frame(self.bubble_frame, bg=color)
        header_frame.pack(fill="x", expand=True)
        
        # Icon based on message type
        if msg_type == "user":
            icon = "👤"
        elif msg_type == "system":
            icon = "🤖"
        else:
            icon = "🎭"
        
        if align_right:
            # For right-aligned bubbles: time on left, sender on right
            time_label = tk.Label(
                header_frame, 
                text=timestamp, 
                font=("Arial", 8),
                bg=color,
                fg="gray",
                anchor="w"
            )
            time_label.pack(side="left")
            
            sender_label = tk.Label(
                header_frame, 
                text=f"{sender} {icon}", 
                font=("Arial", 9, "bold"),
                bg=color,
                anchor="e"
            )
            sender_label.pack(side="right")
        else:
            # For left-aligned bubbles: sender on left, time on right (original layout)
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
        
        # Add message content or loading animation
        if loading:
            self.loading_label = tk.Label(
                self.bubble_frame,
                text="...",
                font=("Arial", 10, "italic"),
                bg=color,
                justify="center",
                anchor="center",
                wraplength=400
            )
            self.loading_label.pack(fill="x", pady=(5, 0))
            self._loading_animation_index = 0
            self._loading_animation_job = None
            self._start_loading_animation()
        else:
            if align_right:
                text_container = tk.Frame(self.bubble_frame, bg=color)
                text_container.pack(fill="x", pady=(5, 0))
                message_label = tk.Label(
                    text_container, 
                    text=message, 
                    font=("Arial", 10),
                    bg=color,
                    justify="left",   # Left-justify text lines within the container
                    anchor="w",       # Anchor text to the left within the container
                    wraplength=400
                )
                message_label.pack(side="right")  # Position the text block to the right within container
            else:
                message_label = tk.Label(
                    self.bubble_frame, 
                    text=message, 
                    font=("Arial", 10),
                    bg=color,
                    justify="left",   # Left-align text content for left bubbles
                    anchor="w",       # Anchor text to the left
                    wraplength=400
                )
                message_label.pack(fill="x", pady=(5, 0))
        
        # Store original color for blinking animation
        self.original_color = color
        self.blink_active = False
        self.blink_job = None
        self.loading = loading
        self._loading_animation_job = None
        
    def _start_loading_animation(self):
        dots = [".", "..", "..."]
        self._loading_animation_index = (self._loading_animation_index + 1) % len(dots)
        self.loading_label.config(text=dots[self._loading_animation_index])
        self._loading_animation_job = self.after(500, self._start_loading_animation)

    def stop_loading_animation(self):
        if self.loading and self._loading_animation_job:
            self.after_cancel(self._loading_animation_job)
            self._loading_animation_job = None

    def start_blink(self):
        """Start blinking animation."""
        if not self.blink_active:
            self.blink_active = True
            self._blink_animate()
    
    def stop_blink(self):
        """Stop blinking animation and restore original color."""
        self.blink_active = False
        if self.blink_job:
            self.after_cancel(self.blink_job)
            self.blink_job = None
        # Restore original color
        self.bubble_frame.config(bg=self.original_color)
        # Update all child widgets' background
        self._update_bg_color(self.original_color)
        # Stop loading animation if present
        self.stop_loading_animation()
    
    def _blink_animate(self):
        """Animate the blinking effect."""
        if not self.blink_active:
            return
        
        # Alternate between original color and a slightly lighter version
        current_bg = self.bubble_frame.cget('bg')
        if current_bg == self.original_color:
            # Make lighter (simple approach - you could use more sophisticated color manipulation)
            lighter_color = self._lighten_color(self.original_color)
            new_color = lighter_color
        else:
            new_color = self.original_color
        
        self.bubble_frame.config(bg=new_color)
        self._update_bg_color(new_color)
        
        # Schedule next blink
        self.blink_job = self.after(500, self._blink_animate)  # Blink every 500ms
    
    def _lighten_color(self, color):
        """Create a lighter version of the given color."""
        try:
            # More sophisticated color lightening
            # Convert color name to RGB if it's a hex color
            if color.startswith('#'):
                # Remove the # and convert to RGB
                hex_color = color[1:]
                if len(hex_color) == 6:
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    
                    # Lighten by adding 30 to each component (max 255)
                    r = min(255, r + 30)
                    g = min(255, g + 30)
                    b = min(255, b + 30)
                    
                    return f"#{r:02x}{g:02x}{b:02x}"
        except:
            pass
        
        # Fallback for specific known colors
        color_map = {
            "#E8F4FD": "#F0F8FF",  # Light blue to lighter blue
            "#E8F8E8": "#F0FFF0",  # Light green to lighter green
            "#FFE8E8": "#FFF0F0",  # Light red to lighter red
            "#FFF8DC": "#FFFACD",  # Light yellow to lighter yellow
            "#E6E6FA": "#F8F8FF",  # Light purple to lighter purple
            "#F0E68C": "#F5F5DC",  # Light brown to lighter brown
            "#F0FFFF": "#F5FFFA"   # Light cyan to lighter cyan
        }
        
        return color_map.get(color, "#F8F8F8")  # Default very light gray
    
    def _update_bg_color(self, color):
        """Update background color for all child widgets."""
        for child in self.bubble_frame.winfo_children():
            if hasattr(child, 'config'):
                try:
                    child.config(bg=color)
                except tk.TclError:
                    pass  # Some widgets might not support bg config
            # Recursively update nested widgets
            self._update_child_bg(child, color)
    
    def _update_child_bg(self, widget, color):
        """Recursively update background color for nested widgets."""
        for child in widget.winfo_children():
            if hasattr(child, 'config'):
                try:
                    child.config(bg=color)
                except tk.TclError:
                    pass
            self._update_child_bg(child, color)
        
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
        
        # Track message bubbles by message_id for blinking animations
        self.message_bubbles = {}  # message_id -> ChatBubble
        self.agent_loading_bubbles = {}  # agent_id -> loading ChatBubble
        
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
        
    def add_bubble(self, sender, message, timestamp, msg_type="ai", color=None, align_right=False, message_id=None, loading=False, agent_id=None):
        """Add a new chat bubble to the canvas."""
        # Add extra spacing between messages from different senders
        if self.previous_sender is not None and self.previous_sender != sender:
            # Add a small spacer frame between different agents' messages
            spacer = tk.Frame(self.bubble_frame, bg=UI_COLORS["chat_background"], height=10)
            spacer.pack(fill="x", expand=True)
          # Update the previous sender
        self.previous_sender = sender
        # Remove loading bubble for this agent if not loading (actual message)
        if not loading and agent_id and agent_id in self.agent_loading_bubbles:
            old_bubble = self.agent_loading_bubbles[agent_id]
            if hasattr(old_bubble, "stop_loading_animation"):
                old_bubble.stop_loading_animation()
            old_bubble.destroy()
            del self.agent_loading_bubbles[agent_id]
        bubble = ChatBubble(
            self.bubble_frame,
            sender, 
            message, 
            timestamp,
            msg_type,
            color,
            align_right=align_right,
            loading=loading
        )
        bubble.pack(fill="x", expand=True)
        if loading and agent_id:
            self.agent_loading_bubbles[agent_id] = bubble
        if message_id:
            self.message_bubbles[message_id] = bubble
        self.bubble_frame.update_idletasks()
        self.update_idletasks()
        self.configure(scrollregion=self.bbox("all"))
        self.after_idle(lambda: self.yview_moveto(1.0))
        return bubble
        
    def clear(self):
        """Clear all chat bubbles."""
        for widget in self.bubble_frame.winfo_children():
            widget.destroy()
        # Reset previous sender tracking
        self.previous_sender = None
        # Clear message bubble references
        self.message_bubbles.clear()
    
    def start_bubble_blink(self, message_id: str):
        """Start blinking animation for a message bubble."""
        if message_id in self.message_bubbles:
            bubble = self.message_bubbles[message_id]
            bubble.start_blink()
    
    def stop_bubble_blink(self, message_id: str):
        """Stop blinking animation for a message bubble."""
        if message_id in self.message_bubbles:
            bubble = self.message_bubbles[message_id]
            bubble.stop_blink()
            # Clean up the reference
            del self.message_bubbles[message_id]
    
    def stop_all_blinking(self):
        """Stop all blinking animations for safety during pause."""
        print(f"DEBUG: Stopping all blinking animations ({len(self.message_bubbles)} bubbles)")
        # Get a copy of the keys to avoid modification during iteration
        bubble_ids = list(self.message_bubbles.keys())
        for message_id in bubble_ids:
            if message_id in self.message_bubbles:
                bubble = self.message_bubbles[message_id]
                bubble.stop_blink()
                del self.message_bubbles[message_id]
        print(f"DEBUG: All blinking animations stopped")
    
    def auto_scroll(self):
        """Automatically scroll to the bottom of the chat."""
        try:
            print("DEBUG: auto_scroll called")
            self.update_idletasks()  # Ensure the canvas is updated
            self.yview_moveto(1.0)  # Scroll to the bottom
            print("DEBUG: auto_scroll completed")
        except Exception as e:
            print(f"ERROR in auto_scroll: {e}")
