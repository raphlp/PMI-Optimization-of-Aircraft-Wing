"""
PMI Wing Optimization - Main Application

A modern GUI for CFD-based deep learning optimization of aircraft wings.
Supports CFD, PINNs, and LSTM simulation types.
"""

import customtkinter as ctk
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gui.theme import COLORS, FONTS, PADDING, SIMULATION_TYPES
from src.gui.components.cfd_panel import CFDPanel
from src.gui.components.pinns_panel import PINNsPanel
from src.gui.components.lstm_panel import LSTMPanel


class PMIApp(ctk.CTk):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("PMI - Wing Optimization Suite")
        self.geometry("1100x750")
        self.minsize(900, 600)
        
        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Current panel
        self.current_panel = None
        self.current_sim_type = "cfd"
        
        # Create UI
        self.create_sidebar()
        self.create_main_area()
        
        # Show CFD panel by default
        self.show_panel("cfd")
    
    def create_sidebar(self):
        """Create the left sidebar with navigation."""
        sidebar = ctk.CTkFrame(
            self,
            width=250,
            corner_radius=0,
            fg_color=COLORS["bg_medium"]
        )
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        
        # Logo/Title
        logo_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", pady=PADDING["lg"], padx=PADDING["md"])
        
        logo = ctk.CTkLabel(
            logo_frame,
            text="✈",
            font=("Segoe UI", 48)
        )
        logo.pack()
        
        title = ctk.CTkLabel(
            logo_frame,
            text="PMI Wing\nOptimization",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"],
            justify="center"
        )
        title.pack(pady=(PADDING["xs"], 0))
        
        subtitle = ctk.CTkLabel(
            logo_frame,
            text="CFD + Deep Learning",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        subtitle.pack()
        
        # Separator
        sep = ctk.CTkFrame(sidebar, height=1, fg_color=COLORS["border"])
        sep.pack(fill="x", padx=PADDING["md"], pady=PADDING["md"])
        
        # Navigation label
        nav_label = ctk.CTkLabel(
            sidebar,
            text="SIMULATION TYPE",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        nav_label.pack(anchor="w", padx=PADDING["lg"], pady=(PADDING["sm"], PADDING["xs"]))
        
        # Navigation buttons
        self.nav_buttons = {}
        
        for sim_type, config in SIMULATION_TYPES.items():
            btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
            btn_frame.pack(fill="x", padx=PADDING["sm"], pady=2)
            
            btn = ctk.CTkButton(
                btn_frame,
                text=f"  {config['name']}",
                font=FONTS["body"],
                anchor="w",
                height=45,
                corner_radius=8,
                fg_color="transparent",
                text_color=COLORS["text_secondary"],
                hover_color=COLORS["bg_light"],
                command=lambda t=sim_type: self.show_panel(t)
            )
            btn.pack(fill="x", padx=PADDING["xs"])
            
            # Add "coming soon" badge if not enabled
            if not config["enabled"]:
                badge = ctk.CTkLabel(
                    btn_frame,
                    text="Soon",
                    font=("Segoe UI", 9),
                    text_color=COLORS["bg_medium"],
                    fg_color=COLORS["text_muted"],
                    corner_radius=4,
                    width=35,
                    height=18
                )
                badge.place(relx=0.88, rely=0.5, anchor="center")
            
            self.nav_buttons[sim_type] = btn
        
        # Bottom section
        bottom_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        bottom_frame.pack(side="bottom", fill="x", pady=PADDING["md"])
        
        # Separator
        sep2 = ctk.CTkFrame(bottom_frame, height=1, fg_color=COLORS["border"])
        sep2.pack(fill="x", padx=PADDING["md"], pady=PADDING["sm"])
        
        # Info
        info = ctk.CTkLabel(
            bottom_frame,
            text="IPSA Master Project\n2024-2025",
            font=FONTS["small"],
            text_color=COLORS["text_muted"],
            justify="center"
        )
        info.pack(pady=PADDING["sm"])
    
    def create_main_area(self):
        """Create the main content area."""
        self.main_frame = ctk.CTkFrame(
            self,
            corner_radius=0,
            fg_color=COLORS["bg_dark"]
        )
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        
        # Content container with padding
        self.content_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent"
        )
        self.content_frame.pack(fill="both", expand=True, padx=PADDING["xl"], pady=PADDING["lg"])
    
    def show_panel(self, sim_type):
        """Switch to a different simulation panel."""
        # Update button states
        for t, btn in self.nav_buttons.items():
            if t == sim_type:
                btn.configure(
                    fg_color=COLORS["primary"],
                    text_color=COLORS["text_primary"]
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=COLORS["text_secondary"]
                )
        
        # Remove current panel
        if self.current_panel:
            self.current_panel.destroy()
        
        # Create new panel
        if sim_type == "cfd":
            self.current_panel = CFDPanel(self.content_frame, self)
        elif sim_type == "pinns":
            self.current_panel = PINNsPanel(self.content_frame, self)
        elif sim_type == "lstm":
            self.current_panel = LSTMPanel(self.content_frame, self)
        
        self.current_panel.pack(fill="both", expand=True)
        self.current_sim_type = sim_type


def run_app():
    """Launch the application."""
    app = PMIApp()
    app.mainloop()


if __name__ == "__main__":
    run_app()
