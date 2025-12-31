"""
PMI Wing Optimization - Main Application

Two-step workflow:
1. Generate/Import Fields (CFD, PINNs, or LSTM)
2. Predict CL/CD (CNN)
"""

import customtkinter as ctk
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gui.theme import COLORS, FONTS, PADDING, DATA_SOURCES, PREDICTION_METHODS
from src.gui.components.cfd_panel import CFDPanel
from src.gui.components.pinns_panel import PINNsPanel
from src.gui.components.lstm_panel import LSTMPanel
from src.gui.components.prediction_panel import PredictionPanel


class PMIApp(ctk.CTk):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.title("PMI - Wing Optimization Suite")
        self.geometry("1100x750")
        self.minsize(900, 600)
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self.current_panel = None
        self.current_section = "cfd"
        
        self.create_sidebar()
        self.create_main_area()
        
        self.show_panel("cfd")
    
    def create_sidebar(self):
        """Create the left sidebar with two-step navigation."""
        sidebar = ctk.CTkFrame(
            self,
            width=260,
            corner_radius=0,
            fg_color=COLORS["bg_medium"]
        )
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        
        # Logo
        logo_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", pady=PADDING["lg"], padx=PADDING["md"])
        
        logo = ctk.CTkLabel(logo_frame, text="✈", font=("Segoe UI", 48))
        logo.pack()
        
        title = ctk.CTkLabel(
            logo_frame,
            text="PMI Wing\nOptimization",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"],
            justify="center"
        )
        title.pack(pady=(PADDING["xs"], 0))
        
        # Step 1: Data Source
        sep1 = ctk.CTkFrame(sidebar, height=1, fg_color=COLORS["border"])
        sep1.pack(fill="x", padx=PADDING["md"], pady=PADDING["md"])
        
        step1_label = ctk.CTkLabel(
            sidebar,
            text="STEP 1: FIELD DATA SOURCE",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        step1_label.pack(anchor="w", padx=PADDING["lg"], pady=(PADDING["sm"], PADDING["xs"]))
        
        self.nav_buttons = {}
        
        for source_id, config in DATA_SOURCES.items():
            btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
            btn_frame.pack(fill="x", padx=PADDING["sm"], pady=2)
            
            btn = ctk.CTkButton(
                btn_frame,
                text=f"  {config['name']}",
                font=FONTS["body"],
                anchor="w",
                height=40,
                corner_radius=8,
                fg_color="transparent",
                text_color=COLORS["text_secondary"],
                hover_color=COLORS["bg_light"],
                command=lambda t=source_id: self.show_panel(t)
            )
            btn.pack(fill="x", padx=PADDING["xs"])
            
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
            
            self.nav_buttons[source_id] = btn
        
        # Step 2: Prediction
        sep2 = ctk.CTkFrame(sidebar, height=1, fg_color=COLORS["border"])
        sep2.pack(fill="x", padx=PADDING["md"], pady=PADDING["md"])
        
        step2_label = ctk.CTkLabel(
            sidebar,
            text="STEP 2: PREDICTION",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        step2_label.pack(anchor="w", padx=PADDING["lg"], pady=(PADDING["sm"], PADDING["xs"]))
        
        for pred_id, config in PREDICTION_METHODS.items():
            btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
            btn_frame.pack(fill="x", padx=PADDING["sm"], pady=2)
            
            btn = ctk.CTkButton(
                btn_frame,
                text=f"  {config['name']}",
                font=FONTS["body"],
                anchor="w",
                height=40,
                corner_radius=8,
                fg_color="transparent",
                text_color=COLORS["text_secondary"],
                hover_color=COLORS["bg_light"],
                command=lambda t=pred_id: self.show_panel(t)
            )
            btn.pack(fill="x", padx=PADDING["xs"])
            
            self.nav_buttons[pred_id] = btn
        
        # Bottom
        bottom_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        bottom_frame.pack(side="bottom", fill="x", pady=PADDING["md"])
        
        sep3 = ctk.CTkFrame(bottom_frame, height=1, fg_color=COLORS["border"])
        sep3.pack(fill="x", padx=PADDING["md"], pady=PADDING["sm"])
        
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
        
        self.content_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent"
        )
        self.content_frame.pack(fill="both", expand=True, padx=PADDING["xl"], pady=PADDING["lg"])
    
    def show_panel(self, section_id):
        """Switch to a different panel."""
        for t, btn in self.nav_buttons.items():
            if t == section_id:
                btn.configure(
                    fg_color=COLORS["primary"],
                    text_color=COLORS["text_primary"]
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=COLORS["text_secondary"]
                )
        
        if self.current_panel:
            self.current_panel.destroy()
        
        if section_id == "cfd":
            self.current_panel = CFDPanel(self.content_frame, self)
        elif section_id == "pinns":
            self.current_panel = PINNsPanel(self.content_frame, self)
        elif section_id == "lstm":
            self.current_panel = LSTMPanel(self.content_frame, self)
        elif section_id == "cnn":
            self.current_panel = PredictionPanel(self.content_frame, self)
        
        self.current_panel.pack(fill="both", expand=True)
        self.current_section = section_id


def run_app():
    """Launch the application."""
    app = PMIApp()
    app.mainloop()


if __name__ == "__main__":
    run_app()
