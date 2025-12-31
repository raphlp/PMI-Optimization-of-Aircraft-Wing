"""
LSTM Panel - Long Short-Term Memory networks interface.
Currently a placeholder for future implementation.
"""

import customtkinter as ctk
from src.gui.theme import COLORS, FONTS, PADDING


class LSTMPanel(ctk.CTkFrame):
    """Panel for LSTM time-series prediction (coming soon)."""
    
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title = ctk.CTkLabel(
            self,
            text="LSTM Time-Series Prediction",
            font=FONTS["title"],
            text_color=COLORS["text_primary"]
        )
        title.pack(pady=(0, PADDING["md"]))
        
        # Coming soon card
        card = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=15)
        card.pack(fill="both", expand=True, pady=PADDING["lg"], padx=PADDING["lg"])
        
        # Coming soon text
        coming = ctk.CTkLabel(
            card,
            text="Coming Soon",
            font=("Segoe UI", 28, "bold"),
            text_color=COLORS["warning"]
        )
        coming.pack(pady=(PADDING["xl"], PADDING["md"]))
        
        # Description
        desc = ctk.CTkLabel(
            card,
            text="LSTM networks will enable time-series prediction\n"
                 "for unsteady aerodynamic phenomena and\n"
                 "temporal evolution of flow fields.",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            justify="center"
        )
        desc.pack(pady=PADDING["md"])
        
        # Features list
        features_frame = ctk.CTkFrame(card, fg_color=COLORS["bg_light"], corner_radius=10)
        features_frame.pack(pady=PADDING["lg"], padx=PADDING["xl"])
        
        features = [
            "Unsteady flow prediction",
            "Temporal sequence modeling",
            "Dynamic stall prediction",
            "Flutter analysis support"
        ]
        
        for feature in features:
            label = ctk.CTkLabel(
                features_frame,
                text=f"  {feature}",
                font=FONTS["body"],
                text_color=COLORS["text_secondary"],
                anchor="w"
            )
            label.pack(padx=PADDING["md"], pady=PADDING["xs"], anchor="w")
