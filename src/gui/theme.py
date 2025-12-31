"""
Theme configuration for the PMI application.
"""

# Color Palette
COLORS = {
    "primary": "#1E88E5",
    "primary_hover": "#1565C0",
    "primary_dark": "#0D47A1",
    
    "bg_dark": "#1A1A2E",
    "bg_medium": "#16213E",
    "bg_light": "#0F3460",
    "bg_card": "#1F2940",
    
    "accent": "#E94560",
    "success": "#00C853",
    "warning": "#FFB300",
    
    "text_primary": "#FFFFFF",
    "text_secondary": "#B0BEC5",
    "text_muted": "#607D8B",
    
    "border": "#2D3A4F",
    "border_active": "#1E88E5",
}

FONTS = {
    "title": ("Segoe UI", 24, "bold"),
    "heading": ("Segoe UI", 16, "bold"),
    "body": ("Segoe UI", 12),
    "small": ("Segoe UI", 10),
    "mono": ("Consolas", 11),
}

PADDING = {
    "xs": 4,
    "sm": 8,
    "md": 16,
    "lg": 24,
    "xl": 32,
}

# Data sources for Step 1
DATA_SOURCES = {
    "cfd": {
        "name": "CFD (Fluent)",
        "description": "Import field data from ANSYS Fluent simulations",
        "color": "#1E88E5",
        "enabled": True,
    },
    "pinns": {
        "name": "PINNs",
        "description": "Generate fields using Physics-Informed Neural Networks",
        "color": "#9C27B0",
        "enabled": True,
    },
    "lstm": {
        "name": "LSTM",
        "description": "Predict field sequences (Coming Soon)",
        "color": "#FF9800",
        "enabled": False,
    },
}

# Prediction methods for Step 2
PREDICTION_METHODS = {
    "cnn": {
        "name": "CNN Prediction",
        "description": "Predict CL/CD from field data using Convolutional Neural Network",
        "color": "#00C853",
        "enabled": True,
    },
}
