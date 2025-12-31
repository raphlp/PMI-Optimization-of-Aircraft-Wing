"""
Theme configuration for the PMI application.
Aeronautical-inspired color palette with dark mode.
"""

# Color Palette
COLORS = {
    # Primary colors
    "primary": "#1E88E5",           # Aviation blue
    "primary_hover": "#1565C0",
    "primary_dark": "#0D47A1",
    
    # Background colors
    "bg_dark": "#1A1A2E",           # Deep navy
    "bg_medium": "#16213E",         # Dark blue-gray
    "bg_light": "#0F3460",          # Lighter panel
    "bg_card": "#1F2940",           # Card background
    
    # Accent colors
    "accent": "#E94560",            # Warning/accent red
    "success": "#00C853",           # Success green
    "warning": "#FFB300",           # Warning yellow
    
    # Text colors
    "text_primary": "#FFFFFF",
    "text_secondary": "#B0BEC5",
    "text_muted": "#607D8B",
    
    # Border colors
    "border": "#2D3A4F",
    "border_active": "#1E88E5",
}

# Typography
FONTS = {
    "title": ("Segoe UI", 24, "bold"),
    "heading": ("Segoe UI", 16, "bold"),
    "body": ("Segoe UI", 12),
    "small": ("Segoe UI", 10),
    "mono": ("Consolas", 11),
}

# Spacing
PADDING = {
    "xs": 4,
    "sm": 8,
    "md": 16,
    "lg": 24,
    "xl": 32,
}

# Simulation types configuration
SIMULATION_TYPES = {
    "cfd": {
        "name": "CFD",
        "description": "Computational Fluid Dynamics\nANSYS Fluent data",
        "color": "#1E88E5",
        "enabled": True,
    },
    "pinns": {
        "name": "PINNs",
        "description": "Physics-Informed Neural Networks\nPhysics-constrained learning",
        "color": "#9C27B0",
        "enabled": True,
    },
    "lstm": {
        "name": "LSTM",
        "description": "Long Short-Term Memory\nTime-series prediction",
        "color": "#FF9800",
        "enabled": False,
    },
}
