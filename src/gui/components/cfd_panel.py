"""
CFD Panel - Import and extract field data from ANSYS Fluent.
Step 1: Data Source (CFD)
"""

import customtkinter as ctk
import threading
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.gui.theme import COLORS, FONTS, PADDING

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class CFDPanel(ctk.CTkFrame):
    """Panel for CFD data import and extraction."""
    
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.is_running = False
        
        self.create_widgets()
        self.refresh_status()
    
    def create_widgets(self):
        # Title
        ctk.CTkLabel(
            self,
            text="Step 1: CFD Data Import",
            font=FONTS["title"],
            text_color=COLORS["text_primary"]
        ).pack(pady=(0, PADDING["xs"]))
        
        # Main scrollable frame
        main_frame = ctk.CTkScrollableFrame(
            self, 
            fg_color="transparent",
            scrollbar_button_color=COLORS["bg_light"]
        )
        main_frame.pack(fill="both", expand=True)
        
        # Status card
        status_card = ctk.CTkFrame(main_frame, fg_color=COLORS["bg_card"], corner_radius=10)
        status_card.pack(fill="x", pady=PADDING["xs"])
        
        status_header = ctk.CTkFrame(status_card, fg_color="transparent")
        status_header.pack(fill="x", padx=PADDING["sm"], pady=PADDING["xs"])
        
        ctk.CTkLabel(
            status_header,
            text="Raw Data",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        ).pack(side="left")
        
        self.status_indicator = ctk.CTkLabel(
            status_header,
            text="",
            font=FONTS["small"],
            text_color=COLORS["success"]
        )
        self.status_indicator.pack(side="right")
        
        self.raw_label = ctk.CTkLabel(
            status_card,
            text="Checking...",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
            justify="left"
        )
        self.raw_label.pack(padx=PADDING["sm"], pady=(0, PADDING["xs"]), anchor="w")
        
        # Processed card
        processed_card = ctk.CTkFrame(main_frame, fg_color=COLORS["bg_card"], corner_radius=10)
        processed_card.pack(fill="x", pady=PADDING["xs"])
        
        ctk.CTkLabel(
            processed_card,
            text="Processed Data",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        ).pack(padx=PADDING["sm"], pady=PADDING["xs"], anchor="w")
        
        self.processed_label = ctk.CTkLabel(
            processed_card,
            text="Checking...",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
            justify="left"
        )
        self.processed_label.pack(padx=PADDING["sm"], pady=(0, PADDING["xs"]), anchor="w")
        
        # Settings
        settings_frame = ctk.CTkFrame(main_frame, fg_color=COLORS["bg_card"], corner_radius=10)
        settings_frame.pack(fill="x", pady=PADDING["xs"])
        
        grid_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        grid_frame.pack(fill="x", padx=PADDING["sm"], pady=PADDING["xs"])
        
        ctk.CTkLabel(
            grid_frame,
            text="Grid Size:",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        ).pack(side="left")
        
        self.grid_size_var = ctk.StringVar(value="128")
        ctk.CTkOptionMenu(
            grid_frame,
            values=["64", "128", "256"],
            variable=self.grid_size_var,
            width=80,
            fg_color=COLORS["bg_light"],
            button_color=COLORS["primary"]
        ).pack(side="right")
        
        # Buttons
        buttons_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        buttons_frame.pack(fill="x", pady=PADDING["xs"])
        
        self.extract_btn = ctk.CTkButton(
            buttons_frame,
            text="Extract & Process",
            font=FONTS["body"],
            fg_color=COLORS["bg_light"],
            hover_color=COLORS["primary_hover"],
            height=35,
            state="disabled",
            command=self.run_extraction
        )
        self.extract_btn.pack(side="left", expand=True, fill="x", padx=(0, PADDING["xs"]))
        
        self.preview_btn = ctk.CTkButton(
            buttons_frame,
            text="Preview",
            font=FONTS["body"],
            fg_color=COLORS["bg_light"],
            hover_color=COLORS["primary"],
            height=35,
            state="disabled",
            command=self.show_preview
        )
        self.preview_btn.pack(side="left", expand=True, fill="x")
        
        # Progress
        self.progress = ctk.CTkProgressBar(
            main_frame,
            fg_color=COLORS["bg_card"],
            progress_color=COLORS["primary"],
            height=4
        )
        self.progress.pack(fill="x", pady=PADDING["xs"])
        self.progress.set(0)
        
        # Log
        self.log_area = ctk.CTkTextbox(
            main_frame,
            height=80,
            font=FONTS["mono"],
            fg_color=COLORS["bg_card"],
            text_color=COLORS["text_secondary"],
            corner_radius=8
        )
        self.log_area.pack(fill="both", expand=True, pady=PADDING["xs"])
    
    def get_raw_cases(self):
        """Get list of raw data cases."""
        data_dir = os.path.join("data", "raw")
        cases = []
        
        if os.path.exists(data_dir):
            for folder in os.listdir(data_dir):
                folder_path = os.path.join(data_dir, folder)
                if os.path.isdir(folder_path) and not folder.startswith('.'):
                    cases.append(folder)
        
        return cases
    
    def get_processed_cases(self):
        """Get list of processed CFD cases."""
        cfd_dir = os.path.join("data", "processed", "cfd")
        cases = []
        
        if os.path.exists(cfd_dir):
            for profile in os.listdir(cfd_dir):
                profile_dir = os.path.join(cfd_dir, profile)
                if os.path.isdir(profile_dir):
                    for angle_folder in os.listdir(profile_dir):
                        angle_dir = os.path.join(profile_dir, angle_folder)
                        if os.path.isdir(angle_dir):
                            info_path = os.path.join(angle_dir, "info.json")
                            if os.path.exists(info_path):
                                with open(info_path) as f:
                                    info = json.load(f)
                                cases.append({
                                    'profile': profile,
                                    'angle': info.get('angle_of_attack', 0),
                                    'cl': info.get('cl'),
                                    'cd': info.get('cd'),
                                    'path': angle_dir
                                })
        
        return cases
    
    def refresh_status(self):
        """Update status display."""
        raw_cases = self.get_raw_cases()
        processed_cases = self.get_processed_cases()
        
        # Raw data
        if raw_cases:
            raw_text = "\n".join(f"• {c}" for c in raw_cases[:4])
            if len(raw_cases) > 4:
                raw_text += f"\n(+{len(raw_cases) - 4} more)"
            self.raw_label.configure(text=raw_text)
            self.status_indicator.configure(text=f"{len(raw_cases)} case(s)")
        else:
            self.raw_label.configure(text="No data. Add folders to data/raw/")
            self.status_indicator.configure(text="", text_color=COLORS["warning"])
        
        # Processed data
        if processed_cases:
            proc_text = ""
            for case in processed_cases[:4]:
                cl = f"{case['cl']:.4f}" if case['cl'] else "?"
                cd = f"{case['cd']:.4f}" if case['cd'] else "?"
                proc_text += f"• {case['profile']} @ {case['angle']}° (CL={cl}, CD={cd})\n"
            self.processed_label.configure(text=proc_text.strip(), text_color=COLORS["success"])
            self.preview_btn.configure(state="normal", fg_color=COLORS["primary"])
        else:
            self.processed_label.configure(text="No processed data", text_color=COLORS["text_muted"])
            self.preview_btn.configure(state="disabled", fg_color=COLORS["bg_light"])
        
        # Buttons
        if not raw_cases:
            self.extract_btn.configure(state="disabled", fg_color=COLORS["bg_light"])
            self.log("Add CFD folders to data/raw/")
        elif len(processed_cases) < len(raw_cases):
            self.extract_btn.configure(state="normal", fg_color=COLORS["primary"])
            self.log(f"Ready to process {len(raw_cases)} case(s)")
        else:
            self.extract_btn.configure(state="normal", fg_color=COLORS["warning"], text="Re-process")
            self.log("All processed. Ready for Step 2.")
    
    def log(self, message):
        self.log_area.delete("1.0", "end")
        self.log_area.insert("1.0", message)
    
    def append_log(self, message):
        self.log_area.insert("end", "\n" + message)
        self.log_area.see("end")
    
    def run_extraction(self):
        """Extract CFD data."""
        if self.is_running:
            return
        
        def task():
            self.is_running = True
            self.extract_btn.configure(state="disabled")
            self.progress.set(0)
            self.log("Extracting...")
            
            try:
                from src.extract_data import extract_dataset
                grid_size = int(self.grid_size_var.get())
                
                self.progress.set(0.2)
                X, y = extract_dataset("data/raw", "data/processed", grid_size=grid_size)
                
                self.progress.set(1.0)
                if X is not None:
                    self.append_log(f"Done! {X.shape[0]} samples")
                else:
                    self.append_log("No data extracted")
                    
            except Exception as e:
                self.append_log(f"Error: {e}")
            
            finally:
                self.is_running = False
                self.refresh_status()
        
        threading.Thread(target=task, daemon=True).start()
    
    def show_preview(self):
        """Show field preview."""
        processed_cases = self.get_processed_cases()
        if not processed_cases:
            return
        
        case = processed_cases[0]
        fields = np.load(os.path.join(case['path'], "fields.npy"))
        
        popup = ctk.CTkToplevel(self)
        popup.title(f"{case['profile']} @ {case['angle']}°")
        popup.geometry("800x500")
        popup.configure(fg_color=COLORS["bg_dark"])
        popup.grab_set()
        
        fig = Figure(figsize=(9, 5), dpi=100, facecolor=COLORS["bg_card"])
        
        titles = ['Pressure', 'X-Velocity', 'Y-Velocity', 'Velocity Mag']
        cmaps = ['coolwarm', 'RdBu_r', 'RdBu_r', 'jet']
        
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1)
            im = ax.imshow(fields[:, :, i], cmap=cmaps[i], aspect='auto')
            ax.set_title(titles[i], color='white', fontsize=10)
            ax.axis('off')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=PADDING["sm"], pady=PADDING["sm"])
        
        ctk.CTkButton(popup, text="Close", command=popup.destroy).pack(pady=PADDING["sm"])
