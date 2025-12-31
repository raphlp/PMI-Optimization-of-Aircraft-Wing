"""
PINNs Panel - Generate additional field data for existing profiles.
Step 1: Data Source (PINNs)
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


class PINNsPanel(ctk.CTkFrame):
    """Panel for generating additional field data using PINNs."""
    
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.is_running = False
        
        self.create_widgets()
        self.refresh_status()
    
    def create_widgets(self):
        # Title
        title = ctk.CTkLabel(
            self,
            text="Step 1: PINNs Data Augmentation",
            font=FONTS["title"],
            text_color=COLORS["text_primary"]
        )
        title.pack(pady=(0, PADDING["sm"]))
        
        # Description
        desc = ctk.CTkLabel(
            self,
            text="Generate fields for new angles using Physics-Informed Neural Networks",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        )
        desc.pack(pady=(0, PADDING["lg"]))
        
        # Requirement notice
        notice = ctk.CTkFrame(self, fg_color=COLORS["bg_light"], corner_radius=10)
        notice.pack(fill="x", pady=PADDING["sm"])
        
        ctk.CTkLabel(
            notice,
            text="Requires CFD data first. PINNs uses existing CFD to guide generation.",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        ).pack(padx=PADDING["md"], pady=PADDING["sm"])
        
        # Status
        status_card = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=10)
        status_card.pack(fill="x", pady=PADDING["sm"])
        
        ctk.CTkLabel(
            status_card,
            text="Available Profiles",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        ).pack(padx=PADDING["md"], pady=(PADDING["sm"], 0), anchor="w")
        
        self.status_label = ctk.CTkLabel(
            status_card,
            text="Checking...",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            justify="left"
        )
        self.status_label.pack(padx=PADDING["md"], pady=(PADDING["xs"], PADDING["sm"]), anchor="w")
        
        # Configuration
        config_card = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=10)
        config_card.pack(fill="x", pady=PADDING["sm"])
        
        ctk.CTkLabel(
            config_card,
            text="Generation Settings",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        ).pack(padx=PADDING["md"], pady=PADDING["sm"], anchor="w")
        
        # Profile dropdown
        profile_frame = ctk.CTkFrame(config_card, fg_color="transparent")
        profile_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["xs"])
        
        ctk.CTkLabel(
            profile_frame,
            text="Base Profile:",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        ).pack(side="left")
        
        self.profile_var = ctk.StringVar(value="No profiles")
        self.profile_menu = ctk.CTkOptionMenu(
            profile_frame,
            values=["No profiles"],
            variable=self.profile_var,
            width=180,
            fg_color=COLORS["bg_light"],
            button_color=COLORS["primary"],
            button_hover_color=COLORS["primary_hover"],
            state="disabled",
            command=self.on_profile_change
        )
        self.profile_menu.pack(side="right")
        
        # Existing angles info
        self.existing_label = ctk.CTkLabel(
            config_card,
            text="",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        self.existing_label.pack(padx=PADDING["md"], anchor="w")
        
        # New angle
        angle_frame = ctk.CTkFrame(config_card, fg_color="transparent")
        angle_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["xs"])
        
        ctk.CTkLabel(
            angle_frame,
            text="New Angle of Attack (°):",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        ).pack(side="left")
        
        self.angle_var = ctk.StringVar(value="5")
        ctk.CTkEntry(
            angle_frame,
            textvariable=self.angle_var,
            width=100,
            fg_color=COLORS["bg_light"],
            border_color=COLORS["border"]
        ).pack(side="right")
        
        # Epochs
        epochs_frame = ctk.CTkFrame(config_card, fg_color="transparent")
        epochs_frame.pack(fill="x", padx=PADDING["md"], pady=(PADDING["xs"], PADDING["sm"]))
        
        ctk.CTkLabel(
            epochs_frame,
            text="Training Epochs:",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        ).pack(side="left")
        
        self.epochs_var = ctk.StringVar(value="2000")
        ctk.CTkEntry(
            epochs_frame,
            textvariable=self.epochs_var,
            width=100,
            fg_color=COLORS["bg_light"],
            border_color=COLORS["border"]
        ).pack(side="right")
        
        # Buttons
        buttons_frame = ctk.CTkFrame(self, fg_color="transparent")
        buttons_frame.pack(fill="x", pady=PADDING["sm"])
        
        self.generate_btn = ctk.CTkButton(
            buttons_frame,
            text="Generate New Angle",
            font=FONTS["heading"],
            fg_color=COLORS["bg_light"],
            hover_color="#7B1FA2",
            height=45,
            state="disabled",
            command=self.run_generation
        )
        self.generate_btn.pack(side="left", expand=True, fill="x", padx=(0, PADDING["xs"]))
        
        self.preview_btn = ctk.CTkButton(
            buttons_frame,
            text="Preview",
            font=FONTS["body"],
            fg_color=COLORS["bg_light"],
            hover_color=COLORS["primary"],
            height=45,
            state="disabled",
            command=self.show_preview
        )
        self.preview_btn.pack(side="left", expand=True, fill="x", padx=(PADDING["xs"], 0))
        
        # Progress
        self.progress = ctk.CTkProgressBar(
            self,
            fg_color=COLORS["bg_card"],
            progress_color="#9C27B0",
            height=6
        )
        self.progress.pack(fill="x", pady=PADDING["xs"])
        self.progress.set(0)
        
        # Log
        self.log_area = ctk.CTkTextbox(
            self,
            height=100,
            font=FONTS["mono"],
            fg_color=COLORS["bg_card"],
            text_color=COLORS["text_secondary"],
            corner_radius=10
        )
        self.log_area.pack(fill="both", expand=True, pady=(PADDING["xs"], 0))
    
    def get_cfd_profiles(self):
        """Get available CFD profiles and their angles."""
        cfd_dir = os.path.join("data", "processed", "cfd")
        profiles = {}
        
        if os.path.exists(cfd_dir):
            for profile in os.listdir(cfd_dir):
                profile_dir = os.path.join(cfd_dir, profile)
                if os.path.isdir(profile_dir):
                    angles = []
                    for angle_folder in os.listdir(profile_dir):
                        if angle_folder.startswith("AoA_"):
                            try:
                                angle = float(angle_folder.replace("AoA_", ""))
                                angles.append(angle)
                            except:
                                pass
                    if angles:
                        profiles[profile] = sorted(angles)
        
        return profiles
    
    def get_pinn_cases(self):
        """Get generated PINN cases."""
        pinns_dir = os.path.join("data", "processed", "pinns")
        cases = []
        
        if os.path.exists(pinns_dir):
            for profile in os.listdir(pinns_dir):
                profile_dir = os.path.join(pinns_dir, profile)
                if os.path.isdir(profile_dir):
                    for angle_folder in os.listdir(profile_dir):
                        angle_dir = os.path.join(profile_dir, angle_folder)
                        if os.path.isdir(angle_dir):
                            cases.append({
                                'profile': profile,
                                'angle_folder': angle_folder,
                                'path': angle_dir
                            })
        
        return cases
    
    def refresh_status(self):
        """Update status display."""
        profiles = self.get_cfd_profiles()
        pinn_cases = self.get_pinn_cases()
        
        if not profiles:
            self.status_label.configure(
                text="No CFD profiles found.\nFirst extract data in CFD tab.",
                text_color=COLORS["warning"]
            )
            self.profile_menu.configure(values=["No profiles"], state="disabled")
            self.generate_btn.configure(state="disabled", fg_color=COLORS["bg_light"])
            self.log("No CFD data. Use CFD tab first.")
        else:
            status_text = f"{len(profiles)} profile(s) available:\n"
            for profile, angles in profiles.items():
                angles_str = ", ".join(f"{a}°" for a in angles)
                status_text += f"  • {profile}: {angles_str}\n"
            
            if pinn_cases:
                status_text += f"\nPINN generated: {len(pinn_cases)} case(s)"
            
            self.status_label.configure(text=status_text.strip(), text_color=COLORS["success"])
            self.profile_menu.configure(values=list(profiles.keys()), state="normal")
            self.profile_var.set(list(profiles.keys())[0])
            self.generate_btn.configure(state="normal", fg_color="#9C27B0")
            self.on_profile_change(list(profiles.keys())[0])
            self.log(f"Select a profile and enter a new angle to generate.")
            
            if pinn_cases:
                self.preview_btn.configure(state="normal", fg_color=COLORS["primary"])
    
    def on_profile_change(self, profile):
        """Update when profile changes."""
        profiles = self.get_cfd_profiles()
        if profile in profiles:
            angles = profiles[profile]
            self.existing_label.configure(
                text=f"  Existing angles: {', '.join(f'{a}°' for a in angles)}"
            )
    
    def log(self, message):
        self.log_area.delete("1.0", "end")
        self.log_area.insert("1.0", message)
    
    def append_log(self, message):
        self.log_area.insert("end", "\n" + message)
        self.log_area.see("end")
    
    def run_generation(self):
        """Generate fields for new angle."""
        if self.is_running:
            return
        
        def task():
            self.is_running = True
            self.generate_btn.configure(state="disabled")
            self.progress.set(0)
            
            profile = self.profile_var.get()
            new_angle = float(self.angle_var.get())
            epochs = int(self.epochs_var.get())
            
            # Extract NACA code from profile name  
            naca_code = profile.replace("NACA", "")
            
            self.log(f"Generating {profile} @ AoA {new_angle}°...")
            self.append_log(f"NACA code: {naca_code}")
            self.append_log(f"Epochs: {epochs}")
            
            try:
                from src.train_pinns import train_pinns
                
                self.progress.set(0.1)
                
                model, trainer, history = train_pinns(
                    naca_code=naca_code,
                    angle_of_attack=new_angle,
                    epochs=epochs,
                    use_cfd_data=True,
                    data_dir="data/raw"
                )
                
                self.progress.set(0.8)
                
                # Save in organized structure
                self.save_pinn_result(profile, new_angle)
                
                # Rebuild combined dataset
                from src.extract_data import build_combined_dataset
                build_combined_dataset("data/processed")
                
                self.progress.set(1.0)
                self.append_log(f"\nGeneration complete!")
                self.append_log(f"Saved to: data/processed/pinns/{profile}/AoA_{new_angle}/")
                
            except Exception as e:
                self.append_log(f"\nError: {str(e)}")
                import traceback
                self.append_log(traceback.format_exc())
            
            finally:
                self.is_running = False
                self.generate_btn.configure(state="normal")
                self.refresh_status()
        
        threading.Thread(target=task, daemon=True).start()
    
    def save_pinn_result(self, profile, angle):
        """Save PINN result in organized structure."""
        pred_path = os.path.join("data", "processed", "pinns_predictions.npz")
        if not os.path.exists(pred_path):
            return
        
        # Load predictions
        data = np.load(pred_path)
        u = data['u']
        v = data['v']
        p = data['p']
        
        # Create field array
        vel_mag = np.sqrt(u**2 + v**2)
        
        # Normalize
        fields = []
        for field in [p, u, v, vel_mag]:
            f_min, f_max = field.min(), field.max()
            if f_max - f_min > 1e-10:
                field = (field - f_min) / (f_max - f_min)
            fields.append(field)
        fields = np.stack(fields, axis=-1)
        
        # Save in organized structure
        output_dir = os.path.join("data", "processed", "pinns", profile, f"AoA_{angle}")
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, "fields.npy"), fields)
        
        info = {
            "source": "PINN",
            "profile": profile,
            "angle_of_attack": angle,
            "grid_size": fields.shape[0]
        }
        with open(os.path.join(output_dir, "info.json"), 'w') as f:
            json.dump(info, f, indent=2)
    
    def show_preview(self):
        """Show PINN generated fields."""
        pinn_cases = self.get_pinn_cases()
        if not pinn_cases:
            return
        
        case = pinn_cases[-1]  # Show latest
        fields_path = os.path.join(case['path'], "fields.npy")
        if not os.path.exists(fields_path):
            return
        
        fields = np.load(fields_path)
        
        popup = ctk.CTkToplevel(self)
        popup.title(f"PINN Preview: {case['profile']} {case['angle_folder']}")
        popup.geometry("900x600")
        popup.configure(fg_color=COLORS["bg_dark"])
        popup.grab_set()
        
        fig = Figure(figsize=(10, 6), dpi=100, facecolor=COLORS["bg_card"])
        
        titles = ['Pressure', 'X-Velocity', 'Y-Velocity', 'Velocity Magnitude']
        cmaps = ['coolwarm', 'RdBu_r', 'RdBu_r', 'jet']
        
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1)
            ax.set_facecolor(COLORS["bg_light"])
            im = ax.imshow(fields[:, :, i], cmap=cmaps[i], aspect='auto')
            ax.set_title(titles[i], color='white')
            ax.tick_params(colors='white')
            fig.colorbar(im, ax=ax)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=PADDING["md"], pady=PADDING["sm"])
        
        ctk.CTkButton(
            popup,
            text="Close",
            fg_color=COLORS["primary"],
            command=popup.destroy
        ).pack(pady=PADDING["sm"])
