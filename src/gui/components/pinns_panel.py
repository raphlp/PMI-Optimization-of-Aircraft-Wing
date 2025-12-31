"""
PINNs Panel - Physics-Informed Neural Networks interface.
"""

import customtkinter as ctk
import threading
import os
import sys
import numpy as np
import pandas as pd
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.gui.theme import COLORS, FONTS, PADDING

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class PINNsPanel(ctk.CTkFrame):
    """Panel for Physics-Informed Neural Networks."""
    
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.is_running = False
        self.training_history = None
        self.predictions = None
        self.cfd_data = None
        
        self.create_widgets()
        self.check_existing_model()
        self.refresh_cfd_status()
    
    def create_widgets(self):
        # Title
        title = ctk.CTkLabel(
            self,
            text="Physics-Informed Neural Networks",
            font=FONTS["title"],
            text_color=COLORS["text_primary"]
        )
        title.pack(pady=(0, PADDING["sm"]))
        
        # Description
        desc = ctk.CTkLabel(
            self,
            text="Train neural networks with embedded Navier-Stokes equations",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        )
        desc.pack(pady=(0, PADDING["md"]))
        
        # Settings card
        settings_card = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=10)
        settings_card.pack(fill="x", pady=PADDING["sm"])
        
        ctk.CTkLabel(
            settings_card,
            text="Configuration",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        ).pack(padx=PADDING["md"], pady=PADDING["sm"], anchor="w")
        
        # NACA code
        naca_frame = ctk.CTkFrame(settings_card, fg_color="transparent")
        naca_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["xs"])
        
        ctk.CTkLabel(
            naca_frame,
            text="NACA Profile:",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        ).pack(side="left")
        
        self.naca_var = ctk.StringVar(value="23015")
        naca_entry = ctk.CTkEntry(
            naca_frame,
            textvariable=self.naca_var,
            width=100,
            fg_color=COLORS["bg_light"],
            border_color=COLORS["border"]
        )
        naca_entry.pack(side="right")
        
        # Bind to check CFD availability when profile changes
        self.naca_var.trace_add("write", lambda *args: self.refresh_cfd_status())
        
        # CFD Status indicator
        self.cfd_status_frame = ctk.CTkFrame(settings_card, fg_color="transparent")
        self.cfd_status_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["xs"])
        
        self.cfd_status_label = ctk.CTkLabel(
            self.cfd_status_frame,
            text="CFD Data: Checking...",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        self.cfd_status_label.pack(side="left")
        
        # Angle of attack
        aoa_frame = ctk.CTkFrame(settings_card, fg_color="transparent")
        aoa_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["xs"])
        
        ctk.CTkLabel(
            aoa_frame,
            text="Angle of Attack (deg):",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        ).pack(side="left")
        
        self.aoa_var = ctk.StringVar(value="0")
        aoa_entry = ctk.CTkEntry(
            aoa_frame,
            textvariable=self.aoa_var,
            width=100,
            fg_color=COLORS["bg_light"],
            border_color=COLORS["border"]
        )
        aoa_entry.pack(side="right")
        
        # Epochs
        epochs_frame = ctk.CTkFrame(settings_card, fg_color="transparent")
        epochs_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["xs"])
        
        ctk.CTkLabel(
            epochs_frame,
            text="Training Epochs:",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        ).pack(side="left")
        
        self.epochs_var = ctk.StringVar(value="2000")
        epochs_entry = ctk.CTkEntry(
            epochs_frame,
            textvariable=self.epochs_var,
            width=100,
            fg_color=COLORS["bg_light"],
            border_color=COLORS["border"]
        )
        epochs_entry.pack(side="right")
        
        # Use CFD data checkbox
        cfd_frame = ctk.CTkFrame(settings_card, fg_color="transparent")
        cfd_frame.pack(fill="x", padx=PADDING["md"], pady=(PADDING["xs"], PADDING["sm"]))
        
        self.use_cfd_var = ctk.BooleanVar(value=False)
        self.cfd_check = ctk.CTkCheckBox(
            cfd_frame,
            text="Use CFD data (Hybrid Mode)",
            variable=self.use_cfd_var,
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"]
        )
        self.cfd_check.pack(side="left")
        
        # Buttons
        buttons_frame = ctk.CTkFrame(self, fg_color="transparent")
        buttons_frame.pack(fill="x", pady=PADDING["sm"])
        
        self.train_btn = ctk.CTkButton(
            buttons_frame,
            text="Train PINN",
            font=FONTS["body"],
            fg_color="#9C27B0",
            hover_color="#7B1FA2",
            height=45,
            command=self.run_training
        )
        self.train_btn.pack(side="left", expand=True, fill="x", padx=(0, PADDING["xs"]))
        
        self.visualize_btn = ctk.CTkButton(
            buttons_frame,
            text="Compare Flow",
            font=FONTS["body"],
            fg_color=COLORS["bg_light"],
            hover_color=COLORS["primary"],
            height=45,
            state="disabled",
            command=self.show_comparison
        )
        self.visualize_btn.pack(side="left", expand=True, fill="x", padx=(PADDING["xs"], 0))
        
        # Progress
        self.progress = ctk.CTkProgressBar(
            self,
            fg_color=COLORS["bg_card"],
            progress_color="#9C27B0",
            height=6
        )
        self.progress.pack(fill="x", pady=PADDING["xs"])
        self.progress.set(0)
        
        self.status_label = ctk.CTkLabel(
            self,
            text="Status: Ready",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        self.status_label.pack(anchor="w")
        
        # Log area
        self.log_area = ctk.CTkTextbox(
            self,
            height=120,
            font=FONTS["mono"],
            fg_color=COLORS["bg_card"],
            text_color=COLORS["text_secondary"],
            corner_radius=10
        )
        self.log_area.pack(fill="both", expand=True, pady=(PADDING["xs"], 0))
        self.log("PINNs ready. Configure parameters and click 'Train PINN'.")
    
    def refresh_cfd_status(self):
        """Check if CFD data exists for the current NACA profile."""
        naca_code = self.naca_var.get()
        cfd_folder = self.find_cfd_folder(naca_code)
        
        if cfd_folder:
            self.cfd_status_label.configure(
                text=f"CFD Data: Available ({os.path.basename(cfd_folder)})",
                text_color=COLORS["success"]
            )
            self.cfd_check.configure(state="normal")
            self.cfd_data = self.load_cfd_data(cfd_folder)
        else:
            self.cfd_status_label.configure(
                text=f"CFD Data: Not found for NACA {naca_code}",
                text_color=COLORS["warning"]
            )
            self.use_cfd_var.set(False)
            self.cfd_check.configure(state="disabled")
            self.cfd_data = None
    
    def find_cfd_folder(self, naca_code):
        """Find CFD data folder matching the NACA code."""
        data_dir = os.path.join("data", "raw")
        if not os.path.exists(data_dir):
            return None
        
        for folder in os.listdir(data_dir):
            if naca_code.upper() in folder.upper():
                folder_path = os.path.join(data_dir, folder)
                if os.path.isdir(folder_path):
                    # Check if it has fields.csv
                    fields = glob.glob(os.path.join(folder_path, "*.csv")) + \
                             glob.glob(os.path.join(folder_path, "*fields*"))
                    if fields:
                        return folder_path
        return None
    
    def load_cfd_data(self, folder_path):
        """Load CFD data from folder."""
        try:
            fields_files = glob.glob(os.path.join(folder_path, "*.csv")) + \
                           glob.glob(os.path.join(folder_path, "*fields*"))
            
            if not fields_files:
                return None
            
            df = pd.read_csv(fields_files[0], skipinitialspace=True)
            df.columns = [col.strip().lower().replace('-', '_') for col in df.columns]
            
            return {
                'x': df['x_coordinate'].values,
                'y': df['y_coordinate'].values,
                'u': df['x_velocity'].values,
                'v': df['y_velocity'].values,
                'p': df['pressure'].values
            }
        except Exception as e:
            self.log(f"Error loading CFD data: {e}")
            return None
    
    def check_existing_model(self):
        """Check if a trained PINN model exists."""
        pred_path = os.path.join("data", "processed", "pinns_predictions.npz")
        
        if os.path.exists(pred_path):
            self.predictions = np.load(pred_path)
            self.visualize_btn.configure(state="normal", fg_color=COLORS["primary"])
            self.log("Existing PINN predictions found.")
    
    def log(self, message):
        self.log_area.insert("end", message + "\n")
        self.log_area.see("end")
    
    def set_status(self, status, color=None):
        self.status_label.configure(
            text=f"Status: {status}",
            text_color=color or COLORS["text_muted"]
        )
    
    def run_training(self):
        if self.is_running:
            return
        
        def task():
            self.is_running = True
            self.train_btn.configure(state="disabled")
            self.set_status("Training...", COLORS["warning"])
            self.progress.set(0)
            
            self.log("\n" + "="*50)
            self.log("Starting PINN training...")
            
            try:
                from src.train_pinns import train_pinns
                
                naca_code = self.naca_var.get()
                aoa = float(self.aoa_var.get())
                epochs = int(self.epochs_var.get())
                use_cfd = self.use_cfd_var.get()
                
                self.log(f"Profile: NACA {naca_code}")
                self.log(f"Angle of Attack: {aoa} deg")
                self.log(f"Mode: {'Hybrid (with CFD)' if use_cfd else 'Pure Physics'}")
                self.log(f"Epochs: {epochs}")
                
                def progress_callback(epoch, losses):
                    if epoch % 100 == 0:
                        self.progress.set(epoch / epochs)
                        self.log(f"Epoch {epoch}: Loss = {losses['total']:.6f}")
                
                model, trainer, history = train_pinns(
                    naca_code=naca_code,
                    angle_of_attack=aoa,
                    epochs=epochs,
                    use_cfd_data=use_cfd,
                    data_dir="data/raw",
                    output_dir="data/processed",
                    callback=progress_callback
                )
                
                self.training_history = history
                self.progress.set(1.0)
                
                # Load predictions
                pred_path = os.path.join("data", "processed", "pinns_predictions.npz")
                if os.path.exists(pred_path):
                    self.predictions = np.load(pred_path)
                    self.visualize_btn.configure(state="normal", fg_color=COLORS["primary"])
                
                self.log("\nTraining complete!")
                self.log(f"Final Loss: {history['loss'][-1]:.6f}")
                self.set_status("Training complete!", COLORS["success"])
                
            except Exception as e:
                self.log(f"Error: {str(e)}")
                self.set_status("Error!", COLORS["accent"])
                import traceback
                self.log(traceback.format_exc())
            
            finally:
                self.is_running = False
                self.train_btn.configure(state="normal")
        
        threading.Thread(target=task, daemon=True).start()
    
    def show_comparison(self):
        """Show PINN predictions vs CFD ground truth."""
        if self.predictions is None:
            return
        
        popup = ctk.CTkToplevel(self)
        popup.title("Flow Comparison: PINN vs CFD")
        popup.geometry("1200x800")
        popup.configure(fg_color=COLORS["bg_dark"])
        popup.grab_set()
        
        # Title
        title_text = "Flow Comparison"
        if self.cfd_data is not None:
            title_text += " (PINN Prediction vs CFD Ground Truth)"
        else:
            title_text += " (PINN Prediction Only - No CFD data)"
        
        title = ctk.CTkLabel(
            popup,
            text=title_text,
            font=FONTS["title"],
            text_color=COLORS["text_primary"]
        )
        title.pack(pady=PADDING["sm"])
        
        # Create figure
        fig = Figure(figsize=(14, 10), dpi=100, facecolor=COLORS["bg_card"])
        
        X = self.predictions['x']
        Y = self.predictions['y']
        u_pinn = self.predictions['u']
        v_pinn = self.predictions['v']
        p_pinn = self.predictions['p']
        
        vel_pinn = np.sqrt(u_pinn**2 + v_pinn**2)
        
        # Get airfoil shape
        from src.models.naca_geometry import generate_naca_profile
        x_up, y_up, x_lo, y_lo = generate_naca_profile(self.naca_var.get(), 100)
        airfoil_x = np.concatenate([x_up, x_lo[::-1]])
        airfoil_y = np.concatenate([y_up, y_lo[::-1]])
        
        if self.cfd_data is not None:
            # Side-by-side comparison
            from scipy.interpolate import griddata
            
            # Interpolate CFD data to same grid
            cfd_points = np.column_stack([self.cfd_data['x'], self.cfd_data['y']])
            
            vel_cfd_raw = np.sqrt(self.cfd_data['u']**2 + self.cfd_data['v']**2)
            vel_cfd = griddata(cfd_points, vel_cfd_raw, (X, Y), method='linear')
            p_cfd = griddata(cfd_points, self.cfd_data['p'], (X, Y), method='linear')
            
            # Row 1: Velocity comparison
            ax1 = fig.add_subplot(231)
            ax1.set_facecolor(COLORS["bg_light"])
            c1 = ax1.contourf(X, Y, vel_cfd, levels=30, cmap='jet')
            ax1.fill(airfoil_x, airfoil_y, color='gray', alpha=0.8)
            ax1.set_title('CFD: Velocity (m/s)', color='white', fontweight='bold')
            ax1.set_xlabel('x/c', color='white')
            ax1.set_ylabel('y/c', color='white')
            ax1.tick_params(colors='white')
            ax1.set_aspect('equal')
            fig.colorbar(c1, ax=ax1)
            
            ax2 = fig.add_subplot(232)
            ax2.set_facecolor(COLORS["bg_light"])
            c2 = ax2.contourf(X, Y, vel_pinn, levels=30, cmap='jet')
            ax2.fill(airfoil_x, airfoil_y, color='gray', alpha=0.8)
            ax2.set_title('PINN: Velocity (m/s)', color='white', fontweight='bold')
            ax2.set_xlabel('x/c', color='white')
            ax2.tick_params(colors='white')
            ax2.set_aspect('equal')
            fig.colorbar(c2, ax=ax2)
            
            # Error
            vel_error = np.abs(vel_pinn - vel_cfd)
            ax3 = fig.add_subplot(233)
            ax3.set_facecolor(COLORS["bg_light"])
            c3 = ax3.contourf(X, Y, vel_error, levels=30, cmap='Reds')
            ax3.fill(airfoil_x, airfoil_y, color='gray', alpha=0.8)
            ax3.set_title('Error: |PINN - CFD|', color='white', fontweight='bold')
            ax3.set_xlabel('x/c', color='white')
            ax3.tick_params(colors='white')
            ax3.set_aspect('equal')
            fig.colorbar(c3, ax=ax3)
            
            # Row 2: Pressure comparison
            ax4 = fig.add_subplot(234)
            ax4.set_facecolor(COLORS["bg_light"])
            c4 = ax4.contourf(X, Y, p_cfd, levels=30, cmap='coolwarm')
            ax4.fill(airfoil_x, airfoil_y, color='gray', alpha=0.8)
            ax4.set_title('CFD: Pressure (Pa)', color='white', fontweight='bold')
            ax4.set_xlabel('x/c', color='white')
            ax4.set_ylabel('y/c', color='white')
            ax4.tick_params(colors='white')
            ax4.set_aspect('equal')
            fig.colorbar(c4, ax=ax4)
            
            ax5 = fig.add_subplot(235)
            ax5.set_facecolor(COLORS["bg_light"])
            c5 = ax5.contourf(X, Y, p_pinn, levels=30, cmap='coolwarm')
            ax5.fill(airfoil_x, airfoil_y, color='gray', alpha=0.8)
            ax5.set_title('PINN: Pressure (Pa)', color='white', fontweight='bold')
            ax5.set_xlabel('x/c', color='white')
            ax5.tick_params(colors='white')
            ax5.set_aspect('equal')
            fig.colorbar(c5, ax=ax5)
            
            # Pressure error
            p_error = np.abs(p_pinn - p_cfd)
            ax6 = fig.add_subplot(236)
            ax6.set_facecolor(COLORS["bg_light"])
            c6 = ax6.contourf(X, Y, p_error, levels=30, cmap='Reds')
            ax6.fill(airfoil_x, airfoil_y, color='gray', alpha=0.8)
            ax6.set_title('Error: |PINN - CFD|', color='white', fontweight='bold')
            ax6.set_xlabel('x/c', color='white')
            ax6.tick_params(colors='white')
            ax6.set_aspect('equal')
            fig.colorbar(c6, ax=ax6)
            
        else:
            # PINN only (no CFD to compare)
            ax1 = fig.add_subplot(221)
            ax1.set_facecolor(COLORS["bg_light"])
            c1 = ax1.contourf(X, Y, vel_pinn, levels=50, cmap='jet')
            ax1.fill(airfoil_x, airfoil_y, color='gray', alpha=0.8)
            ax1.set_title('PINN: Velocity Magnitude (m/s)', color='white')
            ax1.set_xlabel('x/c', color='white')
            ax1.set_ylabel('y/c', color='white')
            ax1.tick_params(colors='white')
            ax1.set_aspect('equal')
            fig.colorbar(c1, ax=ax1)
            
            ax2 = fig.add_subplot(222)
            ax2.set_facecolor(COLORS["bg_light"])
            c2 = ax2.contourf(X, Y, p_pinn, levels=50, cmap='coolwarm')
            ax2.fill(airfoil_x, airfoil_y, color='gray', alpha=0.8)
            ax2.set_title('PINN: Pressure (Pa)', color='white')
            ax2.set_xlabel('x/c', color='white')
            ax2.tick_params(colors='white')
            ax2.set_aspect('equal')
            fig.colorbar(c2, ax=ax2)
            
            ax3 = fig.add_subplot(223)
            ax3.set_facecolor(COLORS["bg_light"])
            c3 = ax3.contourf(X, Y, u_pinn, levels=50, cmap='RdBu_r')
            ax3.fill(airfoil_x, airfoil_y, color='gray', alpha=0.8)
            ax3.set_title('PINN: U Velocity (m/s)', color='white')
            ax3.set_xlabel('x/c', color='white')
            ax3.set_ylabel('y/c', color='white')
            ax3.tick_params(colors='white')
            ax3.set_aspect('equal')
            fig.colorbar(c3, ax=ax3)
            
            ax4 = fig.add_subplot(224)
            ax4.set_facecolor(COLORS["bg_light"])
            c4 = ax4.contourf(X, Y, v_pinn, levels=50, cmap='RdBu_r')
            ax4.fill(airfoil_x, airfoil_y, color='gray', alpha=0.8)
            ax4.set_title('PINN: V Velocity (m/s)', color='white')
            ax4.set_xlabel('x/c', color='white')
            ax4.tick_params(colors='white')
            ax4.set_aspect('equal')
            fig.colorbar(c4, ax=ax4)
            
            # Add info text
            info = ctk.CTkLabel(
                popup,
                text="No CFD data available for comparison. Add CFD data to data/raw/NACA23015_AoA0/",
                font=FONTS["body"],
                text_color=COLORS["warning"]
            )
            info.pack()
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=PADDING["md"], pady=PADDING["sm"])
        
        # Close button
        close_btn = ctk.CTkButton(
            popup,
            text="Close",
            font=FONTS["body"],
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            width=120,
            command=popup.destroy
        )
        close_btn.pack(pady=PADDING["sm"])
