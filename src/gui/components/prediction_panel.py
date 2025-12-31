"""
Prediction Panel - CNN-based CL/CD prediction.
Step 2: Predict coefficients from field data.
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


class PredictionPanel(ctk.CTkFrame):
    """Panel for CNN-based CL/CD prediction."""
    
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
            text="Step 2: Predict CL/CD",
            font=FONTS["title"],
            text_color=COLORS["text_primary"]
        )
        title.pack(pady=(0, PADDING["sm"]))
        
        # Description
        desc = ctk.CTkLabel(
            self,
            text="Train CNN on CFD data and predict coefficients",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        )
        desc.pack(pady=(0, PADDING["lg"]))
        
        # Model status
        model_card = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=10)
        model_card.pack(fill="x", pady=PADDING["sm"])
        
        ctk.CTkLabel(
            model_card,
            text="Model Status",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        ).pack(padx=PADDING["md"], pady=(PADDING["sm"], 0), anchor="w")
        
        self.model_label = ctk.CTkLabel(
            model_card,
            text="Checking...",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            justify="left"
        )
        self.model_label.pack(padx=PADDING["md"], pady=(PADDING["xs"], PADDING["sm"]), anchor="w")
        
        # Data status
        data_card = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=10)
        data_card.pack(fill="x", pady=PADDING["sm"])
        
        ctk.CTkLabel(
            data_card,
            text="Available Data",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        ).pack(padx=PADDING["md"], pady=(PADDING["sm"], 0), anchor="w")
        
        self.data_label = ctk.CTkLabel(
            data_card,
            text="Checking...",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            justify="left"
        )
        self.data_label.pack(padx=PADDING["md"], pady=(PADDING["xs"], PADDING["sm"]), anchor="w")
        
        # Settings
        settings_card = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=10)
        settings_card.pack(fill="x", pady=PADDING["sm"])
        
        epochs_frame = ctk.CTkFrame(settings_card, fg_color="transparent")
        epochs_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["sm"])
        
        ctk.CTkLabel(
            epochs_frame,
            text="Training Epochs:",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        ).pack(side="left")
        
        self.epochs_var = ctk.StringVar(value="50")
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
        
        self.train_btn = ctk.CTkButton(
            buttons_frame,
            text="Train CNN",
            font=FONTS["heading"],
            fg_color=COLORS["bg_light"],
            hover_color="#00A045",
            height=45,
            state="disabled",
            command=self.run_training
        )
        self.train_btn.pack(side="left", expand=True, fill="x", padx=(0, PADDING["xs"]))
        
        self.results_btn = ctk.CTkButton(
            buttons_frame,
            text="View Results",
            font=FONTS["body"],
            fg_color=COLORS["bg_light"],
            hover_color=COLORS["primary"],
            height=45,
            state="disabled",
            command=self.show_results
        )
        self.results_btn.pack(side="left", expand=True, fill="x", padx=(PADDING["xs"], 0))
        
        # Progress
        self.progress = ctk.CTkProgressBar(
            self,
            fg_color=COLORS["bg_card"],
            progress_color=COLORS["success"],
            height=6
        )
        self.progress.pack(fill="x", pady=PADDING["xs"])
        self.progress.set(0)
        
        # Log
        self.log_area = ctk.CTkTextbox(
            self,
            height=120,
            font=FONTS["mono"],
            fg_color=COLORS["bg_card"],
            text_color=COLORS["text_secondary"],
            corner_radius=10
        )
        self.log_area.pack(fill="both", expand=True, pady=(PADDING["xs"], 0))
    
    def get_manifest(self):
        """Load combined dataset manifest."""
        manifest_path = os.path.join("data", "processed", "combined", "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                return json.load(f)
        return None
    
    def get_model_info(self):
        """Get trained model info."""
        info_path = os.path.join("data", "processed", "models", "cnn_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                return json.load(f)
        return None
    
    def refresh_status(self):
        """Update status display."""
        manifest = self.get_manifest()
        model_info = self.get_model_info()
        model_path = os.path.join("data", "processed", "models", "cnn_model.keras")
        
        if not manifest:
            self.data_label.configure(
                text="No data available.\nGo to Step 1 first.",
                text_color=COLORS["warning"]
            )
            self.train_btn.configure(state="disabled", fg_color=COLORS["bg_light"])
            self.log("No data. Extract CFD data in Step 1.")
            return
        
        cfd_count = manifest.get('cfd_samples', 0)
        pinn_count = manifest.get('pinn_samples', 0)
        
        data_text = f"Total samples: {manifest['total_samples']}\n"
        data_text += f"  • CFD (trainable): {cfd_count}\n"
        data_text += f"  • PINN (inference only): {pinn_count}"
        
        self.data_label.configure(text=data_text, text_color=COLORS["text_secondary"])
        
        if not os.path.exists(model_path):
            self.model_label.configure(
                text=f"Not trained\n{cfd_count} CFD samples available",
                text_color=COLORS["warning"]
            )
            self.train_btn.configure(
                state="normal" if cfd_count > 0 else "disabled",
                fg_color=COLORS["success"] if cfd_count > 0 else COLORS["bg_light"],
                text="Train CNN"
            )
            self.results_btn.configure(state="disabled", fg_color=COLORS["bg_light"])
            self.log(f"Ready to train on {cfd_count} CFD samples.")
            
        elif model_info:
            model_text = f"Trained ({model_info.get('epochs', '?')} epochs)\n"
            model_text += f"MAE: CL={model_info.get('mae_cl', 0):.6f}, CD={model_info.get('mae_cd', 0):.6f}\n"
            model_text += f"Samples: {model_info.get('num_samples', '?')}"
            
            self.model_label.configure(text=model_text, text_color=COLORS["success"])
            self.train_btn.configure(state="normal", fg_color=COLORS["warning"], text="Re-train")
            self.results_btn.configure(state="normal", fg_color=COLORS["primary"])
            self.log("Model ready. Click 'View Results' to see predictions.")
            
        else:
            self.model_label.configure(text="Model found (no info)", text_color=COLORS["warning"])
            self.train_btn.configure(state="normal", fg_color=COLORS["warning"], text="Re-train")
            self.results_btn.configure(state="normal", fg_color=COLORS["primary"])
    
    def log(self, message):
        self.log_area.delete("1.0", "end")
        self.log_area.insert("1.0", message)
    
    def append_log(self, message):
        self.log_area.insert("end", "\n" + message)
        self.log_area.see("end")
    
    def run_training(self):
        """Train CNN on CFD data."""
        if self.is_running:
            return
        
        def task():
            self.is_running = True
            self.train_btn.configure(state="disabled")
            self.progress.set(0)
            
            self.log("Starting CNN training...")
            self.append_log("Training on CFD data only")
            
            try:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
                from tensorflow.keras.optimizers import Adam
                
                X = np.load("data/processed/combined/X.npy")
                y = np.load("data/processed/combined/y.npy")
                
                valid_mask = ~np.isnan(y).any(axis=1)
                X_train = X[valid_mask]
                y_train = y[valid_mask]
                
                self.append_log(f"\nTraining samples: {len(X_train)}")
                self.progress.set(0.1)
                
                model = Sequential([
                    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
                    MaxPooling2D((2, 2)),
                    Conv2D(64, (3, 3), activation='relu', padding='same'),
                    MaxPooling2D((2, 2)),
                    Conv2D(64, (3, 3), activation='relu', padding='same'),
                    MaxPooling2D((2, 2)),
                    Flatten(),
                    Dense(128, activation='relu'),
                    Dropout(0.3),
                    Dense(64, activation='relu'),
                    Dense(2)
                ])
                
                model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
                
                epochs = int(self.epochs_var.get())
                self.append_log(f"Training for {epochs} epochs...")
                
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=min(32, len(X_train)),
                    validation_split=0.2 if len(X_train) > 5 else 0,
                    verbose=0
                )
                
                self.progress.set(0.9)
                
                models_dir = os.path.join("data", "processed", "models")
                os.makedirs(models_dir, exist_ok=True)
                model.save(os.path.join(models_dir, "cnn_model.keras"))
                
                predictions = model.predict(X_train, verbose=0)
                mae_cl = float(np.mean(np.abs(predictions[:, 0] - y_train[:, 0])))
                mae_cd = float(np.mean(np.abs(predictions[:, 1] - y_train[:, 1])))
                
                with open(os.path.join(models_dir, "cnn_info.json"), 'w') as f:
                    json.dump({
                        'epochs': epochs,
                        'num_samples': len(X_train),
                        'mae_cl': mae_cl,
                        'mae_cd': mae_cd,
                        'final_loss': float(history.history['loss'][-1])
                    }, f, indent=2)
                
                self.progress.set(1.0)
                self.append_log(f"\nComplete! MAE CL: {mae_cl:.6f}, CD: {mae_cd:.6f}")
                
            except Exception as e:
                self.append_log(f"\nError: {str(e)}")
                import traceback
                self.append_log(traceback.format_exc())
            
            finally:
                self.is_running = False
                self.refresh_status()
        
        threading.Thread(target=task, daemon=True).start()
    
    def show_results(self):
        """Show enhanced results popup."""
        try:
            from tensorflow.keras.models import load_model
            
            model = load_model("data/processed/models/cnn_model.keras", compile=False)
            X = np.load("data/processed/combined/X.npy")
            y = np.load("data/processed/combined/y.npy")
            
            with open("data/processed/combined/manifest.json") as f:
                manifest = json.load(f)
            
            predictions = model.predict(X, verbose=0)
            
        except Exception as e:
            self.append_log(f"Error: {e}")
            return
        
        ResultsPopup(self, X, y, predictions, manifest)


class ResultsPopup(ctk.CTkToplevel):
    """Enhanced results popup with sample selector."""
    
    def __init__(self, parent, X, y, predictions, manifest):
        super().__init__(parent)
        
        self.X = X
        self.y = y
        self.predictions = predictions
        self.manifest = manifest
        self.samples = manifest.get('samples', [])
        
        self.title("Prediction Results")
        self.geometry("1100x800")
        self.configure(fg_color=COLORS["bg_dark"])
        self.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Header
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_medium"], corner_radius=0)
        header.pack(fill="x")
        
        ctk.CTkLabel(
            header,
            text="CL/CD Prediction Results",
            font=FONTS["title"],
            text_color=COLORS["text_primary"]
        ).pack(side="left", padx=PADDING["lg"], pady=PADDING["md"])
        
        # Close button
        ctk.CTkButton(
            header,
            text="Close",
            width=80,
            fg_color=COLORS["accent"],
            command=self.destroy
        ).pack(side="right", padx=PADDING["lg"], pady=PADDING["md"])
        
        # Sample selector
        selector_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=10)
        selector_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["sm"])
        
        ctk.CTkLabel(
            selector_frame,
            text="Select Sample:",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        ).pack(side="left", padx=PADDING["md"], pady=PADDING["sm"])
        
        # Build sample options
        sample_options = []
        for i, sample in enumerate(self.samples):
            source = sample.get('source', 'Unknown')
            profile = sample.get('profile', f'Sample {i+1}')
            angle = sample.get('angle', '?')
            label = f"{i+1}. {profile} @ {angle}° ({source})"
            sample_options.append(label)
        
        if not sample_options:
            sample_options = ["No samples"]
        
        self.sample_var = ctk.StringVar(value=sample_options[0])
        self.sample_menu = ctk.CTkOptionMenu(
            selector_frame,
            values=sample_options,
            variable=self.sample_var,
            width=300,
            fg_color=COLORS["bg_light"],
            button_color=COLORS["primary"],
            command=self.on_sample_change
        )
        self.sample_menu.pack(side="left", padx=PADDING["sm"], pady=PADDING["sm"])
        
        # Quick nav buttons
        ctk.CTkButton(
            selector_frame,
            text="◀ Prev",
            width=70,
            fg_color=COLORS["bg_light"],
            command=self.prev_sample
        ).pack(side="left", padx=PADDING["xs"])
        
        ctk.CTkButton(
            selector_frame,
            text="Next ▶",
            width=70,
            fg_color=COLORS["bg_light"],
            command=self.next_sample
        ).pack(side="left", padx=PADDING["xs"])
        
        # Main content
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=PADDING["md"], pady=PADDING["sm"])
        
        self.update_content()
    
    def get_current_index(self):
        """Get current sample index."""
        try:
            current = self.sample_var.get()
            return int(current.split('.')[0]) - 1
        except:
            return 0
    
    def prev_sample(self):
        idx = max(0, self.get_current_index() - 1)
        options = self.sample_menu.cget("values")
        self.sample_var.set(options[idx])
        self.update_content()
    
    def next_sample(self):
        idx = min(len(self.samples) - 1, self.get_current_index() + 1)
        options = self.sample_menu.cget("values")
        self.sample_var.set(options[idx])
        self.update_content()
    
    def on_sample_change(self, value):
        self.update_content()
    
    def update_content(self):
        """Update content for selected sample."""
        # Clear existing
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        idx = self.get_current_index()
        if idx >= len(self.samples):
            return
        
        sample = self.samples[idx]
        pred = self.predictions[idx]
        actual = self.y[idx]
        fields = self.X[idx]
        
        is_cfd = sample.get('source') == 'CFD' and not np.isnan(actual).any()
        
        # Info card
        info_card = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"], corner_radius=10)
        info_card.pack(fill="x", pady=PADDING["xs"])
        
        profile = sample.get('profile', 'Unknown')
        angle = sample.get('angle', '?')
        source = sample.get('source', 'Unknown')
        source_color = COLORS["primary"] if source == 'CFD' else "#9C27B0"
        
        info_left = ctk.CTkFrame(info_card, fg_color="transparent")
        info_left.pack(side="left", padx=PADDING["md"], pady=PADDING["sm"])
        
        ctk.CTkLabel(
            info_left,
            text=f"{profile}",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            info_left,
            text=f"Angle of Attack: {angle}°",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        ).pack(anchor="w")
        
        # Source badge
        ctk.CTkLabel(
            info_card,
            text=source,
            font=FONTS["body"],
            text_color="white",
            fg_color=source_color,
            corner_radius=6,
            width=60,
            height=28
        ).pack(side="right", padx=PADDING["md"], pady=PADDING["sm"])
        
        # Results section
        results_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        results_frame.pack(fill="both", expand=True, pady=PADDING["xs"])
        
        # Left: Coefficients
        coef_frame = ctk.CTkFrame(results_frame, fg_color=COLORS["bg_card"], corner_radius=10)
        coef_frame.pack(side="left", fill="both", expand=True, padx=(0, PADDING["xs"]))
        
        ctk.CTkLabel(
            coef_frame,
            text="Predicted Coefficients",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        ).pack(padx=PADDING["md"], pady=PADDING["sm"], anchor="w")
        
        # CL
        cl_frame = ctk.CTkFrame(coef_frame, fg_color=COLORS["bg_light"], corner_radius=8)
        cl_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["xs"])
        
        ctk.CTkLabel(
            cl_frame,
            text="Lift Coefficient (CL)",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        ).pack(anchor="w", padx=PADDING["sm"], pady=(PADDING["xs"], 0))
        
        cl_value_frame = ctk.CTkFrame(cl_frame, fg_color="transparent")
        cl_value_frame.pack(fill="x", padx=PADDING["sm"], pady=PADDING["xs"])
        
        ctk.CTkLabel(
            cl_value_frame,
            text=f"Predicted: {pred[0]:.6f}",
            font=("Segoe UI", 18, "bold"),
            text_color=COLORS["accent"]
        ).pack(side="left")
        
        if is_cfd:
            ctk.CTkLabel(
                cl_value_frame,
                text=f"Actual: {actual[0]:.6f}",
                font=FONTS["body"],
                text_color=COLORS["text_secondary"]
            ).pack(side="right")
        
        # CD
        cd_frame = ctk.CTkFrame(coef_frame, fg_color=COLORS["bg_light"], corner_radius=8)
        cd_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["xs"])
        
        ctk.CTkLabel(
            cd_frame,
            text="Drag Coefficient (CD)",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        ).pack(anchor="w", padx=PADDING["sm"], pady=(PADDING["xs"], 0))
        
        cd_value_frame = ctk.CTkFrame(cd_frame, fg_color="transparent")
        cd_value_frame.pack(fill="x", padx=PADDING["sm"], pady=PADDING["xs"])
        
        ctk.CTkLabel(
            cd_value_frame,
            text=f"Predicted: {pred[1]:.6f}",
            font=("Segoe UI", 18, "bold"),
            text_color=COLORS["accent"]
        ).pack(side="left")
        
        if is_cfd:
            ctk.CTkLabel(
                cd_value_frame,
                text=f"Actual: {actual[1]:.6f}",
                font=FONTS["body"],
                text_color=COLORS["text_secondary"]
            ).pack(side="right")
        
        # Error section (CFD only)
        if is_cfd:
            error_frame = ctk.CTkFrame(coef_frame, fg_color=COLORS["bg_light"], corner_radius=8)
            error_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["xs"])
            
            cl_error = abs(pred[0] - actual[0])
            cd_error = abs(pred[1] - actual[1])
            cl_rel = cl_error / (abs(actual[0]) + 1e-10) * 100
            cd_rel = cd_error / (abs(actual[1]) + 1e-10) * 100
            
            ctk.CTkLabel(
                error_frame,
                text="Prediction Error",
                font=FONTS["small"],
                text_color=COLORS["text_muted"]
            ).pack(anchor="w", padx=PADDING["sm"], pady=(PADDING["xs"], 0))
            
            ctk.CTkLabel(
                error_frame,
                text=f"CL Error: {cl_error:.6f} ({cl_rel:.2f}%)  |  CD Error: {cd_error:.6f} ({cd_rel:.2f}%)",
                font=FONTS["body"],
                text_color=COLORS["warning"] if (cl_rel > 10 or cd_rel > 10) else COLORS["success"]
            ).pack(anchor="w", padx=PADDING["sm"], pady=PADDING["xs"])
        else:
            # PINN notice
            notice_frame = ctk.CTkFrame(coef_frame, fg_color=COLORS["bg_light"], corner_radius=8)
            notice_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["xs"])
            
            ctk.CTkLabel(
                notice_frame,
                text="PINN-generated sample - No ground truth available for comparison",
                font=FONTS["small"],
                text_color=COLORS["text_muted"]
            ).pack(padx=PADDING["sm"], pady=PADDING["sm"])
        
        # Right: Field visualization
        field_frame = ctk.CTkFrame(results_frame, fg_color=COLORS["bg_card"], corner_radius=10)
        field_frame.pack(side="left", fill="both", expand=True, padx=(PADDING["xs"], 0))
        
        ctk.CTkLabel(
            field_frame,
            text="Input Fields",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        ).pack(padx=PADDING["md"], pady=PADDING["sm"], anchor="w")
        
        fig = Figure(figsize=(6, 4), dpi=100, facecolor=COLORS["bg_card"])
        
        titles = ['Pressure', 'X-Vel', 'Y-Vel', 'Vel Mag']
        cmaps = ['coolwarm', 'RdBu_r', 'RdBu_r', 'jet']
        
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1)
            ax.imshow(fields[:, :, i], cmap=cmaps[i], aspect='auto')
            ax.set_title(titles[i], color='white', fontsize=9)
            ax.axis('off')
        
        fig.tight_layout(pad=1)
        
        canvas = FigureCanvasTkAgg(fig, master=field_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=PADDING["sm"], pady=PADDING["sm"])
        
        # Bottom: Summary stats (for all CFD)
        if self.manifest.get('cfd_samples', 0) > 0:
            stats_frame = ctk.CTkFrame(self.content_frame, fg_color=COLORS["bg_card"], corner_radius=10)
            stats_frame.pack(fill="x", pady=PADDING["xs"])
            
            # Calculate overall stats for CFD
            valid_mask = ~np.isnan(self.y).any(axis=1)
            if valid_mask.any():
                pred_valid = self.predictions[valid_mask]
                actual_valid = self.y[valid_mask]
                
                mae_cl = np.mean(np.abs(pred_valid[:, 0] - actual_valid[:, 0]))
                mae_cd = np.mean(np.abs(pred_valid[:, 1] - actual_valid[:, 1]))
                
                ctk.CTkLabel(
                    stats_frame,
                    text=f"Overall Performance (CFD samples): MAE CL = {mae_cl:.6f}  |  MAE CD = {mae_cd:.6f}",
                    font=FONTS["body"],
                    text_color=COLORS["text_secondary"]
                ).pack(padx=PADDING["md"], pady=PADDING["sm"])
