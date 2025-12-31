"""
CFD Panel - Interface for CFD data pipeline.
"""

import customtkinter as ctk
import threading
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.gui.theme import COLORS, FONTS, PADDING

# Matplotlib setup for embedding in tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class CFDPanel(ctk.CTkFrame):
    """Panel for CFD data extraction and CNN training."""
    
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.is_running = False
        self.last_results = None
        self.loss_history = None
        
        self.create_widgets()
        self.refresh_data_info()
        self.check_existing_results()
    
    def create_widgets(self):
        # Title
        title = ctk.CTkLabel(
            self,
            text="CFD Data Pipeline",
            font=FONTS["title"],
            text_color=COLORS["text_primary"]
        )
        title.pack(pady=(0, PADDING["md"]))
        
        # Description
        desc = ctk.CTkLabel(
            self,
            text="Extract CFD field data and train a CNN to predict CL/CD coefficients",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        )
        desc.pack(pady=(0, PADDING["lg"]))
        
        # Data info card
        self.data_card = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=10)
        self.data_card.pack(fill="x", pady=PADDING["sm"])
        
        data_title = ctk.CTkLabel(
            self.data_card,
            text="Data Directory",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        )
        data_title.pack(padx=PADDING["md"], pady=(PADDING["sm"], 0), anchor="w")
        
        self.data_label = ctk.CTkLabel(
            self.data_card,
            text="Loading...",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            justify="left"
        )
        self.data_label.pack(padx=PADDING["md"], pady=(PADDING["xs"], PADDING["sm"]), anchor="w")
        
        # Settings frame
        settings_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=10)
        settings_frame.pack(fill="x", pady=PADDING["sm"])
        
        settings_label = ctk.CTkLabel(
            settings_frame,
            text="Settings",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        )
        settings_label.pack(padx=PADDING["md"], pady=PADDING["sm"], anchor="w")
        
        # Grid size
        grid_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        grid_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["xs"])
        
        ctk.CTkLabel(
            grid_frame,
            text="Grid Size:",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        ).pack(side="left")
        
        self.grid_size_var = ctk.StringVar(value="128")
        grid_menu = ctk.CTkOptionMenu(
            grid_frame,
            values=["64", "128", "256"],
            variable=self.grid_size_var,
            width=100,
            fg_color=COLORS["bg_light"],
            button_color=COLORS["primary"],
            button_hover_color=COLORS["primary_hover"]
        )
        grid_menu.pack(side="right")
        
        # Epochs
        epochs_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        epochs_frame.pack(fill="x", padx=PADDING["md"], pady=(PADDING["xs"], PADDING["sm"]))
        
        ctk.CTkLabel(
            epochs_frame,
            text="Training Epochs:",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        ).pack(side="left")
        
        self.epochs_var = ctk.StringVar(value="50")
        epochs_entry = ctk.CTkEntry(
            epochs_frame,
            textvariable=self.epochs_var,
            width=100,
            fg_color=COLORS["bg_light"],
            border_color=COLORS["border"]
        )
        epochs_entry.pack(side="right")
        
        # Buttons frame
        buttons_frame = ctk.CTkFrame(self, fg_color="transparent")
        buttons_frame.pack(fill="x", pady=PADDING["sm"])
        
        self.extract_btn = ctk.CTkButton(
            buttons_frame,
            text="Extract Dataset",
            font=FONTS["body"],
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            height=40,
            command=self.run_extraction
        )
        self.extract_btn.pack(side="left", expand=True, fill="x", padx=(0, PADDING["xs"]))
        
        self.train_btn = ctk.CTkButton(
            buttons_frame,
            text="Train Model",
            font=FONTS["body"],
            fg_color=COLORS["success"],
            hover_color="#00A045",
            height=40,
            command=self.run_training
        )
        self.train_btn.pack(side="left", expand=True, fill="x", padx=PADDING["xs"])
        
        self.results_btn = ctk.CTkButton(
            buttons_frame,
            text="Results",
            font=FONTS["body"],
            fg_color=COLORS["bg_light"],
            hover_color=COLORS["primary"],
            height=40,
            state="disabled",
            command=self.show_results
        )
        self.results_btn.pack(side="left", expand=True, fill="x", padx=(PADDING["xs"], 0))
        
        # Full pipeline button
        self.pipeline_btn = ctk.CTkButton(
            self,
            text="Run Full Pipeline (Extract + Train)",
            font=FONTS["heading"],
            fg_color=COLORS["accent"],
            hover_color="#C62848",
            height=45,
            command=self.run_full_pipeline
        )
        self.pipeline_btn.pack(fill="x", pady=PADDING["sm"])
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(
            self,
            fg_color=COLORS["bg_card"],
            progress_color=COLORS["primary"],
            height=6
        )
        self.progress.pack(fill="x", pady=PADDING["xs"])
        self.progress.set(0)
        
        # Status label
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
            height=100,
            font=FONTS["mono"],
            fg_color=COLORS["bg_card"],
            text_color=COLORS["text_secondary"],
            corner_radius=10
        )
        self.log_area.pack(fill="both", expand=True, pady=(PADDING["xs"], 0))
        self.log("Ready. Add simulation data to data/raw/ and click Extract.")
    
    def check_existing_results(self):
        """Check if model and data already exist."""
        model_path = os.path.join("data", "processed", "cnn_model.keras")
        x_path = os.path.join("data", "processed", "CFD_X.npy")
        y_path = os.path.join("data", "processed", "CFD_y.npy")
        
        if os.path.exists(model_path) and os.path.exists(x_path) and os.path.exists(y_path):
            try:
                from tensorflow.keras.models import load_model
                model = load_model(model_path, compile=False)
                X = np.load(x_path)
                y = np.load(y_path)
                
                predictions = model.predict(X[:5], verbose=0)
                
                self.last_results = {
                    'input_shape': X.shape[1:],
                    'params': model.count_params(),
                    'epochs': 'N/A',
                    'final_loss': 'N/A',
                    'predictions': predictions.tolist(),
                    'actuals': y[:5].tolist(),
                    'all_predictions': model.predict(X, verbose=0).tolist(),
                    'all_actuals': y.tolist()
                }
                
                self.results_btn.configure(state="normal", fg_color=COLORS["primary"])
                self.log("Existing model found. Click 'Results' to view.")
                
            except Exception as e:
                self.log(f"Could not load existing model: {e}")
    
    def refresh_data_info(self):
        """Scan data directory and update info."""
        data_dir = os.path.join("data", "raw")
        cases = []
        
        if os.path.exists(data_dir):
            for folder in os.listdir(data_dir):
                folder_path = os.path.join(data_dir, folder)
                if os.path.isdir(folder_path) and not folder.startswith('.'):
                    cases.append(folder)
        
        if cases:
            info = f"{len(cases)} case(s): " + ", ".join(cases[:3])
            if len(cases) > 3:
                info += f" (+{len(cases) - 3} more)"
        else:
            info = "No simulation data found - Add folders to data/raw/"
        
        self.data_label.configure(text=info)
    
    def log(self, message):
        """Add message to log area."""
        self.log_area.insert("end", message + "\n")
        self.log_area.see("end")
    
    def set_status(self, status, color=None):
        """Update status label."""
        self.status_label.configure(
            text=f"Status: {status}",
            text_color=color or COLORS["text_muted"]
        )
    
    def set_buttons_state(self, enabled):
        """Enable/disable action buttons."""
        state = "normal" if enabled else "disabled"
        self.extract_btn.configure(state=state)
        self.train_btn.configure(state=state)
        self.pipeline_btn.configure(state=state)
    
    def show_results(self):
        """Show results in a popup window with graphs."""
        if not self.last_results:
            return
        
        popup = ctk.CTkToplevel(self)
        popup.title("Training Results")
        popup.geometry("900x700")
        popup.configure(fg_color=COLORS["bg_dark"])
        popup.grab_set()
        
        # Title
        title = ctk.CTkLabel(
            popup,
            text="Training Results",
            font=FONTS["title"],
            text_color=COLORS["text_primary"]
        )
        title.pack(pady=PADDING["sm"])
        
        # Tabs
        tabview = ctk.CTkTabview(
            popup,
            fg_color=COLORS["bg_card"],
            segmented_button_fg_color=COLORS["bg_medium"],
            segmented_button_selected_color=COLORS["primary"]
        )
        tabview.pack(fill="both", expand=True, padx=PADDING["md"], pady=PADDING["sm"])
        
        # Tab 1: Comparison Chart
        tab_chart = tabview.add("Comparison")
        self.create_comparison_chart(tab_chart)
        
        # Tab 2: Error Analysis
        tab_error = tabview.add("Error Analysis")
        self.create_error_analysis(tab_error)
        
        # Tab 3: Training History (if available)
        if self.loss_history:
            tab_history = tabview.add("Training History")
            self.create_training_history(tab_history)
        
        # Tab 4: Data Table
        tab_data = tabview.add("Data Table")
        self.create_data_table(tab_data)
        
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
    
    def create_comparison_chart(self, parent):
        """Create bar chart comparing predictions vs actual."""
        predictions = np.array(self.last_results.get('all_predictions', self.last_results['predictions']))
        actuals = np.array(self.last_results.get('all_actuals', self.last_results['actuals']))
        
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=COLORS["bg_card"])
        
        # CL comparison
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(COLORS["bg_light"])
        
        x = np.arange(min(len(predictions), 10))
        width = 0.35
        
        pred_cl = predictions[:10, 0]
        actual_cl = actuals[:10, 0]
        
        bars1 = ax1.bar(x - width/2, actual_cl, width, label='Actual', color=COLORS["primary"])
        bars2 = ax1.bar(x + width/2, pred_cl, width, label='Predicted', color=COLORS["accent"])
        
        ax1.set_xlabel('Sample', color='white')
        ax1.set_ylabel('CL', color='white')
        ax1.set_title('Lift Coefficient (CL)', color='white', fontweight='bold')
        ax1.legend(facecolor=COLORS["bg_card"], labelcolor='white')
        ax1.tick_params(colors='white')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(i+1) for i in x])
        
        # CD comparison
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor(COLORS["bg_light"])
        
        pred_cd = predictions[:10, 1]
        actual_cd = actuals[:10, 1]
        
        bars3 = ax2.bar(x - width/2, actual_cd, width, label='Actual', color=COLORS["primary"])
        bars4 = ax2.bar(x + width/2, pred_cd, width, label='Predicted', color=COLORS["accent"])
        
        ax2.set_xlabel('Sample', color='white')
        ax2.set_ylabel('CD', color='white')
        ax2.set_title('Drag Coefficient (CD)', color='white', fontweight='bold')
        ax2.legend(facecolor=COLORS["bg_card"], labelcolor='white')
        ax2.tick_params(colors='white')
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(i+1) for i in x])
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def create_error_analysis(self, parent):
        """Create error analysis visualization."""
        predictions = np.array(self.last_results.get('all_predictions', self.last_results['predictions']))
        actuals = np.array(self.last_results.get('all_actuals', self.last_results['actuals']))
        
        # Calculate errors
        cl_error = np.abs(predictions[:, 0] - actuals[:, 0])
        cd_error = np.abs(predictions[:, 1] - actuals[:, 1])
        
        cl_rel_error = np.mean(cl_error / (np.abs(actuals[:, 0]) + 1e-10)) * 100
        cd_rel_error = np.mean(cd_error / (np.abs(actuals[:, 1]) + 1e-10)) * 100
        
        cl_mae = np.mean(cl_error)
        cd_mae = np.mean(cd_error)
        
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=COLORS["bg_card"])
        
        # Error bars
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(COLORS["bg_light"])
        
        metrics = ['MAE\nCL', 'MAE\nCD']
        values = [cl_mae, cd_mae]
        colors = [COLORS["primary"], COLORS["accent"]]
        
        bars = ax1.bar(metrics, values, color=colors)
        ax1.set_ylabel('Mean Absolute Error', color='white')
        ax1.set_title('Prediction Errors', color='white', fontweight='bold')
        ax1.tick_params(colors='white')
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', color='white', fontsize=10)
        
        # Relative error pie
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor(COLORS["bg_light"])
        
        rel_errors = [cl_rel_error, cd_rel_error]
        labels = [f'CL Error\n{cl_rel_error:.1f}%', f'CD Error\n{cd_rel_error:.1f}%']
        
        wedges, texts = ax2.pie(rel_errors, labels=labels, 
                                colors=[COLORS["primary"], COLORS["accent"]],
                                textprops={'color': 'white'})
        ax2.set_title('Relative Error Distribution', color='white', fontweight='bold')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Summary text
        summary = ctk.CTkLabel(
            parent,
            text=f"Mean CL Error: {cl_mae:.6f} ({cl_rel_error:.2f}%)  |  Mean CD Error: {cd_mae:.6f} ({cd_rel_error:.2f}%)",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        )
        summary.pack(pady=PADDING["sm"])
    
    def create_training_history(self, parent):
        """Create training loss history plot."""
        if not self.loss_history:
            return
        
        fig = Figure(figsize=(8, 4), dpi=100, facecolor=COLORS["bg_card"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(COLORS["bg_light"])
        
        epochs = range(1, len(self.loss_history) + 1)
        ax.plot(epochs, self.loss_history, color=COLORS["primary"], linewidth=2, label='Training Loss')
        
        ax.set_xlabel('Epoch', color='white')
        ax.set_ylabel('Loss (MSE)', color='white')
        ax.set_title('Training Loss Over Time', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.legend(facecolor=COLORS["bg_card"], labelcolor='white')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def create_data_table(self, parent):
        """Create data table with all predictions."""
        predictions = self.last_results.get('all_predictions', self.last_results['predictions'])
        actuals = self.last_results.get('all_actuals', self.last_results['actuals'])
        
        # Info
        info_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_light"], corner_radius=10)
        info_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["sm"])
        
        info_text = f"Model: {self.last_results.get('params', 'N/A'):,} parameters  |  "
        info_text += f"Input: {self.last_results.get('input_shape', 'N/A')}  |  "
        info_text += f"Samples: {len(predictions)}"
        
        info_label = ctk.CTkLabel(
            info_frame,
            text=info_text,
            font=FONTS["body"],
            text_color=COLORS["text_primary"]
        )
        info_label.pack(pady=PADDING["sm"])
        
        # Table
        table_text = ctk.CTkTextbox(
            parent,
            font=FONTS["mono"],
            fg_color=COLORS["bg_light"],
            text_color=COLORS["text_primary"]
        )
        table_text.pack(fill="both", expand=True, padx=PADDING["md"], pady=PADDING["sm"])
        
        header = f"{'#':<5}{'Pred CL':<14}{'Actual CL':<14}{'Pred CD':<14}{'Actual CD':<14}{'CL Err%':<10}{'CD Err%':<10}\n"
        table_text.insert("1.0", header)
        table_text.insert("end", "-" * 80 + "\n")
        
        for i in range(len(predictions)):
            pred = predictions[i]
            actual = actuals[i]
            cl_err = abs(pred[0] - actual[0]) / (abs(actual[0]) + 1e-10) * 100
            cd_err = abs(pred[1] - actual[1]) / (abs(actual[1]) + 1e-10) * 100
            
            row = f"{i+1:<5}{pred[0]:<14.6f}{actual[0]:<14.6f}{pred[1]:<14.6f}{actual[1]:<14.6f}{cl_err:<10.2f}{cd_err:<10.2f}\n"
            table_text.insert("end", row)
        
        table_text.configure(state="disabled")
    
    def run_extraction(self):
        """Run data extraction in background thread."""
        if self.is_running:
            return
        
        def task():
            self.is_running = True
            self.set_buttons_state(False)
            self.set_status("Extracting...", COLORS["warning"])
            self.progress.set(0)
            self.log("\n" + "="*50)
            self.log("Starting extraction...")
            
            try:
                from src.extract_data import extract_dataset
                grid_size = int(self.grid_size_var.get())
                
                self.progress.set(0.3)
                X, y = extract_dataset("data/raw", "data/processed", grid_size=grid_size)
                
                self.progress.set(1.0)
                if X is not None:
                    self.log(f"Extraction complete: {X.shape[0]} samples")
                    self.set_status("Extraction complete!", COLORS["success"])
                else:
                    self.log("Warning: No data extracted")
                    self.set_status("No data found", COLORS["warning"])
                    
            except Exception as e:
                self.log(f"Error: {str(e)}")
                self.set_status("Error!", COLORS["accent"])
            
            finally:
                self.is_running = False
                self.set_buttons_state(True)
                self.refresh_data_info()
        
        threading.Thread(target=task, daemon=True).start()
    
    def run_training(self):
        """Run model training in background thread."""
        if self.is_running:
            return
        
        def task():
            self.is_running = True
            self.set_buttons_state(False)
            self.set_status("Training...", COLORS["warning"])
            self.progress.set(0)
            self.log("\n" + "="*50)
            self.log("Starting training...")
            
            try:
                from src.train_cnn import train_cnn_model
                epochs = int(self.epochs_var.get())
                
                self.progress.set(0.1)
                model, history = train_cnn_model("data/processed", epochs=epochs)
                
                self.progress.set(1.0)
                self.log("Training complete!")
                self.log("Model saved to data/processed/cnn_model.keras")
                self.set_status("Training complete!", COLORS["success"])
                
                # Store results
                X = np.load("data/processed/CFD_X.npy")
                y = np.load("data/processed/CFD_y.npy")
                all_predictions = model.predict(X, verbose=0)
                
                self.last_results = {
                    'input_shape': X.shape[1:],
                    'params': model.count_params(),
                    'epochs': len(history.history['loss']),
                    'final_loss': history.history['loss'][-1],
                    'predictions': all_predictions[:5].tolist(),
                    'actuals': y[:5].tolist(),
                    'all_predictions': all_predictions.tolist(),
                    'all_actuals': y.tolist()
                }
                
                self.loss_history = history.history['loss']
                self.results_btn.configure(state="normal", fg_color=COLORS["primary"])
                
            except Exception as e:
                self.log(f"Error: {str(e)}")
                self.set_status("Error!", COLORS["accent"])
            
            finally:
                self.is_running = False
                self.set_buttons_state(True)
        
        threading.Thread(target=task, daemon=True).start()
    
    def run_full_pipeline(self):
        """Run full extraction + training pipeline."""
        if self.is_running:
            return
        
        def task():
            self.is_running = True
            self.set_buttons_state(False)
            self.set_status("Running pipeline...", COLORS["warning"])
            self.progress.set(0)
            self.log("\n" + "="*50)
            self.log("Starting full pipeline...")
            
            try:
                from src.extract_data import extract_dataset
                grid_size = int(self.grid_size_var.get())
                
                self.log("\nStep 1: Extraction")
                self.progress.set(0.2)
                X, y = extract_dataset("data/raw", "data/processed", grid_size=grid_size)
                
                if X is None:
                    self.log("Warning: No data to train on")
                    self.set_status("No data", COLORS["warning"])
                    return
                
                self.log(f"  {X.shape[0]} samples extracted")
                
                from src.train_cnn import train_cnn_model
                epochs = int(self.epochs_var.get())
                
                self.log("\nStep 2: Training")
                self.progress.set(0.5)
                model, history = train_cnn_model("data/processed", epochs=epochs)
                
                self.progress.set(1.0)
                self.log("\nPipeline complete!")
                self.set_status("Pipeline complete!", COLORS["success"])
                
                all_predictions = model.predict(X, verbose=0)
                
                self.last_results = {
                    'input_shape': X.shape[1:],
                    'params': model.count_params(),
                    'epochs': len(history.history['loss']),
                    'final_loss': history.history['loss'][-1],
                    'predictions': all_predictions[:5].tolist(),
                    'actuals': y[:5].tolist(),
                    'all_predictions': all_predictions.tolist(),
                    'all_actuals': y.tolist()
                }
                
                self.loss_history = history.history['loss']
                self.results_btn.configure(state="normal", fg_color=COLORS["primary"])
                
            except Exception as e:
                self.log(f"Error: {str(e)}")
                self.set_status("Error!", COLORS["accent"])
            
            finally:
                self.is_running = False
                self.set_buttons_state(True)
                self.refresh_data_info()
        
        threading.Thread(target=task, daemon=True).start()
