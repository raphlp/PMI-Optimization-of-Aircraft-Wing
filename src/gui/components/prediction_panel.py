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
        """Train CNN on CFD data with production-grade optimization."""
        if self.is_running:
            return
        
        def task():
            self.is_running = True
            self.train_btn.configure(state="disabled")
            self.progress.set(0)
            
            self.log("Initializing optimized CNN training...")
            
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Model
                from tensorflow.keras.layers import (
                    Input, Conv2D, BatchNormalization, Activation, 
                    MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout,
                    Add, Concatenate
                )
                from tensorflow.keras.optimizers import AdamW
                from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
                from tensorflow.keras.regularizers import l2
                
                # Load data
                X = np.load("data/processed/combined/X.npy")
                y = np.load("data/processed/combined/y.npy")
                
                valid_mask = ~np.isnan(y).any(axis=1)
                X_data = X[valid_mask].astype(np.float32)
                y_data = y[valid_mask].astype(np.float32)
                
                n_samples = len(X_data)
                self.log(f"Loaded {n_samples} samples")
                self.progress.set(0.05)
                
                # ========== DATA AUGMENTATION ==========
                def augment_data(X, y, augment_factor=2):
                    """Apply data augmentation: horizontal flip + noise."""
                    X_aug = [X]
                    y_aug = [y]
                    
                    # Horizontal flip (valid for symmetric flow)
                    X_flipped = np.flip(X, axis=2)  # Flip along width
                    # Flip y-velocity sign
                    X_flipped_corrected = X_flipped.copy()
                    X_flipped_corrected[:, :, :, 2] *= -1  # v velocity
                    X_aug.append(X_flipped_corrected)
                    y_aug.append(y)  # CL/CD same for symmetric flip
                    
                    # Add Gaussian noise
                    for _ in range(augment_factor - 2):
                        noise = np.random.normal(0, 0.02, X.shape).astype(np.float32)
                        X_noisy = np.clip(X + noise, 0, 1)
                        X_aug.append(X_noisy)
                        # Slight noise on targets too
                        y_noise = np.random.normal(0, 0.005, y.shape).astype(np.float32)
                        y_aug.append(y + y_noise)
                    
                    return np.concatenate(X_aug), np.concatenate(y_aug)
                
                if n_samples < 100:
                    X_data, y_data = augment_data(X_data, y_data, augment_factor=3)
                    self.append_log(f"Augmented to {len(X_data)} samples")
                
                # ========== OUTPUT NORMALIZATION ==========
                y_mean = y_data.mean(axis=0)
                y_std = y_data.std(axis=0) + 1e-8
                y_normalized = (y_data - y_mean) / y_std
                
                # Save normalization params
                models_dir = os.path.join("data", "processed", "models")
                os.makedirs(models_dir, exist_ok=True)
                np.save(os.path.join(models_dir, "y_mean.npy"), y_mean)
                np.save(os.path.join(models_dir, "y_std.npy"), y_std)
                
                # ========== TRAIN/VAL SPLIT ==========
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_data, y_normalized, test_size=0.2, random_state=42
                )
                
                self.append_log(f"Train: {len(X_train)}, Val: {len(X_val)}")
                self.progress.set(0.1)
                # ========== SQUEEZE-EXCITATION ATTENTION (simpler than CBAM) ==========
                def se_block(x, ratio=8):
                    """Squeeze-Excitation block: channel attention without Lambda layers."""
                    from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Multiply
                    
                    channels = x.shape[-1]
                    
                    # Squeeze: global average pooling
                    se = GlobalAveragePooling2D()(x)
                    
                    # Excitation: two FC layers
                    se = Dense(channels // ratio, activation='relu')(se)
                    se = Dense(channels, activation='sigmoid')(se)
                    
                    # Reshape and scale
                    se = Reshape((1, 1, channels))(se)
                    
                    return Multiply()([x, se])
                
                # ========== ATTENTION-ENHANCED RESIDUAL BLOCK ==========
                def attention_residual_block(x, filters, kernel_size=3, stride=1, use_attention=True):
                    """Residual block with SE attention."""
                    shortcut = x
                    
                    x = Conv2D(filters, kernel_size, strides=stride, padding='same',
                               kernel_regularizer=l2(1e-4))(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)
                    
                    x = Conv2D(filters, kernel_size, padding='same',
                               kernel_regularizer=l2(1e-4))(x)
                    x = BatchNormalization()(x)
                    
                    # Apply SE attention
                    if use_attention:
                        x = se_block(x)
                    
                    # Match dimensions if needed
                    if stride != 1 or shortcut.shape[-1] != filters:
                        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
                        shortcut = BatchNormalization()(shortcut)
                    
                    x = Add()([x, shortcut])
                    x = Activation('relu')(x)
                    return x
                
                # ========== BUILD BAYESIAN CNN WITH SE-ATTENTION ==========
                print("\n" + "="*60, flush=True)
                print("Building Bayesian CNN with SE-Attention", flush=True)
                print("="*60, flush=True)
                
                inputs = Input(shape=X_train.shape[1:])
                
                # Initial convolution
                x = Conv2D(32, 7, strides=2, padding='same', kernel_regularizer=l2(1e-4))(inputs)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = MaxPooling2D(3, strides=2, padding='same')(x)
                
                # Attention-enhanced residual blocks
                x = attention_residual_block(x, 64, use_attention=True)
                x = attention_residual_block(x, 64, use_attention=False)
                x = attention_residual_block(x, 128, stride=2, use_attention=True)
                x = attention_residual_block(x, 128, use_attention=False)
                x = attention_residual_block(x, 256, stride=2, use_attention=True)
                x = attention_residual_block(x, 256, use_attention=True)
                
                # Global pooling
                x = GlobalAveragePooling2D()(x)
                
                # Dense layers with Dropout (for MC inference, we'll run with training=True)
                x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
                x = Dropout(0.3)(x)
                x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
                x = Dropout(0.3)(x)
                x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
                x = Dropout(0.2)(x)
                
                # Output: 2 values (CL, CD)
                outputs = Dense(2, activation='linear', name='predictions')(x)
                
                model = Model(inputs, outputs, name='BayesianCNN_CBAM')
                
                # ========== OPTIMIZER WITH WEIGHT DECAY ==========
                epochs = int(self.epochs_var.get())
                initial_lr = 0.001
                
                optimizer = AdamW(
                    learning_rate=initial_lr,
                    weight_decay=1e-4
                )
                
                model.compile(
                    optimizer=optimizer,
                    loss='huber',  # More robust than MSE
                    metrics=['mae']
                )
                
                self.append_log(f"\nModel: {model.count_params():,} parameters")
                self.append_log(f"Training for {epochs} epochs...")
                
                # ========== CALLBACKS ==========
                panel = self
                
                class GUIProgressCallback(tf.keras.callbacks.Callback):
                    def __init__(cb_self):
                        super().__init__()
                        cb_self.best_val_loss = float('inf')
                    
                    def on_epoch_end(cb_self, epoch, logs=None):
                        loss = logs.get('loss', 0)
                        val_loss = logs.get('val_loss', 0)
                        mae = logs.get('mae', 0)
                        val_mae = logs.get('val_mae', 0)
                        
                        try:
                            lr = float(tf.keras.backend.get_value(cb_self.model.optimizer.learning_rate))
                        except:
                            lr = 0.001
                        
                        progress = 0.1 + (epoch + 1) / epochs * 0.8
                        
                        # Track best
                        if val_loss < cb_self.best_val_loss:
                            cb_self.best_val_loss = val_loss
                            best_marker = " ★ BEST"
                        else:
                            best_marker = ""
                        
                        # Terminal log
                        print(f"Epoch {epoch+1:3d}/{epochs} - loss: {loss:.5f} - val_loss: {val_loss:.5f} - mae: {mae:.5f}{best_marker}", flush=True)
                        
                        msg = f"Epoch {epoch+1}/{epochs}{best_marker}\n"
                        msg += f"Loss: {loss:.5f} | Val: {val_loss:.5f}\n"
                        msg += f"MAE:  {mae:.5f} | Val: {val_mae:.5f}\n"
                        msg += f"LR: {lr:.2e}"
                        
                        # Schedule GUI update
                        try:
                            panel.after(0, lambda: panel.progress.set(progress))
                            panel.after(0, lambda: panel.log(f"Training...\n{msg}"))
                        except:
                            pass
                
                callbacks = [
                    GUIProgressCallback(),
                    EarlyStopping(
                        monitor='val_loss',
                        patience=50,
                        restore_best_weights=True,
                        verbose=0
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=20,
                        min_lr=1e-6,
                        verbose=0
                    )
                ]
                
                # ========== TRAINING ==========
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=min(32, len(X_train) // 4 + 1),
                    callbacks=callbacks,
                    verbose=0
                )
                
                self.progress.set(0.9)
                
                # ========== SAVE MODEL ==========
                model.save(os.path.join(models_dir, "cnn_model.keras"))
                
                # ========== EVALUATE ON ORIGINAL SCALE ==========
                # Predict on all data
                X_all = X[valid_mask].astype(np.float32)
                y_all = y[valid_mask].astype(np.float32)
                
                pred_normalized = model.predict(X_all, verbose=0)
                predictions = pred_normalized * y_std + y_mean  # Denormalize
                
                mae_cl = float(np.mean(np.abs(predictions[:, 0] - y_all[:, 0])))
                mae_cd = float(np.mean(np.abs(predictions[:, 1] - y_all[:, 1])))
                
                # Relative errors
                rel_cl = mae_cl / (np.abs(y_all[:, 0]).mean() + 1e-8) * 100
                rel_cd = mae_cd / (np.abs(y_all[:, 1]).mean() + 1e-8) * 100
                
                # R² scores
                ss_res_cl = np.sum((predictions[:, 0] - y_all[:, 0])**2)
                ss_tot_cl = np.sum((y_all[:, 0] - y_all[:, 0].mean())**2)
                r2_cl = 1 - ss_res_cl / (ss_tot_cl + 1e-8)
                
                ss_res_cd = np.sum((predictions[:, 1] - y_all[:, 1])**2)
                ss_tot_cd = np.sum((y_all[:, 1] - y_all[:, 1].mean())**2)
                r2_cd = 1 - ss_res_cd / (ss_tot_cd + 1e-8)
                
                # Save info
                with open(os.path.join(models_dir, "cnn_info.json"), 'w') as f:
                    json.dump({
                        'epochs': len(history.history['loss']),
                        'epochs_requested': epochs,
                        'num_samples': n_samples,
                        'num_augmented': len(X_data),
                        'mae_cl': mae_cl,
                        'mae_cd': mae_cd,
                        'rel_error_cl': rel_cl,
                        'rel_error_cd': rel_cd,
                        'r2_cl': float(r2_cl),
                        'r2_cd': float(r2_cd),
                        'final_loss': float(history.history['loss'][-1]),
                        'best_val_loss': float(min(history.history['val_loss'])),
                        'architecture': 'ResNet-style with BatchNorm'
                    }, f, indent=2)
                
                self.progress.set(1.0)
                
                result_msg = f"✅ Training Complete!\n\n"
                result_msg += f"🔧 Architecture: Bayesian CNN + CBAM\n"
                result_msg += f"📊 Epochs: {len(history.history['loss'])}/{epochs}\n"
                result_msg += f"📁 Samples: {n_samples} (aug: {len(X_data)})\n\n"
                result_msg += f"═══ CL Performance ═══\n"
                result_msg += f"MAE: {mae_cl:.6f}\n"
                result_msg += f"Relative Error: {rel_cl:.2f}%\n"
                result_msg += f"R²: {r2_cl:.4f}\n\n"
                result_msg += f"═══ CD Performance ═══\n"
                result_msg += f"MAE: {mae_cd:.6f}\n"
                result_msg += f"Relative Error: {rel_cd:.2f}%\n"
                result_msg += f"R²: {r2_cd:.4f}\n\n"
                result_msg += f"🎯 Bayesian inference ready!\n"
                result_msg += f"Use View Results for uncertainty estimates."
                
                self.log(result_msg)
                
            except Exception as e:
                error_msg = f"\nError: {str(e)}"
                self.append_log(error_msg)
                import traceback
                tb = traceback.format_exc()
                self.append_log(tb)
                # Also print to terminal
                print(f"\n{'='*60}", flush=True)
                print(f"TRAINING ERROR: {e}", flush=True)
                print(tb, flush=True)
                print(f"{'='*60}", flush=True)
            
            finally:
                self.is_running = False
                self.refresh_status()
        
        threading.Thread(target=task, daemon=True).start()
    
    def show_results(self):
        """Show enhanced results popup with Bayesian uncertainty."""
        try:
            from tensorflow.keras.models import load_model
            
            self.log("Loading model and computing uncertainties...")
            
            model = load_model("data/processed/models/cnn_model.keras", compile=False)
            X = np.load("data/processed/combined/X.npy").astype(np.float32)
            y = np.load("data/processed/combined/y.npy")
            
            with open("data/processed/combined/manifest.json") as f:
                manifest = json.load(f)
            
            # Load normalization params
            y_mean_path = "data/processed/models/y_mean.npy"
            y_std_path = "data/processed/models/y_std.npy"
            
            if os.path.exists(y_mean_path) and os.path.exists(y_std_path):
                y_mean = np.load(y_mean_path)
                y_std = np.load(y_std_path)
            else:
                y_mean = np.array([0, 0])
                y_std = np.array([1, 1])
            
            # ========== BAYESIAN MC DROPOUT INFERENCE ==========
            import tensorflow as tf
            
            n_mc_samples = 30  # Number of Monte Carlo samples
            mc_predictions = []
            
            self.log(f"Running {n_mc_samples} MC Dropout samples...")
            
            for i in range(n_mc_samples):
                # training=True keeps dropout active for uncertainty
                pred = model(X, training=True)
                mc_predictions.append(pred.numpy())
            
            mc_predictions = np.array(mc_predictions)  # (n_mc, n_samples, 2)
            
            # Denormalize
            mc_predictions = mc_predictions * y_std + y_mean
            
            # Calculate mean and std
            predictions = np.mean(mc_predictions, axis=0)  # Mean prediction
            uncertainties = np.std(mc_predictions, axis=0)  # Uncertainty
            
            self.log("Uncertainty quantification complete!")
            
        except Exception as e:
            self.append_log(f"Error: {e}")
            import traceback
            self.append_log(traceback.format_exc())
            return
        
        ResultsPopup(self, X, y, predictions, uncertainties, manifest)


class ResultsPopup(ctk.CTkToplevel):
    """Enhanced results popup with Bayesian uncertainty display."""
    
    def __init__(self, parent, X, y, predictions, uncertainties, manifest):
        super().__init__(parent)
        
        self.X = X
        self.y = y
        self.predictions = predictions
        self.uncertainties = uncertainties
        self.manifest = manifest
        self.samples = manifest.get('samples', [])
        
        self.title("Bayesian Prediction Results")
        self.geometry("1100x850")
        self.configure(fg_color=COLORS["bg_dark"])
        self.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Separate samples by source
        self.cfd_samples = [(i, s) for i, s in enumerate(self.samples) if s.get('source') == 'CFD']
        self.pinn_samples = [(i, s) for i, s in enumerate(self.samples) if s.get('source') == 'PINN']
        
        # Header
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_medium"], corner_radius=0)
        header.pack(fill="x")
        
        ctk.CTkLabel(
            header,
            text="🎯 Bayesian Prediction Results",
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
        
        # ========== TWO-LEVEL SELECTOR ==========
        selector_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=10)
        selector_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["sm"])
        
        # Source selector (CFD or PINN)
        ctk.CTkLabel(
            selector_frame,
            text="Source:",
            font=FONTS["heading"],
            text_color=COLORS["text_primary"]
        ).pack(side="left", padx=PADDING["md"], pady=PADDING["sm"])
        
        self.source_var = ctk.StringVar(value="CFD")
        self.source_menu = ctk.CTkOptionMenu(
            selector_frame,
            values=["CFD", "PINN"],
            variable=self.source_var,
            width=100,
            fg_color=COLORS["primary"],
            button_color=COLORS["primary"],
            command=self.on_source_change
        )
        self.source_menu.pack(side="left", padx=PADDING["xs"], pady=PADDING["sm"])
        
        # Model selector
        ctk.CTkLabel(
            selector_frame,
            text="Model:",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        ).pack(side="left", padx=(PADDING["md"], PADDING["xs"]), pady=PADDING["sm"])
        
        self.model_var = ctk.StringVar()
        self.model_menu = ctk.CTkOptionMenu(
            selector_frame,
            values=["Loading..."],
            variable=self.model_var,
            width=350,
            fg_color=COLORS["bg_light"],
            button_color=COLORS["primary"],
            command=self.on_model_change
        )
        self.model_menu.pack(side="left", padx=PADDING["xs"], pady=PADDING["sm"])
        
        # Quick nav buttons
        ctk.CTkButton(
            selector_frame,
            text="◀",
            width=40,
            fg_color=COLORS["bg_light"],
            command=self.prev_sample
        ).pack(side="left", padx=PADDING["xs"])
        
        ctk.CTkButton(
            selector_frame,
            text="▶",
            width=40,
            fg_color=COLORS["bg_light"],
            command=self.next_sample
        ).pack(side="left", padx=PADDING["xs"])
        
        # Sample count label
        self.count_label = ctk.CTkLabel(
            selector_frame,
            text="",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        )
        self.count_label.pack(side="right", padx=PADDING["md"], pady=PADDING["sm"])
        
        # Main content
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=PADDING["md"], pady=PADDING["sm"])
        
        # Initialize with CFD samples
        self.update_model_options()
        self.update_content()
    
    def on_source_change(self, value):
        """Handle source dropdown change."""
        self.update_model_options()
        self.update_content()
    
    def on_model_change(self, value):
        """Handle model dropdown change."""
        self.update_content()
    
    def update_model_options(self):
        """Update model dropdown based on selected source."""
        source = self.source_var.get()
        
        if source == "CFD":
            samples_list = self.cfd_samples
            source_color = COLORS["primary"]
        else:
            samples_list = self.pinn_samples
            source_color = "#9C27B0"  # Purple for PINN
        
        # Build options
        options = []
        for idx, sample in samples_list:
            profile = sample.get('profile', 'Unknown')
            angle = sample.get('angle', '?')
            label = f"{idx+1}. {profile} @ {angle}°"
            options.append(label)
        
        if not options:
            options = [f"No {source} samples available"]
        
        # Update dropdown
        self.model_menu.configure(values=options)
        self.model_var.set(options[0])
        self.count_label.configure(text=f"{len(samples_list)} {source} samples")
    
    def get_current_index(self):
        """Get current sample index from the model selection."""
        try:
            current = self.model_var.get()
            return int(current.split('.')[0]) - 1
        except:
            return 0
    
    def get_current_samples_list(self):
        """Get the current filtered samples list."""
        source = self.source_var.get()
        return self.cfd_samples if source == "CFD" else self.pinn_samples
    
    def prev_sample(self):
        samples_list = self.get_current_samples_list()
        if not samples_list:
            return
        options = self.model_menu.cget("values")
        current_idx = 0
        for i, opt in enumerate(options):
            if opt == self.model_var.get():
                current_idx = i
                break
        new_idx = max(0, current_idx - 1)
        self.model_var.set(options[new_idx])
        self.update_content()
    
    def next_sample(self):
        samples_list = self.get_current_samples_list()
        if not samples_list:
            return
        options = self.model_menu.cget("values")
        current_idx = 0
        for i, opt in enumerate(options):
            if opt == self.model_var.get():
                current_idx = i
                break
        new_idx = min(len(options) - 1, current_idx + 1)
        self.model_var.set(options[new_idx])
        self.update_content()
    
    def update_content(self):
        """Update content for selected sample with Bayesian uncertainty."""
        # Clear existing
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        idx = self.get_current_index()
        
        # Handle invalid index or empty samples
        if idx < 0 or idx >= len(self.samples):
            ctk.CTkLabel(
                self.content_frame,
                text="No samples available for this source.",
                font=FONTS["heading"],
                text_color=COLORS["text_muted"]
            ).pack(pady=50)
            return
        
        sample = self.samples[idx]
        pred = self.predictions[idx]
        uncert = self.uncertainties[idx]
        actual = self.y[idx]
        fields = self.X[idx]
        
        source = self.source_var.get()
        is_cfd = source == 'CFD' and not np.isnan(actual).any()
        
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
            text="Bayesian Predictions (with uncertainty)",
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
        
        # Show prediction with uncertainty
        cl_95_low = pred[0] - 1.96 * uncert[0]
        cl_95_high = pred[0] + 1.96 * uncert[0]
        
        ctk.CTkLabel(
            cl_value_frame,
            text=f"{pred[0]:.4f} ± {uncert[0]:.4f}",
            font=("Segoe UI", 18, "bold"),
            text_color=COLORS["accent"]
        ).pack(side="left")
        
        ctk.CTkLabel(
            cl_value_frame,
            text=f"95% CI: [{cl_95_low:.4f}, {cl_95_high:.4f}]",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        ).pack(side="left", padx=PADDING["md"])
        
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
        
        # Show prediction with uncertainty
        cd_95_low = pred[1] - 1.96 * uncert[1]
        cd_95_high = pred[1] + 1.96 * uncert[1]
        
        ctk.CTkLabel(
            cd_value_frame,
            text=f"{pred[1]:.6f} ± {uncert[1]:.6f}",
            font=("Segoe UI", 18, "bold"),
            text_color=COLORS["accent"]
        ).pack(side="left")
        
        ctk.CTkLabel(
            cd_value_frame,
            text=f"95% CI: [{cd_95_low:.6f}, {cd_95_high:.6f}]",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        ).pack(side="left", padx=PADDING["md"])
        
        if is_cfd:
            ctk.CTkLabel(
                cd_value_frame,
                text=f"Actual: {actual[1]:.6f}",
                font=FONTS["body"],
                text_color=COLORS["text_secondary"]
            ).pack(side="right")
        
        # Aerodynamic Efficiency (CL/CD)
        eff_frame = ctk.CTkFrame(coef_frame, fg_color=COLORS["bg_light"], corner_radius=8)
        eff_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["xs"])
        
        ctk.CTkLabel(
            eff_frame,
            text="Aerodynamic Efficiency (CL/CD) - for Bayesian Optimization",
            font=FONTS["small"],
            text_color=COLORS["text_muted"]
        ).pack(anchor="w", padx=PADDING["sm"], pady=(PADDING["xs"], 0))
        
        eff_value_frame = ctk.CTkFrame(eff_frame, fg_color="transparent")
        eff_value_frame.pack(fill="x", padx=PADDING["sm"], pady=PADDING["xs"])
        
        # Calculate efficiency with uncertainty propagation
        efficiency = pred[0] / (pred[1] + 1e-8)
        # Propagated uncertainty: σ(CL/CD) ≈ |CL/CD| * sqrt((σ_CL/CL)² + (σ_CD/CD)²)
        eff_uncertainty = abs(efficiency) * np.sqrt(
            (uncert[0] / (abs(pred[0]) + 1e-8))**2 + 
            (uncert[1] / (abs(pred[1]) + 1e-8))**2
        )
        
        ctk.CTkLabel(
            eff_value_frame,
            text=f"{efficiency:.2f} ± {eff_uncertainty:.2f}",
            font=("Segoe UI", 18, "bold"),
            text_color="#00E676"  # Green for efficiency
        ).pack(side="left")
        
        if is_cfd:
            actual_eff = actual[0] / (actual[1] + 1e-8)
            ctk.CTkLabel(
                eff_value_frame,
                text=f"Actual: {actual_eff:.2f}",
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
            
            # Check if actual is within confidence interval
            cl_in_ci = cl_95_low <= actual[0] <= cl_95_high
            cd_in_ci = cd_95_low <= actual[1] <= cd_95_high
            
            ctk.CTkLabel(
                error_frame,
                text="Prediction Error & Calibration",
                font=FONTS["small"],
                text_color=COLORS["text_muted"]
            ).pack(anchor="w", padx=PADDING["sm"], pady=(PADDING["xs"], 0))
            
            error_text = f"CL Error: {cl_error:.5f} ({cl_rel:.1f}%)  |  CD Error: {cd_error:.5f} ({cd_rel:.1f}%)"
            ctk.CTkLabel(
                error_frame,
                text=error_text,
                font=FONTS["body"],
                text_color=COLORS["warning"] if (cl_rel > 10 or cd_rel > 10) else COLORS["success"]
            ).pack(anchor="w", padx=PADDING["sm"], pady=(0, PADDING["xs"]))
            
            # Calibration check
            calib_cl = "✓" if cl_in_ci else "✗"
            calib_cd = "✓" if cd_in_ci else "✗"
            ctk.CTkLabel(
                error_frame,
                text=f"95% CI contains actual? CL: {calib_cl}  CD: {calib_cd}",
                font=FONTS["small"],
                text_color=COLORS["success"] if (cl_in_ci and cd_in_ci) else COLORS["warning"]
            ).pack(anchor="w", padx=PADDING["sm"], pady=(0, PADDING["xs"]))
        else:
            # PINN notice
            pinn_frame = ctk.CTkFrame(coef_frame, fg_color="#9C27B0", corner_radius=8)
            pinn_frame.pack(fill="x", padx=PADDING["md"], pady=PADDING["xs"])
            
            ctk.CTkLabel(
                pinn_frame,
                text="PINN Prediction",
                font=FONTS["heading"],
                text_color="white"
            ).pack(anchor="w", padx=PADDING["sm"], pady=(PADDING["xs"], 0))
            
            ctk.CTkLabel(
                pinn_frame,
                text="These coefficients are predicted for a simulated airfoil profile.\nNo ground truth CFD data available for comparison.",
                font=FONTS["small"],
                text_color="#E1BEE7"
            ).pack(anchor="w", padx=PADDING["sm"], pady=(0, PADDING["sm"]))
        
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
