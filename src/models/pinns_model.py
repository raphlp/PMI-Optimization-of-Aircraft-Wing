"""
Physics-Informed Neural Network for 2D Incompressible Flow

Solves the Navier-Stokes equations around an airfoil using neural networks.
Supports pure physics mode or hybrid mode with CFD data.

All quantities are NON-DIMENSIONALIZED:
- Length scale: chord c = 1
- Velocity scale: freestream U_inf = 1
- Pressure scale: rho * U_inf^2
- Time scale: c / U_inf
- Reynolds number: Re = U_inf * c / nu
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


class PINNModel(Model):
    """
    Physics-Informed Neural Network for 2D incompressible flow.
    Works in NON-DIMENSIONAL variables.
    
    Inputs: (x, y) coordinates (normalized by chord)
    Outputs: (u, v, p) dimensionless velocity and pressure
    """
    
    def __init__(self, layers_config=[64, 64, 64, 64], activation='tanh', reynolds=1e6):
        super().__init__()
        
        self.hidden_layers = []
        for units in layers_config:
            self.hidden_layers.append(layers.Dense(units, activation=activation))
        
        # Output layer: u, v, p (dimensionless)
        self.output_layer = layers.Dense(3, activation=None)
        
        # Reynolds number (dimensionless)
        self.Re = reynolds
    
    def call(self, inputs):
        """Forward pass through the network."""
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
    
    @tf.function
    def compute_gradients(self, x, y):
        """Compute flow variables and their gradients."""
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y])
            
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y])
                
                inputs = tf.stack([x, y], axis=1)
                outputs = self(inputs, training=True)
                
                u = outputs[:, 0]
                v = outputs[:, 1]
                p = outputs[:, 2]
            
            u_x = tape1.gradient(u, x)
            u_y = tape1.gradient(u, y)
            v_x = tape1.gradient(v, x)
            v_y = tape1.gradient(v, y)
            p_x = tape1.gradient(p, x)
            p_y = tape1.gradient(p, y)
            
            del tape1
        
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)
        v_xx = tape2.gradient(v_x, x)
        v_yy = tape2.gradient(v_y, y)
        
        del tape2
        
        return u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy
    
    def physics_loss(self, x, y):
        """
        Compute dimensionless Navier-Stokes residuals.
        
        Continuity: du/dx + dv/dy = 0
        Momentum X: u*du/dx + v*du/dy = -dp/dx + 1/Re * (d2u/dx2 + d2u/dy2)
        Momentum Y: u*dv/dx + v*dv/dy = -dp/dy + 1/Re * (d2v/dx2 + d2v/dy2)
        """
        u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy = self.compute_gradients(x, y)
        
        # Continuity
        continuity = u_x + v_y
        
        # Momentum (dimensionless form)
        inv_Re = 1.0 / self.Re
        momentum_x = u * u_x + v * u_y + p_x - inv_Re * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y - inv_Re * (v_xx + v_yy)
        
        loss_continuity = tf.reduce_mean(tf.square(continuity))
        loss_momentum_x = tf.reduce_mean(tf.square(momentum_x))
        loss_momentum_y = tf.reduce_mean(tf.square(momentum_y))
        
        return loss_continuity, loss_momentum_x, loss_momentum_y
    
    def boundary_loss(self, x_wall, y_wall, x_inlet, y_inlet, angle_of_attack=0.0):
        """
        Boundary conditions (dimensionless).
        
        Wall: u = 0, v = 0
        Inlet: u = cos(AoA), v = sin(AoA) (normalized freestream = 1)
        """
        # Wall (no-slip)
        inputs_wall = tf.stack([x_wall, y_wall], axis=1)
        outputs_wall = self(inputs_wall, training=True)
        u_wall = outputs_wall[:, 0]
        v_wall = outputs_wall[:, 1]
        
        loss_wall = tf.reduce_mean(tf.square(u_wall)) + tf.reduce_mean(tf.square(v_wall))
        
        # Inlet (freestream)
        aoa_rad = angle_of_attack * np.pi / 180
        u_inf = tf.constant(np.cos(aoa_rad), dtype=tf.float32)
        v_inf = tf.constant(np.sin(aoa_rad), dtype=tf.float32)
        
        inputs_inlet = tf.stack([x_inlet, y_inlet], axis=1)
        outputs_inlet = self(inputs_inlet, training=True)
        u_inlet = outputs_inlet[:, 0]
        v_inlet = outputs_inlet[:, 1]
        
        loss_inlet = tf.reduce_mean(tf.square(u_inlet - u_inf)) + tf.reduce_mean(tf.square(v_inlet - v_inf))
        
        return loss_wall, loss_inlet
    
    def data_loss(self, x_data, y_data, u_data, v_data, p_data):
        """Data fitting loss (all inputs should be normalized)."""
        inputs = tf.stack([x_data, y_data], axis=1)
        outputs = self(inputs, training=True)
        
        u_pred = outputs[:, 0]
        v_pred = outputs[:, 1]
        p_pred = outputs[:, 2]
        
        loss_u = tf.reduce_mean(tf.square(u_pred - u_data))
        loss_v = tf.reduce_mean(tf.square(v_pred - v_data))
        loss_p = tf.reduce_mean(tf.square(p_pred - p_data))
        
        return loss_u + loss_v + loss_p


class PINNTrainer:
    """Trainer for Physics-Informed Neural Networks."""
    
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Loss weights
        self.lambda_physics = 1.0
        self.lambda_bc = 10.0
        self.lambda_data = 10.0  # Higher weight for data fitting
        
        self.history = {
            'loss': [],
            'physics_loss': [],
            'bc_loss': [],
            'data_loss': []
        }
    
    @tf.function
    def train_step(self, domain_points, boundary_points, inlet_points, 
                   angle_of_attack, cfd_data=None):
        """Single training step."""
        x_domain = tf.cast(domain_points[:, 0], tf.float32)
        y_domain = tf.cast(domain_points[:, 1], tf.float32)
        
        x_wall = tf.cast(boundary_points[:, 0], tf.float32)
        y_wall = tf.cast(boundary_points[:, 1], tf.float32)
        
        x_inlet = tf.cast(inlet_points[:, 0], tf.float32)
        y_inlet = tf.cast(inlet_points[:, 1], tf.float32)
        
        with tf.GradientTape() as tape:
            loss_cont, loss_mom_x, loss_mom_y = self.model.physics_loss(x_domain, y_domain)
            physics_loss = loss_cont + loss_mom_x + loss_mom_y
            
            loss_wall, loss_inlet = self.model.boundary_loss(
                x_wall, y_wall, x_inlet, y_inlet, angle_of_attack
            )
            bc_loss = loss_wall + loss_inlet
            
            data_loss = tf.constant(0.0)
            if cfd_data is not None:
                data_loss = self.model.data_loss(
                    cfd_data['x'], cfd_data['y'],
                    cfd_data['u'], cfd_data['v'], cfd_data['p']
                )
            
            total_loss = (self.lambda_physics * physics_loss + 
                         self.lambda_bc * bc_loss + 
                         self.lambda_data * data_loss)
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, physics_loss, bc_loss, data_loss
    
    def train(self, domain_points, boundary_points, inlet_points, 
              angle_of_attack=0.0, epochs=1000, cfd_data=None, 
              batch_size=1000, callback=None):
        """Train the PINN model."""
        n_domain = len(domain_points)
        
        for epoch in range(epochs):
            idx = np.random.choice(n_domain, min(batch_size, n_domain), replace=False)
            batch_domain = domain_points[idx]
            
            total_loss, physics_loss, bc_loss, data_loss = self.train_step(
                batch_domain, boundary_points, inlet_points,
                angle_of_attack, cfd_data
            )
            
            self.history['loss'].append(float(total_loss))
            self.history['physics_loss'].append(float(physics_loss))
            self.history['bc_loss'].append(float(bc_loss))
            self.history['data_loss'].append(float(data_loss))
            
            if callback:
                callback(epoch, {
                    'total': float(total_loss),
                    'physics': float(physics_loss),
                    'bc': float(bc_loss),
                    'data': float(data_loss)
                })
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.6f} "
                      f"(Physics: {physics_loss:.6f}, BC: {bc_loss:.6f}, Data: {data_loss:.6f})")
        
        return self.history
    
    def predict(self, x, y):
        """Predict flow field at given points."""
        inputs = tf.stack([x, y], axis=1)
        outputs = self.model(inputs, training=False)
        
        return outputs[:, 0].numpy(), outputs[:, 1].numpy(), outputs[:, 2].numpy()


def create_pinn_model(hidden_layers=[64, 64, 64, 64], learning_rate=1e-3, reynolds=1e6):
    """Factory function to create PINN model and trainer."""
    model = PINNModel(layers_config=hidden_layers, reynolds=reynolds)
    trainer = PINNTrainer(model, learning_rate=learning_rate)
    
    return model, trainer


if __name__ == "__main__":
    model, trainer = create_pinn_model()
    
    test_input = tf.constant([[0.5, 0.1], [0.3, -0.05]], dtype=tf.float32)
    output = model(test_input)
    print(f"Test output shape: {output.shape}")
    print(f"Test output (u, v, p): {output.numpy()}")
