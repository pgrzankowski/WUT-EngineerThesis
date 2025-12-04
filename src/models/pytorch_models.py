"""PyTorch model implementations."""

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from .base_model import BaseModel


class MLPRegressorModel(BaseModel):
    """Multi-Layer Perceptron regressor using PyTorch."""
    
    def __init__(self, use_log_transform=True, hidden_layers=[50, 30, 10], 
                 dropout=0.15, batch_size=128, num_epochs=50, 
                 learning_rate=1e-3, device=None, **kwargs):
        """
        Initialize MLP model.
        
        Args:
            use_log_transform: Whether to apply log1p transform to target
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None
            **kwargs: Additional parameters (not used, for compatibility)
        """
        super().__init__(use_log_transform=use_log_transform)
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.input_size = None
        self.model = None
    
    def _create_model(self, **kwargs):
        """Create the PyTorch model architecture."""
        if self.input_size is None:
            raise ValueError("Input size must be set before creating model")
        
        return MLPNetwork(
            input_size=self.input_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout
        ).to(self.device)
    
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            verbose: Whether to print training progress
        """
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Set input size
        self.input_size = X.shape[1]
        
        # Apply log transform if specified
        if self.use_log_transform:
            y = np.log1p(y)
        
        # Create model
        self.model = self._create_model()
        
        # Create datasets
        train_dataset = data.TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float()
        )
        train_dataloader = data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        val_dataloader = None
        if X_val is not None and y_val is not None:
            if hasattr(X_val, 'values'):
                X_val = X_val.values
            if hasattr(y_val, 'values'):
                y_val = y_val.values
            
            if self.use_log_transform:
                y_val = np.log1p(y_val)
            
            val_dataset = data.TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).float()
            )
            val_dataloader = data.DataLoader(
                val_dataset, 
                batch_size=self.batch_size
            )
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_module = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            running_train_loss = 0.0
            n_train = 0
            
            for inputs, labels in train_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                preds = self.model(inputs)
                preds = preds.squeeze(dim=1)
                
                loss = loss_module(preds, labels.float())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_size = labels.shape[0]
                running_train_loss += loss.item() * batch_size
                n_train += batch_size
            
            epoch_train_loss = running_train_loss / max(1, n_train)
            train_losses.append(epoch_train_loss)
            
            # Validation
            epoch_val_loss = None
            if val_dataloader is not None:
                self.model.eval()
                running_val_loss = 0.0
                n_val = 0
                
                with torch.no_grad():
                    for inputs, labels in val_dataloader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        preds = self.model(inputs)
                        preds = preds.squeeze(dim=1)
                        
                        val_loss = loss_module(preds, labels.float())
                        
                        batch_size = labels.shape[0]
                        running_val_loss += val_loss.item() * batch_size
                        n_val += batch_size
                
                epoch_val_loss = running_val_loss / max(1, n_val)
                val_losses.append(epoch_val_loss)
            
            if verbose:
                if epoch_val_loss is not None:
                    print(f'Epoch {epoch:03d} | train_loss: {epoch_train_loss:.4f} | val_loss: {epoch_val_loss:.4f}')
                else:
                    print(f'Epoch {epoch:03d} | train_loss: {epoch_train_loss:.4f}')
        
        self.is_fitted = True
        self.train_losses = train_losses
        self.val_losses = val_losses if val_losses else None
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        
        # Create dataset and dataloader
        test_dataset = data.TensorDataset(torch.from_numpy(X).float())
        test_dataloader = data.DataLoader(test_dataset, batch_size=self.batch_size)
        
        # Predict
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for (inputs,) in test_dataloader:
                inputs = inputs.to(self.device)
                preds = self.model(inputs)
                preds = preds.squeeze(dim=1)
                all_preds.append(preds.cpu())
        
        preds = torch.cat(all_preds, dim=0).numpy()
        
        # Inverse log transform if needed
        if self.use_log_transform:
            preds = np.expm1(preds)
        
        return preds
    
    def get_params(self, deep=True):
        """Get model parameters."""
        params = {
            'use_log_transform': self.use_log_transform,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'device': str(self.device),
        }
        return params
    
    def set_params(self, **params):
        """Set model parameters."""
        if 'use_log_transform' in params:
            self.use_log_transform = params.pop('use_log_transform')
        if 'hidden_layers' in params:
            self.hidden_layers = params.pop('hidden_layers')
        if 'dropout' in params:
            self.dropout = params.pop('dropout')
        if 'batch_size' in params:
            self.batch_size = params.pop('batch_size')
        if 'num_epochs' in params:
            self.num_epochs = params.pop('num_epochs')
        if 'learning_rate' in params:
            self.learning_rate = params.pop('learning_rate')
        if 'device' in params:
            self.device = torch.device(params.pop('device'))
        return self


class MLPNetwork(nn.Module):
    """PyTorch MLP network architecture."""
    
    def __init__(self, input_size, hidden_layers=[50, 30, 10], dropout=0.15):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

