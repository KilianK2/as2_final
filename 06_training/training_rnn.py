import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add parent directory to path to import splitting_data
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '05_splitting'))
from splitting_data import load_feature_data, session_aware_train_test_split, load_train_test_data

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom Dataset for IMU data
class IMUDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Reshape input to (batch_size, sequence_length, input_size)
        # For our feature vectors, we'll treat each sample as a sequence of length 1
        x = x.unsqueeze(1)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the outputs from the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

# Training loop function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=10):
    """
    Train the RNN model.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on (cuda or cpu)
        patience: Early stopping patience
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate average loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate average loss and accuracy
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - start_time
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        print('-' * 50)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, history

def evaluate_model(model, test_loader, device, label_encoder, class_names=None):
    """
    Evaluate the RNN model on the test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to evaluate on (cuda or cpu)
        label_encoder: LabelEncoder for transforming class names
        class_names: List of class names (default: None)
    """
    print("\nEvaluating model on test set...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to original class labels
    if label_encoder is not None:
        all_preds = label_encoder.inverse_transform(all_preds)
        all_labels = label_encoder.inverse_transform(all_labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Get unique class names for plot labels
    if class_names is None:
        unique_classes = sorted(np.unique(all_labels))
    else:
        # Ensure class_names is a list, not an array
        unique_classes = sorted(class_names) if isinstance(class_names, list) else sorted(list(class_names))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_classes,
                yticklabels=unique_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix plot
    results_folder = os.path.join('results', 'rnn')
    os.makedirs(results_folder, exist_ok=True)
    plt.savefig(os.path.join(results_folder, 'confusion_matrix.png'))
    plt.close()
    
    return accuracy, all_preds, all_labels

def plot_learning_curves(history, results_folder):
    """Plot and save the learning curves."""
    # Plot training & validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'learning_curves.png'))
    plt.close()

def save_model(model, optimizer, label_encoder, scaler, output_path):
    """Save the trained model and related objects to disk"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the model state_dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_architecture': {
            'input_size': model.lstm.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'num_classes': model.fc.out_features
        }
    }, output_path)
    
    print(f"\nModel saved to {output_path}")

def main():
    # Use absolute paths for robust file location
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct absolute paths to the train and test files
    train_file = os.path.join(project_root, "05_splitting", "train_data.csv")
    test_file = os.path.join(project_root, "05_splitting", "test_data.csv")
    
    print(f"Project root: {project_root}")
    print(f"Looking for train file at: {train_file}")
    print(f"Looking for test file at: {test_file}")
    
    # First check if the pre-split files exist
    if os.path.exists(train_file) and os.path.exists(test_file):
        print("Using pre-split train and test datasets")
        X_train, X_test, y_train, y_test = load_train_test_data(train_file, test_file)
    else:
        # Fall back to on-the-fly splitting if files don't exist
        print("Pre-split datasets not found. Performing splitting on the fly...")
        
        # Construct absolute path to the feature file
        feature_file = os.path.join(project_root, "04_feature extraction", "data_by_features_per_window.csv")
        print(f"Looking for feature file at: {feature_file}")
        
        # Check if file exists
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature file not found at: {feature_file}")
            
        data = load_feature_data(feature_file)
        
        # Perform session-aware train-test split
        X_train, X_test, y_train, y_test = session_aware_train_test_split(data, test_size=0.3, random_state=42)
    
    # Encode target labels to integers for PyTorch
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"Class mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Apply normalization using StandardScaler
    print("Applying StandardScaler normalization to features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Print feature statistics
    print(f"Features normalized. Mean: {np.mean(X_train_scaled):.5f}, Std: {np.std(X_train_scaled):.5f}")
    
    # Create datasets and dataloaders
    train_dataset = IMUDataset(pd.DataFrame(X_train_scaled, columns=X_train.columns), 
                              pd.Series(y_train_encoded))
    test_dataset = IMUDataset(pd.DataFrame(X_test_scaled, columns=X_test.columns), 
                             pd.Series(y_test_encoded))
    
    # Split train into train and validation (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # Use random_split with generator for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=generator)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # Initialize model
    input_size = X_train.shape[1]
    hidden_size = 128
    num_layers = 2
    num_classes = len(label_encoder.classes_)
    dropout = 0.3
    
    model = RNNModel(input_size, hidden_size, num_layers, num_classes, dropout).to(device)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model
    print("\nStarting RNN training...")
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs=100,  # Maximum epochs (early stopping may reduce this)
        device=device,
        patience=10  # Early stopping patience
    )
    
    # Create results directory
    results_folder = os.path.join('results', 'rnn')
    os.makedirs(results_folder, exist_ok=True)
    
    # Evaluate model
    class_names = list(label_encoder.classes_)
    accuracy, _, _ = evaluate_model(model, test_loader, device, label_encoder, class_names)
    
    # Plot learning curves
    plot_learning_curves(history, results_folder)
    
    # Save model and related objects
    model_output_path = os.path.join(results_folder, 'model.pth')
    save_model(model, optimizer, label_encoder, scaler, model_output_path)
    
    # Save label encoder and scaler
    joblib.dump(label_encoder, os.path.join(results_folder, 'label_encoder.joblib'))
    joblib.dump(scaler, os.path.join(results_folder, 'standard_scaler.joblib'))
    
    # Save model hyperparameters
    with open(os.path.join(results_folder, 'hyperparameters.txt'), 'w') as f:
        f.write(f"input_size: {input_size}\n")
        f.write(f"hidden_size: {hidden_size}\n")
        f.write(f"num_layers: {num_layers}\n")
        f.write(f"dropout: {dropout}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"optimizer: Adam\n")
        f.write(f"learning_rate: 0.001\n")
        f.write(f"weight_decay: 1e-5\n")
        f.write(f"test_accuracy: {accuracy:.4f}\n")
    
    print("\nRNN training and evaluation completed successfully!")
    print(f"All results saved to {os.path.abspath(results_folder)}")

if __name__ == "__main__":
    main()
