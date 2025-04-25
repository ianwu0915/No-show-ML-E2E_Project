import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data.data_preparation import NoShowDataset, prepare_data
from src.models.neural_net import NoShowNet
from src.evaluate import evaluate_model, evaluate_nn, evaluate_threshold
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def run_nn_pipeline(X, y, numerical_features, config=None):
    """Run the entire neural network training pipeline"""
    # 初始化wandb
    wandb.init(
        project="no_show_prediction",
        name=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config or {
            "learning_rate": 0.001,
            "epochs": 30,
            "batch_size": 64,
            "weight_decay": 0.01,
            "hidden_dims": [256, 128, 64],
            "dropout_rates": [0.4, 0.3, 0.2],
            "pos_weight": 8395/932,
            "patience": 10
        }
    )
    config = wandb.config
    
    # Prepare data: preprocess data and split into training and validation sets
    X_train, X_test, y_train, y_test = prepare_data(X, y, numerical_features)
    
    # Create datasets for training and validation
    train_dataset = NoShowDataset(X_train, y_train)
    val_dataset = NoShowDataset(X_test, y_test)
    
    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Set device: use MPS if available, otherwise use CPU (MAC)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = NoShowNet(input_dim=input_dim).to(device)
    wandb.watch(model)  # track model parameters
    
    # Set training parameters
    pos_weight = torch.tensor([config.pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Adam or AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.1,
        min_lr=1e-6
    )
    
    # Training loop
    best_f1 = 0
    best_model_state = None
    no_improve = 0
    
    for epoch in range(config.epochs):
        # Training phase
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation phase
        val_preds, val_probs, val_labels = evaluate_nn(model, val_loader, device)
        val_loss = F.binary_cross_entropy_with_logits(
            torch.tensor(val_probs, dtype=torch.float32).reshape(-1, 1),
            torch.tensor(val_labels, dtype=torch.float32).reshape(-1, 1)
        ).item()
        
        # Calculate metrics
        metrics = evaluate_model(val_labels, val_preds, val_probs, f"Epoch {epoch+1}")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": metrics['f1'],
            "val_precision": metrics['precision'],
            "val_recall": metrics['recall'],
            "val_accuracy": metrics['accuracy'],
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Adjust learning rate if validation loss doesn't improve
        scheduler.step(val_loss)
        
        # Early stopping and model saving
        if metrics['recall'] > best_f1:
            best_f1 = metrics['recall']
            
            # Save best model
            best_model_state = model.state_dict().copy()
            model_path = f"best_model_{wandb.run.id}.pth"
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Metrics: {metrics}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # After training loop
    print("\nEvaluating different thresholds...")
    
    # Get validation set predictions
    
    
    model.eval()
    with torch.no_grad():
        val_probs = []
        val_labels = []
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            val_probs.extend(probs)
            val_labels.extend(y_batch.numpy())
    
    val_probs = np.array(val_probs).flatten()
    val_labels = np.array(val_labels)
    
    # Evaluate different thresholds based on cost savings (business metric)
    threshold_results = evaluate_threshold(val_labels, val_probs)
    
    # Find best threshold based on cost savings (business metric)
    best_threshold = threshold_results.loc[threshold_results['cost_savings'].idxmax(), 'threshold']
    best_metrics = threshold_results.loc[threshold_results['cost_savings'].idxmax()].to_dict()
    
    # Log to wandb
    wandb.log({
        "best_threshold": best_threshold,
        "best_cost_savings": best_metrics['cost_savings'],
        "best_precision": best_metrics['precision'],
        "best_recall": best_metrics['recall'],
        "threshold_results": wandb.Table(dataframe=threshold_results)
    })
    
    # Plot threshold evaluation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Precision-Recall curve
    ax1.plot(threshold_results['threshold'], threshold_results['precision'], label='Precision')
    ax1.plot(threshold_results['threshold'], threshold_results['recall'], label='Recall')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision and Recall vs Threshold')
    ax1.legend()
    
    # Cost savings curve
    ax2.plot(threshold_results['threshold'], threshold_results['cost_savings'])
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Cost Savings ($)')
    ax2.set_title('Cost Savings vs Threshold')
    
    plt.tight_layout()
    wandb.log({"threshold_evaluation": wandb.Image(fig)})
    plt.show()
    plt.close()
    
    print(f"\nBest threshold: {best_threshold:.3f}")
    print(f"Cost savings: ${best_metrics['cost_savings']:.2f}")
    print(f"Precision: {best_metrics['precision']:.3f}")
    print(f"Recall: {best_metrics['recall']:.3f}")
    
    wandb.finish()
    return model, best_threshold, best_metrics, val_probs, val_labels

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).reshape(-1, 1)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())
    
    return total_loss / len(train_loader), np.array(all_preds), np.array(all_labels)