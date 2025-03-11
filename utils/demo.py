import os
import time
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from model1 import TeamComparisonModel
from loss import custom_loss
from data_loader import load

def train_model(args):
    # Create output directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set up TensorBoard
    run_name = f"{args.model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(os.path.join(args.log_dir, run_name))
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load and prepare data
    X, y = load()
    print(f"Data loaded - X shape: {X.shape}, y shape: {y.shape}")
    
    # Standardize features
    if args.standardize_features:
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds[stds == 0] = 1  # Prevent division by zero
        X = (X - means) / stds
        print("Features standardized")
    
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    # Split data
    dataset = TensorDataset(X, y)
    dataset_size = len(dataset)
    train_size = int(args.train_split * dataset_size)
    val_size = int(args.val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Data split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Initialize model
    feature_dim = X.shape[2]  # Number of features per team
    print(feature_dim, args.hidden_dim)
    exit()
    model = TeamComparisonModel(feature_dim=feature_dim, hidden_dim=args.hidden_dim)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Add model graph to TensorBoard
    dummy_input = torch.zeros((1, 2, feature_dim), dtype=torch.float32)
    writer.add_graph(model, dummy_input)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay_factor, 
        patience=args.lr_patience, min_lr=args.min_lr
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Training loop
    best_val_loss = float('inf')
    early_stop_counter = 0
    global_step = 0
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_winner_acc = 0.0
        train_odds_mae = 0.0
        start_time = time.time()
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, (data, target) in enumerate(train_progress):
            # Randomly flip teams to avoid order bias (with 50% probability)
            if args.random_flip and np.random.random() < 0.5:
                # Flip team order in data
                data = torch.stack([data[:, 1], data[:, 0]], dim=1)
                # Flip target winner (1 becomes 0, 0 becomes 1)
                target[:, 0] = 1.0 - target[:, 0]
                # Flip bookmaker odds
                target[:, 1] = 1.0 - target[:, 1]
            
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            
            # Calculate loss
            loss = custom_loss(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Calculate winner prediction accuracy
            pred_winner = (output[:, 0] > 0.5).float()
            true_winner = (target[:, 0] > 0.5).float()
            accuracy = (pred_winner == true_winner).float().mean().item()
            train_winner_acc += accuracy
            
            # Calculate odds Mean Absolute Error
            odds_mae = torch.abs(output[:, 1] - target[:, 1]).mean().item()
            train_odds_mae += odds_mae
            
            # Update TensorBoard every few steps
            if batch_idx % args.log_interval == 0:
                writer.add_scalar('train/batch_loss', loss.item(), global_step)
                writer.add_scalar('train/batch_winner_acc', accuracy, global_step)
                writer.add_scalar('train/batch_odds_mae', odds_mae, global_step)
                
                # Log gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'gradients/{name}', param.grad, global_step)
                
                # Update progress bar
                train_progress.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy:.4f}",
                    'odds_mae': f"{odds_mae:.4f}"
                })
            
            global_step += 1
        
        # Average training metrics
        train_loss /= len(train_loader)
        train_winner_acc /= len(train_loader)
        train_odds_mae /= len(train_loader)
        train_time = time.time() - start_time
        
        # Log epoch training metrics
        writer.add_scalar('train/epoch_loss', train_loss, epoch)
        writer.add_scalar('train/epoch_winner_acc', train_winner_acc, epoch)
        writer.add_scalar('train/epoch_odds_mae', train_odds_mae, epoch)
        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('time/train_epoch_time', train_time, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_winner_acc = 0.0
        val_odds_acc = 0.0
        val_odds_mae = 0.0
        start_time = time.time()
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for data, target in val_progress:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = custom_loss(output, target)
                val_loss += loss.item()
                pred_winner = (output[:, 0] > 0.5).float()
                true_winner = (target[:, 0] > 0.5).float()
                true_odds = (target[:, 1] > 0.5).float()
                accuracy = (pred_winner == true_winner).float().mean().item()
                odds_accuracy = (pred_winner == true_odds).float().mean().item()
                val_winner_acc += accuracy
                val_odds_acc += odds_accuracy
                
                # Calculate odds Mean Absolute Error
                odds_mae = torch.abs(output[:, 1] - target[:, 1]).mean().item()
                val_odds_mae += odds_mae
                
                # Update progress bar
                val_progress.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy:.4f}",
                    'odds_mae': f"{odds_mae:.4f}",
                    'odds_acc': f"{odds_accuracy:.4f}",
                })
        
        # Average validation metrics
        val_loss /= len(val_loader)
        val_winner_acc /= len(val_loader)
        val_odds_acc /= len(val_loader)
        val_odds_mae /= len(val_loader)
        val_time = time.time() - start_time
        
        # Log epoch validation metrics
        writer.add_scalar('val/epoch_loss', val_loss, epoch)
        writer.add_scalar('val/epoch_winner_acc', val_winner_acc, epoch)
        writer.add_scalar('val/epoch_odds_acc', val_odds_acc, epoch)
        writer.add_scalar('val/epoch_odds_mae', val_odds_mae, epoch)
        writer.add_scalar('time/val_epoch_time', val_time, epoch)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Winner Acc: {train_winner_acc:.4f}, Odds MAE: {train_odds_mae:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Winner Acc: {val_winner_acc:.4f}, Odds MAE: {val_odds_mae:.4f}")
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # Save model checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{run_name}_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_winner_acc': val_winner_acc,
            'val_odds_acc': val_odds_acc,
            'val_odds_mae': val_odds_mae,
        }, checkpoint_path)
        
        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.checkpoint_dir, f"{run_name}_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_winner_acc': val_winner_acc,
                'val_odds_acc': val_odds_acc,
                'val_odds_mae': val_odds_mae,
            }, best_model_path)
            print(f"New best model saved (val_loss: {val_loss:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{args.early_stop_patience}")
        
        # Early stopping
        if early_stop_counter >= args.early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Test phase
    print("\nEvaluating best model on test set...")
    
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    test_loss = 0.0
    test_winner_acc = 0.0
    test_odds_mae = 0.0
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="Testing")
        for data, target in test_progress:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = custom_loss(output, target)
            
            # Update metrics
            test_loss += loss.item()
            
            # Calculate winner prediction accuracy
            pred_winner = (output[:, 0] > 0.5).float()
            true_winner = (target[:, 0] > 0.5).float()
            accuracy = (pred_winner == true_winner).float().mean().item()
            test_winner_acc += accuracy
            
            # Calculate odds Mean Absolute Error
            odds_mae = torch.abs(output[:, 1] - target[:, 1]).mean().item()
            test_odds_mae += odds_mae
            
            # Save predictions and targets for further analysis
            test_predictions.append(output.cpu().numpy())
            test_targets.append(target.cpu().numpy())
            
            # Update progress bar
            test_progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy:.4f}",
                'odds_mae': f"{odds_mae:.4f}"
            })
    
    # Average test metrics
    test_loss /= len(test_loader)
    test_winner_acc /= len(test_loader)
    test_odds_mae /= len(test_loader)
    
    # Log test metrics
    writer.add_scalar('test/loss', test_loss, 0)
    writer.add_scalar('test/winner_acc', test_winner_acc, 0)
    writer.add_scalar('test/odds_mae', test_odds_mae, 0)
    
    # Print test summary
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Winner Prediction Accuracy: {test_winner_acc:.4f}")
    print(f"Odds Mean Absolute Error: {test_odds_mae:.4f}")
    
    # Save test predictions for further analysis
    test_predictions = np.concatenate(test_predictions)
    test_targets = np.concatenate(test_targets)
    np.save(os.path.join(args.checkpoint_dir, f"{run_name}_test_predictions.npy"), test_predictions)
    np.save(os.path.join(args.checkpoint_dir, f"{run_name}_test_targets.npy"), test_targets)
    
    # Close TensorBoard writer
    writer.close()
    
    return {
        'test_loss': test_loss,
        'test_winner_acc': test_winner_acc,
        'test_odds_mae': test_odds_mae,
        'best_model_path': best_model_path
    }

def main():
    parser = argparse.ArgumentParser(description="Team Comparison Model Training")
    
    # Data parameters
    parser.add_argument('--train-split', type=float, default=0.7, help='Proportion of data to use for training')
    parser.add_argument('--val-split', type=float, default=0.15, help='Proportion of data to use for validation')
    parser.add_argument('--standardize-features', action='store_true', help='Standardize input features')
    parser.add_argument('--random-flip', action='store_true', help='Randomly flip team order during training')
    
    # Model parameters
    parser.add_argument('--model-name', type=str, default='team_comparison', help='Model name')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 penalty)')
    parser.add_argument('--lr-decay-factor', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--lr-patience', type=int, default=5, help='Patience for learning rate scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--early-stop-patience', type=int, default=15, help='Patience for early stopping')
    
    # System parameters
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--log-interval', type=int, default=10, help='Batches between logging training status')
    parser.add_argument('--log-dir', type=str, default='runs', help='TensorBoard log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Model checkpoint directory')
    
    args = parser.parse_args()
    
    results = train_model(args)
    print(f"Training completed. Best model saved at: {results['best_model_path']}")

if __name__ == "__main__":
    main()