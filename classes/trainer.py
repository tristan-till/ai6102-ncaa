import os

from tqdm import tqdm
import numpy as np
import torch

from utils import const

def train_epoch(model, train_loader, custom_loss, optimizer, epoch):
    train_loss = 0.0
    train_winner_acc = 0.0
    train_odds_mae = 0.0
    
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{const.EPOCHS} [Train]")
    for _, (data, target) in enumerate(train_progress):
        # Randomly flip teams to avoid order bias (with 50% probability)
        if np.random.random() < 0.5:
            data = torch.stack([data[:, 1], data[:, 0]], dim=1)
            target[:, 0] = 1.0 - target[:, 0]
            target[:, 1] = 1.0 - target[:, 1]
        
        data, target = data.to(const.DEVICE), target.to(const.DEVICE)
        
        optimizer.zero_grad()
        output = model(data)
        loss = custom_loss(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), const.GRAD_CLIP)
        
        optimizer.step()
        train_loss += loss.item()
        
        pred_winner = (output[:, 0] > 0.5).float()
        true_winner = (target[:, 0] > 0.5).float()
        accuracy = (pred_winner == true_winner).float().mean().item()
        train_winner_acc += accuracy
        
        odds_mae = torch.abs(output[:, 1] - target[:, 1]).mean().item()
        train_odds_mae += odds_mae
                
    train_loss /= len(train_loader)
    train_winner_acc /= len(train_loader)
    train_odds_mae /= len(train_loader)
    print(f"Epoch {epoch+1}/{const.EPOCHS}")
    print(f"Train - Loss: {train_loss:.4f}, Winner Acc: {train_winner_acc:.4f}, Odds MAE: {train_odds_mae:.4f}")
    
def val_epoch(model, val_loader, custom_loss, scheduler, epoch):
    model.eval()
    val_loss = 0.0
    val_winner_acc = 0.0
    val_odds_acc = 0.0
    val_odds_mae = 0.0
    
    with torch.no_grad():
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{const.EPOCHS} [Val]")
        for data, target in val_progress:
            data, target = data.to(const.DEVICE), target.to(const.DEVICE)
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
            
            odds_mae = torch.abs(output[:, 1] - target[:, 1]).mean().item()
            val_odds_mae += odds_mae
            
            val_progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy:.4f}",
                'odds_mae': f"{odds_mae:.4f}",
                'odds_acc': f"{odds_accuracy:.4f}",
            })
    
    val_loss /= len(val_loader)
    val_winner_acc /= len(val_loader)
    val_odds_acc /= len(val_loader)
    val_odds_mae /= len(val_loader)
    
    print(f"Epoch {epoch+1}/{const.EPOCHS}")
    print(f"Val   - Loss: {val_loss:.4f}, Winner Acc: {val_winner_acc:.4f}, Odds MAE: {val_odds_mae:.4f}")
    
    scheduler.step(val_loss)
    
    # Save model checkpoint
    checkpoint_path = os.path.join(const.CHECKPOINT_DIR, f"{const.RUN_NAME}_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)

def run_epoch(epoch, model, train_loader, val_loader, optimizer, scheduler, custom_loss):
    train_epoch(model, train_loader, custom_loss, optimizer, epoch)
    val_epoch(model, val_loader, custom_loss, scheduler, epoch)
    
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     best_model_path = os.path.join(args.checkpoint_dir, f"{run_name}_best.pt")
    #     torch.save({
    #         'epoch': epoch + 1,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'val_loss': val_loss,
    #         'val_winner_acc': val_winner_acc,
    #         'val_odds_acc': val_odds_acc,
    #         'val_odds_mae': val_odds_mae,
    #     }, best_model_path)
    #     print(f"New best model saved (val_loss: {val_loss:.4f})")
    #     early_stop_counter = 0
    # else:
    #     early_stop_counter += 1
    #     print(f"Early stopping counter: {early_stop_counter}/{args.early_stop_patience}")
    