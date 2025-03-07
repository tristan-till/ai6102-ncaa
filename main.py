from classes import trainer
from losses.CustomLoss import CustomLoss
import torch
import torch.optim as optim

from models.AttentionModel import AttentionModel
import classes.data_loader as data_loader
import utils.const as const


def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = data_loader.load()
    
    model = AttentionModel().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=const.LEARNING_RATE, weight_decay=const.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=const.LR_DECAY_FACTOR, 
        patience=const.LR_PATIENCE, min_lr=const.MIN_LR
    )
    custom_loss = CustomLoss()
    
    for epoch in range(const.EPOCHS):
        trainer.run_epoch(epoch, model, train_loader, val_loader, optimizer, scheduler, custom_loss)
    
    
if __name__ == "__main__":
    main()