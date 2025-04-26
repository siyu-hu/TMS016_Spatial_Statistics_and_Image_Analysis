import torch
from torch.utils.data import DataLoader
from siamese_model import SiameseNetwork
from utils import SiameseDataset, ContrastiveLoss, plot_loss, save_checkpoint
import os
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau 

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for img1, img2, label in tqdm(dataloader, desc="Training", leave=False): 
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = loss_fn(output1, output2, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for img1, img2, label in tqdm(dataloader, desc="Validating", leave=False):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)
            loss = loss_fn(output1, output2, label)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # -------------- Config --------------
    use_augmentation = True  # IMPORTANT: Set to True if training on augmented data
    balance_negatives = True  # IMPORTANT: Should match how you created training pairs
    #num_augments = 4  # Just for record (affects ckpt name)

    if use_augmentation:
        train_data_path = "./project3_fingerprint_fvc2000/data/train_pairs_augmented.npz"
    else:
        train_data_path = "./project3_fingerprint_fvc2000/data/train_pairs.npz"
        
    val_data_path = "./project3_fingerprint_fvc2000/data/val_pairs.npz"

    val_data_path = "./project3_fingerprint_fvc2000/data/val_pairs.npz"
    batch_size = 8
    num_epochs = 20
    learning_rate = 0.001
    margin = 2.0
    ckpt_filename = f"model_aug{use_augmentation}_bl{balance_negatives}_bs{batch_size}_ep{num_epochs}_lr{learning_rate}_mg{margin}.pt"
    ckpt_path = f"./project3_fingerprint_fvc2000/checkpoints/{ckpt_filename}"
    resume = False  # IMPORTANT: Set to True to resume training from the last checkpoint
 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------- Dataset + Dataloader -------------
    train_dataset = SiameseDataset(train_data_path)
    val_dataset = SiameseDataset(val_data_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ------------ Model, Loss, Optimizer --------------
    model = SiameseNetwork().to(device)

    if resume and os.path.exists(ckpt_path):
        print(f" Resuming training from checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))

    loss_fn = ContrastiveLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, verbose=True)
    early_stop_patience = 4      # if validation loss does not improve for this many epochs, stop training
    bad_epochs = 0

    # ------ Training Loop -----------------
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, ckpt_path)
            print(f"Saved improved model to {ckpt_path}")
            bad_epochs = 0           # reset
        else:
            bad_epochs += 1
        # early stopping
        if bad_epochs >= early_stop_patience:
            print("Early stopping triggered.")
            break

    plot_loss(train_losses, val_losses, save_path="./project3_fingerprint_fvc2000/outputs/loss_curve.png")

if __name__ == "__main__":
    main()
