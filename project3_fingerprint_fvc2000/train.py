import torch
from torch.utils.data import DataLoader
from siamese_model import SiameseNetwork
from utils import SiameseDataset, ContrastiveLoss, plot_loss, save_checkpoint
import os
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau 
# --- add CLI -------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train_pairs", default=None)
parser.add_argument("--val_pairs",   default=None)
parser.add_argument("--finetune", action="store_true",
                    help="Continue training from --best_ckpt with typically fewer epochs / lower lr")

parser.add_argument("--lr",          type=float)
parser.add_argument("--num_epochs",  type=int)
parser.add_argument("--use_aug",     action="store_true")  
parser.add_argument("--balance_neg", action="store_true")  
parser.add_argument("--best_ckpt",   default=None)   
args = parser.parse_args()
# --------------------------------------------------------

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
    use_augmentation = args.use_aug # IMPORTANT: Set to True if training on augmented data
    balance_negatives = args.balance_neg  # IMPORTANT: Should match how you created training pairs
    finetune = args.finetune    # IMPORTANT: Set to True to continue training from the best checkpoint


    if use_augmentation:
        default_train_path = "./data/new_train_pairs_augmented.npz"
    else:
        default_train_path = "./data/new_train_pairs.npz"

    train_data_path = args.train_pairs or default_train_path
    val_data_path   = args.val_pairs or "./data/new_val_pairs.npz"


    batch_size = 8
    margin = 2.0

    if args.finetune:                
        default_lr     = 1e-4
        default_epoch  = 10
    else:                            
        default_lr     = 5e-4
        default_epoch  = 20

    learning_rate = args.lr if args.lr is not None else default_lr
    num_epochs    = args.num_epochs if args.num_epochs is not None else default_epoch

    # output ckpt 
    ckpt_filename = f"new_model_ft{finetune}_aug{use_augmentation}_bl{balance_negatives}_bs{batch_size}_ep{num_epochs}_lr{learning_rate}_mg{margin}.pt"
    ckpt_path = f"./checkpoints/{ckpt_filename}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_ckpt_path = args.best_ckpt or ckpt_path
    # -------------- Dataset + Dataloader -------------
    train_dataset = SiameseDataset(train_data_path, root_dir=".")
    val_dataset = SiameseDataset(val_data_path, root_dir=".")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ------------ Model, Loss, Optimizer --------------
    model = SiameseNetwork().to(device)

    if finetune and os.path.exists(best_ckpt_path): 
        print(f" Continue training from checkpoint: {best_ckpt_path}")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print("[INFO] Fine-tuning mode: freezing convolutional backbone...")
        for param in model.features.parameters():
            param.requires_grad = False # freeze conv layers, only train the fc layer

    loss_fn = ContrastiveLoss(margin=margin)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate) 

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, verbose=True)
    early_stop_patience = 7      # if validation loss does not improve for this many epochs, stop training
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

    plot_loss(train_losses, val_losses, save_path="./outputs/loss_curve.png")

if __name__ == "__main__":
    main()
