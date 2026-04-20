import os
import torch
import torch.optim as optim
from args import get_args
import matplotlib.pyplot as plt
from utils import show_batch


def validate_model(model, val_loader, device):
    model.train()  # Faster RCNN needs train() mode to compute losses
    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device, dtype=torch.float32) for image in images]
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64),
                }
                for target in targets
            ]

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            val_loss_sum += loss.item() * len(images)  # += not =
            val_count += len(images)                   # += not =

    val_epoch_loss = val_loss_sum / val_count
    return val_epoch_loss


def train_model(model, train_loader, val_loader, device):
    args = get_args()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):  # epoch loop
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64),
                }
                for target in targets
            ]
            #imgRes = show_batch(images, targets)
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)

        train_epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = validate_model(model, val_loader, device)
        train_losses.append(train_epoch_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.out_dir, exist_ok=True)  # args.save_dir → args.out_dir
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))
            print(f"  → Saved best model (val loss: {best_val_loss:.4f})")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Validation Loss", marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    os.makedirs(args.out_dir, exist_ok=True)
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"))
    plt.show()