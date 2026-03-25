from args import get_args
from dataset import ObjDetectionDataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
from model import build_model
from trainer import train_model

def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
def main():
    args = get_args()

    #1 read the dataframe
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    #2 dataset
    train_dataset = ObjDetectionDataset(train_df, image_size=(512, 512))
    val_dataset = ObjDetectionDataset(val_df, image_size=(512, 512))
    #3create data loaders split the images into batches
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,  num_workers=0,  pin_memory=(torch.cuda.is_available()))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,num_workers=0,  pin_memory=(torch.cuda.is_available()))

    ##4 init model
    model = build_model(args.backbone, num_classes=args.num_classes + 1)

    #5. Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()