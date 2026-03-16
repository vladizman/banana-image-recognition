from args import get_args
from dataset import ObjDetectionDataset
from torch.utils.data import DataLoader
import pandas as pd
from model import build_model

def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
def main():
    args = get_args()

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)

    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    ##init model
    model = build_model(args.backbone)
    print(model)
   # images, targets = next(iter(train_loader))
    #print(images, targets)

if __name__ == "__main__":
    main()