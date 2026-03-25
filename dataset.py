import torch
from PIL import Image
from utils import resize_box_xyxy
from torchvision.transforms.functional import to_tensor, resize


class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_size=(640, 640)):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # TODO 1: Get the row number idx from dataframe
        # your code here
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        w, h = img.size
        new_h, new_w = self.image_size
        image = to_tensor(img)

        boxes, labels = [], []
        with open(row["label_path"]) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h

                # resize box to match resized image
                x1, y1, x2, y2 = resize_box_xyxy(
                    (x1, y1, x2, y2),
                    w, h,
                    new_w, new_h
                )

                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)

                #resize
                img = resize(img, (new_h, new_w))
                image = to_tensor(img)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        # TODO 2: Return what you need from this class
        # your code here
        return image, target
