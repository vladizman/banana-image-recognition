import torch
from PIL import Image, ImageOps
from utils import resize_box_xyxy
from torchvision.transforms.functional import to_tensor, resize


class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_size=(640, 640)):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get row
        row = self.df.iloc[idx]

        # Load image
        img = Image.open(row["image_path"]).convert("RGB")
        img = ImageOps.exif_transpose(img)

        original_w, original_h = img.size
        new_h, new_w = self.image_size

        boxes = []
        labels = []

        # Read label file
        with open(row["label_path"]) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())

                # Convert YOLO -> XYXY (original size)
                x1 = (xc - bw / 2) * original_w
                y1 = (yc - bh / 2) * original_h
                x2 = (xc + bw / 2) * original_w
                y2 = (yc + bh / 2) * original_h

                # Resize box
                x1, y1, x2, y2 = resize_box_xyxy(
                    (x1, y1, x2, y2),
                    original_w, original_h,
                    new_w, new_h
                )

                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)  # background = 0

        # Resize image ONCE
        img = resize(img, (new_h, new_w))
        image = to_tensor(img)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        return image, target