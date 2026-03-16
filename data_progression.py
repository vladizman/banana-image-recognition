import os
import random
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    images_dir = "./Data/images"
    labels_dir = "./Data/labels/train"   # where your YOLO .txt files currently are
    out_dir = "./Data/csv"
    val_split = 0.2
    seed = 42

    os.makedirs(out_dir, exist_ok=True)

    # collect valid (image, label) pairs
    pairs = []
    for fname in os.listdir(images_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMG_EXTS:
            continue

        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(images_dir, fname)
        label_path = os.path.join(labels_dir, stem + ".txt")

        if os.path.exists(label_path):
            pairs.append({"image_path": img_path, "label_path": label_path})
        else:
            # If you want to allow images without labels, comment out this print
            print(f"Skipping (no label): {img_path}")

    if len(pairs) == 0:
        raise RuntimeError("No image/label pairs found. Check folder paths and filenames.")

    random.seed(seed)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * (1 - val_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_csv = os.path.join(out_dir, "train.csv")
    val_csv = os.path.join(out_dir, "val.csv")

    pd.DataFrame(train_pairs).to_csv(train_csv, index=False)
    pd.DataFrame(val_pairs).to_csv(val_csv, index=False)

    print(f"Saved: {train_csv} ({len(train_pairs)} samples)")
    print(f"Saved: {val_csv} ({len(val_pairs)} samples)")

if __name__ == "__main__":
    main()