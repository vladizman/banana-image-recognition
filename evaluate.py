import os
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor

TEST_DIR = "./test_images"
OUT_DIR = "./test_results"
MODEL_PATH = "./sessions/best_model.pth"
THRESHOLD = 0.8

def load_model():
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # background + banana

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_model()

    image_files = [
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for fname in image_files:
        path = os.path.join(TEST_DIR, fname)
        image = Image.open(path).convert("RGB")
        image_tensor = to_tensor(image)

        with torch.no_grad():
            outputs = model([image_tensor])[0]

        draw = ImageDraw.Draw(image)

        boxes = outputs["boxes"]
        scores = outputs["scores"]

        if len(scores) > 0:
            best_idx = int(torch.argmax(scores).item())
            best_score = float(scores[best_idx])

            if best_score >= THRESHOLD:
                x1, y1, x2, y2 = boxes[best_idx].tolist()
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, max(0, y1 - 15)), f"banana {best_score:.2f}", fill="red")

        out_path = os.path.join(OUT_DIR, fname)
        image.save(out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
