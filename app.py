import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import streamlit as st
from PIL import Image
import gdown
import os

DEVICE = "cpu"

FDI_CLASSES = [
    11,12,13,14,15,16,17,18,
    21,22,23,24,25,26,27,28,
    31,32,33,34,35,36,37,38,
    41,42,43,44,45,46,47,48
]
class_names = {idx: str(fdi) for idx, fdi in enumerate(FDI_CLASSES)}

UNET_FILE_ID = "1OF4ipYCqqlF5Cz_ocW9Egc9E_O8sS-Fd"
YOLO_FILE_ID = "1cCEj-Fg0w7-gWfonFITRxs4AsyNWcdWh"


@st.cache_resource
def load_models():

    if not os.path.exists("best_model.pth"):
        gdown.download(id=UNET_FILE_ID, output="best_model.pth", quiet=False)

    if not os.path.exists("best.pt"):
        gdown.download(id=YOLO_FILE_ID, output="best.pt", quiet=False)

    if os.path.getsize("best.pt") < 100_000:
        os.remove("best.pt")
        raise RuntimeError(
            "best.pt download failed (got HTML warning page instead of the model). "
            "Try re-sharing the file or use a direct download link."
        )

    if os.path.getsize("best_model.pth") < 100_000:
        os.remove("best_model.pth")
        raise RuntimeError(
            "best_model.pth download failed (got HTML warning page instead of the model)."
        )

    unet_model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    unet_model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    unet_model.eval()

    yolo_model = YOLO("best.pt")

    return unet_model, yolo_model


unet_model, yolo_model = load_models()

inference_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def predict(image):
    original = np.array(image.convert("RGB"))
    orig_h, orig_w = original.shape[:2]

    augmented = inference_transform(image=original)
    input_tensor = augmented["image"].unsqueeze(0)

    with torch.no_grad():
        output = unet_model(input_tensor)
        pred_mask = torch.sigmoid(output).squeeze().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

    pred_mask_full = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    seg_overlay = original.copy()
    seg_overlay[pred_mask_full == 1] = [255, 0, 0]
    seg_overlay = cv2.addWeighted(original, 0.6, seg_overlay, 0.4, 0)

    results = yolo_model.predict(source=original, imgsz=640, conf=0.25, verbose=False)

    combined = seg_overlay.copy()
    for box in results[0].boxes:
        xyxy = box.xyxy[0].numpy().astype(int)
        cls_id = int(box.cls[0].numpy())
        label = class_names.get(cls_id, str(cls_id))
        cv2.rectangle(combined, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(combined, label, (xyxy[0], max(xyxy[1]-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return combined


st.title("Dental X-ray Tooth Segmentation & Detection")
st.caption("Model: EfficientNet-B0 (U-Net) + YOLOv8n")
st.write(
    "Upload a panoramic dental X-ray to see U-Net segmentation and YOLO tooth "
    "detection with FDI numbers.\n\n"
    "⚠️ Note: trained on standard panoramic dental X-rays only. Results on other "
    "image types may be unreliable. For educational/demo purposes only."
)

uploaded_file = st.file_uploader("Upload Dental X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    with st.spinner("Running prediction..."):
        result = predict(image)

    st.image(result, caption="Segmentation + Detection Result", use_column_width=True)
