# v1-v3/inference.py 〔HFなし・同梱pt直読み版〕
from pathlib import Path
from typing import Union, IO
import torch
from torchvision import transforms
from PIL import Image
from models.model_def import CNN  # 学習時に使ったクラス

# ===== デバイス =====
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 前処理（v3の設定）=====
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # [1,H,W] -> [3,H,W]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===== モデルロード（同梱 .pt を直接読む）=====
MODEL_PATH = Path(__file__).resolve().parent / "models" / "kinoko_takenoko_v3.pt"

def load_model() -> torch.nn.Module:
    dev = _device()
    model = CNN()  # 必要なら num_classes=2 など学習時の引数に合わせる
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(dev).eval()
    return model

# ===== 予測 =====
@torch.inference_mode()
def predict(image_path_or_file: Union[str, Path, IO[bytes]]) -> dict:
    """
    image_path_or_file: 画像パス or BytesIO/UploadedFile どちらでもOK
    return: {"label": str, "confidence": float, "probs": list[float]}
    """
    dev = _device()
    model = load_model()

    img = Image.open(image_path_or_file).convert("L")
    x = transform(img).unsqueeze(0).to(dev)  # [1,3,224,224]

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())

    labels = {0: "kinoko", 1: "takenoko"}  # ←学習時のラベル順に必ず合わせる
    return {"label": labels[idx], "confidence": float(probs[idx]), "probs": probs.tolist()}
