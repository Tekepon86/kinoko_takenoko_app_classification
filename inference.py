from pathlib import Path
from typing import Union, IO
import torch
from torchvision import transforms
from PIL import Image

from models.model_def import CNN
from utils.weights import ensure_weights

# === あなたのHF直リンクに置き換える ===
HF_URL = "https://huggingface.co/Tetsushi86/kinoko-takenoko-v3/resolve/main/kinoko_takenoko_v3.pt"
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "kinoko_takenoko_v3.pt"

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# v3の前処理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_model() -> torch.nn.Module:
    dev = _device()
    # 初回のみDL、以降は weights/ から読む
    w = ensure_weights(WEIGHTS_PATH, HF_URL, sha256=None)
    model = CNN()  # 学習時の引数があれば合わせる（例: num_classes=2）
    state = torch.load(w, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(dev).eval()
    return model

@torch.inference_mode()
def predict(image_path_or_file: Union[str, Path, IO[bytes]]) -> dict:
    dev = _device()
    model = load_model()

    img = Image.open(image_path_or_file).convert("L")
    x = transform(img).unsqueeze(0).to(dev)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())

    labels = {0: "kinoko", 1: "takenoko"}  # 学習時のラベル順に合わせる
    return {"label": labels[idx], "confidence": float(probs[idx]), "probs": probs.tolist()}
