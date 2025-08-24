from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from models.model_def import CNN

# ① 追加：自動DLヘルパー（utils/weights.py を作ってある前提）
from utils.weights import ensure_weights

# ② Hugging Face の直リンクに置き換える（自分のURLに差し替え）
HF_URL = "https://huggingface.co/<your-username>/<your-model-repo>/resolve/main/kinoko_takenoko_v3.pt"
HF_SHA256 = None  # 任意で入れる（検証を強めたいとき）

# ③ 参照先はリポ直下の weights/ に統一（Gitには載せない）
WEIGHTS_PATH = Path("weights/kinoko_takenoko_v3.pt")

# ④ 推論デバイス（Cloudは基本CPU。ローカルはGPUがあれば自動で使う）
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⑤ v3の前処理（あなたのコードをそのまま使用）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # [1,H,W] -> [3,H,W]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ⑥ モデルのロード（state_dict 保存/フルモデル保存 どちらにも対応）
def load_model() -> torch.nn.Module:
    # 初回のみDL。2回目以降はweights/から即読み込み
    w = ensure_weights(WEIGHTS_PATH, HF_URL, HF_SHA256)

    dev = _device()
    # A) torch.save(model) で保存されている場合
    try:
        m = torch.load(w, map_location=dev)
        m.eval()
        return m
    except Exception:
        # B) torch.save(model.state_dict()) の場合はこちら
        m = CNN()  # 必要なら引数（num_classes等）を合わせて
        state = torch.load(w, map_location="cpu")
        m.load_state_dict(state, strict=True)
        m.to(dev).eval()
        return m

# ⑦ 画像パスを受け取って予測（ファイル/BytesIOどちらでもOK）
@torch.inference_mode()
def predict(image_path_or_file) -> dict:
    dev = _device()
    model = load_model()

    # PILで開く（file-like でもパスでもOK）
    img = Image.open(image_path_or_file).convert("L")
    x = transform(img).unsqueeze(0).to(dev)  # [1,3,224,224]

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())

    # ★クラス割当は学習と必ず合わせる
    # いまは v3 = {0: kinoko, 1: takenoko} 前提
    labels = {0: "kinoko", 1: "takenoko"}
    return {"label": labels[idx], "confidence": float(probs[idx]), "probs": probs.tolist()}
