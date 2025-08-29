# 🖼️ Image Classification: Kinoko vs Takenoko

PyTorch の ResNet18 を用いた **画像分類モデル** の構築プロジェクトです。  
市販菓子「きのこの山」「たけのこの里」を題材に、**独自データ収集 → 前処理 → 学習 → 精度改善** を実践しました。

---

## 📊 成果
- ResNet18 をファインチューニングし、独自データセットで学習
- データ拡張により未知データへの耐性を向上（汎化性能の改善）
- グレースケール学習で色依存を抑え、形状ベースの識別が安定
- 一部条件下のテストで Accuracy=1.0（シンプル画像中心）  ※詳細はスライド参照
  アプリ: https://kinokotakenokoapp-nwqnhp3hpw2wgnx7dbk23y.streamlit.app/）
---

## 🛠️ 使用技術
- Python (PyTorch, torchvision), ResNet18 (ImageNet pretrained)
- Augmentation（左右反転/回転/明度・コントラスト・彩度・色相）
- Stratified K-Fold による汎化評価の安定化、Early Stopping

---

## 📂 プロジェクト構成
随時記載予定



---

## 🔍 学習・改善プロセス
# 🔍 学習・改善プロセス
- Baseline → Augmentation → Grayscale の順に段階的に改善
- 「分類の限界（複数物体・複雑背景）」は別途 YOLO で対応
- 詳細は[スライド資料参照](https://github.com/Tekepon86/kinoko_takenoko_app_classification/raw/main/docs/app_slide.pdf)
---

## 📌 今後の課題
- 背景が複雑な画像へのロバスト性向上  
- クラス数を増やした多分類タスクへの応用  
