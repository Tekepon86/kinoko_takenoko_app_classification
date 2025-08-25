# 🖼️ Image Classification: Kinoko vs Takenoko

PyTorch の ResNet18 を用いた **画像分類モデル** の構築プロジェクトです。  
市販菓子「きのこの山」「たけのこの里」を題材に、**独自データ収集 → 前処理 → 学習 → 精度改善** を実践しました。

---

## 📊 成果
- **ResNet18** をファインチューニングし、独自データセットで学習  
- **データ拡張（augmentation）** により精度を改善  
- **グレースケール学習** により形状ベースの分類性能を向上  
- **Test Accuracy 1.0** を達成（一部条件下）

---

## 🛠️ 使用技術
- Python (PyTorch, torchvision)
- ResNet18（pretrained on ImageNet）
- データ拡張（左右反転, 回転, 明度/コントラスト/彩度/色相の調整）
- Stratified-K Fold Cross Validation
- Early Stopping

---

## 📂 プロジェクト構成
随時記載予定



---

## 🔍 学習・改善プロセス
- **Baseline**: ResNet18をImageNet重みでファインチューニング → Test Acc ~0.90  
- **データ拡張**: Flip, Rotation, Brightness, Contrast など → Test Acc 0.95  
- **グレースケール化**: 色特徴を捨て、形状に基づく分類 → Test Acc 1.0  
- **不均衡データ対応**: Stratified-K fold を導入して安定評価

👉 詳細は [スライド資料](./slides/classification-process.pdf) をご覧ください

---

## 📌 今後の課題
- 背景が複雑な画像へのロバスト性向上  
- クラス数を増やした多分類タスクへの応用  
