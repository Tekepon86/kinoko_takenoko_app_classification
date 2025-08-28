# 🖼️ Image Classification: Kinoko vs Takenoko

PyTorch の ResNet18 を用いた **画像分類モデル** の構築プロジェクトです。  
市販菓子「きのこの山」「たけのこの里」を題材に、**独自データ収集 → 前処理 → 学習 → 精度改善** を実践しました。

---

## 📊 成果
- **ResNet18** をファインチューニングし、独自データセットで学習  
- **データ拡張（augmentation）** により精度を改善  
- **グレースケール学習** により形状ベースの分類性能を向上  
- **Test Accuracy 1.0** を達成（一部条件下）
- **アプリはこちら（https://kinokotakenokoapp-nwqnhp3hpw2wgnx7dbk23y.streamlit.app/）
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

👉 詳細は [スライド資料]（https://github.com/Tekepon86/kinoko_takenoko_app_classification/raw/main/docs/%E3%80%90%E7%94%BB%E5%83%8F%E6%A4%9C%E7%9F%A5%E3%82%A2%E3%83%97%E3%83%AA%E3%80%91%E3%82%B9%E3%83%A9%E3%82%A4%E3%83%89_github%EF%BC%88%E8%A6%81%E7%B4%84%E7%89%88%EF%BC%89.pdf）をご覧ください

---

## 📌 今後の課題
- 背景が複雑な画像へのロバスト性向上  
- クラス数を増やした多分類タスクへの応用  
