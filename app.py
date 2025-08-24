import streamlit as st
from PIL import Image
from inference import predict
import io
import time

st.title("きのこたけのこ判別アプリ（画像分類） 🍄🎋")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # ファイルのバイトデータを一度読み込む
        bytes_data = uploaded_file.read()

        # バイトデータから画像を開く
        image = Image.open(io.BytesIO(bytes_data))
        image.verify()

        # 表示用に画像を再度開く（verifyで閉じてしまうため）
        image = Image.open(io.BytesIO(bytes_data))
        st.image(image, caption="アップロードされた画像", use_container_width=True)

    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
        st.stop()

    # ファイル保存
    with open("temp.jpg", "wb") as f:
        f.write(bytes_data)
    # プログレスバーの表示
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(100):
            latest_iteration.text(f'処理中... {i + 1}%')
            bar.progress(i + 1)
            time.sleep(0.05)  # 実際の処理時間に応じて調整
    # 推論
    label = predict("temp.jpg")
    st.success(f"予測結果: **{label}**")
