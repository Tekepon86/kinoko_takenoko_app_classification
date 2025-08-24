import streamlit as st
from PIL import Image
from inference import predict  # predict は dict を返す前提
import io
import time

st.set_page_config(page_title="きのこたけのこ判別 v3", layout="wide")
st.title("きのこたけのこ判別アプリ（画像分類） 🍄🎋")

uploaded_file = st.file_uploader(
    "画像をアップロードしてください", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # バイト→PIL画像（verifyで妥当性チェック）
        bytes_data = uploaded_file.read()
        img = Image.open(io.BytesIO(bytes_data))
        img.verify()  # ここで例外になったら不正画像
        img = Image.open(io.BytesIO(bytes_data))  # verify後は再オープンが必要

        st.image(img, caption="アップロードされた画像", use_column_width=True)

    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
        st.stop()

    if st.button("判定する", use_container_width=True):
        # 簡易プログレス（体感用）
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(100):
            latest_iteration.text(f"処理中... {i + 1}%")
            bar.progress(i + 1)
            time.sleep(0.01)

        # ★ ここがポイント：temp.jpg を作らず BytesIO のまま渡す
        res = predict(io.BytesIO(bytes_data))  # predict は PIL/BytesIO/Path に対応済みの想定
        # predict が dict を返す想定に合わせて表示
        st.success(f"予測結果: **{res['label']}**（確信度 {res['confidence']:.3f}）")
