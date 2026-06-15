import streamlit as st
import pickle
from PIL import Image
import numpy as np

# 加载训练好的图像分类模型
with open("model.pkl", "rb") as f:
    clf = pickle.load(f)

st.title("实验五 图像分类在线演示")
upload_img = st.file_uploader("上传图片（jpg/png）", type=["jpg", "png"])

if upload_img:
    img = Image.open(upload_img)
    st.image(img)
    # 把图片转为模型输入格式，28,28改成你实验里的图片尺寸
    img_resized = img.resize((28, 28))
    data = np.array(img_resized).flatten().reshape(1, -1)
    pred = clf.predict(data)
    st.write(f"预测分类结果：{pred[0]}")