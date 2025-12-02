import glob
import os
import random
import time

import pandas as pd
import streamlit as st

# 设置页面
st.set_page_config(page_title="猫咪分类工具", layout="centered")

# --- 初始化状态 ---
if "image_list" not in st.session_state:
    # 定义所有需要测试的数据集来源
    dataset_sources = [("Original", "dataset/test")]
    
    # 添加对比度变体 (1-5)
    for i in range(1, 6):
        dataset_sources.append((f"Contrast_L{i}", f"dataset_variants/contrast_level_{i}/test"))
        
    # 添加噪声变体 (1-6)
    for i in range(1, 7):
        dataset_sources.append((f"Noise_L{i}", f"dataset_variants/noise_level_{i}/test"))

    categories = {
        "persian_cat": 0,
        "siamese_cat": 1,
    }

    img_list = []
    # 加载图片逻辑
    for condition, base_dir in dataset_sources:
        if not os.path.exists(base_dir):
            continue
            
        for cat, label in categories.items():
            p = os.path.join(base_dir, cat)
            if os.path.exists(p):
                files = [
                    os.path.join(p, f)
                    for f in os.listdir(p)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))
                ]
                for f in files:
                    # 保存 (路径, 标签, 条件)
                    img_list.append((f, label, condition))

    # 去重并打乱
    img_list = list(set(img_list))
    random.shuffle(img_list)

    st.session_state.image_list = img_list
    st.session_state.idx = 0
    st.session_state.results = []
    st.session_state.show_image = True  # 控制图片是否显示

# --- 结束逻辑 ---
if st.session_state.idx >= len(st.session_state.image_list):
    st.success("测试结束！")

    # 保存结果
    res_df = pd.DataFrame(st.session_state.results)
    if not res_df.empty:
        csv_path = "classification_results_web.csv"
        res_df.to_csv(csv_path, index=False)
        st.write(f"结果已保存至: `{csv_path}`")

        # 简单计算准确率
        acc = (res_df["True"] == res_df["Pred"]).mean()
        st.metric("最终准确率", f"{acc:.2%}")
        st.dataframe(res_df)
    st.stop()

# --- 获取当前图片 ---
current_img_path, current_true_label, current_condition = st.session_state.image_list[st.session_state.idx]

st.title(f"图片 {st.session_state.idx + 1} / {len(st.session_state.image_list)}")

# 定义黑色遮挡块的 HTML
black_box_html = """
    <div style="width:400px;height:400px;background-color:black;color:white;text-align:center;line-height:400px;margin:auto;">
    (图片已隐藏，请凭记忆分类)
    </div>
"""

# --- 核心逻辑：自动 500ms 隐藏 ---
# 使用 st.empty() 创建一个可变的容器
image_placeholder = st.empty()

if st.session_state.show_image:
    try:
        # 1. 先显示图片
        # use_container_width=False 配合 width=400 保持尺寸一致
        image_placeholder.image(
            current_img_path, caption="请快速记忆特征...", width=400
        )

        # 2. 暂停 0.5 秒 (500ms)
        time.sleep(0.5)

        # 3. 立即用黑块覆盖图片
        image_placeholder.markdown(black_box_html, unsafe_allow_html=True)

        # 4. 更新状态，这样如果用户随便点哪里触发刷新，图片也不会重新显示出来
        st.session_state.show_image = False

    except Exception as e:
        image_placeholder.error(f"无法加载: {current_img_path} ({e})")
else:
    # 如果状态已经是“不显示”，直接显示黑块
    image_placeholder.markdown(black_box_html, unsafe_allow_html=True)

st.write("### 请分类：")
col1, col2 = st.columns(2)


def next_step(pred):
    # 记录结果
    st.session_state.results.append(
        {
            "Image": current_img_path, 
            "True": current_true_label, 
            "Pred": pred, 
            "Condition": current_condition
        }
    )
    
    # 实时保存 (每做完一个就追加写入/覆盖写入)
    # 为了避免频繁IO影响性能，也可以每10个存一次，但这里数据量不大，实时存最安全
    temp_df = pd.DataFrame(st.session_state.results)
    temp_df.to_csv("classification_results_web.csv", index=False)
    
    # 切换下一张
    st.session_state.idx += 1
    # 设置下一张图片为“需要显示”，这样下次运行脚本开头时会进入 show_image=True 的逻辑
    st.session_state.show_image = True


with col1:
    if st.button("Persian Cat (波斯猫)", use_container_width=True):
        next_step(0)
        st.rerun()

with col2:
    if st.button("Siamese Cat (暹罗猫)", use_container_width=True):
        next_step(1)
        st.rerun()
