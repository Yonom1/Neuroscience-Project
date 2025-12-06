import os
import random
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="猫咪分类工具", layout="centered")

components.html(
    """
    <script>
    const doc = window.parent.document;
    
    if (window.parent._keyHandler) {
        doc.removeEventListener('keydown', window.parent._keyHandler);
    }

    window.parent._keyHandler = function(e) {
        if (e.key === 'ArrowLeft') {
            const buttons = Array.from(doc.querySelectorAll('button'));
            const btn = buttons.find(b => b.innerText.includes("Persian Cat"));
            if (btn) btn.click();
        } else if (e.key === 'ArrowRight') {
            const buttons = Array.from(doc.querySelectorAll('button'));
            const btn = buttons.find(b => b.innerText.includes("Siamese Cat"));
            if (btn) btn.click();
        }
    };

    doc.addEventListener('keydown', window.parent._keyHandler);
    </script>
    """,
    height=0,
    width=0,
)

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if "image_list" not in st.session_state:
    dataset_sources = [("Original", os.path.join(PROJECT_ROOT, "dataset", "test"))]
    
    for i in range(1, 6):
        dataset_sources.append((f"Contrast_L{i}", os.path.join(PROJECT_ROOT, "dataset_variants", f"contrast_level_{i}", "test")))
        
    for i in range(1, 7):
        dataset_sources.append((f"Noise_L{i}", os.path.join(PROJECT_ROOT, "dataset_variants", f"noise_level_{i}", "test")))

    for i in range(1, 5):
        dataset_sources.append((f"Jigsaw_L{i}", os.path.join(PROJECT_ROOT, "dataset_variants", f"jigsaw_level_{i}", "test")))

    for i in range(1, 6):
        dataset_sources.append((f"Eidolon_L{i}", os.path.join(PROJECT_ROOT, "dataset_variants", f"eidolon_level_{i}", "test")))

    categories = {
        "persian_cat": 0,
        "siamese_cat": 1,
    }

    img_list = []
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
                    img_list.append((f, label, condition))

    print(f"totally loaded {len(img_list)} images for testing.")

    img_list = list(set(img_list))
    random.shuffle(img_list)

    st.session_state.image_list = img_list
    st.session_state.idx = 0
    st.session_state.results = []
    st.session_state.show_image = True

if st.session_state.idx >= len(st.session_state.image_list):
    st.success("Testing completed! Thank you for your participation.")

    res_df = pd.DataFrame(st.session_state.results)
    if not res_df.empty:
        csv_path = "classification_results.csv"
        res_df.to_csv(csv_path, index=False)
        st.write(f"Results saved to: `{csv_path}`")

        acc = (res_df["True"] == res_df["Pred"]).mean()
        st.metric("Final Accuracy", f"{acc:.2%}")
        st.dataframe(res_df)

        summary = pd.read_csv(csv_path)
        TN = ((summary["True"] == 0) & (summary["Pred"] == 0)).sum()
        TP = ((summary["True"] == 1) & (summary["Pred"] == 1)).sum()
        FN = ((summary["True"] == 1) & (summary["Pred"] == 0)).sum()
        FP = ((summary["True"] == 0) & (summary["Pred"] == 1)).sum()
        summary_df = pd.DataFrame(
            [
                {
                    "Dataset": mark,
                    "Model": "human",
                    "TN": TN,
                    "FP": FP,
                    "FN": FN,
                    "TP": TP,
                }
            ]
        )

        os.remove(csv_path)

        old_summary = (
            pd.read_csv("summary_results.csv")
            if os.path.exists("summary_results.csv")
            else pd.DataFrame()
        )

        old_summary = pd.concat([old_summary, summary_df], ignore_index=True)
        summary_path = "summary_results.csv"
        old_summary.to_csv(summary_path, index=False)

    st.stop()

if st.session_state.get("needs_rest", False):
    st.info(f"Already completed {st.session_state.idx} images, take a break!")
    if st.button("Continue Testing"):
        st.session_state.needs_rest = False
        st.rerun()
    st.stop()

current_img_path, current_true_label, current_condition = st.session_state.image_list[st.session_state.idx]

st.title(f"图片 {st.session_state.idx + 1} / {len(st.session_state.image_list)}")

black_box_html = """
    <div style="width:400px;height:400px;background-color:black;color:white;text-align:center;line-height:400px;margin:auto;">
    (图片已隐藏，请凭记忆分类)
    </div>
"""


image_placeholder = st.empty()

if st.session_state.show_image:
    try:

        image_placeholder.image(
            current_img_path, caption="Please quickly memorize the features...", width=400
        )

        time.sleep(1)

        image_placeholder.markdown(black_box_html, unsafe_allow_html=True)

        st.session_state.show_image = False

    except Exception as e:
        image_placeholder.error(f"Failed to load: {current_img_path} ({e})")
else:
    image_placeholder.markdown(black_box_html, unsafe_allow_html=True)

st.write("### 请分类 (支持键盘 ← / →)：")
col1, col2 = st.columns(2)


def next_step(pred):
    st.session_state.results.append(
        {
            "Image": current_img_path, 
            "True": current_true_label, 
            "Pred": pred, 
            "Condition": current_condition
        }
    )
    
    temp_df = pd.DataFrame(st.session_state.results)
    temp_df.to_csv("classification_results_web.csv", index=False)
    
    st.session_state.idx += 1
    st.session_state.show_image = True
    
    if st.session_state.idx > 0 and st.session_state.idx % 50 == 0:
        st.session_state.needs_rest = True


with col1:
    if st.button("Persian Cat (波斯猫)", use_container_width=True):
        next_step(0)
        st.rerun()

with col2:
    if st.button("Siamese Cat (暹罗猫)", use_container_width=True):
        next_step(1)
        st.rerun()
