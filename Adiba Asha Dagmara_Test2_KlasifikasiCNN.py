import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image

ART = "artifacts"
st.set_page_config(page_title="Klasifikasi Kehidupan Laut (CNN)", page_icon="🐠", layout="wide")

# ---------- Load artefak ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(f"{ART}/oceanic_cnn.keras")

@st.cache_data
def load_all():
    rd = lambda n: json.load(open(f"{ART}/{n}"))
    return (rd("class_names.json"), rd("history.json"), rd("metrics.json"), rd("summary.json"),
            np.load(f"{ART}/confusion_matrix.npy"), pd.read_csv(f"{ART}/class_distribution.csv"))

class_names, hist, metrics, summ, cm, dist = load_all()
model = load_model()
S = summ["img_size"]

def grad_cam(img_batch, layer_name="last_conv"):
    gm = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = gm(tf.convert_to_tensor(img_batch))
        cls = int(tf.argmax(preds[0]))
        score = preds[:, cls]
    grads = tape.gradient(score, conv_out)
    w = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.nn.relu(tf.reduce_sum(conv_out[0] * w, axis=-1))
    heat = (heat / (tf.reduce_max(heat) + 1e-8)).numpy()
    return heat, preds[0].numpy()

def overlay(img, heat, alpha=0.45):
    h = tf.image.resize(heat[..., None], img.shape[:2]).numpy()[..., 0]
    return np.clip((1 - alpha) * img + alpha * plt.cm.jet(h)[..., :3], 0, 1)

# ---------- Header ----------
st.title("🐠 Klasifikasi Citra Kehidupan Laut dengan CNN")
st.markdown(
    "Dashboard ini menampilkan hasil eksperimen klasifikasi citra organisme laut "
    "(Oceanic Life Dataset, Kaggle) menggunakan **Convolutional Neural Network (CNN)**. "
    "Mulai dari ringkasan dataset, performa pelatihan, evaluasi model, hingga prediksi gambar baru secara langsung."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📦 Dataset", "📈 Pelatihan", "🧮 Evaluasi", "🔥 Grad-CAM", "🔍 Prediksi Gambar Baru"])

# ---------- Tab 1: Dataset ----------
with tab1:
    st.subheader("Ringkasan Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah kelas", summ["n_classes"])
    c2.metric("Total gambar", f"{summ['n_images']:,}")
    c3.metric("Data latih", f"{summ['n_train']:,}")
    c4.metric("Data uji", f"{summ['n_val']:,}")
    st.caption(f"Proporsi latih : uji = {summ['n_train']/summ['n_images']:.0%} : {summ['n_val']/summ['n_images']:.0%} "
               f"| Ukuran input model: {S}x{S} piksel RGB, piksel dinormalisasi 0-1.")

    if summ.get("label_source"):
        st.warning(f"Catatan: label kelas berasal dari {summ['label_source']}, bukan label manusia (ground truth). "
                   "Akurasi model mengukur kesesuaiannya dengan label tersebut.")

    st.subheader("Distribusi Data per Kelas")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(x="kelas", y="total", data=dist, palette="viridis", ax=ax)
    for i, v in enumerate(dist["total"]): ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Kelas"); ax.set_ylabel("Jumlah gambar"); plt.xticks(rotation=40, ha="right")
    st.pyplot(fig)
    top, low = dist.loc[dist["total"].idxmax()], dist.loc[dist["total"].idxmin()]
    st.info(f"**Analisis:** kelas terbanyak adalah **{top['kelas']}** ({top['total']} gambar) dan tersedikit **{low['kelas']}** "
            f"({low['total']} gambar), rasio {summ['imbalance_ratio']:.2f}x. " +
            ("Data **tidak seimbang**, sehingga model berpotensi bias ke kelas mayoritas; "
             "karena itu dipakai class weight dan augmentasi." if summ["imbalance_ratio"] > 1.5
             else "Data relatif **seimbang**, sehingga risiko bias kelas mayoritas kecil."))
    st.dataframe(dist.rename(columns={"kelas": "Kelas", "total": "Total", "train": "Latih", "val": "Uji"}),
                 use_container_width=True, hide_index=True)

# ---------- Tab 2: Pelatihan ----------
with tab2:
    st.subheader("Grafik Akurasi dan Loss")
    ep = range(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(ep, hist["accuracy"], "o-", label="Train"); ax[0].plot(ep, hist["val_accuracy"], "o-", label="Validation")
    ax[0].set_title("Akurasi"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(ep, hist["loss"], "o-", label="Train"); ax[1].plot(ep, hist["val_loss"], "o-", label="Validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    st.pyplot(fig)
    best_ep = int(np.argmax(hist["val_accuracy"])) + 1
    gap = hist["accuracy"][-1] - hist["val_accuracy"][-1]
    st.info(f"**Analisis:** akurasi validasi tertinggi **{max(hist['val_accuracy']):.2%}** pada epoch **{best_ep}** "
            f"(total {len(hist['loss'])} epoch). Selisih akurasi train-validasi di akhir pelatihan {gap:.2%} -> " +
            ("terindikasi **overfitting** (model terlalu hafal data latih)." if gap > 0.1
             else "model **generalisasi dengan baik** dan kurva sudah cenderung konvergen."))

# ---------- Tab 3: Evaluasi ----------
with tab3:
    st.subheader("Hasil Evaluasi Model")
    m = metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.2%}")
    c2.metric("Precision (macro)", f"{m['macro']['precision']:.2%}")
    c3.metric("Recall (macro)", f"{m['macro']['recall']:.2%}")
    c4.metric("F1-Score (macro)", f"{m['macro']['f1']:.2%}")

    rep = pd.DataFrame(m["report"]).T.loc[class_names, ["precision", "recall", "f1-score", "support"]].round(4)
    st.dataframe(rep, use_container_width=True)
    best, worst = rep["f1-score"].idxmax(), rep["f1-score"].idxmin()
    st.info(f"**Analisis:** akurasi {m['accuracy']:.2%} menunjukkan proporsi prediksi benar secara keseluruhan. "
            f"F1 macro {m['macro']['f1']:.3f} (weighted {m['weighted']['f1']:.3f}) menggambarkan keseimbangan precision dan recall. "
            f"Kelas paling mudah dikenali: **{best}** (F1 {rep.loc[best,'f1-score']:.3f}); "
            f"paling sulit: **{worst}** (F1 {rep.loc[worst,'f1-score']:.3f}).")

    st.subheader("Confusion Matrix")
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax[0])
    ax[0].set_title("Jumlah")
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Greens", xticklabels=class_names, yticklabels=class_names, ax=ax[1])
    ax[1].set_title("Dinormalisasi per kelas asli")
    for a in ax: a.set_xlabel("Prediksi"); a.set_ylabel("Label asli"); a.tick_params(axis="x", rotation=40)
    st.pyplot(fig)
    off = cm.copy(); np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    st.info(f"**Analisis:** diagonal yang gelap berarti prediksi benar. Kesalahan terbanyak terjadi saat "
            f"**{class_names[i]}** diprediksi sebagai **{class_names[j]}** ({off[i, j]} gambar), "
            f"kemungkinan karena kemiripan visual (warna/bentuk/latar) kedua kelas.")

# ---------- Tab 4: Grad-CAM penjelasan ----------
with tab4:
    st.subheader("Feature Importance (Grad-CAM)")
    st.markdown(
        "Grad-CAM menyorot area gambar yang paling memengaruhi keputusan model. "
        "**Area merah/kuning** = pengaruh besar, **biru** = pengaruh kecil. "
        "Visualisasi ini dihitung langsung untuk gambar yang kamu unggah di tab **Prediksi Gambar Baru**."
    )
    st.info("**Analisis:** jika area merah berada pada objek (ikan, karang, dll.), model belajar dari fitur yang benar. "
            "Jika banyak berada di latar belakang (air/pasir), model mungkin mengandalkan konteks sehingga rawan salah pada gambar baru.")

# ---------- Tab 5: Prediksi ----------
with tab5:
    st.subheader("Upload Gambar untuk Prediksi")
    up = st.file_uploader("Pilih gambar (JPG/PNG/WEBP)", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if up:
        img = Image.open(up).convert("RGB")
        arr = np.asarray(img.resize((S, S)), dtype="float32") / 255.0
        heat, probs = grad_cam(arr[None])
        k = int(probs.argmax())

        c1, c2 = st.columns(2)
        c1.image(img, caption="Gambar yang diunggah", use_container_width=True)
        c2.image(overlay(arr, heat), caption="Grad-CAM (area fokus model)", use_container_width=True)

        st.success(f"### Prediksi: **{class_names[k]}**  ({probs[k]:.2%})")
        prob_df = pd.DataFrame({"Kelas": class_names, "Probabilitas": probs}).sort_values("Probabilitas", ascending=False)
        st.bar_chart(prob_df.set_index("Kelas"))

        second = prob_df.iloc[1]
        msg = (f"**Analisis:** model memprediksi **{class_names[k]}** dengan keyakinan {probs[k]:.1%}. "
               f"Kandidat terdekat kedua: {second['Kelas']} ({second['Probabilitas']:.1%}). ")
        msg += ("Keyakinan rendah, gambar mungkin ambigu/berbeda dari data latih." if probs[k] < 0.6
                else "Keyakinan tinggi, ciri visual objek cukup jelas bagi model.")
        st.info(msg)
    else:
        st.write("Belum ada gambar. Unggah gambar organisme laut untuk melihat prediksi secara real-time.")