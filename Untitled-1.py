# Jalankan dengan: streamlit run app_statis.py
# Taruh file "pr_curve.png" di folder yang sama dengan script ini.
#
# Versi STATIS -- menampilkan ulang hasil analisis dari notebook (Untitled-1.ipynb).
# Tidak perlu upload CSV, model.pkl, maupun scaler.pkl. Semua angka di bawah adalah
# hasil asli dari proses training yang sudah dijalankan di notebook.
#
# Dependensi: streamlit, pandas, matplotlib, numpy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Deteksi Fraud Kartu Kredit", layout="wide")


def plot_confusion_matrix(cm, title, ax):
    """Gambar confusion matrix langsung dari angka asli (bukan file gambar),
    supaya selalu tajam di resolusi layar berapa pun."""
    im = ax.imshow(cm, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "black" if val > vmax * 0.5 else "yellow"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=11)
    return im


st.title("Deteksi Fraud Kartu Kredit")
st.caption("Ringkasan statis hasil eksperimen SVM dari notebook -- tanpa upload data atau model.")

# =========================================================================
# 1. Ringkasan dataset
# =========================================================================
st.header("1. Ringkasan Dataset")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total baris awal", "284,807")
c2.metric("Baris duplikat", "1,081")
c3.metric("Baris setelah dedup", "283,726")
c4.metric("Missing values", "0")

st.subheader("Distribusi Kelas (setelah hapus duplikat)")
class_df = pd.DataFrame({"Kelas": ["Normal (0)", "Fraud (1)"], "Jumlah": [283253, 473]})
st.bar_chart(class_df.set_index("Kelas"))
st.caption(
    f"Transaksi fraud hanya **{473 / 283726 * 100:.3f}%** dari seluruh data "
    "-> dataset sangat tidak seimbang (imbalanced)."
)

# =========================================================================
# 2. Setup eksperimen
# =========================================================================
st.header("2. Setup Eksperimen")
st.markdown(
    """
- **Fitur**: `V1`-`V28`, `Time`, `Amount`
- **Target**: `Class` (0 = normal, 1 = fraud)
- **Split**: 80% train / 20% test, distratifikasi berdasarkan kelas
- **Scaling**: `StandardScaler`
- **Model**: `SVC` (SVM), dibandingkan dua skenario:
  - **Baseline** -- dilatih pada data asli (imbalanced)
  - **Undersampling** -- dilatih pada data yang diseimbangkan dengan `RandomUnderSampler`
"""
)

# =========================================================================
# 3. Perbandingan metrik
# =========================================================================
st.header("3. Perbandingan Metrik")

metrics_df = pd.DataFrame(
    {
        "Metrik": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Baseline": [0.9994, 0.9846, 0.6737, 0.8000],
        "Undersampling (RUS)": [0.9824, 0.0742, 0.8316, 0.1363],
    }
)

mcol1, mcol2 = st.columns([1, 1])
with mcol1:
    st.dataframe(
        metrics_df.style.format({"Baseline": "{:.4f}", "Undersampling (RUS)": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
with mcol2:
    st.bar_chart(metrics_df.set_index("Metrik"))

st.info(
    "**Insight**: Baseline punya accuracy & precision sangat tinggi tapi recall rendah "
    "(banyak transaksi fraud yang lolos tidak terdeteksi). Undersampling menaikkan recall "
    "(menangkap lebih banyak fraud) tapi precision-nya anjlok drastis (banyak transaksi "
    "normal yang salah ditandai sebagai fraud)."
)

# =========================================================================
# 4. Confusion Matrix (digambar langsung dari angka asli)
# =========================================================================
st.header("4. Confusion Matrix")
st.caption("Data uji: n = 56.746 (hasil split 80/20 stratifikasi)")

cm_baseline = np.array([[56650, 1], [31, 64]])
cm_rus = np.array([[55666, 985], [16, 79]])

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
im0 = plot_confusion_matrix(cm_baseline, "Baseline", axes[0])
im1 = plot_confusion_matrix(cm_rus, "Undersampling", axes[1])
fig.colorbar(im0, ax=axes[0])
fig.colorbar(im1, ax=axes[1])
fig.tight_layout()
st.pyplot(fig, use_container_width=False)
plt.close(fig)

cm1, cm2 = st.columns(2)
with cm1:
    st.markdown("**Baseline**")
    st.table(
        pd.DataFrame(
            cm_baseline,
            index=["Aktual: Normal", "Aktual: Fraud"],
            columns=["Prediksi: Normal", "Prediksi: Fraud"],
        )
    )
with cm2:
    st.markdown("**Undersampling**")
    st.table(
        pd.DataFrame(
            cm_rus,
            index=["Aktual: Normal", "Aktual: Fraud"],
            columns=["Prediksi: Normal", "Prediksi: Fraud"],
        )
    )

# =========================================================================
# 5. Precision-Recall Curve
# =========================================================================
st.header("5. Precision-Recall Curve")
st.image(
    "pr_curve.png",
    caption="Kurva Precision-Recall -- Baseline vs Undersampling",
    width=567,  # ukuran asli, biar tidak buram
)
st.caption(
    "Garis putus-putus abu-abu menunjukkan baseline model acak "
    "(proporsi kelas fraud pada data uji)."
)

# =========================================================================
# 6. Kesimpulan
# =========================================================================
st.header("6. Kesimpulan")
st.markdown(
    """
- Model **baseline (tanpa balancing)** unggul di precision & accuracy -- cocok jika biaya
  *false alarm* (menandai transaksi normal sebagai fraud) dianggap mahal.
- Model **dengan undersampling** unggul di recall -- cocok jika prioritas utama adalah
  **menangkap sebanyak mungkin transaksi fraud**, meski konsekuensinya banyak *false positive*.
- Pemilihan model akhir sebaiknya disesuaikan dengan *cost-benefit* operasional
  (biaya investigasi manual vs kerugian akibat fraud yang lolos terdeteksi).
"""
)

st.caption(
    "Halaman ini statis: seluruh angka diambil langsung dari hasil training "
    "di notebook (Untitled-1.ipynb) dan tidak dihitung ulang saat aplikasi berjalan."
)