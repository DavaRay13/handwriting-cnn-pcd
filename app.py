import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Tulisan Tangan", page_icon="✍️")

# Mapping EMNIST Balanced yang Benar (Sesuai Paper EMNIST)
# 0-9: Angka
# 10-35: Huruf Besar (A-Z)
# 36-46: Huruf Kecil (abdefghnqrt)
LABEL_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

# Load Model dengan Caching biar cepet
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('handwriting_model.h5')
        return model
    except:
        return None

model = load_model()

st.title("✍️ AI Deteksi Tulisan Tangan")
st.subheader("Oleh: Muhamad Dava Rayhan - 231011402488")

col1, col2 = st.columns([1, 1])

with col1:
    st.write("Gambar di sini:")
    # Canvas input
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,       # Ketebalan kuas
        stroke_color="white",  # Warna kuas putih
        background_color="black", # Background hitam
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

def proses_gambar(image_data):
    # 1. Ambil channel warna saja & convert ke grayscale
    # image_data dari canvas itu formatnya RGBA
    img = image_data.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # 2. Resize ke 28x28 pixel
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 3. Normalisasi & Reshape
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    
    return img

with col2:
    st.write("Hasil Prediksi:")
    
    if model is None:
        st.error("⚠️ Model belum ditemukan! Jalanin 'python train_model.py' dulu bro.")
    else:
        if canvas_result.image_data is not None:
            # Tombol Trigger
            if st.button('🔍 Deteksi!'):
                # Ambil gambar
                img_array = canvas_result.image_data
                
                # Cek apakah canvas kosong (semua hitam)
                if np.sum(img_array) == 0:
                    st.warning("Canvas masih kosong, gambar dulu dong!")
                else:
                    # Preprocessing
                    processed_img = proses_gambar(img_array)
                    
                    # Prediksi
                    prediction = model.predict(processed_img)
                    class_idx = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    hasil = LABEL_MAP.get(class_idx, "Misterius")
                    
                    # Tampilan Hasil
                    st.success(f"Huruf/Angka: **{hasil}**")
                    st.info(f"Yakin segini: {confidence*100:.2f}%")
                    
                    # Debugging view (Biar dosen liat prosesnya)
                    with st.expander("Lihat Input Model (28x28)"):
                        st.image(processed_img.reshape(28, 28), width=100, clamp=True)
                        st.caption("Ini gambar yang dilihat oleh AI setelah di-resize.")

st.markdown("---")
st.caption("Dibuat dengan Python, TensorFlow & Streamlit di Linux Mint.")