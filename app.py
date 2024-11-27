import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

# 1. Memuat Dataset
df = pd.read_csv('mbti_1.csv')

# Preprocessing: Membersihkan teks postingan
def clean_posts(posts):
    posts = posts.replace("|||", " ")  # Mengganti delimiter ||| dengan spasi
    return posts.lower()  # Mengubah menjadi huruf kecil

df['cleaned_posts'] = df['posts'].apply(clean_posts)

# 2. Vectorisasi TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_posts'])

# 3. Fungsi CBR
def find_similar_personality(input_text, tfidf_matrix, df, top_n=5):
    # Preprocessing input text
    cleaned_input = clean_posts(input_text)
    
    # Vectorisasi input teks
    input_vector = tfidf_vectorizer.transform([cleaned_input])
    
    # Menghitung kemiripan cosine
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
    
    # Mendapatkan indeks kemiripan tertinggi
    sorted_indices = np.argsort(similarity_scores)[::-1]  # Urutkan dari skor tertinggi ke terendah
    
    # Mengumpulkan hasil unik berdasarkan tipe kepribadian
    seen_types = set()  # Untuk melacak tipe yang sudah dimasukkan
    results = []
    
    for idx in sorted_indices:
        mbti_type = df.iloc[idx]['type']
        if mbti_type not in seen_types:  # Jika tipe belum dimasukkan
            results.append((mbti_type, similarity_scores[idx]))
            seen_types.add(mbti_type)  # Tandai tipe sebagai sudah dimasukkan
        if len(results) == top_n:  # Berhenti jika sudah mendapatkan top_n hasil unik
            break
    
    return results

# 4. Dictionary untuk kepanjangan MBTI
mbti_dict = {
    "INFJ": "Introversion, Intuition, Feeling, Judging",
    "INFP": "Introversion, Intuition, Feeling, Perceiving",
    "INTJ": "Introversion, Intuition, Thinking, Judging",
    "INTP": "Introversion, Intuition, Thinking, Perceiving",
    "ISFJ": "Introversion, Sensing, Feeling, Judging",
    "ISFP": "Introversion, Sensing, Feeling, Perceiving",
    "ISTJ": "Introversion, Sensing, Thinking, Judging",
    "ISTP": "Introversion, Sensing, Thinking, Perceiving",
    "ENFJ": "Extraversion, Intuition, Feeling, Judging",
    "ENFP": "Extraversion, Intuition, Feeling, Perceiving",
    "ENTJ": "Extraversion, Intuition, Thinking, Judging",
    "ENTP": "Extraversion, Intuition, Thinking, Perceiving",
    "ESFJ": "Extraversion, Sensing, Feeling, Judging",
    "ESFP": "Extraversion, Sensing, Feeling, Perceiving",
    "ESTJ": "Extraversion, Sensing, Thinking, Judging",
    "ESTP": "Extraversion, Sensing, Thinking, Perceiving"
}

# 5. Streamlit User Interface
def app():
    st.title('MBTI Personality Type Finder')

    # Input teks dari pengguna
    input_text = st.text_area("Masukkan Kalimat Anda:", "")
    
    # Tombol untuk mencari tipe MBTI
    if st.button('Cari Tipe MBTI'):
        if input_text.strip() != "":
            results = find_similar_personality(input_text, tfidf_matrix, df, top_n=5)

            # Menampilkan hasil dengan format yang diinginkan
            st.subheader("Hasil Tipe MBTI yang Mirip:")
            
            # Menampilkan tipe MBTI pertama dengan highlight
            mbti_type, score = results[0]
            st.success(f"**{mbti_type}** ({score:.4f})")
            st.write(f"**Kepanjangan Tipe MBTI:** {mbti_dict.get(mbti_type, 'Tipe tidak ditemukan dalam dictionary')}")

            # Menampilkan kemungkinan hasil lainnya
            with st.expander("Kemungkinan hasil lainnya:"):
                for mbti_type, score in results[1:]:
                    st.markdown(f"<small>{mbti_type} ({score:.4f})</small>", unsafe_allow_html=True)
                    st.write(f"<small>Kepanjangan: {mbti_dict.get(mbti_type, 'Tipe tidak ditemukan dalam dictionary')}</small>", unsafe_allow_html=True)

        else:
            st.warning("Harap masukkan kalimat untuk mencari tipe MBTI.")

if __name__ == '__main__':
    app()
