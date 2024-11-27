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

# 4. Dictionary untuk kepanjangan MBTI dan deskripsi singkat
mbti_dict = {
    "INFJ": ("Introversion, Intuition, Feeling, Judging", "INFJs are insightful, empathetic, and deeply concerned about their relationships and the world."),
    "INFP": ("Introversion, Intuition, Feeling, Perceiving", "INFPs are idealistic, introspective, and driven by their values."),
    "INTJ": ("Introversion, Intuition, Thinking, Judging", "INTJs are strategic, determined, and highly independent thinkers."),
    "INTP": ("Introversion, Intuition, Thinking, Perceiving", "INTPs are innovative, curious, and prefer abstract concepts over practical applications."),
    "ISFJ": ("Introversion, Sensing, Feeling, Judging", "ISFJs are dedicated, practical, and quietly supportive individuals."),
    "ISFP": ("Introversion, Sensing, Feeling, Perceiving", "ISFPs are artistic, spontaneous, and enjoy living in the moment."),
    "ISTJ": ("Introversion, Sensing, Thinking, Judging", "ISTJs are responsible, organized, and value traditions."),
    "ISTP": ("Introversion, Sensing, Thinking, Perceiving", "ISTPs are logical, analytical, and enjoy problem-solving."),
    "ENFJ": ("Extraversion, Intuition, Feeling, Judging", "ENFJs are charismatic, supportive, and enjoy helping others grow."),
    "ENFP": ("Extraversion, Intuition, Feeling, Perceiving", "ENFPs are enthusiastic, creative, and motivated by their passions."),
    "ENTJ": ("Extraversion, Intuition, Thinking, Judging", "ENTJs are confident, assertive, and excel at leadership and organization."),
    "ENTP": ("Extraversion, Intuition, Thinking, Perceiving", "ENTPs are inventive, curious, and excel at debating new ideas."),
    "ESFJ": ("Extraversion, Sensing, Feeling, Judging", "ESFJs are warm, sociable, and driven by a desire to help others."),
    "ESFP": ("Extraversion, Sensing, Feeling, Perceiving", "ESFPs are outgoing, fun-loving, and enjoy living in the present."),
    "ESTJ": ("Extraversion, Sensing, Thinking, Judging", "ESTJs are organized, practical, and value order and structure."),
    "ESTP": ("Extraversion, Sensing, Thinking, Perceiving", "ESTPs are energetic, adventurous, and excel in fast-paced environments.")
}

# 5. Streamlit User Interface
def app():
    st.title('MBTI Personality Type Finder')

    # Input teks dari pengguna
    input_text = st.text_area("Input text:", "")
    
    # Tombol untuk mencari tipe MBTI
    if st.button('Find MBTI Type'):
        if input_text.strip() != "":
            results = find_similar_personality(input_text, tfidf_matrix, df, top_n=5)

            # Menampilkan hasil dengan format yang diinginkan
            st.subheader("MBTI Result:")
            
            # Menampilkan tipe MBTI pertama dengan highlight
            mbti_type, score = results[0]
            st.success(f"**{mbti_type}** ({mbti_dict.get(mbti_type, ('Tipe tidak ditemukan dalam dictionary', 'Tidak ada deskripsi tersedia'))[0]}), Similarity Score: {score:.4f}")
            st.success(f"{mbti_dict.get(mbti_type, ('Tipe tidak ditemukan dalam dictionary', 'Tidak ada deskripsi tersedia'))[1]}")

            # Menampilkan kemungkinan hasil lainnya
            with st.expander("Kemungkinan hasil lainnya:"):
                for mbti_type, score in results[1:]:
                    st.write(f"{mbti_type} ({mbti_dict.get(mbti_type, ('Tipe tidak ditemukan dalam dictionary', 'Tidak ada deskripsi tersedia'))[0]}), Similarity Score: {score:.4f}")

        else:
            st.warning("Harap masukkan kalimat untuk mencari tipe MBTI.")

if __name__ == '__main__':
    app()
