import streamlit as st
from transformers import pipeline
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt

# Memuat pipeline analisis sentimen dari transformers
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to create a circular image
def make_circular_image(image_path):
    try:
        img = Image.open(image_path).convert("RGBA")
        size = (min(img.size),) * 2  # Ensure the mask is a square
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0) + size, fill=255)
        output = ImageOps.fit(img, size, centering=(0.5, 0.5))
        output.putalpha(mask)
        return output
    except Exception as e:
        st.error(f"Error creating circular image: {e}")
        return None

# Display logo at the top of each menu
def display_logo():
    st.image("Logofix.png", width=200)

# Sidebar for navigation
st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih Menu", ["Deskripsi", "Analisis Sentimen", "Our Team"])

if menu == "Deskripsi":
    display_logo()
    st.title("Deskripsi ESG")
    st.write(""" ESG merupakan panduan harus diterapkan oleh perusahaan yang ingin berinvestasi dengan mempertimbangkan aspek lingkungan, sosial, dan tata kelola. Konsep ini digunakan sebagai acuan  ukur untuk mengevaluasi dampak sosial dan keberlanjutan dari investasi yang dilakukan oleh perusahaan. 
    """)
    st.title("Deskripsi Proyek")
    st.write("""
    ### Proyek Analisis Sentimen ESG
    Proyek ini bertujuan untuk menganalisis sentimen dari teks yang diberikan.
    Dengan menggunakan model pembelajaran mesin dari transformers, kita dapat mengidentifikasi sentimen positif, negatif, atau netral dari teks yang diinput.
    
    Data proyek ini adalah bertipe One-Hot Encoding : Dataset dengan angka 0 dan 1 juga dapat mengindikasikan penggunaan one-hot encoding, di mana setiap kolom mewakili satu kategori dan memiliki nilai 0 atau 1 untuk menunjukkan kehadiran atau ketiadaan kategori tersebut. Misalnya, jika ada kolom-kolom seperti "Kategori Positif", "Kategori Netral ", dan "Kategori Negatif", dengan angka 0 dan 1, maka nilai 0 menunjukkan bahwa kategori tersebut tidak ada atau netral, sementara nilai 1 menunjukkan bahwa kategori tersebut ada.

    """)
    st.image("poster.png", use_column_width=True)

elif menu == "Analisis Sentimen":
    display_logo()
    st.title("Dashboard Analisis Sentimen")
    st.write("""Ini adalah aplikasi untuk analisis sentimen dengan Topik ESG""")

    st.header("Prediksi Kalimat")
    user_input = st.text_area("Masukkan teks untuk analisis sentimen")
    
    if st.button("Analisis"):
        if user_input:
            # Melakukan analisis sentimen
            result = sentiment_pipeline(user_input)
            sentiment = result[0]['label']
            confidence = result[0]['score']
            # Menampilkan hasil analisis sentimen
            st.write(f"*Sentimen:* {sentiment} dengan tingkat keyakinan {confidence:.2f}")
        else:
            st.error("Masukkan teks untuk dianalisis.")

    st.header("Prediksi File")
    uploaded_file = st.file_uploader("Upload file CSV untuk analisis sentimen dalam jumlah banyak", type=["csv"])

    if uploaded_file:
        try:
            # Read the uploaded file, skip bad lines
            df = pd.read_csv(uploaded_file, on_bad_lines='skip')

            # Check if the expected column exists
            if 'text' not in df.columns:
                st.error("File CSV harus memiliki kolom 'text'")
            else:
                # Ensure all values in 'text' column are strings
                df['text'] = df['text'].astype(str)

                # Predict sentiment for each text
                df['Prediksi Sentimen'] = df['text'].apply(lambda x: sentiment_pipeline(x)[0]['label'] if x else 'UNKNOWN')

                # Display the results
                st.write("Hasil Prediksi Sentimen:")
                st.write(df)

                # Plot the pie chart
                sentiment_counts = df['Prediksi Sentimen'].value_counts()
                fig, ax = plt.subplots()
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                st.pyplot(fig)
        except pd.errors.ParserError as e:
            st.error(f"Error reading CSV file: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

elif menu == "Our Team":
    display_logo()
    st.title("Our Team Gcoder")
    col1, col2 = st.columns(2)

    with col1:
        img = make_circular_image("team1.png")
        if img:
            st.image(img, use_column_width=True)
        st.write("**Denisa Septalian**")
        st.write("Project Leader & Analist.")

        img = make_circular_image("team2.png")
        if img:
            st.image(img, use_column_width=True)
        st.write("**Lintang Karunia A.**")
        st.write("Visualization.")

    with col2:
        img = make_circular_image("team3.png")
        if img:
            st.image(img, use_column_width=True)
        st.write("**Bernardinus Rico**")
        st.write("Modeler 1.")

        img = make_circular_image("team4.png")
        if img:
            st.image(img, use_column_width=True)
        st.write("**Khalid Jundullah**")
        st.write("Modeler 2.")
