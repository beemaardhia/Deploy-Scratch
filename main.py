import pandas as pd
import numpy as np
import streamlit as st
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from perpus import KNN
from perpus import tfidf
from perpus import split



stopwords_ind = stopwords.words('indonesian')

# Buat fungsi untuk langkah case folding
def casefolding(text):
    text = text.lower()   # Mengubah teks menjadi lower case    
    text = re.sub(r'#\w+\s*', '', text)
    text = re.sub(r'<[^>]+>|\n', ' ', text)            # Menghapus <USERNAME>                      
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Menghapus URL
    text = re.sub(r'[-+]?[0-9]+', '', text)           # Menghapus angka
    text = re.sub(r'[^\w\s]', '', text)               # Menghapus karakter tanda baca
    text = re.sub(r'\busername\b', 'username', text, flags=re.IGNORECASE)  # Mengganti 'username' dengan 'username' (case-insensitive)
    return text

key_norm = pd.read_csv('key_norm_indo.csv')

def text_normalize(text):
  text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if (key_norm['singkat'] == word).any() else word for word in text.split()])
  text = str.lower(text)
  return text

def remove_stop_words(text):
  clean_words = []
  text = text.split()
  for word in text:
      if word not in stopwords_ind:
          clean_words.append(word)
  return " ".join(clean_words)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Buat fungsi untuk langkah stemming bahasa Indonesia
def stemming(text):
  text = stemmer.stem(text)
  return text

def text_preprocessing_process(text):
  text = casefolding(text)
  text = text_normalize(text)
  text = remove_stop_words(text)
  text = stemming(text)
  return text


X_train = pd.read_csv("X.csv")
y_train = pd.read_csv("y.csv")

X = pd.read_csv("Xraw.csv")
X.drop("Unnamed: 0", axis=1, inplace=True )

df = pd.DataFrame(X)

# Menggunakan TFIDFVectorizer
vectorizer = tfidf()
vectorizer.fit(df['clean_teks'])

y_train.drop("Unnamed: 0", axis=1, inplace=True )
X_train.drop("Unnamed: 0", axis=1, inplace=True )




X_train = X_train.to_numpy()
y_train = y_train.to_numpy().ravel()  # Use ravel() if y_train is a single column to get a flat array



clf = KNN(k=3)
clf.fit(X_train, y_train)



# Mengimpor model dari file .joblib
# model = joblib.load('nlpJarinan.joblib')

def predict_sentiment(text):
    preprocessed_text = text_preprocessing_process(text)
    tf_idf_matrix = vectorizer.transform([preprocessed_text])
    result = clf.predict(tf_idf_matrix)
    return result[0]


# Tampilan Streamlit
st.header('Sentiment Analysis')

with st.expander('Analyze Text'):
    input_text = st.text_area('Text here:', '')  # Menerima input teks dari pengguna

    if st.button('Analyze'):  # Tombol untuk melakukan analisis
        if input_text:
            prediction = predict_sentiment(input_text)
            st.write('Hasil Prediksi:', prediction)

# Bagian untuk menganalisis file CSV yang diunggah
with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file', type='csv')
    

    if upl:        
        df = pd.read_csv(upl, on_bad_lines='skip')

        # Preprocessing pada teks di kolom 'tweets'
        df['preprocessed_text'] = df["Text"].apply(text_preprocessing_process)


        # Terapkan fungsi analisis sentimen pada setiap baris dalam DataFrame
        df['sentiment_analysis'] = df['preprocessed_text'].apply(predict_sentiment)

        combined_df = pd.concat([df[df['sentiment_analysis'] == 'positive'].head(), df[df['sentiment_analysis'] == 'negative'].head()])

        # Menampilkan DataFrame gabungan
        st.write(combined_df)

        # Mendownload hasil analisis sebagai file CSV
        @st.cache_data
        def convert_df(df):
            # Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment_analysis.csv',
            mime='text/csv',
        )