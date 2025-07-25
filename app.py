import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\VS\.vscode\Python\animes0.csv")
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['score'] = df['score'].fillna(df['score'].mean())
    
    # Safely parse genres
    def parse_genres(x):
        try:
            return ' '.join(ast.literal_eval(x.lower()))
        except:
            return ''
    
    df['genres'] = df['genres'].apply(parse_genres)
    df['genres'] = df['genres'].fillna('')

    return df


df = load_data()

# Vectorize genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])

# Fit Nearest Neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(tfidf_matrix)

# Mapping titles to indices (case-insensitive and unique)
anime_indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

# Recommendation function
def get_recommendations(title, top_n=5):
    title = title.lower()
    if title not in anime_indices:
        return None
    idx = anime_indices[title]
    vec = tfidf_matrix[idx]
    distances, indices = model.kneighbors(vec, n_neighbors=top_n + 1)
    indices = indices.flatten()[1:]
    return df.iloc[indices][['title', 'score', 'genres', 'mal_url']]

# Streamlit UI
st.title("🎌 Anime Recommendation System")
st.write("Find similar anime based on genres!")

selected_title = st.selectbox("Choose an Anime", sorted(df['title'].unique()))

if st.button("Get Recommendations"):
    results = get_recommendations(selected_title)
    if results is not None:
        for _, row in results.iterrows():
            st.subheader(f"{row['title']} (⭐ {row['score']:.2f})")
            st.write(f"Genres: {row['genres']}")
            st.markdown(f"[🌐 MyAnimeList Page]({row['mal_url']})")
    else:
        st.warning("Anime not found!")
