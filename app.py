import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("movies.csv")

# Fill missing values
selected_features = ["genres", "keywords", "tagline", "cast", "director", "overview"]
for feature in selected_features:
    df[feature] = df[feature].fillna("")

# Combine selected features
df["combined_features"] = (
    df["genres"] + " " + df["keywords"] + " " + df["tagline"] + " " +
    df["cast"] + " " + df["director"] + " " + df["overview"]
)

# Convert combined text data to numerical vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(df["combined_features"])

# Compute cosine similarity matrix
similarity = cosine_similarity(feature_vectors)

# Streamlit app
st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorites!")

# User input
input_movie = st.text_input("Enter a movie title")

if st.button("Recommend"):
    if input_movie.strip():
        # Convert titles to lowercase for matching
        df["title"] = df["title"].str.lower()
        input_movie = input_movie.lower()

        # Find the closest movie match
        find_close_match = difflib.get_close_matches(input_movie, df["title"])

        if find_close_match:
            close_match = find_close_match[0]
            st.success(f"Closest match found: **{close_match.title()}**")

            # Get movie index and similarity scores
            index_of_movie = df[df.title == close_match].index[0]
            similarity_score = list(enumerate(similarity[index_of_movie]))
            sorted_similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)

            # Show recommendations
            st.subheader("Top 10 Recommended Movies:")
            for i, movie in enumerate(sorted_similarity_score[1:11]):
                index = movie[0]
                title_from_index = df.iloc[index]["title"].title()
                st.write(f"{i + 1}. {title_from_index}")
        else:
            st.error("Movie not found. Please try another title.")
    else:
        st.warning("Please enter a movie title!")
