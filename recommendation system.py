import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("SpotifyFeatures.csv")
print(df.head())


df.drop_duplicates(subset='track_name',inplace=True)



# Remove non-numeric or unhelpful features
audio_features = ['danceability', 'energy', 'loudness','speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']  


# Drop rows with missing values
df = df.dropna(subset=audio_features)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
feature_data = scaler.fit_transform(df[audio_features])


from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(feature_data)
def recommend(track_name, df, similarity_matrix, top_n=5):
    if track_name not in df['track_name'].values:
        return "Track not found in dataset."
    
    idx = df[df['track_name'] == track_name].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1] 

    recommendations = []
    for i, score in sim_scores:
        recommendations.append({
            'track_name': df.iloc[i]['track_name'],
            'artist_name': df.iloc[i]['artist_name'],
            'genre': df.iloc[i]['genre'],
            'similarity': round(score, 2)
        })
    return pd.DataFrame(recommendations)

recommend("Shape of You", df, similarity_matrix, top_n=5)
