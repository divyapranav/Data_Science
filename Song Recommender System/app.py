from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and preprocessor
with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

data = pd.read_csv("dataset.csv")
data['track_name_lower'] = data['track_name'].str.lower()
song_to_index = pd.Series(data.index, index=data['track_name_lower'])

def recommend_songs(song_name, n=5):
    song_name = song_name.lower()

    # Normalize track names in dataset
    data['track_name_lower'] = data['track_name'].str.lower()

    matches = data[data['track_name_lower'] == song_name]

    if matches.empty:
        return [f"No match found for '{song_name}' in dataset."]

    idx = matches.index[0]

    input_row = data.loc[[idx], ['artists', 'danceability', 'energy', 'speechiness',
                                 'acousticness', 'instrumentalness', 'track_genre']]
    
    input_features = preprocessor.transform(input_row)
    distances, indices = knn_model.kneighbors(input_features, n_neighbors=n+1)

    results = []
    for i in range(1, len(indices[0])):
        track = data.iloc[indices[0][i]]
        results.append(f"{track['track_name']} - {track['artists']}")
    
    return results


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    error = None

    if request.method == "POST":
        song_name = request.form.get("song")
        if song_name:
            recommendations = recommend_songs(song_name)
            if not recommendations:
                error = "Song not found. Please check the name."

    return render_template("index.html", recommendations=recommendations, error=error)

if __name__ == "__main__":
    app.run(debug=True)
