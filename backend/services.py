import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_legacy_sklearn_pickle_compatibility() -> None:
    """Support older sklearn pickles on newer sklearn releases.

    Some serialized ColumnTransformer objects reference the private class
    ``_RemainderColsList`` from older scikit-learn versions. Newer releases
    removed it from ``sklearn.compose._column_transformer``. Re-adding a
    minimal compatible stub lets ``joblib.load`` succeed without retraining.
    """
    try:
        from sklearn.compose import _column_transformer as column_transformer_module

        if not hasattr(column_transformer_module, '_RemainderColsList'):
            class _RemainderColsList(list):
                pass

            column_transformer_module._RemainderColsList = _RemainderColsList
    except Exception:
        # The environment may not have scikit-learn or may be using a version
        # with a different module layout. Fail-open here so app startup can
        # continue if there are no legacy pickles to load.
        pass


def _safe_joblib_load(path: str):
    _ensure_legacy_sklearn_pickle_compatibility()
    return joblib.load(path)


class MLService:
    def __init__(self):
        self.pipeline = None
        self.nn_recommender = None
        self.rec_scaler = None
        self.kmeans_mood = None
        self.mood_scaler = None
        self.catalog_df = None
        self.rec_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                             'instrumentalness', 'liveness', 'valence', 'tempo']
        self.load_artifacts()

    def load_artifacts(self):
        pipe_path = os.path.join(BASE_DIR, 'soundscope_popularity_pipeline.joblib')
        nn_path = os.path.join(BASE_DIR, 'soundscope_nn_recommender.joblib')
        rec_scaler_path = os.path.join(BASE_DIR, 'soundscope_rec_scaler.joblib')
        kmeans_path = os.path.join(BASE_DIR, 'soundscope_mood_kmeans.joblib')
        mood_scaler_path = os.path.join(BASE_DIR, 'soundscope_mood_scaler.joblib')
        catalog_path = os.path.join(BASE_DIR, 'soundscope_sample_tracks.csv')

        print("Loading ML models into memory...")
        self.pipeline = _safe_joblib_load(pipe_path)
        self.nn_recommender = _safe_joblib_load(nn_path)
        self.rec_scaler = _safe_joblib_load(rec_scaler_path)
        self.kmeans_mood = _safe_joblib_load(kmeans_path)
        self.mood_scaler = _safe_joblib_load(mood_scaler_path)

        catalog_parquet = os.path.join(BASE_DIR, 'soundscope_clean_catalog.parquet')
        catalog_csv = os.path.join(BASE_DIR, 'soundscope_sample_tracks.csv')

        if os.path.exists(catalog_parquet):
            self.catalog_df = pd.read_parquet(catalog_parquet)
        elif os.path.exists(catalog_csv):
            self.catalog_df = pd.read_csv(catalog_csv)
        else:
            self.catalog_df = pd.read_csv(os.path.join(BASE_DIR, 'dataset.csv')).head(15000)
        print("ML Service successfully initialized with models and catalog!")

    def detect_mood(self, valence: float, energy: float, danceability: float, acousticness: float) -> str:
        if acousticness >= 0.7:
            return 'Chill & Acoustic'
        elif valence >= 0.5 and energy >= 0.6:
            return 'Energetic & Euphoric'
        elif valence < 0.5 and energy >= 0.6:
            return 'Intense & Heavy'
        else:
            return 'Mellow & Reflective'

    def predict_popularity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        val = float(data.get('valence', 0.5))
        enrg = float(data.get('energy', 0.5))
        dance = float(data.get('danceability', 0.5))
        acoust = float(data.get('acousticness', 0.5))
        dur_min = float(data.get('duration_min', data.get('duration_ms', 180000.0) / 60000.0))
        art_count = int(data.get('artist_track_count', 5))

        mood = self.detect_mood(val, enrg, dance, acoust)
        
        input_data = {
            'danceability': [dance],
            'energy': [enrg],
            'loudness': [float(data.get('loudness', -6.0))],
            'speechiness': [float(data.get('speechiness', 0.05))],
            'acousticness': [acoust],
            'instrumentalness': [float(data.get('instrumentalness', 0.0))],
            'liveness': [float(data.get('liveness', 0.12))],
            'valence': [val],
            'tempo': [float(data.get('tempo', 120.0))],
            'duration_min': [dur_min],
            'artist_track_count': [art_count],
            'explicit': [bool(data.get('explicit', False))],
            'key': [int(data.get('key', 0))],
            'mode': [int(data.get('mode', 1))],
            'time_signature': [int(data.get('time_signature', 4))],
            'track_genre': [str(data.get('track_genre', 'pop'))],
            'mood_label': [mood]
        }
        
        df_input = pd.DataFrame(input_data)
        raw_pred = float(self.pipeline.predict(df_input)[0])
        score = round(max(0.0, min(100.0, raw_pred)), 2)

        if score >= 70.0:
            rating = "Viral / Hit Tier"
            tier = "High"
        elif score >= 45.0:
            rating = "Solid Streaming Appeal"
            tier = "Moderate"
        else:
            rating = "Niche / Underground Fit"
            tier = "Specialized"

        return {
            "predicted_popularity": score,
            "mood_label": mood,
            "commercial_rating": rating,
            "confidence_tier": tier
        }

    def simulate_what_if(self, track_name: Optional[str], baseline: Optional[Dict[str, Any]], 
                         modify_feature: str, new_value: float) -> Dict[str, Any]:
        if track_name:
            match = self.catalog_df[self.catalog_df['track_name'].str.lower() == track_name.lower()]
            if match.empty:
                match = self.catalog_df[self.catalog_df['track_name'].str.lower().str.contains(track_name.lower())]
            if match.empty:
                raise ValueError(f"Track '{track_name}' not found.")
            row = match.iloc[0]
            base_dict = row.to_dict()
            name_label = f"'{row['track_name']}' by {row['artists']}"
        elif baseline:
            base_dict = baseline
            name_label = "Custom Production Profile"
        else:
            raise ValueError("Must provide either a track_name or baseline_features.")

        orig_val = float(base_dict.get(modify_feature, 0.0))
        orig_res = self.predict_popularity(base_dict)
        orig_score = orig_res["predicted_popularity"]

        mod_dict = base_dict.copy()
        mod_dict[modify_feature] = new_value
        mod_res = self.predict_popularity(mod_dict)
        mod_score = mod_res["predicted_popularity"]

        return {
            "track_name": name_label,
            "feature_modified": modify_feature,
            "original_value": round(orig_val, 3),
            "simulated_value": round(float(new_value), 3),
            "original_prediction": orig_score,
            "simulated_prediction": mod_score,
            "popularity_delta": round(mod_score - orig_score, 2)
        }

    def recommend(self, track_name: str, top_n: int = 5) -> List[Dict[str, Any]]:
        match = self.catalog_df[self.catalog_df['track_name'].str.lower() == track_name.lower()]
        if match.empty:
            match = self.catalog_df[self.catalog_df['track_name'].str.lower().str.contains(track_name.lower())]
        if match.empty:
            raise ValueError(f"Track '{track_name}' not found in catalog.")

        query_row = match.iloc[0]
        query_vector = self.rec_scaler.transform(pd.DataFrame([query_row[self.rec_features]]))
        distances, indices = self.nn_recommender.kneighbors(query_vector, n_neighbors=top_n + 1)

        recommendations = []
        for dist, idx in zip(distances[0], indices[0]):
            row = self.catalog_df.iloc[idx]
            if row['track_id'] == query_row['track_id']:
                continue
            sim_score = round(float((1.0 - dist) * 100.0), 1)
            recommendations.append({
                "track": row['track_name'],
                "artist": row['artists'],
                "genre": row.get('track_genre', 'N/A'),
                "mood": row.get('mood_label', 'N/A'),
                "popularity": int(row['popularity']),
                "similarity": sim_score
            })
            if len(recommendations) >= top_n:
                break
        return recommendations

    def discover(self, genre: Optional[str] = None, mood: Optional[str] = None, 
                 min_danceability: float = 0.0, min_energy: float = 0.0, 
                 target_tempo: Optional[float] = None, max_results: int = 10) -> List[Dict[str, Any]]:
        df = self.catalog_df.copy()
        if genre and genre != "Any":
            df = df[df['track_genre'].str.lower() == genre.lower()]
        if mood and mood != "Any" and 'mood_label' in df:
            df = df[df['mood_label'].str.lower() == mood.lower()]
            
        df = df[(df['danceability'] >= min_danceability) & (df['energy'] >= min_energy)]

        if df.empty:
            return []

        if target_tempo and target_tempo > 0:
            df['tempo_diff'] = (df['tempo'] - target_tempo).abs()
            df = df.sort_values(by=['tempo_diff', 'popularity'], ascending=[True, False])
        else:
            df = df.sort_values(by='popularity', ascending=False)

        cols = ['track_name', 'artists', 'track_genre', 'mood_label', 'popularity', 'danceability', 'energy', 'tempo']
        avail = [c for c in cols if c in df.columns]
        results = df[avail].head(max_results).to_dict(orient='records')
        return results

    def get_genres(self) -> List[str]:
        if 'track_genre' in self.catalog_df:
            return sorted(self.catalog_df['track_genre'].dropna().unique().tolist())
        return ['pop', 'rock', 'dance', 'acoustic', 'chill', 'hip-hop']

    def search_tracks(self, query: str, limit: int = 8) -> List[Dict[str, str]]:
        if not query:
            return []
        matches = self.catalog_df[self.catalog_df['track_name'].str.lower().str.contains(query.lower())]
        results = []
        for _, row in matches.head(limit).iterrows():
            results.append({
                "track_name": row['track_name'],
                "artists": row['artists'],
                "genre": row.get('track_genre', '')
            })
        return results

ml_service = MLService()
