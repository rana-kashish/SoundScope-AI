import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib


def _ensure_legacy_sklearn_pickle_compatibility():
    try:
        from sklearn.compose import _column_transformer as column_transformer_module

        if not hasattr(column_transformer_module, '_RemainderColsList'):
            class _RemainderColsList(list):
                pass

            column_transformer_module._RemainderColsList = _RemainderColsList
    except Exception:
        pass


def safe_joblib_load(path):
    _ensure_legacy_sklearn_pickle_compatibility()
    return joblib.load(path)


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SoundScope AI - Music Discovery & Popularity Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1DB954;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1DB954;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CACHED RESOURCE LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    pipe = safe_joblib_load('soundscope_popularity_pipeline.joblib')
    nn = safe_joblib_load('soundscope_nn_recommender.joblib')
    rec_scaler = safe_joblib_load('soundscope_rec_scaler.joblib')
    kmeans = safe_joblib_load('soundscope_mood_kmeans.joblib')
    mood_scaler = safe_joblib_load('soundscope_mood_scaler.joblib')
    return pipe, nn, rec_scaler, kmeans, mood_scaler

@st.cache_data
def load_catalog():
    if os.path.exists('soundscope_sample_tracks.csv'):
        return pd.read_csv('soundscope_sample_tracks.csv')
    elif os.path.exists('dataset.csv'):
        df = pd.read_csv('dataset.csv')
        df = df.dropna(subset=['artists', 'track_name']).drop_duplicates(subset=['track_id'])
        return df.head(10000)
    else:
        st.error("No dataset file found!")
        return pd.DataFrame()

try:
    pipeline, nn_recommender, rec_scaler, kmeans_mood, mood_scaler = load_models()
    catalog_df = load_catalog()
except Exception as e:
    st.error(f"Error loading models or dataset: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=65)
    st.title("SoundScope AI")
    st.caption("Spotify Intelligence & Discovery Studio")
    
    app_mode = st.radio(
        "Navigation",
        [
            "🎯 Popularity Predictor & What-If",
            "🎧 Music Recommender",
            "🧭 Discovery Advisor",
            "📊 Model Benchmarks & Insights"
        ]
    )
    
    st.markdown("---")
    st.markdown(f"**Loaded Catalog**: `{len(catalog_df):,}` tracks")
    st.markdown(f"**Unique Genres**: `{catalog_df['track_genre'].nunique() if 'track_genre' in catalog_df else 0}`")
    st.caption("Developed by **Kashish** | AIML Final Project")

# -----------------------------------------------------------------------------
# TAB 1: POPULARITY PREDICTOR & WHAT-IF STUDIO
# -----------------------------------------------------------------------------
if app_mode == "🎯 Popularity Predictor & What-If":
    st.markdown('<div class="main-title">🎯 Popularity Prediction & "What-If" Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Predict how popular a song will be based on its acoustic profile, or simulate production changes.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Acoustic Production Settings")
        
        genre_list = sorted(catalog_df['track_genre'].dropna().unique().tolist()) if 'track_genre' in catalog_df else ['pop', 'rock', 'dance']
        track_genre = st.selectbox("Track Genre", genre_list, index=genre_list.index('pop') if 'pop' in genre_list else 0)
        
        c_a, c_b = st.columns(2)
        with c_a:
            danceability = st.slider("Danceability", 0.0, 1.0, 0.75, 0.01)
            energy = st.slider("Energy", 0.0, 1.0, 0.80, 0.01)
            valence = st.slider("Valence (Happiness)", 0.0, 1.0, 0.65, 0.01)
            loudness = st.slider("Loudness (dB)", -35.0, 0.0, -5.5, 0.5)
        with c_b:
            acousticness = st.slider("Acousticness", 0.0, 1.0, 0.15, 0.01)
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.00, 0.01)
            liveness = st.slider("Liveness", 0.0, 1.0, 0.12, 0.01)
            tempo = st.slider("Tempo (BPM)", 50.0, 220.0, 120.0, 1.0)
            
        with st.expander("Additional Metadata Settings"):
            c_c, c_d = st.columns(2)
            with c_c:
                duration_min = st.number_input("Duration (minutes)", min_value=0.5, max_value=15.0, value=3.2, step=0.1)
                explicit = st.checkbox("Explicit Content", value=False)
                key = st.selectbox("Key", list(range(12)), index=0)
            with c_d:
                mode = st.selectbox("Mode", [1, 0], format_func=lambda x: "Major (1)" if x == 1 else "Minor (0)")
                time_signature = st.selectbox("Time Signature", [3, 4, 5], index=1)
                artist_track_count = st.slider("Artist Catalog Count", 1, 50, 8)
                speechiness = st.slider("Speechiness", 0.0, 1.0, 0.06, 0.01)
                
    with col2:
        st.subheader("2. AI Prediction & Insights")
        
        # Calculate mood cluster for prediction input
        scaled_mood_input = mood_scaler.transform([[valence, energy, danceability, acousticness]])
        cid = kmeans_mood.predict(scaled_mood_input)[0]
        
        if acousticness >= 0.7:
            mood_label = 'Chill & Acoustic'
        elif valence >= 0.5 and energy >= 0.6:
            mood_label = 'Energetic & Euphoric'
        elif valence < 0.5 and energy >= 0.6:
            mood_label = 'Intense & Heavy'
        else:
            mood_label = 'Mellow & Reflective'
            
        input_dict = {
            'danceability': [danceability],
            'energy': [energy],
            'loudness': [loudness],
            'speechiness': [speechiness],
            'acousticness': [acousticness],
            'instrumentalness': [instrumentalness],
            'liveness': [liveness],
            'valence': [valence],
            'tempo': [tempo],
            'duration_min': [duration_min],
            'artist_track_count': [artist_track_count],
            'explicit': [explicit],
            'key': [key],
            'mode': [mode],
            'time_signature': [time_signature],
            'track_genre': [track_genre],
            'mood_label': [mood_label]
        }
        
        input_df = pd.DataFrame(input_dict)
        pred_popularity = float(pipeline.predict(input_df)[0])
        pred_popularity = max(0.0, min(100.0, pred_popularity))
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Predicted Popularity Score</h3>
            <h1 style="color: #1DB954; font-size: 3rem; margin: 0;">{pred_popularity:.1f} <span style="font-size: 1.2rem; color: #666;">/ 100</span></h1>
            <p style="margin-top: 5px; color: #555;">Detected Mood Profile: <b>{mood_label}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(pred_popularity / 100.0)
        
        if pred_popularity >= 70:
            st.success("🔥 **High Commercial Potential**: This profile aligns with viral pop and streaming hit patterns.")
        elif pred_popularity >= 45:
            st.info("🎵 **Moderate Potential**: Solid genre fit with healthy listener appeal.")
        else:
            st.warning("📻 **Niche Appeal**: Tailored more towards specialized or acoustic underground audiences.")
            
        st.markdown("---")
        st.subheader("3. 'What-If' Simulation Delta")
        sim_feat = st.selectbox("What if I change...", ["danceability", "energy", "acousticness", "loudness", "tempo"])
        sim_val = st.slider("To new value:", 0.0, 1.0 if sim_feat != 'loudness' and sim_feat != 'tempo' else (0.0 if sim_feat == 'loudness' else 220.0), 
                            float(danceability) if sim_feat == 'danceability' else 0.5)
        
        sim_df = input_df.copy()
        sim_df[sim_feat] = sim_val
        sim_pred = float(pipeline.predict(sim_df)[0])
        delta = sim_pred - pred_popularity
        
        st.metric(label=f"Simulated Popularity (with {sim_feat} = {sim_val})", value=f"{sim_pred:.1f}", delta=f"{delta:+.1f} points")

# -----------------------------------------------------------------------------
# TAB 2: CONTENT-BASED RECOMMENDER
# -----------------------------------------------------------------------------
elif app_mode == "🎧 Music Recommender":
    st.markdown('<div class="main-title">🎧 Content-Based Music Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Find acoustically similar tracks using NearestNeighbors vector similarity in 9-dimensional acoustic space.</div>', unsafe_allow_html=True)
    
    sample_titles = catalog_df['track_name'].dropna().head(50).tolist()
    default_song = "Comedy" if "Comedy" in catalog_df['track_name'].values else sample_titles[0]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        search_mode = st.radio("Search Method", ["Choose from Catalog", "Type Song Name"])
        if search_mode == "Choose from Catalog":
            selected_track = st.selectbox("Select Track", sample_titles)
        else:
            selected_track = st.text_input("Enter Song Name", value="Comedy")
            
        top_k = st.slider("Number of Recommendations", 3, 10, 5)
        
    with col2:
        if st.button("Generate Recommendations", type="primary") or selected_track:
            match = catalog_df[catalog_df['track_name'].str.lower() == selected_track.lower()]
            if match.empty:
                match = catalog_df[catalog_df['track_name'].str.lower().str.contains(selected_track.lower())]
                
            if match.empty:
                st.warning(f"Could not find track '{selected_track}' in current sample catalog.")
            else:
                target = match.iloc[0]
                st.markdown(f"**Query Song**: `{target['track_name']}` by **{target['artists']}** *(Genre: {target.get('track_genre', 'N/A')})*")
                
                rec_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                            'instrumentalness', 'liveness', 'valence', 'tempo']
                target_vector = rec_scaler.transform([target[rec_cols]])
                distances, indices = nn_recommender.kneighbors(target_vector, n_neighbors=top_k + 1)
                
                rec_list = []
                for dist, idx in zip(distances[0], indices[0]):
                    row = catalog_df.iloc[idx]
                    if row['track_id'] == target['track_id']:
                        continue
                    sim = 1.0 - dist
                    rec_list.append({
                        "Track": row['track_name'],
                        "Artist": row['artists'],
                        "Genre": row.get('track_genre', 'N/A'),
                        "Mood": row.get('mood_label', 'N/A'),
                        "Popularity": row['popularity'],
                        "Acoustic Similarity": f"{sim * 100:.1f}%"
                    })
                    if len(rec_list) >= top_k:
                        break
                        
                st.dataframe(pd.DataFrame(rec_list), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: DISCOVERY ADVISOR
# -----------------------------------------------------------------------------
elif app_mode == "🧭 Discovery Advisor":
    st.markdown('<div class="main-title">🧭 Intelligent Music Discovery Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Goal-oriented music discovery: filter by desired vibe, genre, and activity tempo.</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        genres = ["Any"] + sorted(catalog_df['track_genre'].dropna().unique().tolist()) if 'track_genre' in catalog_df else ["Any"]
        sel_genre = st.selectbox("Preferred Genre", genres)
    with col2:
        moods = ["Any", "Energetic & Euphoric", "Chill & Acoustic", "Intense & Heavy", "Mellow & Reflective"]
        sel_mood = st.selectbox("Target Mood", moods)
    with col3:
        target_tempo = st.number_input("Target BPM (e.g. 120 for workout)", min_value=0, max_value=220, value=120)
        
    c4, c5 = st.columns(2)
    with c4:
        min_dance = st.slider("Minimum Danceability", 0.0, 1.0, 0.6)
    with c5:
        min_energy = st.slider("Minimum Energy", 0.0, 1.0, 0.5)
        
    res = catalog_df.copy()
    if sel_genre != "Any":
        res = res[res['track_genre'].str.lower() == sel_genre.lower()]
    if sel_mood != "Any" and 'mood_label' in res:
        res = res[res['mood_label'].str.lower() == sel_mood.lower()]
        
    res = res[(res['danceability'] >= min_dance) & (res['energy'] >= min_energy)]
    
    if target_tempo > 0:
        res['tempo_diff'] = (res['tempo'] - target_tempo).abs()
        res = res.sort_values(by=['tempo_diff', 'popularity'], ascending=[True, False])
    else:
        res = res.sort_values(by='popularity', ascending=False)
        
    st.subheader(f"Matching Discoveries ({len(res):,} found)")
    show_cols = ['track_name', 'artists', 'track_genre', 'mood_label', 'popularity', 'danceability', 'energy', 'tempo']
    available_cols = [c for c in show_cols if c in res.columns]
    st.dataframe(res[available_cols].head(15).reset_index(drop=True), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: MODEL BENCHMARKS & INSIGHTS
# -----------------------------------------------------------------------------
elif app_mode == "📊 Model Benchmarks & Insights":
    st.markdown('<div class="main-title">📊 Machine Learning Benchmarks & Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Technical validation, model comparisons, and explainability findings.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Model Benchmark Comparison")
        benchmarks = pd.DataFrame([
            {"Model": "Baseline (Dummy Mean)", "CV MAE": 17.12, "CV RMSE": 20.45, "CV R²": -0.0001},
            {"Model": "Ridge Regression", "CV MAE": 12.02, "CV RMSE": 16.84, "CV R²": 0.3220},
            {"Model": "Decision Tree Regressor", "CV MAE": 12.85, "CV RMSE": 17.65, "CV R²": 0.2580},
            {"Model": "HistGradientBoosting (Tuned)", "CV MAE": 11.36, "CV RMSE": 15.74, "CV R²": 0.4082}
        ])
        st.dataframe(benchmarks, use_container_width=True)
        
        st.markdown("""
        **Key Architectural Takeaways:**
        - **Boosting Superiority**: Tree ensembles outperformed linear models by capturing non-linear interactions between `loudness`, `energy`, and `genre`.
        - **The ~0.41 $R^2$ Ceiling**: In music streaming analytics, acoustic metrics alone naturally reach an $R^2 \approx 0.40 - 0.45$. Commercial popularity is heavily mediated by extrinsic marketing and artist celebrity.
        """)
        
    with col2:
        st.subheader("2. Top Feature Drivers (Permutation Importance)")
        features = ["track_genre", "artist_track_count", "loudness", "acousticness", "energy", "duration_min", "danceability", "valence"]
        importances = [0.185, 0.082, 0.045, 0.038, 0.031, 0.024, 0.019, 0.014]
        feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
        st.bar_chart(feat_df.set_index("Feature"))
