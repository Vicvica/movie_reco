from os import lseek
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
import qrcode
from io import BytesIO
df_movie = pd.read_csv('C:/Users/jocel/Desktop/streamlit/df_movie.csv')
df_movie = df_movie.drop_duplicates(subset=['tconst', 'nconst'])
# Choose features
df_neighbors = df_movie[['title','startYear', 'genres_Action', 'genres_Adventure',
       'genres_Animation', 'genres_Biography', 'genres_Comedy', 'genres_Crime',
       'genres_Documentary', 'genres_Drama', 'genres_Family', 'genres_Fantasy',
       'genres_Film-Noir', 'genres_Game-Show', 'genres_History',
       'genres_Horror', 'genres_Music', 'genres_Musical', 'genres_Mystery',
       'genres_News', 'genres_Reality-TV', 'genres_Romance', 'genres_Sci-Fi',
       'genres_Short', 'genres_Sport', 'genres_Talk-Show', 'genres_Thriller',
       'genres_War', 'genres_Western']]
# Instantiate X
X = df_neighbors[['startYear', 'genres_Action', 'genres_Adventure',
       'genres_Animation', 'genres_Biography', 'genres_Comedy', 'genres_Crime',
       'genres_Documentary', 'genres_Drama', 'genres_Family', 'genres_Fantasy',
       'genres_Film-Noir', 'genres_Game-Show', 'genres_History',
       'genres_Horror', 'genres_Music', 'genres_Musical', 'genres_Mystery',
       'genres_News', 'genres_Reality-TV', 'genres_Romance', 'genres_Sci-Fi',
       'genres_Short', 'genres_Sport', 'genres_Talk-Show', 'genres_Thriller',
       'genres_War', 'genres_Western']]
st.markdown("""
    <h1 style='text-align: center;'>Les 100 ans du cinéma</h1>
""", unsafe_allow_html=True)
st.markdown("""
    <h3 style='text-align: center;'>Subtitle: 100 ans de divertissement pour tous</h3>
""", unsafe_allow_html=True)
st.markdown("""
    <h1 style='text-align: center;'>Trouvez nos meilleures recommendations en fonction de votre film préféré</h1>
""", unsafe_allow_html=True)
# Streamlit UI components
user_input_title = st.text_input('Recherchez un film', '')
if user_input_title:
    year_weight = 0.1
    user_movie_row = df_neighbors[df_neighbors['title'] == user_input_title]
    if not user_movie_row.empty:
        user_movie_features = user_movie_row.drop('title', axis=1)
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(X)
        pca = PCA(n_components=0.95)
        features_pca = pca.fit_transform(features_normalized)
        features_without_user_movie = features_pca[X.index != user_movie_row.index[0]]
        year_scaler = MinMaxScaler()
        years_normalized = year_scaler.fit_transform(X[['startYear']])
        years_normalized_subset = years_normalized[:features_without_user_movie.shape[0]]
        features_combined = np.hstack([features_without_user_movie, year_weight * years_normalized_subset])
        k_neighbors = 5
        nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
        nn_model.fit(features_combined)
        user_movie_features_normalized = scaler.transform(user_movie_features)
        user_movie_features_pca = pca.transform(user_movie_features_normalized)
        user_movie_year_normalized = year_scaler.transform(user_movie_row[['startYear']])
        user_movie_features_combined = np.hstack([user_movie_features_pca, year_weight * user_movie_year_normalized])
        distances, indices = nn_model.kneighbors(user_movie_features_combined)
        recommendation_indices = indices[0]
        recommendations = df_neighbors['title'].iloc[recommendation_indices].tolist()
        st.write('Recommendations:')
        for movie_title in recommendations:
            st.write(movie_title)
    else:
        st.write(f"Le film {user_input_title} n'est pas dans notre catalogue.")

###QR CODE
# Function to generate and display the QR code
def generate_and_display_qr_code(data):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=1,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    # Convert the PIL Image to a base64-encoded string
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_str = img_bytes.getvalue()
    # Display the QR code in the Streamlit app
    st.image(img_str, caption="Scan this QR Code to visit the website", use_column_width=True)

# Title and URL for the website
st.title("Website QR Code Generator")
website_url = "https://www.wildcodeschool.com/fr-fr/"

# Display the QR code
generate_and_display_qr_code(website_url)        



##Code for voice recognition

import speech_recognition as sr
import time
import threading

### Draft
# def voice_recognition():
#     recognizer = sr.Recognizer()
#     recording_started = False
#     recording_timeout = 10  # Set the recording timeout in seconds

#     def capture_audio(source, timeout):
#         try:
#             recognizer.adjust_for_ambient_noise(source)
#             audio = recognizer.listen(source, timeout=timeout)
#             query = recognizer.recognize_google(audio, language="fr-FR")
#             st.success(f"Your query: {query}")
#             st.text(f"Transcription: {query}")  # Display transcription
#         except sr.WaitTimeoutError:
#             st.warning("Speech recognition timed out. No speech detected within the timeout.")
#         except sr.UnknownValueError:
#             st.warning("Sorry, could not understand audio.")
#         except sr.RequestError as e:
#             st.error(f"Could not request results from Google Speech Recognition service; {e}")

#     st.title("Speech Recognition with Timeout")

#     start_recording = st.button("Start Recording")

#     if start_recording:
#         recording_started = True
#         st.write("Recording...")

#         with st.empty():
#             remaining_time = recording_timeout
#             while remaining_time > 0:
#                 st.write(f"Time remaining: {remaining_time} seconds")
#                 time.sleep(1)
#                 remaining_time -= 1

#             st.write("Recording completed.")
#             with sr.Microphone() as source:
#                 # Use threading to enforce a time limit
#                 thread = threading.Thread(target=capture_audio, args=(source,), kwargs={'timeout': recording_timeout})
#                 thread.start()
#                 thread.join()

# # Call the voice_recognition function directly
# voice_recognition()

#### Working.
def voice_recognition():
    recognizer = sr.Recognizer()

    st.title("Speech Recognition")

    if 'transcription' not in st.session_state:
        st.session_state.transcription = ""

    start_recording = st.button("Commencer l'enregistrement")
    stop_recording = st.button("Terminer l'enregistrement")

    if start_recording:
        st.session_state.transcription = ""
        st.write("Enregistrement en cours...")

        your_device_index = 1  # Change this to the index of your Bluetooth headset. Il faut lancer le script plus bas à part ton trouver l'index de ton input sur ton device.

        with sr.Microphone(device_index=your_device_index) as source:
            recognizer.adjust_for_ambient_noise(source)

            while not stop_recording:
                audio = recognizer.listen(source)

                try:
                    query = recognizer.recognize_google(audio, language="fr-FR")
                    st.session_state.transcription = query
                except sr.UnknownValueError:
                    st.warning("Désolé, je n'ai pas compris.")
                except sr.RequestError as e:
                    st.error(f"Je n'ai pas pu trouvé de résultat; {e}")

    # Display the transcription
    st.text(f"Transcription: {st.session_state.transcription}")

# Call the voice_recognition function directly
voice_recognition()

# #to check the index of the audio input
# def list_audio_devices():
#     for index, name in enumerate(sr.Microphone.list_microphone_names()):
#         print(f"Index {index}: {name}")

# list_audio_devices()