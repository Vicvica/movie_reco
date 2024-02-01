#cd ~/Desktop/streamlit_projet
#streamlit run projet_2_voc.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import networkx as nx
import plotly.graph_objects as go
from matplotlib_venn import venn2
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import requests
from streamlit_carousel import carousel
from fuzzywuzzy import process
import qrcode
from PIL import Image
from io import BytesIO
import speech_recognition as sr
import pyaudio
import time

TMDB_API_KEY = "b37d6953a05dd68a9492d0ddf5dd87d6"
OMDB_API_KEY = "52d003e7"

# Configuration de la page Streamlit
st.set_page_config(page_title="Les Data Flingueurs", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

##QR CODE
def generate_qr_code():
    # Adresse de votre site
    site_url = "https://odyssey.wildcodeschool.com/"

    obj_qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )

    obj_qr.add_data(site_url)
    obj_qr.make(fit=True)

    # Redimensionner l'image
    qr_img = obj_qr.make_image(fill_color="white", back_color="black").resize((100, 100))

    img_byte_array = BytesIO()
    qr_img.save(img_byte_array, format="PNG")
    img_bytes = img_byte_array.getvalue()

    return img_bytes

# Créer une disposition en colonnes
col1, col2 = st.columns([3, 1])

# Afficher le QR Code en haut à droite
col2.image(generate_qr_code(), caption="Scanner pour voir sur mobile", width=100, clamp=True, channels="BGR")


#titre de la page
st.markdown("""
    <h1 style='text-align: center;'>Les Data Flingueurs 🔫</h1>
""", unsafe_allow_html=True)
st.write("")

st.markdown("<div style='text-align: center; font-size: 1.5em;'>Recommandations de Films 🍿🎬</div>", unsafe_allow_html=True)


# Chargement des données
df_movie = pd.read_csv("/Users/user/Desktop/streamlit_projet/df_movie.csv")
df_movie = df_movie.drop_duplicates(subset=["tconst", "nconst"])
df_cinema = pd.read_csv("/Users/user/Desktop/streamlit_projet/df_cinema.csv")

# Sélection des features
df_neighbors = df_movie[['title','startYear', 'genres_Action', 'genres_Adventure',
       'genres_Animation', 'genres_Biography', 'genres_Comedy', 'genres_Crime',
       'genres_Documentary', 'genres_Drama', 'genres_Family', 'genres_Fantasy',
       'genres_Film-Noir', 'genres_Game-Show', 'genres_History',
       'genres_Horror', 'genres_Music', 'genres_Musical', 'genres_Mystery',
       'genres_News', 'genres_Reality-TV', 'genres_Romance', 'genres_Sci-Fi',
       'genres_Short', 'genres_Sport', 'genres_Talk-Show', 'genres_Thriller',
       'genres_War', 'genres_Western']]

X = df_neighbors[['startYear', 'genres_Action', 'genres_Adventure',
       'genres_Animation', 'genres_Biography', 'genres_Comedy', 'genres_Crime',
       'genres_Documentary', 'genres_Drama', 'genres_Family', 'genres_Fantasy',
       'genres_Film-Noir', 'genres_Game-Show', 'genres_History',
       'genres_Horror', 'genres_Music', 'genres_Musical', 'genres_Mystery',
       'genres_News', 'genres_Reality-TV', 'genres_Romance', 'genres_Sci-Fi',
       'genres_Short', 'genres_Sport', 'genres_Talk-Show', 'genres_Thriller',
       'genres_War', 'genres_Western']]

target = df_neighbors[['title']]



# Barre de navigation latérale
menu = ["Accueil", "Recommandation de Film", "Recherche de Personnalités", "Exploration des Données", "À Propos"]
choice = st.sidebar.selectbox("Menu", menu)

# Contenu principal
if choice == "Accueil":
    st.markdown("<h1 style='text-align: center;'>Bienvenue chez Les Data Flingueurs !</h1>", unsafe_allow_html=True)    
    st.write("")
    st.write("")
    st.markdown("<div style='text-align: center;'>Découvrez des recommandations de films et explorez des données cinématographiques intéressantes.</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<div style='text-align: center;'>À propos de nous</div>", unsafe_allow_html=True)
    st.write("")
    st.write("Les Data Flingueurs sont une équipe passionnée par le cinéma et les données. "
             "Nous utilisons l'analyse de données pour découvrir des tendances intéressantes dans l'industrie cinématographique. "
             "Explorez nos recommandations de films et plongez dans les insights que nous avons trouvés !")


    #carousel 
    st.write("")
    st.write("")
    st.markdown("<div style='text-align: center;'>Anniversaires à souhaiter aujourd'hui!</div>", unsafe_allow_html=True)
    st.write("")
    test_items = [
        dict(
            title="Ellie Bamber",
            text="The Show, The Trial of Christine Keeler",
            interval=None,
            img="https://decider.com/wp-content/uploads/2022/11/ellie-bamber-elora-danan.jpg?quality=80&strip=all&w=1200",
        ),
        dict(
            title="Shakira",
            text="Zootopia, Sprung, It's Bruno!, Freak Out",
            img="https://www.melty.fr/wp-content/uploads/meltyfr/2023/01/308412860_765650828054285_9134301558114735862_n-643x410.jpg"
        ),
        dict(
            title="Tom Blyth",
            text="Hunger Games, Billy the Kid",
            img="https://decider.com/wp-content/uploads/2023/11/tom-blyth-the-hunger-games-ballad-of-songbirds-and-snakes.jpg?quality=75&strip=all"
        ),
    ]

    carousel(items=test_items, width=1)
    



elif choice == "Recommandation de Film":
    st.write("")
    st.write("")
    st.header("Recommandation de Film 🎥🍿")

    # Sélection du film

    # Option de recherche par texte ou reconnaissance vocale
    search_option = st.radio("Choisissez votre méthode de recherche :", ["Texte", "Reconnaissance vocale"])

    liste_film = list(df_movie['title'])

    def voice_recognition():
        recognizer = sr.Recognizer()
        st.title("Reconnaissance vocale")
        st.write("")
        
        if 'transcription' not in st.session_state:
            st.session_state.transcription = ""
        if 'recording' not in st.session_state:
            st.session_state.recording = False

        start_recording = st.button("Commencer l'enregistrement")
        stop_recording = st.button("Terminer l'enregistrement")

        if start_recording:
            st.session_state.transcription = ""
            st.session_state.recording = True
            st.write("Enregistrement en cours...")

        if st.session_state.recording:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)

                audio = recognizer.listen(source)

                try:
                    query = recognizer.recognize_google(audio, language="fr-FR")
                    st.session_state.transcription = query
                except sr.UnknownValueError:
                    st.warning("Cliquez sur Terminer l'enregistrement")
                except sr.RequestError as e:
                    st.error(f"Je n'ai pas pu trouvé de résultat; {e}")

                if stop_recording:
                    st.session_state.recording = False  # Arrêter l'enregistrement si stop_recording est activé

        # Afficher la transcription
        st.text(f"Transcription: {st.session_state.transcription}")
        display_movie_recommendations(st.session_state.transcription)
            

    def trouver_titre_film(entree_utilisateur, liste_film):
        meilleur_match = process.extractOne(entree_utilisateur, liste_film)
        return meilleur_match[0]

    def movie_reco(df_neighbors, X, target, selected_movie):
        model = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
        model.fit(X)

        movie_index = df_neighbors[df_neighbors['title'] == selected_movie].index
        if not movie_index.empty:
            movie_index = movie_index[0]
            distances, indices = model.kneighbors(X.iloc[movie_index, :].values.reshape(1, -1))

            recommendations = []

            for i in range(len(indices.flatten())):
                recommended_movie = df_neighbors['title'].iloc[indices.flatten()[i]]
                recommendations.append(recommended_movie)

            return recommendations[:6]
        else:
            return None

    def get_movie_details(movie_title):
        tmdb_params = {
            'api_key': TMDB_API_KEY,
            'query': movie_title,
            'language': 'fr'
        }
        tmdb_response = requests.get('https://api.themoviedb.org/3/search/movie', params=tmdb_params)
        tmdb_data = tmdb_response.json()
        if 'results' in tmdb_data and tmdb_data['results']:
            tmdb_movie_info = tmdb_data['results'][0]
            poster = f"https://image.tmdb.org/t/p/w500/{tmdb_movie_info['poster_path']}" if 'poster_path' in tmdb_movie_info else None
            overview = tmdb_movie_info.get('overview', 'No overview available')
            release_date = tmdb_movie_info.get('release_date', 'Release date not available')
        else:
            poster, overview, release_date = None, 'No overview available', 'Release date not available'

        omdb_params = {
            'apikey': OMDB_API_KEY,
            't': movie_title,
            'r': 'json',
            'plot': 'full'
        }
        omdb_response = requests.get('http://www.omdbapi.com/', params=omdb_params)
        omdb_data = omdb_response.json()
        cast = omdb_data.get('Actors', None)
        genre = omdb_data.get('Genre', None)
        rating = omdb_data.get('imdbRating', None)
        return poster, overview, cast, genre, rating, release_date

    def display_movie_recommendations(user_input_title=None):
        if user_input_title is None:
            user_input_title = st.text_input("Saisissez le titre du film :")

        if user_input_title:
            titre_trouve = trouver_titre_film(user_input_title, liste_film)
            recommendations = movie_reco(df_neighbors, X, target, titre_trouve)

            if recommendations is not None:
                recommendations = [movie for movie in recommendations if movie != titre_trouve]

            if recommendations:
                st.write(f"## Top 5 films recommandés pour '{titre_trouve}':")
                for i, recommended_movie in enumerate(recommendations[:6]):
                    st.write(f"### {i + 1}: {recommended_movie}")

                    poster, overview, cast, genre, rating, release_date = get_movie_details(recommended_movie)

                    if poster is not None:
                        st.markdown(
                            f'<div style="display: flex; justify-content: center;">'
                            f'<img src="{poster}" alt="Affiche pour {recommended_movie}" style="max-width: 100%; height: auto;">'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    st.markdown(f"**Synopsis:** \n\n {overview}")
                    if cast is not None:
                        st.markdown(f"**Distribution:** {cast}")
                    if genre is not None:
                        st.markdown(f"**Genre:** {genre}")
                    if rating is not None:
                        st.markdown(f"**Note:** {rating}")
                    if release_date is not None:
                        st.markdown(f"**Date de sortie:** {release_date}")

            else:
                st.write(f"Le film '{titre_trouve}' n'est pas dans notre catalogue.")
        else:
            # Rien afficher si l'utilisateur n'a pas encore saisi de titre de film
            pass


    if search_option == "Texte":
        display_movie_recommendations()

    elif search_option == "Reconnaissance vocale":
        voice_recognition()
        


#Recherche par perso

elif choice == "Recherche de Personnalités":
    st.write("")
    st.write("")

    st.header("Recherche de Personnalités 🔍🎬")

    st.write("")
    st.write("")

    sub_df = df_cinema.copy()

    sub_df['star_and_known_movies'] = sub_df['primaryName'] + ' - ' + sub_df['knownForTitles'] + ' - ' + sub_df['characters']

    personality_name = st.text_input("Saisissez le nom de la personnalité :")

    if personality_name:
        filtered_sub_df = sub_df[sub_df['star_and_known_movies'].str.contains(personality_name, case=False, na=False)]

        if not filtered_sub_df.empty:
            selected_movie = st.selectbox(f"Films pour lesquels {personality_name} est mentionné.e:", filtered_sub_df['title'])

            st.write(f"Vous avez sélectionné le film : {selected_movie}")

            if selected_movie:
                year = filtered_sub_df.loc[filtered_sub_df['title'] == selected_movie, 'startYear'].values[0]
                year_int64 = int(year)
                st.write(f"Année: {year_int64}")
        else:
            st.write(f"Aucun résultat trouvé pour {personality_name}.")
    else:
        # Lorsque le nom de la personnalité n'est pas encore renseigné
        st.write("Entrez le nom d'une personnalité pour afficher les résultats.")
        selected_movie = None  # Pour que la boîte de sélection reste vide


#KPI

elif choice == "Exploration des Données":
    st.write("")
    st.write("")
    st.header("Exploration des Données Cinématographiques 📊🎬")
    st.write("")
    st.write("")
    # Sélection du type de visualisation
    exploration_option = st.selectbox("Découvrez une analyse!", [
        "Analyse des acteurs les plus présents par période",
        "Analyse de la durée des films au fil des années",
        "Analyse des acteurs présents au grand et au petit écran",
        "Quelle-est la moyenne d'âge des acteurs?",
        "Quels-sont les films les mieux notés?",
        "Quelles-sont leurs caractéristiques communes?"
    ])

    
    
    def analyse_acteurs_presents_par_periode(df_cinema):
        st.title("Quels-sont les acteurs les plus présents au fil des années?")
        st.write("")
        st.write("")
        df_actors = df_cinema[['category', 'primaryName', 'startYear']]

            # On garde que les acteurs
        df_actors = df_actors.loc[(df_actors['category'] == 'actor') | (df_actors['category'] == 'actress')]
        df_actors = df_actors[["primaryName", "startYear"]]
        df_actors['startYear'] = df_actors['startYear'].astype('int64')

        most_frequent_names = df_actors.groupby('startYear')['primaryName'].agg(lambda x: x.value_counts().idxmax()).reset_index()
        name_counts = df_actors.groupby(['startYear', 'primaryName']).size().reset_index(name='Occurrence')

        result_df = pd.merge(most_frequent_names, name_counts, on=['startYear', 'primaryName'])

        result_df.columns = ['Year', 'MostFrequentName', 'Occurrence']

        st.sidebar.write("Résultats pour l'analyse des acteurs les plus présents par période:")
        st.sidebar.write(result_df)

            # Graphique interactif KPI 1
        colors = px.colors.qualitative.Plotly[:len(result_df['Year'].unique())]

        fig = px.bar(result_df, x='MostFrequentName', y='Occurrence', color='Year',
                     title='Occurrences des acteurs au fil des années',
                     labels={'Occurrence': 'Nombre d\'occurrences par années', 'MostFrequentName': 'Acteurs'},
                     barmode='group', color_discrete_sequence=colors)

        fig.update_layout(legend=dict(title='Année'), showlegend=True, height=600)
        fig.update_xaxes(tickangle=45, tickmode='array')

        st.plotly_chart(fig)


    def analyse_duree_films(df_cinema):
        st.title("Durée moyenne en minutes des films au fil des années")
        st.write("")
        st.write("")

        df_time_movie = df_cinema[['runtimeMinutes', 'startYear']]
        df_time_movie['startYear'] = df_time_movie['startYear'].astype('int64')
        df_mean_by_year = df_time_movie.groupby('startYear')['runtimeMinutes'].mean().reset_index()

        # Utilisation de Streamlit pour afficher le graphique
        fig = px.line(df_mean_by_year, x='startYear', y='runtimeMinutes',
                      labels={'runtimeMinutes': 'Durée moyenne en minutes des films', 'startYear': 'Années'},
                      markers=True, line_shape='linear')

        fig.update_layout(showlegend=False, width=800, height=500)

        st.plotly_chart(fig)

        st.sidebar.write("La durée des films s'allonge avec le temps, elle se stabilise même!")


    def acteurs_cinema_vs_petit_ecran(df_cinema):
        st.title("Les acteurs de cinéma sont-ils les mêmes qu'au petit écran?")
        st.write("")
        st.write("")

        df_serie = df_cinema[['nconst', 'titleType_y', 'category', 'primaryName']]
        df_actor_movie = df_serie[((df_serie["category"] == 'actor') | (df_serie["category"] == 'actress')) & (df_serie["titleType_y"] == 'movie')]
        df_actor_serie = df_serie[((df_serie["category"] == 'actor') | (df_serie["category"] == 'actress')) & (df_serie["titleType_y"] == 'tvSeries')]

        df_actor_movie = df_actor_movie.drop_duplicates()
        df_actor_serie = df_actor_serie.drop_duplicates()

        df_movie_and_serie = pd.merge(df_actor_serie, df_actor_movie, how='inner', on=['nconst'])

        # Créez une nouvelle figure avant de tracer le diagramme de Venn
        fig, ax = plt.subplots(figsize=(8, 8))
        venn_diagram = venn2(subsets=(len(df_actor_serie), len(df_actor_movie), len(df_movie_and_serie)),
                             set_labels=('Acteurs dans des séries', 'Acteurs dans des films'))
        plt.title("Les acteurs de cinéma sont-ils les mêmes qu'au petit écran?\nDans l'histoire du cinéma")

        st.pyplot(fig)

        st.sidebar.write("La réponse est: OUI! On retrouve énormément de stars dans ce cas!")


    def show_age_distribution(df_cinema):
        st.title("Quel-est l'âge moyen des acteurs?")
        st.write("")
        st.write("")

        df1 = df_cinema.copy()

        df1["birthYear"] = pd.to_numeric(df1["birthYear"].replace('\\N', pd.NA))
        df1["deathYear"] = pd.to_numeric(df1["deathYear"].replace('\\N', pd.NA))

        # Création d'un sous-dataframe avec les colonnes nécessaires et ajout de la colonne 'age' et le calcul
        df_rename = df1[['birthYear', 'deathYear']]

        # Filtrer les acteurs vivants en vérifiant si la colonne "deathYear" est nulle et si la colonne "birthYear" n'est pas nulle.
        df_rename = df_rename[df_rename["deathYear"].isna() & ~df_rename["birthYear"].isna()].copy()

        # Calcul de l'âge des acteurs vivants
        df_rename.loc[:, "age"] = pd.Timestamp.now().year - df_rename["birthYear"]

        # Supprimer les outliers : 5 ans à 90 ans
        df_rename = df_rename[(df_rename['age'] >= 5) & (df_rename['age'] <= 90)]

        # Afficher la moyenne d'âge
        average_age = df_rename['age'].mean()
        st.write(f"Moyenne d'âge des acteurs : {average_age:.2f} ans")

        # Créer un histogramme basique avec plotly.graph_objects
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_rename['age'], nbinsx=10))

        st.write(fig)

        st.sidebar.write("La moyenne d'âges des acteurs présents au cinéma aujourd'hui est de 60 ans!")


    def analyse_films_mieux_notes(df_cinema):
        st.title("Quels-sont les films les plus populaires?")
        st.write("")
        st.write("")

        df_movie= df_cinema[["tconst", "directors", "nconst", "writers", "title", "region", "startYear", "runtimeMinutes", "genres", "averageRating", "numVotes", "titleType_y"]]
        df_movie= df_movie[(df_movie["titleType_y"] == "movie")]

        df_movie= df_movie.groupby('tconst').agg({
            'directors': 'first',
            'nconst': lambda x: ','.join(x.astype(str)),
            'writers': lambda x: ','.join(x.astype(str)),
            'title': 'first',
            'region': 'first',
            'startYear': 'first',
            'runtimeMinutes': 'first',
            'genres': 'first',
            'averageRating': 'first',
            'numVotes': 'first'
        }).reset_index()

        df_movies = df_movie[(df_movie["averageRating"] > 8.6) & (df_movie["numVotes"] >= 10000)]
        df_movies = df_movies.drop_duplicates()

        df_movies = df_movies.reset_index(drop=True)

        df_movies_reshaped = df_movies.drop(["tconst", "nconst", "directors", "writers", "region", "startYear"], axis=1)

        fig = px.bar(df_movies_reshaped, y="title", x="averageRating", color="numVotes", text="averageRating", 
                     orientation="h", 
                     title="Films les mieux notés",
                     labels={"averageRating": "Note moyenne", "title": "Titre du film", "numVotes": "Nombre de votes"}
                    )
        st.plotly_chart(fig)

        st.title("Liste complète des films les mieux notés")
        st.write("")
        st.write("")
        st.table(df_movies_reshaped)

        st.sidebar.write("Représentation des films les mieux notés aynat le plus de votes.")



    def best_char_genre(df_movie):
        st.title("Quels sont les genres les plus populaires?")
        st.write("")
        st.write("")

        result_data = {'Drama': 25, 'Action': 13, 'Adventure': 11, 'Crime': 8}
        result = pd.Series(result_data)

        st.write("Les 4 premiers genres avec leur occurrence:")
        for genre, occurrence in result.items():
            st.write(f"{genre}: {occurrence}")

        fig = px.pie(result, names=result.index, values=result.values,
                        title="Distribution des genres les plus populaires", 
                        hole=0.3,
                        )
        st.write("")
        st.write("")
        st.plotly_chart(fig)

        st.sidebar.markdown(f"Le genre le plus apprécié est le **{result.idxmax()}**. Surprenant?")


    def best_char(df_movie, df_cinema):
        st.title("Quels-sont les personnes les plus populaires au cinéma?")
        st.write("")
        st.write("")
        columns = ["directors", "nconst", "writers"]

        for column in columns:
            temp = df_movie[column].str.split(',').apply(lambda x: list(set(x)))
            temp = temp.explode(column).value_counts()

            result = temp[temp == 3]

            # Filtrer les résultats spécifiques
            if column == "directors":
                nconst_list = ["nm0634240", "nm0001392"]
            elif column == "nconst":
                nconst_list = ["nm0001392", "nm0909638", "nm0000704", "nm0089217", "nm0866058", "nm0101991", "nm0005212", "nm0634240"]
            elif column == "writers":
                nconst_list = ["nm0909638", "nm0101991", "nm0634240", "nm0866058", "nm0001392"]

            result_specific = df_cinema[df_cinema["nconst"].isin(nconst_list)][["nconst", "primaryName"]].drop_duplicates()
            
            st.write(f"Résultats spécifiques pour '{column}':")
            st.write(result_specific)


    def display_network_graph():
        st.title("Quelles-sont les relations des personnes les plus populaires au cinéma?")
        st.write("")
        st.write("")
        relations = [
            ("Christopher Nolan", "Elijah Wood"),
            ("Christopher Nolan", "Ian McKellen"),
            ("Christopher Nolan", "Orlando Bloom"),
            ("Peter Jackson", "Elijah Wood"),
            ("Peter Jackson", "Ian McKellen"),
            ("Peter Jackson", "Orlando Bloom"),
            ("Peter Jackson", "Fran Walsh"),
            ("Peter Jackson", "Philippa Boyens"),
            ("Peter Jackson", "J.R.R. Tolkien"),
        ]
        G = nx.Graph()
        G.add_edges_from(relations)

        pos = nx.spring_layout(G)

        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, font_size=8, node_size=700, node_color="skyblue", font_color="black", font_weight="bold", edge_color="gray")
        plt.title("Relations entre Réalisateurs, Scénaristes et Autres")
        st.write("")
        st.write("")
        st.pyplot(fig)
        st.sidebar.write("Seigneurs des Anneaux: dieu des films?")



    if exploration_option == "Analyse des acteurs les plus présents par période":
        analyse_acteurs_presents_par_periode(df_cinema)
    elif exploration_option == "Analyse de la durée des films au fil des années":
        analyse_duree_films(df_cinema)
    elif exploration_option == "Analyse des acteurs présents au grand et au petit écran":
        acteurs_cinema_vs_petit_ecran(df_cinema)
    elif exploration_option == "Quelle-est la moyenne d'âge des acteurs?":
        show_age_distribution(df_cinema)
    elif exploration_option == "Quels-sont les films les mieux notés?":
        analyse_films_mieux_notes(df_cinema)
    elif exploration_option == "Quelles-sont leurs caractéristiques communes?":
        best_char_genre(df_movie)
        best_char(df_movie, df_cinema)
        display_network_graph()


else:
    st.write("")
    st.write("")
    st.header("À Propos de Nous 🎬📊")
    st.write("")
    st.write("")
    st.subheader("Les Data Flingueurs")
    st.write("")
    st.write("")
    st.write("Nous sommes une équipe passionnée par les films et les données: Sylia, Jocelyn, Sara et Victoria! "
             "Notre mission est d'explorer l'univers cinématographique à l'aide de l'analyse de données "
             "et de fournir des recommandations de films personnalisées. "
             "Explorez notre application pour découvrir de nouveaux films et des insights intéressants sur l'industrie cinématographique.")

    st.subheader("Contactez-nous")
    st.write("")
    st.write("📧 Email: contact@lesdataflingueurs.com")

    st.subheader("Suivez-nous sur les Réseaux Sociaux: @data_flingueurs")
    st.write("")
    st.write("")
    st.write("Scannez le QR Code pour Visiter Notre Site Web ⬆️")
    