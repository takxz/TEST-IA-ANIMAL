import streamlit as st
from transformers import pipeline
from PIL import Image
import json
import os
from datetime import datetime
from deep_translator import GoogleTranslator

# --- NOUVEAU : Import pour le filtrage animal ---
import nltk
from nltk.corpus import wordnet

# On télécharge le dictionnaire WordNet au premier lancement
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# --- CONFIGURATION ---
DATA_FILE = "pokedex_data.json"

@st.cache_resource
def load_model():
    # Modèle Google ViT (Très bon, généraliste)
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

# --- FONCTION MAGIQUE : LE FILTRE ---
def is_this_an_animal(label):
    """
    Vérifie si le mot (en anglais) est un descendant de 'animal' dans WordNet.
    """
    # 1. Nettoyage : "Tabby, tabby cat" -> on prend juste "tabby_cat"
    formatted_label = label.split(',')[0].replace(' ', '_').lower()
    
    # 2. On demande à WordNet ce que c'est (les 'synsets')
    synsets = wordnet.synsets(formatted_label)
    
    if not synsets:
        return False # Mot inconnu

    # 3. On remonte l'arbre généalogique pour chaque définition
    for synset in synsets:
        # On regarde tous les ancêtres (hypernyms)
        for path in synset.hypernym_paths():
            # On vérifie si 'animal.n.01' est dans les ancêtres
            for ancestor in path:
                if ancestor.name() == 'animal.n.01':
                    return True
    return False

# ... (Fonctions load_collection et save_animal restent identiques) ...
def load_collection():
    if not os.path.exists(DATA_FILE): return []
    with open(DATA_FILE, "r") as f: return json.load(f)

def save_animal(name, confidence):
    collection = load_collection()
    if not any(d['name'] == name for d in collection):
        entry = {"name": name, "date": datetime.now().strftime("%d/%m/%Y"), "confidence": confidence}
        collection.append(entry)
        with open(DATA_FILE, "w") as f: json.dump(collection, f)
        return True
    return False

# --- INTERFACE ---
st.title("🦁 Google Pokédex (Filtre Animal)")

# Source : caméra ou fichier
mode = st.radio("Source", ["📸 Caméra", "📁 Fichier"], horizontal=True)
img_file_buffer = (
    st.camera_input("Viser l'animal")
    if mode == "📸 Caméra"
    else st.file_uploader("Choisir une photo", type=["png", "jpg", "jpeg"])
)

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    
    with st.spinner('Analyse et vérification biologique...'):
        predictions = classifier(image)
        top_result = predictions[0]
        english_name = top_result['label'] # ex: "Golden retriever"
        score = top_result['score']

        # 1. LE FILTRE : Est-ce un animal ?
        if is_this_an_animal(english_name):
            
            # C'est un animal ! On traduit et on affiche
            try:
                animal_name = GoogleTranslator(source='auto', target='fr').translate(english_name)
                animal_name = animal_name.capitalize()
            except:
                animal_name = english_name

            st.success(f"Espèce détectée : **{animal_name}**")
            st.progress(score)

            if score > 0.4:
                if st.button("🔴 CAPTURER"):
                    if save_animal(animal_name, f"{round(score*100)}%"):
                        st.balloons()
                        st.success("Ajouté au Pokédex !")
                    else:
                        st.warning("Déjà attrapé !")
        
        else:
            # Ce n'est pas un animal
            st.error(f"Objet détecté : {english_name}")
            st.warning("⚠️ Ce n'est pas un animal ! Le Pokédex refuse cette entrée.")

# ... (Affichage de la collection reste identique) ...
st.divider()
st.subheader("📖 Ma Collection")
collection = load_collection()
for animal in reversed(collection):
    st.text(f"- {animal['name']} ({animal['confidence']})")