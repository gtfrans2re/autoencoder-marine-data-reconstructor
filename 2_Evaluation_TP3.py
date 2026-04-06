# **************************************************************************
# INF7370 - Apprentissage automatique
# Fichier d’évaluation du Travail Pratique 3 (TP3)
# réalisé par :
# Francois Gonothi Toure, TOUF23329808
# et
# Martial Zachee Kaljob Kollo, KALM30319404
# ===========================================================================

# ===========================================================================
# Ce script évalue la performance d’un autoencodeur convolutif entraîné
# dans le cadre du TP3 sur un ensemble de données composé d’images RGB
# de deux classes marines : dauphin (label 0) et requin (label 1).
#
# Les étapes principales de l’évaluation sont les suivantes :
#
# --------------------------------------------------------------
# 1) Reconstruction :
#    On affiche les images d’entrée et leurs reconstructions 
#    générées par l’autoencodeur pour évaluer visuellement la
#    fidélité de la reconstruction (Question 2).
#
# 2) Extraction de l’embedding :
#    On extrait les embeddings des images de test en récupérant
#    la sortie du dernier bloc convolutif de l’encodeur (Question 3).
#
# 3) Normalisation :
#    On normalise les embeddings pour qu’ils soient adaptés à des
#    algorithmes de classification linéaires comme SVM (Question 4).
#
# 4) Évaluation par classification (SVM) :
#    On applique un SVM linéaire sur :
#      a. les images originales aplaties
#      b. les embeddings extraits et normalisés
#    et on compare les exactitudes obtenues (Question 5 et 6).
#
# 5) Visualisation TSNE :
#    On réduit les embeddings à deux dimensions avec t-SNE et
#    on génère un scatter plot pour visualiser la séparation
#    des deux classes (Question 7).
# ----------------------------------------------------------------
#
# Ces étapes permettent d’évaluer à la fois la qualité de reconstruction
# du modèle et la pertinence des embeddings extraits pour des tâches
# de classification et de visualisation.
# ===========================================================================

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes et des images
import matplotlib.pyplot as plt

# La librairie numpy
import numpy as np

# Configuration du GPU
import tensorflow as tf

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model

# Utilisé pour normaliser l'embedding
from sklearn.preprocessing import StandardScaler

# Utilisé pour le SVM-Linaire et la visualisation TSEN
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(sess);
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# ==========================================
# ==================MODÈLE==================
# ==========================================
# Chargement du modéle (autoencodeur) sauvegardé dans la section 1 via 1_Modele_TP3.py
model_path = "Model.keras"
autoencoder = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#            RÉPONSES AUX QUESTIONS
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# 1) Aujustement des variables suivantes selon ce problème:
# - mainDataPath
# - number_images
# - number_images_class_x
# - image_scale
# - images_color_mode
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# L'emplacement des images
mainDataPath = "donnees/"

# On évalue le modèle sur les images tests
datapath = mainDataPath + "test"

# Le nombre des images de test à évaluer
number_images = 600 # 600 images
number_images_class_0 = 300  # Dauphins
number_images_class_1 = 300  # Requins

# Les étiquettes (classes) des images
labels = np.array([0] * number_images_class_0 +
                  [1] * number_images_class_1)

# La taille des images
image_scale = 140

# La couleur des images
images_color_mode = "rgb"

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images test
data_generator = ImageDataGenerator(rescale=1. / 255)

generator = data_generator.flow_from_directory(
    datapath, # Place des images d'entrainement
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale), # taille des images
    batch_size=number_images, # nombre d'images total à charger en mémoire
    class_mode=None,
    shuffle=False) # pas besoin de bouleverser les images

x = generator.__next__()

# ***********************************************
#            RÉPONSE À LA QUESTION 2
# ***********************************************

reconstructed = autoencoder.predict(x)

# Affichage d’un dauphin (classe 0)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(x[0])
axs[0].set_title("Dauphin (originale)")
axs[0].axis('off')
axs[1].imshow(reconstructed[0])
axs[1].set_title("Dauphin (reconstruite)")
axs[1].axis('off')
plt.tight_layout()
plt.show()

# Affichage d’un requin (classe 1)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(x[number_images_class_0])
axs[0].set_title("Requin (originale)")
axs[0].axis('off')
axs[1].imshow(reconstructed[number_images_class_0])
axs[1].set_title("Requin (reconstruite)")
axs[1].axis('off')
plt.tight_layout()
plt.show()

# ***********************************************
#              RÉPONSE À LA QUESTION 3
# ***********************************************

# Le dernier MaxPooling2D du modèle est à l’indice 13 (selon model.summary())
encoder = Model(autoencoder.input, autoencoder.layers[13].output)

embedding = encoder.predict(x)

# Flatten l’embedding
embedding_flattened = embedding.reshape(embedding.shape[0], -1)

# ***********************************************
#              RÉPONSE À LA QUESTION 4
# ***********************************************

scaler = StandardScaler()
embedding_normalized = scaler.fit_transform(embedding_flattened)

# ***********************************************
#              RÉPONSE À LA QUESTION 5
# ***********************************************

svm_raw = SVC(kernel='linear')
accuracy_raw = cross_val_score(svm_raw, x.reshape(x.shape[0], -1), labels, cv=5, scoring='accuracy')
print(f"Accuracy SVM sur les images originales : {np.mean(accuracy_raw):.4f}")

# ***********************************************
#              RÉPONSE À LA QUESTION 6
# ***********************************************

svm_embed = SVC(kernel='linear')
accuracy_embed = cross_val_score(svm_embed, embedding_normalized, labels, cv=5, scoring='accuracy')
print(f"Accuracy SVM sur l'embedding : {np.mean(accuracy_embed):.4f}")

# ***********************************************
#              RÉPONSE À LA QUESTION 7
# ***********************************************

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedding_tsne = tsne.fit_transform(embedding_flattened)

plt.figure(figsize=(8, 6))
plt.scatter(embedding_tsne[labels == 0, 0], embedding_tsne[labels == 0, 1], label='Dauphins', alpha=0.6)
plt.scatter(embedding_tsne[labels == 1, 0], embedding_tsne[labels == 1, 1], label='Requins', alpha=0.6)
plt.title("TSNE des embeddings test (2D)")
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()