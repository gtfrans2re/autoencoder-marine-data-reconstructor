# **************************************************************************
# INF7370 - Apprentissage automatique
# Fichier de remise du Travail Pratique 3 (TP3)
# réalisé par :
# Francois Gonothi Toure, TOUF23329808
# et
# Martial Zachee Kaljob Kollo, KALM30319404
# ===========================================================================

# ===========================================================================
# Ce modèle est un autoencodeur convolutif entraîné sur un ensemble de données
# composé d’images marines appartenant à deux classes : dauphin et requin.
# Il a pour objectif de reconstruire les images d'entrée et de générer une
# représentation compacte (embedding) permettant d'évaluer la qualité des
# descripteurs produits via une classification SVM et une visualisation TSNE.
#
# Données:
# ------------------------------------------------
# Entraînement   : 3 240 images (1 620 dauphins + 1 620 requins)
# Validation     : 360 images (180 dauphins + 180 requins)
# Test           : 600 images (300 dauphins + 300 requins)
# ------------------------------------------------
#
# >>> Ce code était initialement prévu pour des données MNIST.
# >>> Nous avons adapté ce fichier pour traiter des images RGB (140x140x3)
# >>> de deux classes marines dans le cadre du TP3.
# >>> Les étapes incluent la reconstruction, l'extraction des embeddings,
# >>> l'évaluation par SVM, et la visualisation des descripteurs en 2D.
# ===========================================================================

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

from tensorflow import keras

# La libraire responsable du chargement des données dans la mémoir
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Le Type de modéle à compiler
from keras.models import Model

# Le type d'optimisateur utilisé dans notre modèle (Adam) :
# L'optimisateur ajuste les poids de notre modèle par descente du gradient
# Chaque optimisateur ayant ses propres paramètres,
# Nous avons testé principalement l'optimisateur Adam et ajusté les paramètres 
# afin d'avoir les meilleurs résultats dans le cadre de notre remise du TP3.
from tensorflow.keras.optimizers import Adam

# Les types des couches utlilisées dans notre modèle
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Cropping2D

# Des outils pour suivre et gérer l'entrainement de notre modèle
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Affichage des graphes
import matplotlib.pyplot as plt

# Utilitaire pour les chemins
import os

# Utilitaire pour mesurer du temps
import time

# ==========================================
# ===============GPU & CPU==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(sess);
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# ==========================================
# ================VARIABLES=================
# ==========================================

# ******************************************************
#               RÉPONSE À LA QUESTION DU TP3
# ******************************************************
# 1) Ajustement des variables selon le problème donné :
# - mainDataPath
# - training_ds_size
# - validation_ds_size
# - image_scale
# - image_channels
# - images_color_mode
# - fit_batch_size
# - fit_epochs
# ******************************************************

# Le dossier principal qui contient les données
mainDataPath = "donnees/"

# Le dossier contenant les images d'entrainement
trainPath = mainDataPath + "entrainement"

# Le dossier contenant les images de validation
validationPath = mainDataPath + "entrainement"

# Le nom du fichier du modèle à sauvegarder
model_path = "Model.keras"

# Le nombre d'images d'entrainement
training_ds_size = 3240
validation_ds_size = 360

# Configuration des  images
image_scale = 140 # la taille des images
image_channels = 3  # le nombre de canaux de couleurs


images_color_mode = "rgb" #(3 pour les images en couleurs (rouge vert bleu) )

# la forme des images d'entrées, ce qui correspond à la couche d'entrée du réseau
image_shape = (image_scale, image_scale, image_channels)

# Configuration des paramètres d'entrainement
fit_batch_size = 16 # le nombre d'images entrainées ensemble: un batch
fit_epochs = 100  # Le nombre d'époques

# ==========================================
# ==================MODÈLE==================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#             RÉPONSES AUX QUESTIONS DU TP3
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Ajustement des deux fonctions:
# 2) encoder
# 3) decoder
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Couche d'entrée:
# Cette couche prend comme paramètre la forme des images (image_shape)
input_layer = Input(shape=image_shape)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       ENCODER
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ===========================================================================
# Définition de l’encodeur (partie descendante de l’autoencodeur)
# L’encodeur extrait progressivement les caractéristiques visuelles
# à travers une série de blocs convolutionnels, réduisant la dimension spatiale
# tout en augmentant la profondeur des représentations.
# ===========================================================================

def encoder(input):
    
    # Bloc 1 :
    # - Extraction initiale des motifs bas niveau (textures, bords)
    # - 32 filtres convolutionnels 3x3 avec padding pour conserver la taille
    # - Normalisation par lot pour stabiliser les activations
    # - Activation ReLU pour introduire la non-linéarité
    # - Réduction spatiale via MaxPooling (2x2)
    # - Dropout à 20% pour la régularisation
    x = Conv2D(32, (3, 3), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)

    # Bloc 2 :
    # - Approfondissement de l’extraction des caractéristiques
    # - 64 filtres pour capter des motifs plus complexes
    # - Même structure que le bloc précédent
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)

    # Bloc 3 :
    # - Dernier niveau d’encodage : motifs abstraits et profonds
    # - 128 filtres avec ReLU, BatchNorm et Pooling
    # - Aucun Dropout ici pour préserver l'information du code latent
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    return encoded

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       DECODER
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ===========================================================================
# Définition du décodeur (partie ascendante de l’autoencodeur)
# Le décodeur prend en entrée le code latent (encoded) et reconstruit
# l’image originale à travers des opérations inverses de celles de l’encodeur :
# convolutions, normalisation, activation et upsampling.
# ===========================================================================

def decoder(encoded):

    # Bloc 1 :
    # - Décompression initiale du code latent
    # - 128 filtres 3x3 pour reconstruire des motifs profonds
    # - Normalisation par lot et activation ReLU
    # - UpSampling (x2) pour doubler la taille spatiale
    x = Conv2D(128, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)

    # Bloc 2 :
    # - Reconstruction intermédiaire avec 64 filtres
    # - Même structure que le bloc précédent
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)

    # Bloc 3 :
    # - Dernier bloc convolutif avec 32 filtres
    # - Prépare les canaux d’image pour la reconstruction finale
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)

    # Couche de sortie :
    # - Reconstruit une image RGB (3 canaux) avec activation sigmoid
    # - Output : taille 144x144x3 (à cause du padding 'same')
    x = Conv2D(image_channels, (3, 3), padding='same')(x)
    x = Activation('sigmoid')(x)

    # Recadrage (cropping) :
    # - Coupe les 2 pixels de chaque côté pour revenir à la taille d’origine : 140x140x3
    x = Cropping2D(((2, 2), (2, 2)))(x)

    return x  # Important : ne pas utiliser une variable nommée "decoded"

# Déclaration du modèle:
# La sortie de l'encodeur sert comme entrée à la partie decodeur
model = Model(input_layer, decoder(encoder(input_layer)))

# Affichage des paramétres du modèle
# Cette commande affiche un tableau avec les détails du modèle
# (nombre de couches et de paramétres ...)
model.summary()

# Compilation du modèle :
# loss: On définit la fonction de perte (généralement on utilise le MSE pour les autoencodeurs standards)
# optimizer: L'optimisateur utilisé avec ses paramétres (Exemple : optimizer=adam(learning_rate=0.001) )
# metrics: La valeur à afficher durant l'entrainement, metrics=['mse']
# On suit le loss (ou la difference) de l'autoencodeur entre les images d'entrée et les images de sortie
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mse'])

# ==========================================
# ==========CHARGEMENT DES IMAGES===========
# ==========================================

# ===========================================================================
# Chargement des images pour l'entraînement et la validation
# Ce bloc utilise ImageDataGenerator pour :
#   - normaliser les images dans [0, 1] (rescale=1./255),
#   - découper automatiquement l'ensemble de données en deux sous-ensembles :
#     • 90 % pour l'entraînement (subset='training')
#     • 10 % pour la validation (subset='validation')
#   - charger dynamiquement les images depuis leur répertoire
#
# Le mode 'class_mode' est ici défini à 'input' car on entraîne un autoencodeur :
#   → les étiquettes sont les mêmes que les entrées (l'image est sa propre cible).
#
# Note :
#   • shuffle=False est utilisé pour garder un ordre constant des images.
#   • La structure du répertoire doit respecter : train/class_x/*.jpg
# ===========================================================================

training_data_generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.1)

training_generator = training_data_generator.flow_from_directory(
    trainPath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=training_ds_size,
    class_mode="input",
    subset='training',
    shuffle=False
)

validation_generator = training_data_generator.flow_from_directory(
    validationPath,
    color_mode=images_color_mode,
    target_size=(image_scale, image_scale),
    batch_size=validation_ds_size,
    class_mode="input",
    subset='validation',
    shuffle=False
)

# Extraction manuelle d'un lot complet d'images depuis les générateurs
# Ces variables contiendront tous les échantillons d'entraînement et validation
# nécessaires à l'entraînement de l'autoencodeur.
(x_train, _) = training_generator.__next__()
(x_val, _) = validation_generator.__next__()

# ==========================================
# ==============ENTRAINEMENT================
# ==========================================

# ===========================================================================
# Phase d'entraînement du modèle (fit) avec callbacks de régulation
#
# Cette étape consiste à entraîner l'autoencodeur à partir des données d’entrée (x_train)
# qui servent également de cible, car il s’agit d’un apprentissage non supervisé.
#
# Trois mécanismes de régulation sont utilisés ici :
#
# - ModelCheckpoint : Sauvegarde automatiquement le meilleur modèle (val_loss minimum).
# - ReduceLROnPlateau : Réduit le taux d’apprentissage si la perte de validation stagne.
# - EarlyStopping : Interrompt précocement l’entraînement pour éviter le surapprentissage.
#
# L’entraînement est réalisé avec :
#   - x_train comme entrée ET cible (autoencodeur)
#   - validation_data sur x_val pour monitorer la performance
#   - shuffle=False pour garder le même ordre des échantillons
#
# La durée d’entraînement est mesurée à l’aide de time.time()
# ===========================================================================

modelcheckpoint = ModelCheckpoint(filepath=model_path,
                                  monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

start_time = time.time()

autoencoder = model.fit(x_train, x_train,
                        epochs=fit_epochs,
                        batch_size=fit_batch_size,
                        verbose=1,
                        callbacks=[modelcheckpoint, reduce_lr, early_stop],
                        shuffle=False,
                        validation_data=(x_val, x_val))

end_time = time.time()

# ==========================================
# ========AFFICHAGE DES RESULTATS===========
# ==========================================

# ===========================================================================
# AFFICHAGE DES RÉSULTATS D'ENTRAÎNEMENT
#
# Cette section permet de :
#   - Calculer et afficher le temps total d’entraînement (en minutes)
#   - Visualiser graphiquement l’évolution de la perte (loss) à chaque époque,
#     pour l’entraînement et la validation.
#
# Cela nous permet de :
#   • Vérifier si la convergence est stable (courbes descendantes)
#   • Détecter le surapprentissage (écart croissant entre les courbes)
#   • Identifier l’époque à laquelle le modèle a obtenu la meilleure validation
#
# Le minimum de la perte de validation est également extrait et affiché,
# car il correspond au meilleur compromis entre apprentissage et généralisation.
# ===========================================================================

# ***********************************************
#                    QUESTION
# ***********************************************

# Afficher le temps d'execution
print(f"\nTemps total d'entraînement : {(end_time - start_time)/60:.2f} minutes")

# ***********************************************
#                    QUESTION
# ***********************************************

# Afficher la courbe de  perte par époque (loss over epochs)
plt.figure(figsize=(8, 5))
plt.plot(autoencoder.history['loss'], label='train')
plt.plot(autoencoder.history['val_loss'], label='validation')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
plt.show()

min_val_loss = min(autoencoder.history['val_loss'])
# Résumé de la performance
print(f"Minimum validation loss atteint : {min_val_loss:.6f}")