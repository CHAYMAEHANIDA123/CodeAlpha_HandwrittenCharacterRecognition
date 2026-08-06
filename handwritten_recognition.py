# === IMPORTATION DES BIBLIOTHÈQUES ===
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("🔄 Chargement du dataset MNIST...")
# 1. CHARGEMENT DES DONNÉES (MNIST est intégré à TensorFlow)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. PRÉTRAITEMENT (Normalisation des pixels entre 0 et 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Redimensionnement pour ajouter la dimension des canaux (nécessaire pour CNN)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("🛠️ Construction du modèle CNN (Convolutional Neural Network)...")
# 3. CONSTRUCTION DU MODÈLE CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2), # Pour éviter le surapprentissage (overfitting)
    tf.keras.layers.Dense(10, activation='softmax') # 10 classes (chiffres 0 à 9)
])

# 4. COMPILATION DU MODÈLE
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("🤖 Entraînement du modèle en cours... (Cela peut prendre 1-2 minutes)")
# 5. ENTRAÎNEMENT
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 6. ÉVALUATION DU MODÈLE
print("\n📈 Évaluation du Modèle sur les données de test :")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"✅ Accuracy sur le test set : {test_acc * 100:.2f}%")

# 7. VISUALISATION DES RÉSULTATS
plt.figure(figsize=(15, 4))

# Courbe de précision (Accuracy) — occupe les 2 premières colonnes d'une grille à 7 colonnes
ax_acc = plt.subplot2grid((1, 7), (0, 0), colspan=2)
ax_acc.plot(history.history['accuracy'], label='Train Accuracy')
ax_acc.plot(history.history['val_accuracy'], label='Test Accuracy')
ax_acc.set_title('Précision du Modèle (Accuracy)')
ax_acc.set_xlabel('Epochs')
ax_acc.set_ylabel('Accuracy')
ax_acc.legend()

# Afficher quelques prédictions — chacune dans sa propre colonne (colonnes 2 à 6)
predictions = model.predict(x_test[:5])
for i in range(5):
    ax_img = plt.subplot2grid((1, 7), (0, 2 + i))
    ax_img.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax_img.set_title(f"Pred: {np.argmax(predictions[i])}\nTrue: {y_test[i]}")
    ax_img.axis('off')

plt.tight_layout()
plt.savefig('mnist_predictions.png') # Sauvegarde de l'image pour LinkedIn
plt.show()

print("\n✨ Task 3 Terminée avec succès! L'image 'mnist_predictions.png' a été sauvegardée.")
