import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

#  Last inn den trente modellen
model = tf.keras.models.load_model("fish_nofish_model_v2.h5")

#  Kompiler modellen (nødvendig for evaluering)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#  Definer test-datasettet
test_dir = "dataset/test"
img_size = (224, 224)
batch_size = 8

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Viktig for evaluering
)

#  Evaluer modellen på testsettet
print("\n Kjører evaluering på testsettet...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print(f"\n Modellnøyaktighet på testsettet: {test_accuracy:.4f}")
print(f" Modell-loss på testsettet: {test_loss:.4f}")

#  Lag forutsigelser på testsettet
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype("int32")  # Threshold på 0.5

#  Hent sanne etiketter
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

#  Generer forvirringsmatrise
conf_matrix = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predikert etikett")
plt.ylabel("Sann etikett")
plt.title("Forvirringsmatrise for Fish/NoFish")
plt.show()

#  Skriv ut klassifikasjonsrapport
print("\n Klassifikasjonsrapport:\n")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print("\n Evaluering fullført!")  # Sikkerhetsmelding for å vite at koden er ferdig
