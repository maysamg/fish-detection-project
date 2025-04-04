import sys
import os
import cv2
from region_proposal import generate_region_proposals
from yolo_model_handler import YOLOModelHandler

try:
    print(" main.py startet!")

    # Finn den absolutte stien til `src2`
    src2_path = os.path.abspath(os.path.dirname(__file__))

    # Sett opp riktig sti for `dataset/test/`
    dataset_test_path = os.path.join(os.path.dirname(src2_path), "dataset", "test")

    # Velg spesifikk kategori
    category = "Fish"
    category_path = os.path.join(dataset_test_path, category)

    # Sett spesifikt bilde
    #image_filename = "Fishdetection_frame_000255.PNG"
    #image_filename = "20230716080335_20230716080344_3.mp4_frame_000028.PNG"
    #image_filename = "Fishdetection_frame_000114.PNG"
    image_filename = "Fishdetection_frame_000037.PNG"
    image_path = os.path.join(category_path, image_filename)

    # Sjekk om bildet eksisterer.
    if not os.path.exists(image_path):
        raise FileNotFoundError(f" Feil: Bildet ble ikke funnet på {image_path}!")

    print(f" Valgt kategori: {category}")
    print(f" Valgt bilde: {image_path}")

    # Finn riktig YOLO-modellmappe
    yolo_model_path = os.path.join(os.path.dirname(src2_path), "yolo_model")

    # Oppdater YOLO filbaner
   # config_path = os.path.join(yolo_model_path, "yolov4-tiny.cfg")
   # weights_path = os.path.join(yolo_model_path, "yolov4-tiny_best.weights")
    weights_path = os.path.join(yolo_model_path, "yolo-fish-2.weights")
    config_path = os.path.join(yolo_model_path, "yolo-fish-2.cfg")
    class_names_path = os.path.join(yolo_model_path, "classes.txt")

    # Test om filene finnes før vi kjører YOLO
    for path in [config_path, weights_path, class_names_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f" Feil: Finner ikke {path}")
    print(" Alle YOLO-filer ble funnet!")

    # Generer region proposals
    print(" Genererer region proposals...")
    proposals = generate_region_proposals(image_path)
    print(f" Antall foreslåtte regioner: {len(proposals)}")

    #  Kjør YOLO deteksjon
    print(" Kjører YOLO-modellen...")
    model_handler = YOLOModelHandler(config_path, weights_path, class_names_path)
    model_handler.detect_objects(image_path)

    print(" Prosessen er ferdig! Resultatene er lagret i detected_fish.jpg.")

except Exception as e:
    print(f" Feil oppstod: {e}")
