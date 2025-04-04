import cv2
from src2.yolo_model_handler import YOLOModelHandler

class ImageProcessor:
    def __init__(self, image_path, config_path, weights_path, class_names_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            raise FileNotFoundError(f" Feil: Bildet ble ikke funnet på {image_path}")

        # Les klassenavn
        with open(class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Initialiser YOLO-modellen.
        self.model_handler = YOLOModelHandler(config_path, weights_path, class_names_path)

    def process_image(self):
        self.model_handler.detect_objects(self.image_path)
        print(" Bildeprosessering fullført.")
