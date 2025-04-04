import cv2
import numpy as np
import os

class YOLOModelHandler:
    def __init__(self, 
                 config_path="yolo_model/yolov4-tiny.cfg",
                 weights_path="yolo_model/yolov4-tiny_best.weights",
                 class_names_path="yolo_model/classes.txt"):
        
        #  Test at filene finnes før vi prøver å laste dem.
        for path in [config_path, weights_path, class_names_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f" Feil: Finner ikke {path}")

        print(f" Laster inn YOLO-modellen fra:\n  - Config: {config_path}\n  - Weights: {weights_path}\n  - Classes: {class_names_path}")

        #  Last inn YOLO-modellen
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

        #  Les klassene fra filen
        with open(class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        print(f" Lastet inn {len(self.class_names)} klasser.")

        layer_names = self.net.getLayerNames()
        
        try:
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        except Exception:
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, image_path):
        print(f" (yolo_model_handler.py) Bruker dette bildet: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f" Feil: Bildet ble ikke funnet på {image_path}")

        print(f" Behandler bildet: {image_path}")

        height, width, _ = image.shape

        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.2:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

        if len(indices) > 0 and isinstance(indices, np.ndarray):
            indices = indices.flatten()
        else:
            indices = []

        print(f" Totalt {len(indices)} objekter detektert.")

        if len(indices) == 0:
            print(" Ingen sikre objekter ble funnet! Kan være at modellen ikke er godt nok trent.")
            cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), 2)  # Test-boks
        else:
            for i in indices:
                box = boxes[i]
                x, y, w, h = box
                label = str(self.class_names[class_ids[i]]) if class_ids[i] < len(self.class_names) else "Unknown"
                confidence = str(round(confidences[i], 2))
                color = [int(c) for c in np.random.randint(0, 255, size=3)]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, f"{label}: {confidence}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_path = os.path.join(os.getcwd(), "detected_fish.jpg")
        cv2.imwrite(output_path, image)
        print(f" Lagret detektert bilde: {output_path}")

        cv2.imshow("Detected Objects", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
