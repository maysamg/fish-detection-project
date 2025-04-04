import cv2
import numpy as np

def generate_region_proposals(image_path, threshold=127):
    """
    Generate region proposals from an image using simple thresholding.

    Parameters:
    - image_path: Path to the input image.
    - threshold: The threshold value for binary segmentation.

    Returns:
    - proposals: List of proposed regions, each defined by [x, y, width, height].
    """
    print(f" (region_proposal.py) Bruker dette bildet: {image_path}")

    #  Last inn bilde.
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f" Feil: Bildet ble ikke funnet på {image_path}")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #  Bruk thresholding for segmentering
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    #  Finn konturer
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #  Generer region proposals
    proposals = [cv2.boundingRect(contour) for contour in contours]

    return proposals
