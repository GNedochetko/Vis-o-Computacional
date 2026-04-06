from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_DIR / "imagens"
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _find_image_path(base_name):
    for extension in SUPPORTED_EXTENSIONS:
        candidate = IMAGES_DIR / f"{base_name}{extension}"
        if candidate.exists():
            return candidate
    return None


def load_fixed_images():
    image1_path = _find_image_path("img1")
    image2_path = _find_image_path("img2")

    if image1_path is None or image2_path is None:
        raise FileNotFoundError(
            "Nao foi possivel encontrar img1 e img2 na pasta 'imagens'. "
            "Use nomes como img1.jpg e img2.jpg."
        )

    image1 = cv2.imread(str(image1_path))
    image2 = cv2.imread(str(image2_path))

    if image1 is None or image2 is None:
        raise ValueError("OpenCV nao conseguiu carregar uma das imagens.")

    return image1, image2, image1_path, image2_path