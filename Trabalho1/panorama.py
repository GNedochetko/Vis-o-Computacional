import time
from pathlib import Path

import cv2
import numpy as np


def build_panorama_canvas(image1, image2, homography):
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    corners_image1 = np.float32(
        [[0, 0], [0, height1], [width1, height1], [width1, 0]]
    ).reshape(-1, 1, 2)
    corners_image2 = np.float32(
        [[0, 0], [0, height2], [width2, height2], [width2, 0]]
    ).reshape(-1, 1, 2)

    # Os cantos transformados definem o tamanho necessario para a panoramica.
    transformed_corners_image1 = cv2.perspectiveTransform(corners_image1, homography)
    all_corners = np.concatenate((transformed_corners_image1, corners_image2), axis=0)

    [x_min, y_min] = np.int32(np.floor(all_corners.min(axis=0).ravel()))
    [x_max, y_max] = np.int32(np.ceil(all_corners.max(axis=0).ravel()))

    canvas_width = max(1, x_max - x_min)
    canvas_height = max(1, y_max - y_min)

    # A translacao evita coordenadas negativas na imagem final.
    translation = [-x_min, -y_min]
    translation_matrix = np.array(
        [[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]],
        dtype=np.float64,
    )

    # WarpPerspective reposiciona a primeira imagem de acordo com a homografia.
    panorama = cv2.warpPerspective(
        image1,
        translation_matrix @ homography,
        (canvas_width, canvas_height),
    )

    y_start = translation[1]
    y_end = translation[1] + height2
    x_start = translation[0]
    x_end = translation[0] + width2

    # Copia a segunda imagem diretamente no canvas, preservando as bordas pretas.
    panorama[y_start:y_end, x_start:x_end] = image2

    return panorama


def create_panorama_orb_bf(image1, image2):
    start_time = time.perf_counter()

    # O detector trabalha em escala de cinza para encontrar pontos de interesse.
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # ORB encontra pontos-chave e gera descritores binarios para cada imagem.
    detector = cv2.ORB_create(nfeatures=3000)
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Nao foi possivel encontrar descritores nas imagens.")

    # BF compara os descritores diretamente usando distancia Hamming.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda match: match.distance)

    if len(matches) < 4:
        raise ValueError("Nao ha correspondencias suficientes para gerar a homografia.")

    # Cada match gera um par de pontos correspondente entre as duas imagens.
    src_points = np.float32(
        [keypoints1[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    dst_points = np.float32(
        [keypoints2[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)

    # A homografia estima como alinhar a primeira imagem no plano da segunda.
    homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if homography is None:
        raise ValueError("Nao foi possivel calcular a homografia.")

    panorama = build_panorama_canvas(image1, image2, homography)
    elapsed_time = time.perf_counter() - start_time

    return panorama, elapsed_time, len(matches), mask


def create_panorama_orb_flann(image1, image2):
    start_time = time.perf_counter()

    # O detector trabalha em escala de cinza para encontrar pontos de interesse.
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # ORB encontra pontos-chave e gera descritores binarios para cada imagem.
    detector = cv2.ORB_create(nfeatures=3000)
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Nao foi possivel encontrar descritores nas imagens.")

    # O FLANN com LSH e a configuracao apropriada para descritores binarios do ORB.
    descriptors1 = np.uint8(descriptors1)
    descriptors2 = np.uint8(descriptors2)

    index_params = dict(
        algorithm=6,
        table_number=6,
        key_size=12,
        multi_probe_level=1,
    )
    search_params = dict(checks=50)

    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # O ratio test filtra correspondencias ambiguas e reduz matches ruins.
    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError("Nao ha correspondencias suficientes para gerar a homografia.")

    # Cada match valido gera um par de pontos correspondente entre as duas imagens.
    src_points = np.float32(
        [keypoints1[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    dst_points = np.float32(
        [keypoints2[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)

    # A homografia estima como alinhar a primeira imagem no plano da segunda.
    homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if homography is None:
        raise ValueError("Nao foi possivel calcular a homografia.")

    panorama = build_panorama_canvas(image1, image2, homography)
    elapsed_time = time.perf_counter() - start_time

    return panorama, elapsed_time, len(good_matches), mask


def create_panorama_sift_bf(image1, image2):
    start_time = time.perf_counter()

    # O detector trabalha em escala de cinza para encontrar pontos de interesse.
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # SIFT encontra pontos-chave mais robustos e gera descritores em ponto flutuante.
    detector = cv2.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Nao foi possivel encontrar descritores nas imagens.")

    # BF compara descritores do SIFT com distancia L2.
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda match: match.distance)

    if len(matches) < 4:
        raise ValueError("Nao ha correspondencias suficientes para gerar a homografia.")

    # Cada match gera um par de pontos correspondente entre as duas imagens.
    src_points = np.float32(
        [keypoints1[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    dst_points = np.float32(
        [keypoints2[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)

    # A homografia estima como alinhar a primeira imagem no plano da segunda.
    homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if homography is None:
        raise ValueError("Nao foi possivel calcular a homografia.")

    panorama = build_panorama_canvas(image1, image2, homography)
    elapsed_time = time.perf_counter() - start_time

    return panorama, elapsed_time, len(matches), mask


def create_panorama_sift_flann(image1, image2):
    start_time = time.perf_counter()

    # O detector trabalha em escala de cinza para encontrar pontos de interesse.
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # SIFT encontra pontos-chave mais robustos e gera descritores em ponto flutuante.
    detector = cv2.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Nao foi possivel encontrar descritores nas imagens.")

    # O FLANN com KDTree e a configuracao apropriada para descritores float do SIFT.
    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # O ratio test filtra correspondencias ambiguas e reduz matches ruins.
    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError("Nao ha correspondencias suficientes para gerar a homografia.")

    # Cada match valido gera um par de pontos correspondente entre as duas imagens.
    src_points = np.float32(
        [keypoints1[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    dst_points = np.float32(
        [keypoints2[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)

    # A homografia estima como alinhar a primeira imagem no plano da segunda.
    homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if homography is None:
        raise ValueError("Nao foi possivel calcular a homografia.")

    panorama = build_panorama_canvas(image1, image2, homography)
    elapsed_time = time.perf_counter() - start_time

    return panorama, elapsed_time, len(good_matches), mask


def save_panorama(panorama, file_name):
    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "resultados"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / file_name
    success, encoded_image = cv2.imencode(output_path.suffix, panorama)
    if not success:
        raise ValueError("OpenCV nao conseguiu codificar a panoramica para salvar.")

    output_path.write_bytes(encoded_image.tobytes())

    return output_path
