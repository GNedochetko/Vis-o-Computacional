import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import pyautogui
except ImportError:
    pyautogui = None


FEATURE_PARAMS = {
    "maxCorners": 80,      # numero maximo de cantos
    "qualityLevel": 0.2,   # qualidade minima aceita
    "minDistance": 7,      # distancia minima entre pontos
    "blockSize": 7,        # tamanho da vizinhanca analisada
}

LK_PARAMS = {
    "winSize": (21, 21),   # tamanho da janela de busca
    "maxLevel": 3,         # niveis da piramide
    "criteria": (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        20,                # maximo de iteracoes
        0.03,              # tolerancia de erro
    ),
}

@dataclass
class GestureState:
    tracking_started: bool = False              # indica se o rastreio iniciou
    previous_gray: Optional[np.ndarray] = None  # frame anterior em cinza
    tracked_points: Optional[np.ndarray] = None # pontos atualmente rastreados
    cumulative_dx: float = 0.0                  # deslocamento horizontal acumulado
    last_command_time: float = 0.0              # instante do ultimo comando


# Calcula a regiao central onde a mao deve ser posicionada.
def _central_roi(frame_shape, scale=0.35):
    height, width = frame_shape[:2]
    roi_width = int(width * scale)
    roi_height = int(height * scale)
    x1 = (width - roi_width) // 2
    y1 = (height - roi_height) // 2
    x2 = x1 + roi_width
    y2 = y1 + roi_height
    return x1, y1, x2, y2


# Detecta pontos de interesse dentro da regiao central.
def _detect_points(gray_frame, roi):
    x1, y1, x2, y2 = roi
    roi_gray = gray_frame[y1:y2, x1:x2]
    corners = cv2.goodFeaturesToTrack(roi_gray, mask=None, **FEATURE_PARAMS)

    if corners is None:
        return None

    # Ajusta os pontos do recorte para o sistema de coordenadas do frame inteiro.
    corners[:, 0, 0] += x1
    corners[:, 0, 1] += y1
    return corners


# Inicializa o rastreamento com os pontos detectados.
def _initialize_tracking(gray_frame, state):
    roi = _central_roi(gray_frame.shape)
    tracked_points = _detect_points(gray_frame, roi)

    if tracked_points is None or len(tracked_points) < 8:
        return False

    state.tracking_started = True
    state.previous_gray = gray_frame
    state.tracked_points = tracked_points
    state.cumulative_dx = 0.0
    return True


# Desenha a interface visual e os pontos rastreados no frame.
def _draw_overlay(frame, state, status_message):
    x1, y1, x2, y2 = _central_roi(frame.shape)
    color = (0, 255, 0) if state.tracking_started else (0, 255, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        frame,
        "S: iniciar  R: recalibrar  Q: sair",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        status_message,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    if state.tracking_started and state.tracked_points is not None:
        for point in state.tracked_points.reshape(-1, 2):
            cv2.circle(frame, tuple(np.int32(point)), 3, (255, 0, 0), -1)


# Envia o comando do slide respeitando o tempo de espera.
def _send_slide_command(direction, state, cooldown_seconds):
    current_time = time.time()
    if current_time - state.last_command_time < cooldown_seconds:
        return

    if direction == "right":
        pyautogui.press("right")
    else:
        pyautogui.press("left")

    state.last_command_time = current_time
    state.cumulative_dx = 0.0


# Atualiza os pontos rastreados e detecta o gesto horizontal.
def _update_tracking(gray_frame, state, movement_threshold, cooldown_seconds):
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        state.previous_gray,
        gray_frame,
        state.tracked_points,
        None,
        **LK_PARAMS,
    )

    if next_points is None or status is None:
        state.tracking_started = False
        state.tracked_points = None
        state.previous_gray = gray_frame
        return "Rastreamento perdido. Pressione S para iniciar novamente."

    valid_new = next_points[status.flatten() == 1]
    valid_old = state.tracked_points[status.flatten() == 1]

    if len(valid_new) < 6:
        state.tracking_started = False
        state.tracked_points = None
        state.previous_gray = gray_frame
        return "Poucos pontos encontrados. Reposicione a mao e pressione S."

    # Usa a mediana para reduzir o impacto de pontos ruidosos ou fora da mao.
    horizontal_displacement = np.median(valid_new[:, 0] - valid_old[:, 0])
    state.cumulative_dx += float(horizontal_displacement)

    if state.cumulative_dx >= movement_threshold:
        _send_slide_command("right", state, cooldown_seconds)
        status_message = "Gesto para direita detectado. Slide avancado."
    elif state.cumulative_dx <= -movement_threshold:
        _send_slide_command("left", state, cooldown_seconds)
        status_message = "Gesto para esquerda detectado. Slide retrocedido."
    else:
        status_message = "Rastreando mao..."

    state.previous_gray = gray_frame
    state.tracked_points = valid_new.reshape(-1, 1, 2)
    return status_message


# Inicia a captura da webcam e o controle de slides por gestos.
def start_gesture_interface(
    camera_index=0,
    movement_threshold=45,
    cooldown_seconds=1.0,
):
    if pyautogui is None:
        raise RuntimeError(
            "A biblioteca pyautogui nao esta instalada. "
            "Instale as dependencias com 'pip install -r requirements.txt'."
        )

    # Desabilita o fail-safe para nao interromper a apresentacao ao tocar
    # acidentalmente os cantos da tela com o mouse.
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Nao foi possivel acessar a webcam.")

    state = GestureState()
    status_message = "Posicione a mao na area central e pressione S."

    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Falha ao capturar frame da webcam.")

            frame = cv2.flip(frame, 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if state.tracking_started and state.tracked_points is not None:
                status_message = _update_tracking(
                    gray_frame,
                    state,
                    movement_threshold,
                    cooldown_seconds,
                )

            _draw_overlay(frame, state, status_message)
            cv2.imshow("Controle de Slides por Gestos", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                state = GestureState()
                status_message = "Calibracao reiniciada. Posicione a mao e pressione S."
            if key == ord("s"):
                if _initialize_tracking(gray_frame, state):
                    status_message = "Rastreamento iniciado. Mova a mao horizontalmente."
                else:
                    status_message = (
                        "Nao foi possivel detectar pontos suficientes. "
                        "Aproxime a mao e tente novamente."
                    )
    finally:
        capture.release()
        cv2.destroyAllWindows()
