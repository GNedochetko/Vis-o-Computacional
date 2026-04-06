import os
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np


ROI_TOP_LEFT = (160, 120)
ROI_BOTTOM_RIGHT = (480, 360)
MAX_CORNERS = 80
MIN_TRACKED_POINTS = 6
GESTURE_THRESHOLD = 90
COMMAND_COOLDOWN_SECONDS = 1.0


def _configure_opencv_display():
    # No GNOME/Wayland, o OpenCV do pip costuma abrir janelas via plugin Qt xcb.
    # Forcamos o uso do XWayland e tentamos localizar o arquivo de autorizacao correto.
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    xauthority = os.environ.get("XAUTHORITY")
    if xauthority and Path(xauthority).exists():
        return

    runtime_dir = Path(f"/run/user/{os.getuid()}")
    candidates = sorted(runtime_dir.glob(".mutter-Xwaylandauth.*"))
    if candidates:
        os.environ["XAUTHORITY"] = str(candidates[0])


def _get_ydotool_socket_path():
    return os.environ.get("YDOTOOL_SOCKET", f"/run/user/{os.getuid()}/.ydotool_socket")


def _ensure_ydotool_available():
    ydotool_path = shutil.which("ydotool")
    if ydotool_path is None:
        raise RuntimeError(
            "O comando 'ydotool' nao esta instalado. "
            "Instale o ydotool e inicie o daemon 'ydotoold' para enviar teclas no Wayland."
        )
    return ydotool_path


def _send_slide_key(direction):
    key_code = "106" if direction == "right" else "105"
    socket_path = _get_ydotool_socket_path()
    command_env = os.environ.copy()
    command_env["YDOTOOL_SOCKET"] = socket_path
    key_name = "Right" if direction == "right" else "Left"

    # O ydotool injeta eventos de teclado via uinput e depende do daemon ydotoold.
    try:
        subprocess.run(
            ["ydotool", "key", f"{key_code}:1", f"{key_code}:0"],
            check=True,
            capture_output=True,
            text=True,
            env=command_env,
        )
        print(f"[gesture] tecla enviada: {key_name}")
    except subprocess.CalledProcessError as error:
        error_message = error.stderr.strip() or error.stdout.strip() or str(error)
        raise RuntimeError(
            "O ydotool falhou ao enviar a tecla para o slide. "
            f"Erro original: {error_message}"
        ) from error


def _select_points(gray_frame, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    roi = gray_frame[y1:y2, x1:x2]
    roi_points = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=MAX_CORNERS,
        qualityLevel=0.2,
        minDistance=7,
        blockSize=7,
    )

    if roi_points is None:
        return None

    # Os pontos sao detectados na ROI e depois convertidos para coordenadas da imagem.
    roi_points[:, 0, 0] += x1
    roi_points[:, 0, 1] += y1
    return roi_points


def _draw_interface(frame, tracking_active, tracked_points, accumulated_dx):
    cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)
    cv2.putText(
        frame,
        "Posicione a mao na area verde e pressione S para iniciar",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        "Pressione R para recalibrar ou Q para sair",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    status_text = "Rastreamento ativo" if tracking_active else "Rastreamento parado"
    cv2.putText(
        frame,
        status_text,
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Pontos rastreados: {tracked_points}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Deslocamento horizontal: {accumulated_dx:.1f}",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


def _reset_tracking_state():
    return None, None, 0.0, False


def start_gesture_interface(camera_index=0):
    _configure_opencv_display()
    _ensure_ydotool_available()

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Nao foi possivel acessar a webcam.")

    previous_gray = None
    tracked_points, previous_points, accumulated_dx, tracking_active = (
        _reset_tracking_state()
    )
    last_command_time = 0.0
    feedback_message = "Pressione S para detectar pontos na mao"
    feedback_color = (255, 255, 255)

    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Nao foi possivel capturar um frame da webcam.")

            frame = cv2.flip(frame, 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if tracking_active and previous_gray is not None and previous_points is not None:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    previous_gray,
                    gray_frame,
                    previous_points,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        30,
                        0.01,
                    ),
                )

                if next_points is None or status is None:
                    tracked_points, previous_points, accumulated_dx, tracking_active = (
                        _reset_tracking_state()
                    )
                else:
                    good_new = next_points[status.flatten() == 1]
                    good_old = previous_points[status.flatten() == 1]

                    if len(good_new) < MIN_TRACKED_POINTS:
                        tracked_points, previous_points, accumulated_dx, tracking_active = (
                            _reset_tracking_state()
                        )
                    else:
                        # A media do deslocamento horizontal dos pontos indica a direcao do gesto.
                        delta_x = np.mean(good_new[:, 0] - good_old[:, 0])
                        accumulated_dx += float(delta_x)

                        for point in good_new:
                            x, y = point.ravel()
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

                        current_time = time.time()
                        if (
                            accumulated_dx >= GESTURE_THRESHOLD
                            and current_time - last_command_time >= COMMAND_COOLDOWN_SECONDS
                        ):
                            _send_slide_key("right")
                            last_command_time = current_time
                            accumulated_dx = 0.0
                            feedback_message = "Gesto para a direita detectado"
                            feedback_color = (0, 255, 0)
                            cv2.putText(
                                frame,
                                "Slide seguinte",
                                (20, 190),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )
                        elif (
                            accumulated_dx <= -GESTURE_THRESHOLD
                            and current_time - last_command_time >= COMMAND_COOLDOWN_SECONDS
                        ):
                            _send_slide_key("left")
                            last_command_time = current_time
                            accumulated_dx = 0.0
                            feedback_message = "Gesto para a esquerda detectado"
                            feedback_color = (0, 255, 0)
                            cv2.putText(
                                frame,
                                "Slide anterior",
                                (20, 190),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )

                        previous_points = good_new.reshape(-1, 1, 2)
                        tracked_points = len(good_new)

            if tracked_points is None:
                tracked_points = 0

            _draw_interface(frame, tracking_active, tracked_points, accumulated_dx)
            cv2.putText(
                frame,
                feedback_message,
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                feedback_color,
                2,
            )
            cv2.imshow("Interface Gestual", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                tracked_points, previous_points, accumulated_dx, tracking_active = (
                    _reset_tracking_state()
                )
                previous_gray = None
                feedback_message = "Rastreamento reiniciado"
                feedback_color = (0, 255, 255)
            if key == ord("s"):
                selected_points = _select_points(
                    gray_frame,
                    ROI_TOP_LEFT,
                    ROI_BOTTOM_RIGHT,
                )

                if selected_points is not None and len(selected_points) >= MIN_TRACKED_POINTS:
                    previous_points = selected_points.astype(np.float32)
                    tracking_active = True
                    accumulated_dx = 0.0
                    tracked_points = len(previous_points)
                    previous_gray = gray_frame.copy()
                    feedback_message = f"Rastreamento iniciado com {tracked_points} pontos"
                    feedback_color = (0, 255, 0)
                else:
                    tracking_active = False
                    tracked_points = 0
                    feedback_message = (
                        "Nao foi possivel detectar pontos suficientes na area verde"
                    )
                    feedback_color = (0, 0, 255)

            if tracking_active:
                previous_gray = gray_frame.copy()

    finally:
        capture.release()
        cv2.destroyAllWindows()
