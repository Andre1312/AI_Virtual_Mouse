"""
MediaPipe 0.10.33 - Virtual Mouse com HandLandmarker (Nova API)
pip install mediapipe==0.10.33
pip install opencv-python
pip install pyautogui
"""
import cv2
import mediapipe as mp
import pyautogui
import time

# Inicializar captura de vídeo
cap = cv2.VideoCapture(0)

# Inicializar a nova API do MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

screen_width, screen_height = pyautogui.size()

index_x = 0
index_y = 0

# Callback para resultados em modo live stream
def on_hand_detected(result, output_image, timestamp_ms):
    global index_x, index_y

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Processar cada landmark
            for id, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                # Ponto 8: ponta do dedo indicador
                if id == 8:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y

                # Ponto 4: ponta do polegar
                if id == 4:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y

                    # Calcular distância
                    distance = abs(index_y - thumb_y)
                    print(f'Distância: {distance:.2f}')

                    # Clicar se muito próximos
                    if distance < 20:
                        pyautogui.click()
                        pyautogui.sleep(0.5)
                    # Mover cursor se proximidade média
                    elif distance < 100:
                        pyautogui.moveTo(index_x, index_y)

# Configurar opções do HandLandmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=None),  # Use modelo padrão
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5
)

# Criar o landmarker
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()

        if not success:
            print("Erro ao capturar frame")
            break

        # Espelhar e obter dimensões
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        # Converter para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Converter para mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detectar landmarks
        detection_result = landmarker.detect(mp_image)

        # Processar resultados
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Desenhar landmarks
                for id, landmark in enumerate(hand_landmarks):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)

                    # Ponto 8: indicador
                    if id == 8:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                        index_x = screen_width / frame_width * x
                        index_y = screen_height / frame_height * y

                    # Ponto 4: polegar
                    if id == 4:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                        thumb_x = screen_width / frame_width * x
                        thumb_y = screen_height / frame_height * y

                        distance = abs(index_y - thumb_y)
                        print(f'Distância: {distance:.2f}')

                        if distance < 20:
                            pyautogui.click()
                            pyautogui.sleep(0.5)
                        elif distance < 100:
                            pyautogui.moveTo(index_x, index_y)

        # Exibir informações
        cv2.putText(frame, "Pressione 'q' para sair", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Virtual Mouse - MediaPipe 0.10.33', frame)

        # Sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
