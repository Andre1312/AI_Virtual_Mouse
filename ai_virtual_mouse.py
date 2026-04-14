import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from datetime import datetime

# Importações necessárias
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

# Variáveis globais para visualização
resultados_atuais = None
ultimo_timestamp = 0
fps_counter = 0
fps_atual = 0

# Configurações do mouse virtual
screen_width, screen_height = pyautogui.size()
prev_x, prev_y = 0, 0
smoothing_factor = 0.75
click_cooldown = 0
last_click_time = 0
click_hold = False

# Índices dos landmarks importantes
INDEX_TIP = 8  # Ponta do indicador
THUMB_TIP = 4  # Ponta do polegar
MIDDLE_TIP = 12  # Ponta do dedo médio
WRIST = 0  # Pulso


def calculate_distance(point1, point2):
    """Calcula a distância euclidiana entre dois pontos"""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def move_mouse(x, y, frame_width, frame_height):
    """Move o mouse para a posição correspondente na tela"""
    global prev_x, prev_y

    # Converter coordenadas da câmera para coordenadas da tela
    screen_x = np.interp(x, [0, frame_width], [0, screen_width])
    screen_y = np.interp(y, [0, frame_height], [0, screen_height])

    # Inverter eixo X para movimento natural
    screen_x = screen_width - screen_x

    # Aplicar suavização
    smooth_x = prev_x * (1 - smoothing_factor) + screen_x * smoothing_factor
    smooth_y = prev_y * (1 - smoothing_factor) + screen_y * smoothing_factor

    pyautogui.moveTo(smooth_x, smooth_y)
    prev_x, prev_y = smooth_x, smooth_y


def detect_click(thumb_pos, index_pos, middle_pos, current_time):
    """Detecta diferentes tipos de clique baseado nos gestos"""
    global last_click_time, click_hold

    # Distância entre polegar e indicador (clique esquerdo)
    pinch_distance = calculate_distance(thumb_pos, index_pos)

    # Distância entre polegar e médio (clique direito)
    middle_distance = calculate_distance(thumb_pos, middle_pos)

    # Clique esquerdo (pinça com indicador)
    if pinch_distance < 30 and (current_time - last_click_time) > 300:
        pyautogui.click(button='left')
        last_click_time = current_time
        print("Clique esquerdo detectado!")
        return "left_click"

    # Clique direito (pinça com dedo médio)
    elif middle_distance < 30 and (current_time - last_click_time) > 300:
        pyautogui.click(button='right')
        last_click_time = current_time
        print("Clique direito detectado!")
        return "right_click"

    # Clique duplo (dedos indicador e médio estendidos)
    elif pinch_distance > 50 and middle_distance > 50:
        # Verificar se indicador e médio estão próximos
        fingers_distance = calculate_distance(index_pos, middle_pos)
        if fingers_distance < 40:
            pyautogui.doubleClick()
            last_click_time = current_time
            print("Clique duplo detectado!")
            return "double_click"

    return None


def detect_scroll(index_pos, middle_pos, frame_height):
    """Detecta movimento de scroll"""
    # Se indicador e médio estiverem estendidos e juntos
    fingers_distance = calculate_distance(index_pos, middle_pos)
    if fingers_distance < 40:
        # Usar posição Y relativa para scroll
        scroll_value = int((index_pos[1] - middle_pos[1]) / 10)
        if abs(scroll_value) > 2:
            pyautogui.scroll(scroll_value)
            return scroll_value
    return 0


# 1. DEFINIÇÃO DO CALLBACK
def callback_resultado(resultado: HandLandmarkerResult,
                       imagem_mp: mp.Image,
                       timestamp_ms: int):
    """
    Esta função é chamada automaticamente a cada quadro processado
    """
    global resultados_atuais, fps_counter, fps_atual

    resultados_atuais = resultado

    # Contador para FPS
    fps_counter += 1
    if timestamp_ms - fps_atual >= 1000:  # Atualiza FPS a cada segundo
        fps_atual = timestamp_ms
        print(f"FPS: {fps_counter}")
        fps_counter = 0

    # Exemplo: Verificar se detectou alguma mão
    if resultado.hand_landmarks:
        num_maos = len(resultado.hand_landmarks)
        # Informações sobre a primeira mão detectada
        primeira_mao = resultado.hand_landmarks[0]
        classificacao = resultado.handedness[0][0].category_name
        confianca = resultado.handedness[0][0].score

        print(f"[{timestamp_ms}ms] {num_maos} mão(s) detectada(s)")
        print(f"  → Mão {classificacao} (confiança: {confianca:.2f})")
        print(f"  → Ponta do indicador: ({primeira_mao[8].x:.3f}, {primeira_mao[8].y:.3f})")


# 2. CONFIGURAÇÃO DO DETECTOR COM CALLBACK
def main():
    global resultados_atuais, prev_x, prev_y

    # Configurações com LIVE_STREAM
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path='hand_landmarker.task'  # Baixe da documentação do MediaPipe
        ),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,  # Usar apenas 1 mão para controle do mouse
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=callback_resultado
    )

    # Configurações do PyAutoGUI
    pyautogui.FAILSAFE = True  # Mover mouse para canto superior esquerdo para emergência

    # Inicializa a webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam")
        return

    # Cria o detector com as opções
    with HandLandmarker.create_from_options(options) as landmarker:
        print("Detector inicializado! Pressione 'q' para sair")
        print("\nControles do Mouse Virtual:")
        print("  - Movimento: Mova sua mão")
        print("  - Clique esquerdo: Junte polegar e indicador")
        print("  - Clique direito: Junte polegar e dedo médio")
        print("  - Clique duplo: Junte indicador e médio")
        print("  - Scroll: Mova indicador e médio juntos para cima/baixo")
        print("  - Pressione ESC para emergência\n")

        timestamp_inicial = datetime.now().timestamp() * 1000  # ms

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Erro ao capturar quadro")
                break

            # Converte BGR (OpenCV) para RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Calcula timestamp atual em milissegundos
            timestamp_ms = int((datetime.now().timestamp() * 1000) - timestamp_inicial)

            # 3. CHAMADA ASSÍNCRONA - O callback será executado depois
            landmarker.detect_async(imagem_mp, timestamp_ms)

            # Processa os resultados para o mouse virtual
            if resultados_atuais and resultados_atuais.hand_landmarks:
                # Pega a primeira mão detectada
                mao_landmarks = resultados_atuais.hand_landmarks[0]
                altura, largura = frame.shape[:2]

                # Obtém coordenadas dos landmarks importantes
                index_tip = (int(mao_landmarks[INDEX_TIP].x * largura),
                             int(mao_landmarks[INDEX_TIP].y * altura))
                thumb_tip = (int(mao_landmarks[THUMB_TIP].x * largura),
                             int(mao_landmarks[THUMB_TIP].y * altura))
                middle_tip = (int(mao_landmarks[MIDDLE_TIP].x * largura),
                              int(mao_landmarks[MIDDLE_TIP].y * altura))
                wrist = (int(mao_landmarks[WRIST].x * largura),
                         int(mao_landmarks[WRIST].y * altura))

                # Controla o mouse (usa posição do indicador para movimento)
                move_mouse(index_tip[0], index_tip[1], largura, altura)

                # Detecta cliques
                click_type = detect_click(thumb_tip, index_tip, middle_tip, timestamp_ms)

                # Detecta scroll
                scroll_amount = detect_scroll(index_tip, middle_tip, altura)

                # Desenha informações na tela
                frame = desenhar_landmarks(frame, resultados_atuais)
                frame = desenhar_interface_mouse(frame, click_type, scroll_amount)

                # Desenha círculo de controle do mouse
                cv2.circle(frame, index_tip, 10, (0, 255, 0), 2)
                cv2.putText(frame, "Mouse Control", (index_tip[0] - 30, index_tip[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Mostra status dos cliques
                if click_type:
                    cv2.putText(frame, f"Action: {click_type}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Mostra FPS na tela
            cv2.putText(frame, f"FPS: {fps_counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mostra instruções na tela
            cv2.putText(frame, "Press 'q' to quit | ESC for emergency", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Virtual Mouse - Hand Tracking', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 27:  # Tecla ESC
                pyautogui.moveTo(0, 0)  # Move mouse para canto superior esquerdo
                print("Modo de emergência ativado!")

        cap.release()
        cv2.destroyAllWindows()


# 3. FUNÇÃO AUXILIAR PARA DESENHAR OS LANDMARKS
def desenhar_landmarks(frame, resultado):
    """
    Desenha os pontos e conexões das mãos no frame
    """
    altura, largura = frame.shape[:2]

    # Cores
    COR_VERMELHA = (0, 0, 255)
    COR_VERDE = (0, 255, 0)
    COR_AZUL = (255, 0, 0)

    for mao_idx, mao_landmarks in enumerate(resultado.hand_landmarks):
        # Desenha cada ponto (landmark)
        for ponto_id, ponto in enumerate(mao_landmarks):
            # Converte coordenadas normalizadas para pixels
            x = int(ponto.x * largura)
            y = int(ponto.y * altura)

            # Destaca landmarks importantes em cores diferentes
            if ponto_id in [INDEX_TIP, THUMB_TIP, MIDDLE_TIP]:
                cv2.circle(frame, (x, y), 8, COR_AZUL, -1)
            else:
                cv2.circle(frame, (x, y), 4, COR_VERMELHA, -1)

            # Adiciona número do landmark para debug
            if ponto_id % 4 == 0:  # Mostra apenas alguns números para não poluir
                cv2.putText(frame, str(ponto_id), (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, COR_VERDE, 1)

        # Informações da mão
        classificacao = resultado.handedness[mao_idx][0].category_name
        confianca = resultado.handedness[mao_idx][0].score
        texto = f"{classificacao} ({confianca:.2f})"

        # Posição da palma da mão (landmark 0)
        palma_x = int(mao_landmarks[0].x * largura)
        palma_y = int(mao_landmarks[0].y * altura)
        cv2.putText(frame, texto, (palma_x - 30, palma_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame


def desenhar_interface_mouse(frame, click_type, scroll_amount):
    """Desenha interface do mouse virtual na tela"""
    # Mostra coordenadas do mouse
    mouse_x, mouse_y = pyautogui.position()
    cv2.putText(frame, f"Mouse: ({mouse_x}, {mouse_y})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Mostra scroll
    if scroll_amount != 0:
        cv2.putText(frame, f"Scroll: {scroll_amount}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


if __name__ == "__main__":
    # Instalação necessária: pip install opencv-python mediapipe pyautogui numpy
    print("Certifique-se de ter instalado: pip install opencv-python mediapipe pyautogui numpy")
    print("\nATENÇÃO: Baixe o arquivo 'hand_landmarker.task' do site do MediaPipe")
    main()