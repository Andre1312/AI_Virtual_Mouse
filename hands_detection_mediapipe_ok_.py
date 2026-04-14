import cv2
import mediapipe as mp
import numpy as np
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
    global resultados_atuais

    # Configurações com LIVE_STREAM
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path='hand_landmarker.task'  # Baixe da documentação do MediaPipe
        ),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=callback_resultado  # 👈 AQUI O CALLBACK É DEFINIDO
    )

    # Inicializa a webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam")
        return

    # Cria o detector com as opções
    with HandLandmarker.create_from_options(options) as landmarker:
        print("Detector inicializado! Pressione 'q' para sair")

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

            # Desenha os resultados no frame (se disponíveis)
            if resultados_atuais and resultados_atuais.hand_landmarks:
                frame = desenhar_landmarks(frame, resultados_atuais)

            # Mostra FPS na tela
            cv2.putText(frame, f"FPS: {fps_counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Hand Tracking - LIVE_STREAM Mode', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# 4. FUNÇÃO AUXILIAR PARA DESENHAR OS LANDMARKS
def desenhar_landmarks(frame, resultado):
    """
    Desenha os pontos e conexões das mãos no frame
    """
    altura, largura = frame.shape[:2]

    # Cores
    COR_VERMELHA = (0, 0, 255)
    COR_VERDE = (0, 255, 0)

    for mao_idx, mao_landmarks in enumerate(resultado.hand_landmarks):
        # Desenha cada ponto (landmark)
        for ponto_id, ponto in enumerate(mao_landmarks):
            # Converte coordenadas normalizadas para pixels
            x = int(ponto.x * largura)
            y = int(ponto.y * altura)

            # Desenha círculo em cada ponto
            cv2.circle(frame, (x, y), 5, COR_VERMELHA, -1)

            # Adiciona número do landmark para debug
            cv2.putText(frame, str(ponto_id), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COR_VERDE, 1)

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


if __name__ == "__main__":
    main()
