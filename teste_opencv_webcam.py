import cv2

# Use CAP_DSHOW (DirectShow) no Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    # Fallback para MSMF (Media Foundation)
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if cap.isOpened():
    print("Webcam aberta com sucesso!")
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()