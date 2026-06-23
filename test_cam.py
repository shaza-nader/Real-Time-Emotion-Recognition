import cv2
from deepface import DeepFace

cap=cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

        if isinstance(result, list):
            dominant_emotion = result[0].get("dominant_emotion", "unknown")
        else:
            dominant_emotion = result.get("dominant_emotion", "unknown")

        cv2.putText(frame, f'Emotion: {dominant_emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    except Exception as e:
        cv2.putText(frame,f'no emotion detected',(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

    cv2.imshow('Live Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()

cv2.destroyAllWindows()