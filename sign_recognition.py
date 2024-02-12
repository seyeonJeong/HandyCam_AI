

import cv2
import imutils as imutils
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont,ImageDraw, Image
import time


actions = ['머리가','다리가','눈이','아픕니다.','기침이 납니다.','어지럼증이 있습니다.','열이 납니다.','허리가','부어오릅니다.']
seq_length = 30

model = load_model('models/model4.h5')
print(model)

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
# out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

seq = []
action_seq = []
sentence = ''
prev_action=''
startTime = time.time()
prev_index = 0
recognizeDelay = 2
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(w, h) # 640, 480
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(w, h) # 640, 480
while cap.isOpened():
    ret, img = cap.read()

    img0 = img.copy()

    # img = imutils.resize(img, width=1280)
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pil_img = Image.fromarray(img)
    font = ImageFont.truetype("fonts/gulim.ttc",40)

    draw = ImageDraw.Draw(pil_img)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                print(lm)
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
            v = v2 - v1  # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle])
            seq.append(d)

            # mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS,)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            print(i_pred)
            action_seq.append(action)

            if len(action_seq) < 3:
                continue


            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action
                print(this_action)

                if time.time() - startTime > recognizeDelay and prev_action != this_action:
                    sentence += this_action
                    prev_action = this_action
                    startTime = time.time()
                    action_seq[-1]=''
                    action_seq[-2] = ''
                    action_seq[-3] = ''
                    print("ㅇㅇ")
            else:
                startTime = time.time()
            cv2.putText(img, f'{this_action}',
                        org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    draw.text((30,50), sentence, font=font, fill=(0,0,0))

    img = np.asarray(pil_img)
    # cv2.putText(img, sentence, (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break
    if cv2.waitKey(1) == ord('a'):
        sentence = ''
        startTime = time.time()
