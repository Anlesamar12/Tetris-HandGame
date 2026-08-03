import cv2
import mediapipe as mp
import time
import pygame


class HandController:

    def __init__(self):

        self.cap = cv2.VideoCapture(0)

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.rotate_ready = True

    def fingers_up(self, hand):

        tips = [8, 12, 16, 20]

        fingers = []

        for tip in tips:
            fingers.append(
                hand.landmark[tip].y <
                hand.landmark[tip-2].y
            )

        return fingers

    def update(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        data = {
            "column": None,
            "rotate": False,
            "down": False,
            "image": None

        }

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            fingers = self.fingers_up(hand)

            x = hand.landmark[8].x
            y = hand.landmark[8].y

            column = int(x * 10)

            column = max(0, min(9, column))

            data["column"] = column

            cv2.circle(
                frame,
                (int(x*w), int(y*h)),
                10,
                (0,255,0),
                -1
            )

            # Índice + medio = rotar
            if fingers[0] and fingers[1]:

                if self.rotate_ready:

                    data["rotate"] = True

                    self.rotate_ready = False

            else:

                self.rotate_ready = True

            # Mano abajo = bajar rápido
            if y > 0.70:

                data["down"] = True

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.resize(frame, (160, 120))

        surface = pygame.image.frombuffer(
            frame.tobytes(),
            frame.shape[1::-1],
            "RGB"
        )

        data["image"] = surface

        return data

    def close(self):

        self.cap.release()

        cv2.destroyAllWindows()