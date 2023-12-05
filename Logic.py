#!/Users/ZoesComputer/.pyenv/versions/3.11.5/bin/python

import cv2
import mediapipe as mp
import os
import time
import numpy as np
import speech_recognition as sr
import threading
from threading import Lock

# Initialize variables for cooldown and flags
last_x_press_time = 0
COOLDOWN_TIME = 1
last_fist_time = 0
FIST_COOLDOWN_TIME = 1
fist_counter = 0
FIST_CONFIDENCE_THRESHOLD = 4
three_fingers_counter = 0
THREE_FINGERS_CONFIDENCE_THRESHOLD = 3
last_three_finger_time = 0
THREE_FINGER_COOLDOWN_TIME = 1
last_record_time = 0
RECORD_COOLDOWN_TIME = 1
is_recording = False
is_recording_lock = Lock()

# Kalman filter parameters
initial_state_covariance = 0.1
process_variance = 0.01
measurement_variance = 0.1


class KalmanFilter:
    def __init__(self):
        self.state_estimate = np.array([0, 0, 0])
        self.error_covariance = np.array(
            [initial_state_covariance, initial_state_covariance, initial_state_covariance])

    def update(self, measurement):
        kalman_gain = self.error_covariance / \
            (self.error_covariance + measurement_variance)
        self.state_estimate = self.state_estimate + \
            kalman_gain * (measurement - self.state_estimate)
        self.error_covariance = (1 - kalman_gain) * \
            self.error_covariance + process_variance
        return self.state_estimate


# Initialize Kalman filter for each landmark
kalman_filters = {i: KalmanFilter() for i in range(21)}


def kalman_filter_process(landmark, kalman_filter):
    measurement = np.array([landmark.x, landmark.y, landmark.z])
    return kalman_filter.update(measurement)


# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
# Flag to track whether Logic Pro X is paused
logic_pro_paused = False

# Function to pause Logic Pro (Flat hand to fist close)


def pause_logic_pro():
    global logic_pro_paused
    if not logic_pro_paused:
        try:
            os.system(
                "osascript '/Users/ZoesComputer/Desktop/Tracking/TogglePlayPause.scpt'")
            logic_pro_paused = True
        except Exception as e:
            print(f"An error occurred: {e}")


def press_b_key():
    try:
        os.system("osascript '/path/to/PressBKey.scpt'")
    except Exception as e:
        print(f"An error occurred: {e}")


def press_r_key():
    try:
        os.system("osascript '/Users/ZoesComputer/Desktop/Tracking/PressRKey.scpt'")
    except Exception as e:
        print(f"An error occurred: {e}")


def press_x_key():
    try:
        os.system("osascript '/Users/ZoesComputer/Desktop/Tracking/PressXKey.scpt'")
    except Exception as e:
        print(f"An error occurred: {e}")


def listen_for_commands():
    global is_recording
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("Listening for command...")
                audio = recognizer.listen(
                    source, timeout=5, phrase_time_limit=3)

            command = recognizer.recognize_google(audio).lower()
            print(f"Recognized command: {command}")

            if "tamber" in command or "timber" in command:
                with is_recording_lock:  # Acquire lock
                    if "start recording" in command and not is_recording:
                        press_r_key()  # Start recording
                        is_recording = True
                    elif "stop recording" in command and is_recording:
                        pause_logic_pro()  # Stop recording
                        is_recording = False
        except sr.WaitTimeoutError:
            print("Listening timed out, no phrase detected.")
            continue
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            continue
        except sr.RequestError as e:
            print(
                f"Could not request results from Google Speech Recognition service; {e}")
            continue


voice_thread = threading.Thread(target=listen_for_commands)
voice_thread.start()

cap = cv2.VideoCapture(0)

# Main processing loop
while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                filtered_landmarks = {}
                for i, landmark in enumerate(hand_landmarks.landmark):
                    filtered_landmarks[i] = kalman_filter_process(
                        landmark, kalman_filters[i])

                thumb_tip = filtered_landmarks.get(
                    mp_hands.HandLandmark.THUMB_TIP.value)
                index_finger_tip = filtered_landmarks.get(
                    mp_hands.HandLandmark.INDEX_FINGER_TIP.value)
                middle_finger_tip = filtered_landmarks.get(
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value)
                ring_finger_tip = filtered_landmarks.get(
                    mp_hands.HandLandmark.RING_FINGER_TIP.value)
                pinky_tip = filtered_landmarks.get(
                    mp_hands.HandLandmark.PINKY_TIP.value)
                index_finger_pip = filtered_landmarks.get(
                    mp_hands.HandLandmark.INDEX_FINGER_PIP.value)
                middle_finger_pip = filtered_landmarks.get(
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP.value)
                ring_finger_pip = filtered_landmarks.get(
                    mp_hands.HandLandmark.RING_FINGER_PIP.value)
                pinky_pip = filtered_landmarks.get(
                    mp_hands.HandLandmark.PINKY_PIP.value)

                # Fist Detection with Confidence
                if (index_finger_tip[1] > index_finger_pip[1] and
                    middle_finger_tip[1] > middle_finger_pip[1] and
                    ring_finger_tip[1] > ring_finger_pip[1] and
                        pinky_tip[1] > pinky_pip[1]):
                    fist_counter += 1
                    if fist_counter >= FIST_CONFIDENCE_THRESHOLD and time.time() - last_fist_time > FIST_COOLDOWN_TIME:
                        print("Pause (Fist is closed)")
                        pause_logic_pro()
                        last_fist_time = time.time()
                        fist_counter = 0
                else:
                    fist_counter = 0

                # Three Fingers Up Detection with Confidence
                if (index_finger_tip[1] < index_finger_pip[1] and
                    middle_finger_tip[1] < middle_finger_pip[1] and
                    ring_finger_tip[1] < ring_finger_pip[1] and
                        pinky_tip[1] > pinky_pip[1]):
                    three_fingers_counter += 1
                    if three_fingers_counter >= THREE_FINGERS_CONFIDENCE_THRESHOLD and time.time() - last_three_finger_time > THREE_FINGER_COOLDOWN_TIME:
                        print("Plug-In Menu (Index, Middle, and Ring fingers are up)")
                        press_x_key()
                        last_three_finger_time = time.time()
                        three_fingers_counter = 0
                else:
                    three_fingers_counter = 0

                # Two Fingers Up Detection with Confidence (for recording)
                if (time.time() - last_record_time > RECORD_COOLDOWN_TIME):
                    if (index_finger_tip[1] < index_finger_pip[1] and
                        middle_finger_tip[1] < middle_finger_pip[1] and
                        ring_finger_tip[1] > ring_finger_pip[1] and
                            pinky_tip[1] > pinky_pip[1]):
                        with is_recording_lock:
                            if not is_recording:
                                print("Start Recording")
                                press_r_key()
                                is_recording = True
                            else:
                                print("Stop Recording")
                                pause_logic_pro()
                                is_recording = False
                        last_record_time = time.time()
                else:
                    # Do not reset the timer here to ensure cooldown works
                    pass  # No operation

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(e)
        break

# [End of main loop]

voice_thread.join()
cap.release()
cv2.destroyAllWindows()
