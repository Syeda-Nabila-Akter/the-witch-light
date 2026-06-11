import cv2
import os
import time
import numpy as np
from collections import deque

class WitchLightMinimalSystem:
    def __init__(self):
        # --- 1. Core Geometry Engine Setup ---
        cascade_file = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        
        try:
            self.facemark = cv2.face.createFacemarkLBF()
        except AttributeError:
            self.facemark = cv2.face_FaceMarkLBF.create()
        
        if os.path.exists("lbfmodel.yaml"):
            self.facemark.loadModel("lbfmodel.yaml")
            
        # Calibration Parameters
        self.is_calibrated = False
        self.calibration_start_time = None
        self.collected_mouth_ratios = []
        self.collected_brow_distances = []
        self.collected_mouth_curves = []
        
        self.base_mouth_ratio = 0.22
        self.base_brow_dist = 20.0
        self.base_mouth_curve = 0.0

        # Moving Average Memory Pools to prevent flickering
        self.memory_window = 8
        self.ratio_history = deque(maxlen=self.memory_window)
        self.brow_history = deque(maxlen=self.memory_window)
        self.curve_history = deque(maxlen=self.memory_window)

    def get_face_metrics(self, points):
        """Calculates precise geometric relationships across facial feature clusters."""
        mouth_left, mouth_right = points[48], points[54]
        mouth_top, mouth_bottom = points[51], points[57]
        mouth_w = max(1.0, np.linalg.norm(mouth_left - mouth_right))
        mouth_h = np.linalg.norm(mouth_top - mouth_bottom)
        mouth_ratio = mouth_h / mouth_w

        eye_center_y = (points[39][1] + points[42][1]) / 2.0
        brow_center_y = (points[21][1] + points[22][1]) / 2.0
        brow_distance = abs(eye_center_y - brow_center_y)

        lip_midpoint_y = (points[51][1] + points[57][1]) / 2.0
        corners_avg_y = (points[48][1] + points[54][1]) / 2.0
        mouth_curvature = lip_midpoint_y - corners_avg_y

        return mouth_ratio, brow_distance, mouth_curvature

    def run_stream(self):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("[CRITICAL] Camera hardware bus initialization failed.")
            return

        print("[SYSTEM ACTIVE] Minimalist High-Accuracy Engine live. Press 'q' to exit.")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5, minSize=(60, 60))

            if len(faces) > 0 and not self.is_calibrated and self.calibration_start_time is None:
                self.calibration_start_time = time.time()

            if len(faces) > 0:
                try:
                    ok, landmarks = self.facemark.fit(gray_frame, faces)
                except Exception:
                    ok = False

                for i, face_box in enumerate(faces):
                    x, y, w, h = face_box
                    points = landmarks[i][0] if (ok and len(landmarks) > i) else None

                    if points is not None:
                        raw_ratio, raw_brow, raw_curve = self.get_face_metrics(points)
                        self.ratio_history.append(raw_ratio)
                        self.brow_history.append(raw_brow)
                        self.curve_history.append(raw_curve)

                        # --- MODIFICATION: Changed tracking dots color tuple to (0, 0, 0) for pure black ---
                        for pt in points:
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 0, 0), -1)

                        # --- PHASE 1: Calibration Mode ---
                        if not self.is_calibrated:
                            self.collected_mouth_ratios.append(raw_ratio)
                            self.collected_brow_distances.append(raw_brow)
                            self.collected_mouth_curves.append(raw_curve)
                            
                            elapsed = time.time() - self.calibration_start_time
                            mood_label = f"Calibrating... Relax ({3 - int(elapsed)}s)"
                            ui_color = (255, 191, 0) # Amber text box
                            commentary = "INITIALIZING MATRIX..."
                            
                            if elapsed >= 3.0 and len(self.collected_mouth_ratios) > 0:
                                self.base_mouth_ratio = sum(self.collected_mouth_ratios) / len(self.collected_mouth_ratios)
                                self.base_brow_dist = sum(self.collected_brow_distances) / len(self.collected_brow_distances)
                                self.base_mouth_curve = sum(self.collected_mouth_curves) / len(self.collected_mouth_curves)
                                self.is_calibrated = True
                                print("[READY] Minimalist setup initialized.")

                        # --- PHASE 2: Live High-Speed Inference ---
                        elif self.is_calibrated:
                            m_ratio = sum(self.ratio_history) / len(self.ratio_history)
                            b_dist = sum(self.brow_history) / len(self.brow_history)
                            m_curve = sum(self.curve_history) / len(self.curve_history)

                            curve_deviation = m_curve - self.base_mouth_curve
                            brow_deviation_ratio = b_dist / max(1.0, self.base_brow_dist)

                            if m_ratio > (self.base_mouth_ratio * 1.45):
                                mood_label = "Surprised"
                                ui_color = (255, 255, 0)  # Cyan Blue
                                commentary = "SYSTEM ALERT: UNEXPECTED COGNITIVE INPUT."
                            elif curve_deviation > 3.5:
                                mood_label = "Happy"
                                ui_color = (0, 255, 0)    # Emerald Green
                                commentary = "SEROTONIN HIGH. PROTOCOL: ENJOY."
                            elif curve_deviation < -2.0:
                                mood_label = "Sad"
                                ui_color = (255, 0, 0)    # Soft Ocean Blue
                                commentary = "VIBE PLUMMET. INITIATING COZY MODE."
                            elif brow_deviation_ratio < 0.85:
                                mood_label = "Angry"
                                ui_color = (0, 0, 255)    # Crimson Red
                                commentary = "STRESS CRITICAL. STEP AWAY FROM KEYBOARD."
                            elif m_ratio < (self.base_mouth_ratio * 0.75):
                                mood_label = "Deep Focus"
                                ui_color = (128, 0, 128)  # Concentration Purple
                                commentary = "FOCUS MODE ACTIVE. DISCONNECTING DISTRACTIONS."
                            else:
                                mood_label = "Neutral State"
                                ui_color = (255, 191, 0)  # Balanced Amber
                                commentary = "SYSTEM BALANCED."

                        # --- PHASE 3: Draw On-Screen Overlays ---
                        cv2.rectangle(frame, (x, y), (x + w, y + h), ui_color, 2)
                        
                        # Note: The text itself remains crisp white so it stays readable over the colored banner
                        cv2.rectangle(frame, (x, y - 35), (x + w, y), ui_color, cv2.FILLED)
                        cv2.putText(frame, mood_label, (x + 6, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Black bottom banner for text commentary
                        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame, commentary, (x + 6, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ui_color, 1, cv2.LINE_AA)

            cv2.imshow("The Witch Light - Precision System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = WitchLightMinimalSystem()
    system.run_stream()