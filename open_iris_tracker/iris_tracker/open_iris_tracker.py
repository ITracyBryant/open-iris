import time
import traceback

import cv2
import mediapipe as mp
import numpy as np


class OpenIrisTracker:
    def __init__(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        # 初始化参数
        self.baseline_duration = 10.0
        self.start_time = time.time()
        self.calibrating = True
        self.left_deltas = []
        self.right_deltas = []
        self.left_baseline = (0, 0)
        self.right_baseline = (0, 0)

        self.last_offset_time = 0
        self.offset_count = 0

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_center(self, landmarks, indices, image_shape):
        """获取标记中心点"""
        h, w = image_shape[:2]
        pts = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in indices
        ])
        center = np.mean(pts, axis=0).astype(int)
        return tuple(center)

    def update_baseline(self):
        """计算平均偏移, 获取基准视线"""

        def avg(delta_list):
            arr = np.array(delta_list)
            avg_arr = np.mean(arr, axis=0)
            avg_arr = np.round(avg_arr).astype(int)
            return tuple(avg_arr)

        self.left_baseline = avg(self.left_deltas)
        self.right_baseline = avg(self.right_deltas)
        self.calibrating = False
        print(f"完成基准视线采样: left_baseline={self.left_baseline}, right_baseline={self.right_baseline}")

    def detect_sight_offset(self, eye_center, iris_center, baseline):
        """sight offset standard threshold
        实时收集的虹膜-眼眶偏移量 对比 初始视线基线偏移量
        todo 策略待优化(?): 视线必须一直聚焦到初始收集的视线基线(均值后可能有差异), 即持续的盯着同一个位置(很难保持), 轻微的移动与偏移都会被识别；眨眼情况未过滤
        """
        dx = abs(iris_center[0] - eye_center[0])
        dy = abs(iris_center[1] - eye_center[1])
        baseline_dx, baseline_dy = baseline

        delta_dx = abs(dx - baseline_dx)
        delta_dy = abs(dy - baseline_dy)

        if delta_dx > baseline_dx * 0.5 and delta_dy > baseline_dy * 0.5:
            current_time = time.time()
            if current_time - self.last_offset_time > 1.5:
                self.last_offset_time = current_time
                self.offset_count += 1
                print(f"视线偏移次数: {self.offset_count}")
                print(
                    f"detect (delta baseline dx,dy):({delta_dx}, {delta_dy}), baseline: {baseline}, (dx,dy): ({dx}, {dy}), eye_center: {eye_center}, iris_center: {iris_center}")
                if self.offset_count > 3:
                    print("[!]检测到多次视线偏移，判断为视线异常")
                    self.offset_count = 0

    def process_eye(self, frame, face_landmarks, eye_idx, iris_idx, baseline_deltas, is_left):
        eye_center = self.get_center(face_landmarks.landmark, eye_idx, frame.shape)
        iris_center = self.get_center(face_landmarks.landmark, iris_idx, frame.shape)

        dx = abs(iris_center[0] - eye_center[0])
        dy = abs(iris_center[1] - eye_center[1])

        if self.calibrating:
            # print(f"collect baseline_delta: ({dx}, {dy})")
            baseline_deltas.append((dx, dy))
        else:
            baseline = self.left_baseline if is_left else self.right_baseline
            self.detect_sight_offset(eye_center, iris_center, baseline)

        # 可视化标记
        # cv2.circle(frame, eye_center, 2, (255, 0, 0), -1)
        cv2.circle(frame, iris_center, 2, (0, 255, 0), -1)

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 左右眼眶轮廓点索引
                left_eye_idx = [33, 133, 158, 159, 160, 173, 246]  # 左眼轮廓（可微调）
                right_eye_idx = [263, 362, 387, 386, 385, 384, 398, 466]  # 右眼轮廓
                # 虹膜点索引
                left_iris_idx = list(range(468, 473))
                right_iris_idx = list(range(473, 478))

                self.process_eye(frame, face_landmarks, left_eye_idx, left_iris_idx, self.left_deltas, is_left=True)
                self.process_eye(frame, face_landmarks, right_eye_idx, right_iris_idx, self.right_deltas, is_left=False)
        return frame

    def test_sight(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # iris landmarks (left: 468-473, right: 473-478)
                # eye landmarks (left: [33, 133, 160, 159, 158, 157, 173, 246], right: [263, 362, 387, 386, 385, 384, 398, 466])
                for idx in range(468, 478):
                    x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                    y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                    # Draw a small circle at each iris landmark
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        return frame

    def run(self):
        print("启动视线偏移检测，请保持注视屏幕...")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                elapsed = time.time() - self.start_time
                if self.calibrating and elapsed > self.baseline_duration:
                    self.update_baseline()

                frame = self.process_frame(frame)
                # frame = self.test_sight(frame)
                if self.calibrating:
                    cv2.putText(frame, f"Collecting sight baseline... {int(self.baseline_duration - elapsed)}s",
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Detecting sight...", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

                cv2.imshow('Sight Offset Tracker', frame)

                # exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as ex:
            print(f"Error: iris tracker run failed: {ex}")
            traceback.print_exc()
        finally:
            self.__del__()
            print("程序关闭, 资源释放...")
