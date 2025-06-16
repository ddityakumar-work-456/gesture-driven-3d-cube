import cv2
import mediapipe as mp
import numpy as np
import math

# --------------------- Utility ---------------------


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


# --------------------- Hand Tracking ---------------------


class HandTracker:
    def __init__(self, max_hands=2, detection_conf=0.6, tracking_conf=0.6):
        """Initializes the MediaPipe hand detector."""
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.drawer = mp.solutions.drawing_utils

    def detect(self, frame):
        """Detects hands in the given frame and returns landmarks."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        hand_centers = []
        hand_landmarks = []

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self.drawer.draw_landmarks(
                    frame, hand, mp.solutions.hands.HAND_CONNECTIONS
                )
                landmarks = [
                    (int(l.x * frame.shape[1]), int(l.y * frame.shape[0]))
                    for l in hand.landmark
                ]
                hand_landmarks.append(landmarks)
                hand_centers.append(landmarks[0])  # Wrist point as center

        return hand_centers, hand_landmarks


# --------------------- Cube Class ---------------------


class Cube:
    def __init__(self, position=(400, 300), size=100):
        """Represents the cube object on screen."""
        self.pos = np.array(position, dtype=np.float32)
        self.target_pos = self.pos.copy()
        self.size = size
        self.angle = 0
        self.picked = False
        self.easing = 0.4

    def move_smoothly(self):
        """Smooth transition to target position."""
        if euclidean_distance(self.pos, self.target_pos) > 10:
            self.pos += self.easing * (self.target_pos - self.pos)
        else:
            self.pos = self.target_pos.copy()

    def scale_and_rotate(self, hand1, hand2, width):
        """Handles scaling and rotation based on two hand gestures."""
        distance_hands = euclidean_distance(hand1, hand2)
        new_size = np.interp(distance_hands, [50, width], [80, 250])
        self.size += self.easing * (new_size - self.size)

        # Rotation based on horizontal difference
        diff_x = hand2[0] - hand1[0]
        new_angle = np.clip(diff_x * 0.3, -90, 90)
        self.angle += self.easing * (new_angle - self.angle)

    def draw(self, frame):
        """Draw a 3D-looking cube on the frame."""
        cx, cy = self.pos.astype(int)
        s = int(self.size // 2)
        offset = 30
        angle_rad = np.deg2rad(self.angle)

        def rotate(pt):
            x, y = pt[0] - cx, pt[1] - cy
            xr = x * np.cos(angle_rad) - y * np.sin(angle_rad)
            yr = x * np.sin(angle_rad) + y * np.cos(angle_rad)
            return int(cx + xr), int(cy + yr)

        front = np.array(
            [[cx - s, cy - s], [cx + s, cy - s], [cx + s, cy + s], [cx - s, cy + s]]
        )
        front = np.array([rotate(pt) for pt in front])
        back = front - [offset, offset]

        edges = (
            [(front[i], front[(i + 1) % 4]) for i in range(4)]
            + [(back[i], back[(i + 1) % 4]) for i in range(4)]
            + [(front[i], back[i]) for i in range(4)]
        )

        # Draw faces and edges
        cv2.fillPoly(frame, [front], (200, 60, 60))
        for e in edges:
            cv2.line(frame, e[0], e[1], (0, 0, 0), 2)

        # Status text
        status = "🟢 PICKED" if self.picked else "⚪ DROP"
        cv2.putText(
            frame, status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3
        )
        cv2.putText(
            frame,
            f"Angle: {int(self.angle)}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (100, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Size: {int(self.size)}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 200, 100),
            2,
        )


# --------------------- Controller ---------------------


class CubeController:
    def __init__(self):
        """Handles webcam, cube, and gesture coordination."""
        self.cap = cv2.VideoCapture(0)
        self.tracker = HandTracker()
        self.cube = Cube()

        # Setup full-screen window
        cv2.namedWindow("3D Cube Controller", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            "3D Cube Controller", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

    def run(self):
        """Main loop for real-time cube interaction."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            hand_centers, hands = self.tracker.detect(frame)

            # Gesture: pinch to pick and move
            for lm in hands:
                thumb_tip, index_tip = lm[4], lm[8]
                if euclidean_distance(thumb_tip, index_tip) < 40:
                    self.cube.picked = True
                    self.cube.target_pos = np.array(
                        [
                            (thumb_tip[0] + index_tip[0]) // 2,
                            (thumb_tip[1] + index_tip[1]) // 2,
                        ]
                    )
                    break
            else:
                self.cube.picked = False

            # Gesture: two-hand scale and rotate
            if len(hand_centers) == 2:
                self.cube.scale_and_rotate(hand_centers[0], hand_centers[1], width)

            # Move and draw cube
            self.cube.move_smoothly()
            self.cube.draw(frame)

            # Show output
            cv2.imshow("3D Cube Controller", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()


# --------------------- Run ---------------------

if __name__ == "__main__":
    CubeController().run()
