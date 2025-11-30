"""
main.py - VITAL-SENSE: Non-Contact Real-Time Heart Rate Monitor (rPPG)

This application detects the user's pulse (BPM) by analyzing sub-pixel color
changes in their forehead skin caused by blood flow using remote photoplethysmography.

Architecture:
- Face Locking: MediaPipe Face Mesh for face detection and ROI extraction
- Signal Extraction: Mean Green Intensity from forehead region
- Signal Processing: Detrending, Normalization, Bandpass Filter, FFT
- HUD Display: Futuristic Medical/Military style interface

Author: VITAL-SENSE Team
License: Apache 2.0
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
from signal_utils import process_signal, calculate_signal_quality

# Constants
BUFFER_SIZE = 300  # Sliding window of 300 frames (~10 seconds at 30 fps)
MIN_FRAMES_FOR_BPM = 90  # Minimum frames needed to calculate BPM (~3 seconds)
TARGET_FPS = 30.0  # Target frame rate for signal processing

# Color scheme (BGR format for OpenCV)
CYAN = (255, 255, 0)  # #00FFFF in BGR
RED = (0, 0, 255)  # #FF0000 in BGR
DARK_CYAN = (150, 150, 0)  # Darker cyan for secondary elements
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# MediaPipe Face Mesh forehead landmarks
# These landmarks define the forehead region between eyes and hairline
FOREHEAD_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

# Refined forehead landmarks (smaller, more stable region)
FOREHEAD_CORE_LANDMARKS = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]


class VitalSenseMonitor:
    """Main class for the VITAL-SENSE heart rate monitor."""
    
    def __init__(self):
        """Initialize the heart rate monitor."""
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Signal buffer for green channel values
        self.signal_buffer = deque(maxlen=BUFFER_SIZE)
        
        # BPM and stability tracking
        self.current_bpm = 0.0
        self.bpm_history = deque(maxlen=10)  # For smoothing BPM readings
        self.signal_quality = 0.0
        self.last_face_landmarks = None
        self.face_movement = 0.0
        
        # Timing
        self.frame_times = deque(maxlen=30)
        self.last_time = time.time()
        
        # Filtered signal for display
        self.filtered_signal = np.array([])
    
    def get_forehead_roi(self, frame: np.ndarray, landmarks) -> tuple:
        """
        Extract the forehead Region of Interest (ROI) from the frame.
        
        Args:
            frame: The input video frame (BGR)
            landmarks: MediaPipe face landmarks
            
        Returns:
            Tuple of (roi_mask, roi_points, bounding_box)
        """
        h, w = frame.shape[:2]
        
        # Get forehead landmark coordinates
        points = []
        for idx in FOREHEAD_CORE_LANDMARKS:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Create convex hull for the forehead region
        hull = cv2.convexHull(points)
        
        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(hull)
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        
        return mask, hull, (x, y, bw, bh)
    
    def extract_green_channel_mean(self, frame: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate the mean green intensity of the ROI.
        
        Green light is absorbed best by hemoglobin, making it ideal for
        detecting blood volume changes.
        
        Args:
            frame: The input video frame (BGR)
            mask: Binary mask of the ROI
            
        Returns:
            Mean green channel intensity
        """
        # Extract green channel
        green_channel = frame[:, :, 1]
        
        # Apply mask and calculate mean
        masked_green = cv2.bitwise_and(green_channel, green_channel, mask=mask)
        non_zero_pixels = mask > 0
        
        if np.sum(non_zero_pixels) == 0:
            return 0.0
        
        mean_green = np.mean(masked_green[non_zero_pixels])
        return mean_green
    
    def calculate_movement(self, landmarks) -> float:
        """
        Calculate face movement between frames for stability detection.
        
        Args:
            landmarks: Current face landmarks
            
        Returns:
            Movement score (higher = more movement)
        """
        if self.last_face_landmarks is None:
            return 0.0
        
        # Calculate displacement of key landmarks
        key_indices = [1, 4, 5, 6, 10, 152]  # Nose tip, nose bridge, forehead, chin
        total_movement = 0.0
        
        for idx in key_indices:
            curr = landmarks.landmark[idx]
            prev = self.last_face_landmarks.landmark[idx]
            
            dx = curr.x - prev.x
            dy = curr.y - prev.y
            total_movement += np.sqrt(dx**2 + dy**2)
        
        return total_movement / len(key_indices)
    
    def get_fps(self) -> float:
        """Calculate current FPS."""
        if len(self.frame_times) < 2:
            return TARGET_FPS
        
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff == 0:
            return TARGET_FPS
        
        return len(self.frame_times) / time_diff
    
    def draw_hud(self, frame: np.ndarray, face_bbox: tuple, is_stable: bool) -> np.ndarray:
        """
        Draw the futuristic HUD (Heads-Up Display) on the frame.
        
        Style: Medical/Military with cyan graphics and red BPM text.
        
        Args:
            frame: The input video frame
            face_bbox: Face bounding box (x, y, w, h)
            is_stable: Whether the face is stable enough for reading
            
        Returns:
            Frame with HUD overlay
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw targeting box around face
        if face_bbox:
            x, y, fw, fh = face_bbox
            
            # Expand the bounding box
            padding = 40
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + fw + padding)
            y2 = min(h, y + fh + padding)
            
            # Corner length
            corner_len = 30
            
            # Draw corner brackets (targeting box style)
            # Top-left corner
            cv2.line(overlay, (x1, y1), (x1 + corner_len, y1), CYAN, 2)
            cv2.line(overlay, (x1, y1), (x1, y1 + corner_len), CYAN, 2)
            
            # Top-right corner
            cv2.line(overlay, (x2, y1), (x2 - corner_len, y1), CYAN, 2)
            cv2.line(overlay, (x2, y1), (x2, y1 + corner_len), CYAN, 2)
            
            # Bottom-left corner
            cv2.line(overlay, (x1, y2), (x1 + corner_len, y2), CYAN, 2)
            cv2.line(overlay, (x1, y2), (x1, y2 - corner_len), CYAN, 2)
            
            # Bottom-right corner
            cv2.line(overlay, (x2, y2), (x2 - corner_len, y2), CYAN, 2)
            cv2.line(overlay, (x2, y2), (x2, y2 - corner_len), CYAN, 2)
            
            # Draw crosshair lines (dashed effect with gaps)
            # Horizontal center lines
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            
            # Decorative dashed lines
            dash_len = 10
            gap = 5
            for i in range(x1, x1 + corner_len, dash_len + gap):
                cv2.line(overlay, (i, mid_y), (min(i + dash_len, x1 + corner_len), mid_y), DARK_CYAN, 1)
            for i in range(x2 - corner_len, x2, dash_len + gap):
                cv2.line(overlay, (i, mid_y), (min(i + dash_len, x2), mid_y), DARK_CYAN, 1)
        
        # Draw status bar at top
        cv2.rectangle(overlay, (0, 0), (w, 40), BLACK, -1)
        cv2.rectangle(overlay, (0, 0), (w, 40), CYAN, 1)
        
        # Title
        cv2.putText(overlay, "VITAL-SENSE", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
        
        # FPS display
        fps = self.get_fps()
        cv2.putText(overlay, f"FPS: {fps:.1f}", (w - 100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
        
        # Draw BPM display
        bpm_box_y = 60
        cv2.rectangle(overlay, (w - 180, bpm_box_y), (w - 10, bpm_box_y + 80), BLACK, -1)
        cv2.rectangle(overlay, (w - 180, bpm_box_y), (w - 10, bpm_box_y + 80), CYAN, 2)
        
        # BPM label
        cv2.putText(overlay, "HEART RATE", (w - 170, bpm_box_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)
        
        # BPM value
        if self.current_bpm > 0 and is_stable:
            bpm_text = f"{int(self.current_bpm)}"
            cv2.putText(overlay, bpm_text, (w - 150, bpm_box_y + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 3)
            cv2.putText(overlay, "BPM", (w - 60, bpm_box_y + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
        else:
            cv2.putText(overlay, "---", (w - 130, bpm_box_y + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, DARK_CYAN, 2)
        
        # Signal quality indicator
        quality_y = bpm_box_y + 100
        cv2.rectangle(overlay, (w - 180, quality_y), (w - 10, quality_y + 30), BLACK, -1)
        cv2.rectangle(overlay, (w - 180, quality_y), (w - 10, quality_y + 30), CYAN, 1)
        cv2.putText(overlay, "SIGNAL:", (w - 175, quality_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)
        
        # Quality bar
        bar_width = int(80 * self.signal_quality)
        bar_color = GREEN if self.signal_quality > 0.5 else (0, 165, 255)  # Orange if low
        cv2.rectangle(overlay, (w - 100, quality_y + 8), (w - 100 + bar_width, quality_y + 22), bar_color, -1)
        cv2.rectangle(overlay, (w - 100, quality_y + 8), (w - 20, quality_y + 22), CYAN, 1)
        
        # Stability warning
        if not is_stable:
            warning_text = "STABILIZE TARGET"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            # Background for warning
            cv2.rectangle(overlay, (text_x - 10, text_y - 30), 
                         (text_x + text_size[0] + 10, text_y + 10), BLACK, -1)
            cv2.rectangle(overlay, (text_x - 10, text_y - 30), 
                         (text_x + text_size[0] + 10, text_y + 10), RED, 2)
            cv2.putText(overlay, warning_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
        
        # Draw pulse wave graph at the bottom
        self.draw_pulse_graph(overlay)
        
        # Blend overlay with original frame
        alpha = 0.85
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame
    
    def draw_pulse_graph(self, frame: np.ndarray) -> None:
        """
        Draw a scrolling line graph of the filtered pulse wave.
        
        Args:
            frame: The frame to draw on
        """
        h, w = frame.shape[:2]
        
        # Graph dimensions
        graph_height = 100
        graph_y = h - graph_height - 20
        graph_x = 20
        graph_width = w - 40
        
        # Draw graph background
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), 
                      BLACK, -1)
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), 
                      CYAN, 1)
        
        # Graph title
        cv2.putText(frame, "PULSE WAVEFORM", (graph_x + 5, graph_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)
        
        # Draw grid lines
        for i in range(1, 4):
            y_line = graph_y + int(graph_height * i / 4)
            cv2.line(frame, (graph_x, y_line), (graph_x + graph_width, y_line), DARK_CYAN, 1)
        
        for i in range(1, 8):
            x_line = graph_x + int(graph_width * i / 8)
            cv2.line(frame, (x_line, graph_y), (x_line, graph_y + graph_height), DARK_CYAN, 1)
        
        # Draw the pulse wave
        if len(self.filtered_signal) > 1:
            # Normalize signal for display
            display_signal = self.filtered_signal[-min(len(self.filtered_signal), graph_width):]
            
            if len(display_signal) > 1:
                # Normalize to fit graph height
                sig_min = np.min(display_signal)
                sig_max = np.max(display_signal)
                sig_range = sig_max - sig_min
                
                if sig_range > 0:
                    normalized = (display_signal - sig_min) / sig_range
                else:
                    normalized = np.ones_like(display_signal) * 0.5
                
                # Scale to graph dimensions
                points = []
                for i, val in enumerate(normalized):
                    x = graph_x + int(i * graph_width / len(normalized))
                    y = graph_y + graph_height - int(val * (graph_height - 20)) - 10
                    points.append([x, y])
                
                # Draw the line
                if len(points) > 1:
                    points = np.array(points, dtype=np.int32)
                    cv2.polylines(frame, [points], False, CYAN, 2)
                    
                    # Draw glow effect
                    cv2.polylines(frame, [points], False, (100, 100, 0), 4)
                    cv2.polylines(frame, [points], False, CYAN, 2)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single video frame.
        
        Args:
            frame: Input video frame (BGR)
            
        Returns:
            Processed frame with HUD overlay
        """
        # Update timing
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face
        results = self.face_mesh.process(rgb_frame)
        
        face_bbox = None
        is_stable = True
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Calculate face movement
            self.face_movement = self.calculate_movement(landmarks)
            self.last_face_landmarks = landmarks
            
            # Check stability (movement threshold)
            is_stable = self.face_movement < 0.01
            
            # Get forehead ROI
            mask, hull, bbox = self.get_forehead_roi(frame, landmarks)
            face_bbox = bbox
            
            # Extract green channel mean
            green_mean = self.extract_green_channel_mean(frame, mask)
            self.signal_buffer.append(green_mean)
            
            # Process signal if we have enough frames
            if len(self.signal_buffer) >= MIN_FRAMES_FOR_BPM:
                raw_signal = np.array(self.signal_buffer)
                fps = self.get_fps()
                
                bpm, filtered = process_signal(raw_signal, fs=fps)
                
                if bpm > 40 and bpm < 200:  # Physiologically valid range
                    self.bpm_history.append(bpm)
                    # Smooth BPM with moving average
                    self.current_bpm = np.mean(self.bpm_history)
                
                self.filtered_signal = filtered
                self.signal_quality = calculate_signal_quality(filtered)
            
            # Draw forehead ROI outline (optional debug visualization)
            # cv2.polylines(frame, [hull], True, GREEN, 1)
        else:
            # No face detected
            is_stable = False
            self.signal_quality = 0.0
        
        # Draw HUD
        frame = self.draw_hud(frame, face_bbox, is_stable)
        
        return frame
    
    def run(self) -> None:
        """Run the main application loop."""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            print("Please ensure a webcam is connected and accessible.")
            return
        
        # Set camera properties for optimal performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("=" * 50)
        print("VITAL-SENSE Heart Rate Monitor")
        print("=" * 50)
        print("Instructions:")
        print("1. Position your face in the camera view")
        print("2. Keep still for accurate readings")
        print("3. Ensure good lighting on your face")
        print("4. Press 'q' to quit")
        print("=" * 50)
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Could not read frame from webcam.")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display the result
                cv2.imshow("VITAL-SENSE", processed_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()
            print("\nVITAL-SENSE terminated.")


def main():
    """Entry point for the application."""
    monitor = VitalSenseMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
