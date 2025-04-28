import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
import math
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from typing import List, Tuple, Dict, Optional
import pandas as pd
from datetime import datetime
import os
from prefect import task, flow

# -------------------- Constants --------------------
SKILL_THRESHOLDS = {
    'MIN_AROUND_WORLD_FRAMES': 15,
    'MAX_AROUND_WORLD_FRAMES': 45,
    'AROUND_WORLD_RADIUS_THRESHOLD': 120,
    'MIN_CIRCULAR_POINTS': 8,
    'CIRCLE_FIT_THRESHOLD': 0.85,
    'DRIBBLE_TIME_THRESHOLD': 0.3,
    'MIN_BALL_HEIGHT_DRIBBLE': 20,
    'MAX_BALL_HEIGHT_DRIBBLE': 150,
    'JUMP_HEIGHT_THRESHOLD': 15,
    'JUMP_FRAME_THRESHOLD': 5
}

# -------------------- Base Analyzer Class --------------------
class BaseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
    
    def _get_foot_positions(self, landmarks, frame_width, frame_height):
        left_foot = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        left_heel = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL]
        
        left_pos = np.array([left_foot.x * frame_width, left_foot.y * frame_height])
        right_pos = np.array([right_foot.x * frame_width, right_foot.y * frame_height])
        left_heel_pos = np.array([left_heel.x * frame_width, left_heel.y * frame_height])
        right_heel_pos = np.array([right_heel.x * frame_width, right_heel.y * frame_height])
        
        return left_pos, right_pos, left_heel_pos, right_heel_pos

# -------------------- Detector Classes --------------------

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results

class BallDetector:
    def __init__(self, model_path='yolo11n.pt'):
        self.model = YOLO(model_path)
        
    def detect_ball(self, frame, conf_threshold=0.4):
        detections = self.model.predict(frame, conf=conf_threshold, verbose=False)
        ball_positions = []
        
        for r in detections:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                
                if label.lower() in ['ball', 'sports ball']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    ball_positions.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'height': (y2 - y1)
                    })
        
        return ball_positions

# -------------------- Skill Analyzers --------------------

class JuggleAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.juggle_count = 0
        self.ball_positions = deque(maxlen=50)
        self.in_air = False
        self.ball_velocity = deque(maxlen=10)
        self.juggle_data = []
        self.leg_touches = {'left': 0, 'right': 0}
        
    def analyze(self, ball_pos, landmarks,frame_width, frame_height, timestamp):
        if ball_pos is None or landmarks is None:
            return
            
        self.ball_positions.append(ball_pos)
        ball_height = frame_height - ball_pos[1]
        
        if len(self.ball_positions) > 5:
            recent_y = [pos[1] for pos in self.ball_positions]
            vertical_velocity = recent_y[-1] - recent_y[-2]
            self.ball_velocity.append(vertical_velocity)
            
            if len(self.ball_velocity) > 1:
                velocity = self.ball_velocity[-1]
                prev_velocity = self.ball_velocity[-2]
                
                if velocity > 0 and prev_velocity <= 0 and not self.in_air:
                    self.in_air = True
                elif velocity < 0 and prev_velocity >= 0 and self.in_air:
                    self.in_air = False
                    self.juggle_count += 1
                    self.juggle_data.append((timestamp, ball_pos[0], ball_pos[1]))
                    
                    left_pos, right_pos, _, _ = self._get_foot_positions(landmarks, frame_width, frame_height)
                    ball_np = np.array(ball_pos)
                    dist_left = np.linalg.norm(ball_np - left_pos)
                    dist_right = np.linalg.norm(ball_np - right_pos)
                    
                    if dist_left < dist_right:
                        self.leg_touches['left'] += 1
                    else:
                        self.leg_touches['right'] += 1

class DribbleAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.dribble_count = 0
        self.last_leg_used = None
        self.last_leg_switch_time = None
        
    def analyze(self, ball_pos, landmarks, frame_width, frame_height, timestamp):
        if ball_pos is None or landmarks is None:
            return
            
        left_pos, right_pos, _, _ = self._get_foot_positions(landmarks, frame_width, frame_height)
        ball_height = frame_height - ball_pos[1]
        
        dist_left = np.linalg.norm(np.array(ball_pos) - left_pos)
        dist_right = np.linalg.norm(np.array(ball_pos) - right_pos)
        
        current_leg = 'left' if dist_left < dist_right else 'right'
        
        if (SKILL_THRESHOLDS['MIN_BALL_HEIGHT_DRIBBLE'] < ball_height < 
            SKILL_THRESHOLDS['MAX_BALL_HEIGHT_DRIBBLE']):
            if self.last_leg_used is not None and current_leg != self.last_leg_used:
                current_time = timestamp
                
                if self.last_leg_switch_time is not None:
                    time_since_last_switch = current_time - self.last_leg_switch_time
                    
                    if time_since_last_switch < SKILL_THRESHOLDS['DRIBBLE_TIME_THRESHOLD']:
                        self.dribble_count += 1
                
                self.last_leg_switch_time = current_time
            
            self.last_leg_used = current_leg

class AroundWorldAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.around_world_count = 0
        self.around_world_start = None
        self.around_world_leg = None
        self.around_world_positions = []
        
    def analyze(self, ball_pos, landmarks, frame_width, frame_height, frame_count):
        if ball_pos is None or landmarks is None:
            return
            
        left_pos, right_pos, _, _ = self._get_foot_positions(landmarks, frame_width, frame_height)
        ball_np = np.array(ball_pos)
        
        dist_left = np.linalg.norm(ball_np - left_pos)
        dist_right = np.linalg.norm(ball_np - right_pos)
        
        current_leg = 'left' if dist_left < dist_right else 'right'
        foot_pos = left_pos if current_leg == 'left' else right_pos
        
        if (dist_left < SKILL_THRESHOLDS['AROUND_WORLD_RADIUS_THRESHOLD'] or 
            dist_right < SKILL_THRESHOLDS['AROUND_WORLD_RADIUS_THRESHOLD']):
            
            if self.around_world_start is None:
                self.around_world_start = frame_count
                self.around_world_leg = current_leg
                self.around_world_positions = [foot_pos]
            elif current_leg == self.around_world_leg:
                self.around_world_positions.append(foot_pos)
                
                frame_diff = frame_count - self.around_world_start
                if (SKILL_THRESHOLDS['MIN_AROUND_WORLD_FRAMES'] <= frame_diff <= 
                    SKILL_THRESHOLDS['MAX_AROUND_WORLD_FRAMES']):
                    if self._detect_circular_motion(self.around_world_positions):
                        self.around_world_count += 1
                        self.around_world_start = None
                        self.around_world_positions = []
                elif frame_diff > SKILL_THRESHOLDS['MAX_AROUND_WORLD_FRAMES']:
                    self.around_world_start = None
                    self.around_world_positions = []
            else:
                self.around_world_start = None
                self.around_world_positions = []
        else:
            self.around_world_start = None
            self.around_world_positions = []
    
    def _detect_circular_motion(self, points):
        if len(points) < SKILL_THRESHOLDS['MIN_CIRCULAR_POINTS']:
            return False
        
        points_array = np.array(points)
        mean = np.mean(points_array, axis=0)
        points_centered = points_array - mean
        
        x = points_centered[:, 0]
        y = points_centered[:, 1]
        z = x**2 + y**2
        A = np.column_stack((x, y, np.ones(len(x))))
        b = z
        res = np.linalg.lstsq(A, b, rcond=None)[0]
        
        xc = res[0]/2
        yc = res[1]/2
        r = np.sqrt(res[2] + xc**2 + yc**2)
        
        residuals = np.abs(np.sqrt((x-xc)**2 + (y-yc)**2) - r)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((z - np.mean(z))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return r_squared > SKILL_THRESHOLDS['CIRCLE_FIT_THRESHOLD']

# -------------------- Posture Analyzer --------------------

class PostureAnalyzer:
    def __init__(self):
        self.balance_history = []
        self.movement_speed_history = []
        self.last_shoulder_pos = None
        self.last_hip_pos = None
        self.movement_speeds = deque(maxlen=10)
        self.posture_data = []
        self.jump_count = 0
        self.last_floor_y = None
        self.jumping = False
        self.mp_pose = mp.solutions.pose
    
    def analyze(self, landmarks, frame_width, frame_height, frame_count, fps):
        if not landmarks:
            return None, 0
            
        left_shoulder, right_shoulder = self._get_shoulder_positions(landmarks, frame_width, frame_height)
        left_hip, right_hip = self._get_hip_positions(landmarks, frame_width, frame_height)
        
        # Calculate balance score
        balance_score = self._calculate_balance_score(left_shoulder, right_shoulder, left_hip, right_hip)
        self.balance_history.append(balance_score)
        
        # Calculate movement speed
        current_position = (left_shoulder + right_shoulder) / 2
        current_speed = 0
        
        if self.last_shoulder_pos is not None:
            distance = np.linalg.norm(current_position - self.last_shoulder_pos)
            current_speed = distance * fps
            self.movement_speeds.append(current_speed)
            self.movement_speed_history.append(current_speed)
        
        self.last_shoulder_pos = current_position
        
        # Jump detection
        current_floor_y = self._estimate_floor_position(landmarks, frame_height)
        
        if self.last_floor_y is not None:
            if abs(current_floor_y - self.last_floor_y) > SKILL_THRESHOLDS['JUMP_HEIGHT_THRESHOLD']:
                if not self.jumping:
                    self.jumping = True
                    self.jump_count += 1
            else:
                if self.jumping:
                    self.jumping = False
        
        self.last_floor_y = current_floor_y
        
        # Store posture data
        timestamp = frame_count / fps
        self.posture_data.append((timestamp, balance_score, current_speed))
        
        return balance_score, current_speed
    
    def _get_shoulder_positions(self, landmarks, frame_width, frame_height):
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_pos = np.array([left_shoulder.x * frame_width, left_shoulder.y * frame_height])
        right_pos = np.array([right_shoulder.x * frame_width, right_shoulder.y * frame_height])
        return left_pos, right_pos
    
    def _get_hip_positions(self, landmarks, frame_width, frame_height):
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_pos = np.array([left_hip.x * frame_width, left_hip.y * frame_height])
        right_pos = np.array([right_hip.x * frame_width, right_hip.y * frame_height])
        return left_pos, right_pos
    
    def _calculate_balance_score(self, left_shoulder, right_shoulder, left_hip, right_hip):
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
        hip_diff = abs(left_hip[1] - right_hip[1])
        avg_diff = (shoulder_diff + hip_diff) / 2
        return max(0, min(100, 50 - (avg_diff / 10 * 50)))
    
    def _estimate_floor_position(self, landmarks, frame_height):
        left_heel = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL]
        left_foot_index = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot_index = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        
        positions = [
            left_heel.y * frame_height,
            right_heel.y * frame_height,
            left_foot_index.y * frame_height,
            right_foot_index.y * frame_height
        ]
        
        return max(positions)

# -------------------- Video Processor --------------------

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video at {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.display_width = 400
        self.scale_factor = self.display_width / self.frame_width
        
        # Initialize detectors and analyzers
        self.pose_detector = PoseDetector()
        self.ball_detector = BallDetector()
        self.juggle_analyzer = JuggleAnalyzer()
        self.dribble_analyzer = DribbleAnalyzer()
        self.around_world_analyzer = AroundWorldAnalyzer()
        self.posture_analyzer = PostureAnalyzer()
        
        # Tracking variables
        self.frame_count = 0
        self.last_ball_pos = None
        self.movement_count = 0
        self.control_start_time = None
        self.control_periods = []
    
    def process_frame(self, frame):
        self.frame_count += 1
        timestamp = self.frame_count / self.fps
        
        # Pose detection
        pose_results = self.pose_detector.process_frame(frame)
        landmarks = pose_results.pose_landmarks if pose_results.pose_landmarks else None
        
        # Ball detection
        ball_detections = self.ball_detector.detect_ball(frame)
        current_ball_pos = ball_detections[0]['center'] if ball_detections else None
        
        # Movement detection
        if self.last_ball_pos is not None and current_ball_pos is not None:
            movement = np.linalg.norm(np.array(current_ball_pos) - np.array(self.last_ball_pos))
            if movement > 5:  # Threshold of 5 pixels
                self.movement_count += 1
        self.last_ball_pos = current_ball_pos
        
        # Ball control periods
        if current_ball_pos is not None:
            if self.control_start_time is None:
                self.control_start_time = timestamp
        else:
            if self.control_start_time is not None:
                self.control_periods.append((self.control_start_time, timestamp))
                self.control_start_time = None
        
        # Analyze skills
        if landmarks and current_ball_pos:
            self.juggle_analyzer.analyze(
                current_ball_pos, landmarks, self.frame_width, self.frame_height, timestamp)
            self.dribble_analyzer.analyze(
                current_ball_pos, landmarks,self.frame_width, self.frame_height, timestamp)
            self.around_world_analyzer.analyze(
                current_ball_pos, landmarks, self.frame_width, self.frame_height, self.frame_count)
        
        # Posture analysis
        balance_score, current_speed = self.posture_analyzer.analyze(
            landmarks, self.frame_width, self.frame_height, self.frame_count, self.fps)
        
        return {
            'frame_count': self.frame_count,
            'timestamp': timestamp,
            'ball_pos': current_ball_pos,
            'landmarks': landmarks,
            'balance_score': balance_score,
            'current_speed': current_speed,
            'jumping': self.posture_analyzer.jumping
        }

    def get_skill_counts(self):
        return {
            'juggles': self.juggle_analyzer.juggle_count,
            'dribbles': self.dribble_analyzer.dribble_count,
            'around_worlds': self.around_world_analyzer.around_world_count,
            'jumps': self.posture_analyzer.jump_count,
            'movements': self.movement_count,
            'leg_touches': self.juggle_analyzer.leg_touches,
            'control_periods': self.control_periods,
            'posture_data': self.posture_analyzer.posture_data
        }

    def release(self):
        self.cap.release()

# -------------------- Prefect Tasks --------------------

@task(name="Process Video for Performance Analysis")
def process_video_performance(video_path: str) -> Dict:
    """Task to process video and extract player performance metrics."""
    processor = VideoProcessor(video_path)
    
    while processor.cap.isOpened():
        ret, frame = processor.cap.read()
        if not ret:
            break
        
        frame_data = processor.process_frame(frame)
    
    skill_counts = processor.get_skill_counts()
    processor.release()
    
    return skill_counts

@task(name="Visualize Performance Results")
def visualize_results_performance(skill_counts: Dict, output_dir: str = "output") -> None:
    """Task to visualize and save performance analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Skill counts visualization
    plt.figure(figsize=(10, 6))
    skills = ['Juggles', 'Dribbles', 'Around Worlds', 'Jumps']
    counts = [
        skill_counts['juggles'],
        skill_counts['dribbles'],
        skill_counts['around_worlds'],
        skill_counts['jumps']
    ]
    
    plt.bar(skills, counts, color=['blue', 'green', 'orange', 'red'])
    plt.title('Football Skills Count')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'skill_counts.png'))
    plt.close()
    
    # Leg touches visualization
    plt.figure(figsize=(6, 6))
    plt.pie(
        [skill_counts['leg_touches']['left'], skill_counts['leg_touches']['right']],
        labels=['Left Foot', 'Right Foot'],
        autopct='%1.1f%%',
        colors=['lightblue', 'lightgreen']
    )
    plt.title('Foot Usage for Juggles')
    plt.savefig(os.path.join(output_dir, 'foot_usage.png'))
    plt.close()
    
    # Posture data visualization
    if skill_counts['posture_data']:
        posture_df = pd.DataFrame(
            skill_counts['posture_data'], 
            columns=['timestamp', 'balance', 'speed'])
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(posture_df['timestamp'], posture_df['balance'])
        plt.title('Balance Score Over Time')
        plt.ylabel('Balance (0-100)')
        
        plt.subplot(2, 1, 2)
        plt.plot(posture_df['timestamp'], posture_df['speed'])
        plt.title('Movement Speed Over Time')
        plt.ylabel('Speed (px/s)')
        plt.xlabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'posture_metrics.png'))
        plt.close()
    
    mlflow.log_artifacts(output_dir)

@task(name="Track Performance Metrics")
def track_metrics(skill_counts: Dict) -> None:
    """Task to log performance metrics to MLflow."""
    mlflow.log_metrics({
        'juggles': skill_counts['juggles'],
        'dribbles': skill_counts['dribbles'],
        'around_worlds': skill_counts['around_worlds'],
        'jumps': skill_counts['jumps'],
        'movements': skill_counts['movements'],
        'left_foot_touches': skill_counts['leg_touches']['left'],
        'right_foot_touches': skill_counts['leg_touches']['right']
    })
    
    # Log control periods
    if skill_counts['control_periods']:
        control_durations = [end - start for start, end in skill_counts['control_periods']]
        mlflow.log_metrics({
            'avg_control_duration': np.mean(control_durations),
            'max_control_duration': np.max(control_durations),
            'total_control_time': np.sum(control_durations)
        })