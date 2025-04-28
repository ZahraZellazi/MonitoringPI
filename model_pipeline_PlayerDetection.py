import matplotlib.pyplot as plt
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from ultralytics import YOLO
from collections import defaultdict
import os
import csv
from datetime import datetime
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from typing import Dict, List, Tuple
from prefect import task
from prefect.logging import get_run_logger

# -------------------- YOLO Detector --------------------

class YOLODetector:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        self.class_names = {0: "player", 32: "ball"}
        
    def detect_objects(self, frame):
        results = self.model(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                detections.append({
                    'class': self.class_names.get(class_id, "unknown"),
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class_id': class_id
                })
        return detections

# -------------------- GNN Models --------------------

class PlayerMovementGNN(torch.nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=2):
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        return self.lin(x)

class TacticalRefiner(torch.nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=2):
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_channels, hidden_channels)
        self.tactical_conv = torch_geometric.nn.GCNConv(hidden_channels, hidden_channels)
        self.fc_refinement = torch.nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        movement_features = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        tactical_features = F.relu(self.tactical_conv(movement_features, edge_index, edge_weight=edge_weight))
        combined = torch.cat([movement_features, tactical_features], dim=1)
        return self.fc_refinement(combined)

# -------------------- Tactical Analyzer --------------------

class TacticalAnalyzer:
    def __init__(self):
        self.pitch_size = (600, 400)
        self.team_colors = {'Team1': (255, 0, 0), 'Team2': (0, 255, 255), 'Goalkeeper': (255, 255, 0)}
        
    def detect_team(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        lower_cyan = np.array([85, 150, 50])
        upper_cyan = np.array([95, 255, 255])
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        blue_pixels = cv2.countNonZero(blue_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        cyan_pixels = cv2.countNonZero(cyan_mask)
        
        counts = {'Team1': blue_pixels, 'Team2': yellow_pixels, 'Goalkeeper': cyan_pixels}
        team = max(counts, key=counts.get) if max(counts.values()) > 50 else "Unknown"
        return team

    def calculate_team_metrics(self, positions, team_assignments):
        team_positions = defaultdict(list)
        for pos, team in zip(positions, team_assignments):
            team_positions[team].append(pos)
        
        metrics = {}
        for team, positions in team_positions.items():
            if positions:
                centroid = np.mean(positions, axis=0)
                distances = [np.linalg.norm(np.array(pos) - centroid) for pos in positions]
                metrics[f'{team}_centroid'] = centroid
                metrics[f'{team}_compactness'] = np.mean(distances)
                metrics[f'{team}_spread'] = np.std(distances)
            else:
                metrics[f'{team}_centroid'] = (0, 0)
                metrics[f'{team}_compactness'] = 0
                metrics[f'{team}_spread'] = 0
        
        if 'Team1' in team_positions and 'Team2' in team_positions:
            metrics['inter_team_distance'] = np.linalg.norm(
                metrics['Team1_centroid'] - metrics['Team2_centroid'])
        else:
            metrics['inter_team_distance'] = 0
            
        return metrics

# -------------------- Tasks --------------------

@task(name="process_video", retries=2, retry_delay_seconds=10)
def process_video(video_path_1: str, frame_skip: int = 3) -> str:
    logger = get_run_logger()
    logger.info(f"Starting video processing for {video_path_1}")
    
    output_csv = "detections.csv"
    with mlflow.start_run(nested=True):
        mlflow.log_param("processing_frame_skip", frame_skip)
        
        detector = YOLODetector()
        analyzer = TacticalAnalyzer()
        cap = cv2.VideoCapture(video_path_1)
        
        if not cap.isOpened():
            error_msg = f"Could not open video at {video_path_1}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        last_positions = {}
        
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["frame", "time", "class", "team", "x1", "y1", "x2", "y2", "cx", "cy", "speed"])
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                    
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                frame = cv2.resize(frame, (1000, 550))
                detections = detector.detect_objects(frame)
                
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    speed = 0
                    
                    if det['class'] == "player":
                        team = analyzer.detect_team(frame, det['bbox'])
                        key = (team, det['class_id'])
                        
                        if team != "Unknown" and key in last_positions:
                            last_cx, last_cy, last_time = last_positions[key]
                            dist = np.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
                            time_diff = current_time - last_time
                            speed = dist / time_diff if time_diff > 0 else 0
                        
                        last_positions[key] = (cx, cy, current_time)
                    else:
                        team = "ball"
                    
                    writer.writerow([frame_count, current_time, det['class'], team, x1, y1, x2, y2, cx, cy, speed])
                    
        cap.release()
        mlflow.log_artifact(output_csv)
        logger.info(f"Video processing completed. Results saved to {output_csv}")
        return output_csv

@task(name="prepare_graph_data")
def prepare_graph_data(csv_path: str) -> Tuple[List[Data], List[float]]:
    logger = get_run_logger()
    logger.info(f"Preparing graph data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    graphs = []
    ball_speeds = []
    
    for frame_id in sorted(df['frame'].unique()):
        frame_data = df[df['frame'] == frame_id]
        node_features = []
        positions = []
        team_assignments = []
        for _, row in frame_data.iterrows():
            feature = [row['cx'], row['cy'], row['speed'], int(row['team'] == "Team1"), int(row['team'] == "Team2")]
            node_features.append(feature)
            positions.append((row['cx'], row['cy']))
            team_assignments.append(row['team'])
            if row['class'] == "ball":
                ball_speeds.append(row['speed'])
        
        if len(node_features) == 0:
            continue
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        edges = []
        edge_attr = []
        for i in range(len(positions)):
            for j in range(len(positions)):
                if i != j:
                    xi, yi = positions[i]
                    xj, yj = positions[j]
                    distance = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    if distance < 150:
                        edges.append([i, j])
                        if team_assignments[i] == team_assignments[j]:
                            edge_attr.append(1.0)
                        else:
                            edge_attr.append(0.3)
        
        if len(edges) == 0:
            continue
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_attr, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        graphs.append(data)
        
    logger.info(f"Created {len(graphs)} graph samples with {len(ball_speeds)} ball speed measurements")
    return graphs, ball_speeds

@task(name="train_models")
def train_models(graph_data: List[Data], epochs: int = 20):
    logger = get_run_logger()
    logger.info(f"Starting model training with {epochs} epochs")
    
    movement_model = PlayerMovementGNN()
    refiner_model = TacticalRefiner()
    optimizer = torch.optim.Adam(
        list(movement_model.parameters()) + list(refiner_model.parameters()), 
        lr=1e-3
    )
    loader = DataLoader(graph_data, batch_size=4, shuffle=True)
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            movement_out = movement_model(batch.x, batch.edge_index, batch.edge_attr)
            refinement_out = refiner_model(batch.x, batch.edge_index, batch.edge_attr)
            combined_out = movement_out * 0.7 + refinement_out * 0.3
            targets = torch.zeros_like(combined_out)
            loss = F.mse_loss(combined_out, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
    
    logger.info("Model training completed")
    return movement_model, refiner_model, losses

@task(name="evaluate_detections")
def evaluate_detections(csv_path: str):
    logger = get_run_logger()
    logger.info("Evaluating detection performance")
    
    df = pd.read_csv(csv_path)
    y_true = df['class']
    y_pred = df['team'].apply(lambda t: 'player' if t in ['Team1', 'Team2'] else 'ball')
    
    player_mask = (y_true == 'player')
    ball_mask = (y_true == 'ball')
    
    player_precision = precision_score(y_true[player_mask], y_pred[player_mask], pos_label="player", zero_division=0)
    player_recall = recall_score(y_true[player_mask], y_pred[player_mask], pos_label="player", zero_division=0)
    player_f1 = f1_score(y_true[player_mask], y_pred[player_mask], pos_label="player", zero_division=0)
    
    ball_precision = precision_score(y_true[ball_mask], y_pred[ball_mask], pos_label="ball", zero_division=0)
    ball_recall = recall_score(y_true[ball_mask], y_pred[ball_mask], pos_label="ball", zero_division=0)
    ball_f1 = f1_score(y_true[ball_mask], y_pred[ball_mask], pos_label="ball", zero_division=0)
    
    acc = accuracy_score(y_true, y_pred)
    
    metrics = {
        "player_precision": player_precision,
        "player_recall": player_recall,
        "player_f1": player_f1,
        "ball_precision": ball_precision,
        "ball_recall": ball_recall,
        "ball_f1": ball_f1,
        "accuracy": acc
    }
    
    mlflow.log_metrics(metrics)
    logger.info("\n📊 Detection Performance Metrics:")
    logger.info(f"Player - Precision: {player_precision:.2f}, Recall: {player_recall:.2f}, F1: {player_f1:.2f}")
    logger.info(f"Ball - Precision: {ball_precision:.2f}, Recall: {ball_recall:.2f}, F1: {ball_f1:.2f}")
    logger.info(f"Overall Accuracy: {acc:.2f}")
    
    return metrics

@task(name="visualize_results")
def visualize_results(graph_data: List[Data], models: Tuple, ball_speeds: List[float]):
    logger = get_run_logger()
    logger.info("Visualizing results")
    
    movement_model, refiner_model = models
    
    plt.figure(figsize=(10, 5))
    plt.plot(ball_speeds)
    plt.title("Ball Speed Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Speed (pixels/frame)")
    mlflow.log_figure(plt.gcf(), "ball_speed_plot.png")
    plt.close()
    
    sample_data = graph_data[:5]
    movement_model.eval()
    refiner_model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(sample_data):
            movement_pred = movement_model(data.x, data.edge_index, data.edge_attr)
            refinement_pred = refiner_model(data.x, data.edge_index, data.edge_attr)
            orig_pos = data.x[:, :2].numpy()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax1.scatter(orig_pos[:, 0], orig_pos[:, 1], c='blue', label='Original')
            ax1.scatter(movement_pred[:, 0], movement_pred[:, 1], c='red', marker='x', label='Predicted')
            ax1.set_title(f"Movement Prediction - Sample {i+1}")
            ax1.legend()
            
            ax2.scatter(orig_pos[:, 0], orig_pos[:, 1], c='blue', label='Original')
            ax2.scatter(refinement_pred[:, 0], refinement_pred[:, 1], c='green', marker='s', label='Refinement')
            ax2.set_title(f"Tactical Refinement - Sample {i+1}")
            ax2.legend()
            
            plt.tight_layout()
            mlflow.log_figure(fig, f"predictions_sample_{i+1}.png")
            plt.close()
    
    analyzer = TacticalAnalyzer()
    pitch = np.zeros((analyzer.pitch_size[1], analyzer.pitch_size[0], 3), dtype=np.uint8)
    pitch[:] = (34, 139, 34)
    
    cv2.rectangle(pitch, (50, 50), (analyzer.pitch_size[0]-50, analyzer.pitch_size[1]-50), (255, 255, 255), 2)
    cv2.line(pitch, (analyzer.pitch_size[0]//2, 50), (analyzer.pitch_size[0]//2, analyzer.pitch_size[1]-50), (255, 255, 255), 2)
    cv2.circle(pitch, (analyzer.pitch_size[0]//2, analyzer.pitch_size[1]//2), 50, (255, 255, 255), 2)
    
    last_frame = graph_data[-1]
    positions = last_frame.x[:, :2].numpy()
    teams = ["Team1" if x[3] == 1 else "Team2" for x in last_frame.x.numpy()]
    
    for pos, team in zip(positions, teams):
        color = analyzer.team_colors.get(team, (255, 255, 255))
        cv2.circle(pitch, (int(pos[0]), int(pos[1])), 5, color, -1)
    
    plt.imshow(cv2.cvtColor(pitch, cv2.COLOR_BGR2RGB))
    plt.title("Final Tactical Overview")
    mlflow.log_figure(plt.gcf(), "final_tactical_overview.png")
    plt.close()
    
    logger.info("Visualization completed")
