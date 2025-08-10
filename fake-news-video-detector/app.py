import torch
import torch.nn as nn 
import cv2
import numpy as np
from pathlib import Path
import time
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from transformers import ViTModel


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Frame Extraction
def extract_frames(video_path, frame_rate=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // frame_rate)
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb)
            frames.append(frame_tensor)
        count += 1
    
    cap.release()
    return torch.stack(frames) if len(frames) > 0 else None

# ViT Feature Extractor
class ViTFeatureExtractorModel(nn.Module):
    def __init__(self):
        super(ViTFeatureExtractorModel, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
    def forward(self, x):
        with torch.no_grad():
            outputs = self.vit(pixel_values=x).last_hidden_state[:, 0, :]
        return outputs

# LSTM Classifier
class VideoFakeNewsDetector(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_layers=2, num_classes=2):
        super(VideoFakeNewsDetector, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

def load_best_model(model_path, model, optimizer=None):
    """
    Load the best saved model with proper device mapping
    """
    print(f"Loading model from {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded (Best accuracy: {checkpoint['accuracy']:.2f}%, Epoch: {checkpoint['epoch']})")
    return model

def load_video_frames(video_path, max_frames=32):
    """
    Load frames from a video file
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames could be loaded from {video_path}")
        
    frames = torch.stack(frames)
    print(f"Loaded {len(frames)} frames from video")
    return frames

def predict_video(video_path, model, vit_extractor, device, display_frames=False):
    """
    Make prediction on a single video
    """
    model.eval()
    start_time = time.time()
    
    try:
        # Load and process video frames
        print(f"Processing video: {video_path}")
        frames = load_video_frames(video_path)
        
        # Process frames
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        processed_frames = torch.stack([transform(frame) for frame in frames])
        processed_frames = processed_frames.to(device)
        
        # Make prediction
        with torch.no_grad():
            features = vit_extractor(processed_frames)
            outputs = model(features)
            
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get class probabilities
            class_probs = probabilities[0].cpu().numpy()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare results
        result = {
            'prediction': 'Real' if predicted.item() == 1 else 'Fake',
            'confidence': float(confidence.item()),
            'processing_time': float(processing_time),
            'class_probabilities': {
                'Fake': float(class_probs[0]),
                'Real': float(class_probs[1])
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None

# Load model globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # Initialize model and load weights
    video_model = VideoFakeNewsDetector().to(device)
    vit_extractor = ViTFeatureExtractorModel().to(device)
    
    # Load the best model
    model_path = 'best_video_model.pth'
    video_model = load_best_model(model_path, video_model)
    video_model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = predict_video(filepath, video_model, vit_extractor, device)
            os.remove(filepath)  # Clean up the uploaded file
            
            if result is None:
                return jsonify({'error': 'Error processing video'}), 500
                
            return jsonify(result)
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
