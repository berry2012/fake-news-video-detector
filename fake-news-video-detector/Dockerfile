FROM python:3.11

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt

# Copy application files
COPY app.py /app/
COPY best_video_model.pth /app/
COPY templates/ /app/templates/

# Create uploads directory
RUN mkdir -p /app/uploads && \
    chmod 777 /app/uploads

# Download the ViT model during build
RUN python -c "from transformers import ViTModel; ViTModel.from_pretrained('google/vit-base-patch16-224')"


# Make port 8080 available
EXPOSE 8080

# Run the application
CMD ["python", "/app/app.py"]