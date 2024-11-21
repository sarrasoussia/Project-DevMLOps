# Use a Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the Flask app port
EXPOSE 5001

# Run the Flask app
CMD ["python", "vgg19_service.py"]