FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files to the image
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Run the main script with collection path mounted at runtime
ENTRYPOINT ["python", "-m", "src", "--collection_path", "/app/input"]
