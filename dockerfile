# Step 1: Use an official Python runtime as a parent image
# Using a 'slim' version keeps our final image size smaller.
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file into the container
COPY requirements.txt .

# Step 4: Install the Python dependencies
# --no-cache-dir ensures we don't store the download cache, saving space.
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy our application code and data into the container
# This includes main.py, our saved embeddings, and text chunks.
COPY . .

# Step 6: Expose the port the app runs on
EXPOSE 8000

# Step 7: Define the command to run the application
# We use uvicorn to run our FastAPI app.
# The --host 0.0.0.0 makes the container accessible from outside.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]