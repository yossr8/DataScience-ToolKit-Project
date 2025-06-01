# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
