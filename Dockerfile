
# Use the official Python image as base
FROM python:3

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir Flask numpy pandas scikit-learn yfinance

# Expose the port on which your Flask app will run
EXPOSE 5000

# Define the command to run your Flask app when the container starts
CMD ["python", "app.py"]
