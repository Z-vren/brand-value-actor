# Use the official Apify Python image
FROM apify/actor-python:3.11

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . ./

# Set the entry point to run the Python module
CMD ["python", "-m", "src.main"]

