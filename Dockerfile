# Use the specified python base image compatible with linux/amd64
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install required python libraries with compatible versions
# This pins numpy to a version before ComplexWarning was removed.
RUN pip install --no-cache-dir scikit-learn==1.3.0 pandas PyMuPDF joblib "numpy<2.0"

# Copy the processing script and the training data into the container
COPY process_pdfs.py .
COPY training_data/ ./training_data/

# Command to run when the container starts
CMD ["python", "process_pdfs.py"]