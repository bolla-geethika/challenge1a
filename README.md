# Challenge 1a: PDF Outline Extractor (ML Approach)

This project is a solution for Challenge 1a of the Adobe India Hackathon 2025. It provides a Dockerized application that extracts a structured outline (title and headings) from PDF documents and outputs the results as JSON files.

## Approach

This solution uses a **Machine Learning (ML) model** (`RandomForestClassifier`) to classify lines of text as titles, headings, or paragraphs.

### 1. Training

The Docker container includes a `training_data` directory with sample PDFs and their corresponding correct JSON outputs.

- When the container runs for the first time, the script checks if a trained model file (`document_outline_model.joblib`) exists.
- If not, it runs a training routine using all the examples in the `training_data` folder.
- The newly trained model is saved inside the container for subsequent use.

### 2. Feature Extraction

For each line of text in a PDF, a set of features is calculated to help the model make its decision. These include:

- **Font Style**: `font_size`, `rel_font_size` (relative to body text), `is_bold`.
- **Text Content**: `word_count`, `char_count`, `is_all_caps`, `is_numbered`.
- **Layout**: `x_pos` and `y_pos` (normalized position on page).

### 3. Prediction and Post-processing

- The trained model predicts a label (e.g., H1, H2, Paragraph) for every line of text.
- Post-processing rules are applied to clean up the output, remove duplicates, and ensure a logical heading hierarchy.

## How to Build and Run

### 1. Organize Project Files

Your project must have the following structure:
