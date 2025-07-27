import os
import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd
import fitz  # PyMuPDF
import joblib
from sklearn.ensemble import RandomForestClassifier

# --- 1. Feature Extraction ---
def extract_features(pdf_path):
    doc = fitz.open(pdf_path)
    features = []
    all_sizes = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        all_sizes.append(round(s["size"]))
    body_size = Counter(all_sizes).most_common(1)[0][0] if all_sizes else 10

    for page_num, page in enumerate(doc):
        page_width, page_height = page.rect.width, page.rect.height
        blocks = page.get_text("dict", flags=11)["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if not line["spans"] or not line["spans"][0]["text"].strip():
                        continue
                    span = line["spans"][0]
                    text = " ".join([s["text"] for s in line["spans"]]).strip()
                    font_size = round(span["size"])
                    is_all_caps = 1 if text.isupper() and len(text) > 1 else 0
                    is_numbered_heading = 1 if re.match(r'^\s*(\d+(\.\d+)*)\s+', text) else 0
                    features.append({
                        "text": text, "font_size": font_size,
                        "rel_font_size": font_size / body_size,
                        "is_bold": 1 if "bold" in span["font"].lower() else 0,
                        "word_count": len(text.split()), "char_count": len(text),
                        "x_pos": span["origin"][0] / page_width, "y_pos": span["origin"][1] / page_height,
                        "is_numbered": is_numbered_heading, "is_all_caps": is_all_caps,
                        "page_num": page_num
                    })
    return pd.DataFrame(features)

# --- 2. Model Training ---
def train_model(training_input_dir, training_output_dir, model_path):
    all_features = []
    print(f"Starting model training...")
    for pdf_filename in os.listdir(training_input_dir):
        if not pdf_filename.endswith(".pdf"): continue
        base_name = os.path.splitext(pdf_filename)[0]
        pdf_path = os.path.join(training_input_dir, pdf_filename)
        json_path = os.path.join(training_output_dir, base_name + ".json")
        if not os.path.exists(json_path): continue
        print(f"  - Learning from {pdf_filename}")
        with open(json_path, 'r', encoding='utf-8') as f: json_data = json.load(f)
        df = extract_features(pdf_path)
        if df.empty: continue
        labels = []
        title_text = " ".join(json_data["title"].strip().split())
        outline_texts = {" ".join(o["text"].strip().split()): o["level"] for o in json_data["outline"]}
        for _, row in df.iterrows():
            text = " ".join(row["text"].strip().split())
            if title_text and text == title_text: labels.append("Title")
            elif text in outline_texts: labels.append(outline_texts[text])
            else: labels.append("Paragraph")
        df["label"] = labels
        all_features.append(df)
    
    if not all_features:
        raise RuntimeError("No training data was successfully processed. Model cannot be trained.")
    
    training_data = pd.concat(all_features, ignore_index=True)
    feature_cols = ['font_size', 'rel_font_size', 'is_bold', 'word_count', 'char_count', 'x_pos', 'y_pos', 'is_numbered', 'is_all_caps']
    X = training_data[feature_cols]
    y = training_data['label']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")
    return model

# --- 3. Prediction ---
def generate_outline(pdf_path, model):
    doc = fitz.open(pdf_path)
    title = ""
    if len(doc) > 0:
        first_page = doc.load_page(0)
        page_height = first_page.rect.height
        max_font_size = 0
        blocks = first_page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b and b['bbox'][1] < page_height * 0.40:
                for l in b["lines"]:
                    for s in l["spans"]:
                        if s["size"] > max_font_size: max_font_size = s["size"]
        if max_font_size > 0:
            title_texts = []
            for b in blocks:
                if "lines" in b and b['bbox'][1] < page_height * 0.40:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            if abs(s["size"] - max_font_size) < 0.1: title_texts.append(s["text"].strip())
            title = " ".join(title_texts)

    features_df = extract_features(pdf_path)
    if features_df.empty: return {"title": title, "outline": []}
    
    feature_cols = ['font_size', 'rel_font_size', 'is_bold', 'word_count', 'char_count', 'x_pos', 'y_pos', 'is_numbered', 'is_all_caps']
    predictions = model.predict(features_df[feature_cols])

    outline = []
    seen_headings = set()
    for i, prediction in enumerate(predictions):
        if prediction in ["H1", "H2", "H3"]:
            item = features_df.iloc[i]
            text = item["text"]
            if text in seen_headings: continue
            level = prediction
            match = re.match(r'^\s*(\d+(\.\d+)*)\s+', text)
            if match:
                depth = match.group(1).count('.')
                if depth == 0: level = "H1"
                elif depth == 1: level = "H2"
                else: level = "H3"
            outline.append({"level": level, "text": text, "page": int(item["page_num"])})
            seen_headings.add(text)
            
    return {"title": title, "outline": outline}


# --- Main Execution Logic ---
def main():
    # Define directories relative to the container's /app folder
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    model_path = Path("/app/document_outline_model.joblib")
    training_input_dir = Path("/app/training_data/inputs")
    training_output_dir = Path("/app/training_data/outputs")
    
    # --- Step 1: Get the Model ---
    # If a pre-trained model doesn't exist, train one using the bundled data.
    if not model_path.exists():
        print(f"No existing model found at {model_path}.")
        if not training_input_dir.exists():
            raise FileNotFoundError("Training data folder not found. Cannot train new model.")
        train_model(training_input_dir, training_output_dir, model_path)
    else:
        print(f"Loading existing model from {model_path}")
        
    model = joblib.load(model_path)

    # --- Step 2: Process PDFs ---
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
        
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        processed_data = generate_outline(pdf_file, model)
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, "w") as f:
            json.dump(processed_data, f, indent=4)
        print(f"  -> Saved output to {output_file.name}")

if __name__ == "__main__":
    print("Starting PDF processing...")
    main()
    print("Completed PDF processing.")