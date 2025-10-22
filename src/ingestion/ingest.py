import os
from pathlib import Path
import pandas as pd
import textract

def extract_text(file_path: str) -> str:
    try:
        text = textract.process(file_path).decode('utf-8')
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        text = ""
    return text

def ingest_folder(folder: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for p in Path(folder).rglob('*'):
        if p.is_file() and p.suffix.lower() in ['.pdf', '.txt', '.docx']:
            text = extract_text(str(p))
            out_path = Path(out_dir) / (p.stem + '.txt')
            out_path.write_text(text, encoding='utf-8')
            rows.append({'doc_id': p.stem, 'path': str(out_path), 'source': str(p)})
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir)/'metadata.csv', index=False)
    print(f"Ingested {len(rows)} files")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw')
    parser.add_argument('--output', default='data/processed')
    args = parser.parse_args()
    ingest_folder(args.input, args.output)
