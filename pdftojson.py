import requests
import pdfplumber
import json
import io
import sys
from typing import List, Dict, Any

def download_pdf(sas_url: str) -> bytes:
    """
    Fetches the PDF bytes from ADLS via the provided SAS URL.
    """
    resp = requests.get(sas_url)
    resp.raise_for_status()
    return resp.content

def parse_pdf_to_json(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Opens the PDF in-memory and extracts:
      - page_number
      - text (full page text)
      - tables (list of row-lists)
    Returns a dict suitable for JSON serialization.
    """
    document = {"pages": []}
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # Extract full text
            text = page.extract_text() or ""
            
            # Extract all tables on the page
            tables: List[List[List[str]]] = []
            for table in page.extract_tables():
                # table is a list of rows, each row is list of cell-strings
                tables.append(table)
            
            document["pages"].append({
                "page_number": i,
                "text": text,
                "tables": tables
            })
    return document

def save_json(obj: Dict[str, Any], path: str):
    """
    Writes the given object to a JSON file with pretty formatting.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main(sas_url: str, output_path: str = "output.json"):
    try:
        print("⏬ Downloading PDF...")
        pdf_bytes = download_pdf(sas_url)
        
        print("🔍 Parsing PDF content...")
        doc_json = parse_pdf_to_json(pdf_bytes)
        
        print(f"💾 Saving JSON to {output_path}...")
        save_json(doc_json, output_path)
        
        print("✅ Extraction complete!")
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_json.py <sas_url> [<output_path>]")
        sys.exit(1)
    sas_url_arg = sys.argv[1]
    out_path_arg = sys.argv[2] if len(sys.argv) > 2 else "output.json"
    main(sas_url_arg, out_path_arg)
