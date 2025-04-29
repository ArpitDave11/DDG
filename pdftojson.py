# 1. Install dependencies (only the first time)
# 1. Install dependencies (only the first time)
%pip install pdfminer.six requests

# 2. Imports
import requests
import json
from io import BytesIO
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine

# 3. Your SAS URL
sas_url = "https://<your-storage-account>.dfs.core.windows.net/<container>/<path>.pdf?<sas-token>"

# 4. Download PDF into memory
resp = requests.get(sas_url)
resp.raise_for_status()
pdf_stream = BytesIO(resp.content)

# 5. Set up pdfminer parser & device
parser = PDFParser(pdf_stream)
doc = PDFDocument(parser)
rsrcmgr = PDFResourceManager()
laparams = LAParams()  # tweak if you need finer control over layout analysis
device = PDFPageAggregator(rsrcmgr, laparams=laparams)
interpreter = PDFPageInterpreter(rsrcmgr, device)

# 6. Extract every page
output = []
for page_no, page in enumerate(PDFPage.create_pages(doc), start=1):
    interpreter.process_page(page)
    layout = device.get_result()
    page_items = []
    for elem in layout:
        if isinstance(elem, (LTTextBox, LTTextLine)):
            page_items.append({
                "type": elem.__class__.__name__,
                "bbox": elem.bbox,
                "text": elem.get_text().rstrip("\n")
            })
    output.append({
        "page_number": page_no,
        "items": page_items
    })

# 7. Write JSON to DBFS (Databricks filesystem)
dbfs_path = "/dbfs/tmp/extracted_pdf.json"
with open(dbfs_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Extraction complete. JSON file written to {dbfs_path}")
