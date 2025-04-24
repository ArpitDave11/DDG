import pdfplumber

tables_data = []  # will hold tables from all pages
with pdfplumber.open(pdf_data) as pdf:   # pdf_data is the BytesIO from earlier
    for page in pdf.pages:
        # Extract tables on this page
        page_tables = page.extract_tables()
        for table in page_tables:
            # Optionally, filter tables by structure (e.g., number of columns)
            if table and len(table[0]) > 1:  # simple check: more than one column
                tables_data.append(table)
print(f"Found {len(tables_data)} tables in PDF.")
