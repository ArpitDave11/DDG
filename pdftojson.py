another part 
import fitz  # PyMuPDF
import pandas as pd
import sys

pdf_path = 'Mansfield--70-21009048 - ConvertToExcel.pdf'
try:
    doc = fitz.open(pdf_path)
except Exception as e:
    print(f"❌ Could not open PDF: {e}")
    sys.exit(1)

all_blocks = []

for page in doc:
    # 1. Grab all text blocks: each is (x0, y0, x1, y1, text, block_no, block_type)
    blocks = page.get_text("blocks")
    if not blocks:
        continue

    # 2. For each block, extract its words and reassemble in reading order
    words = page.get_text("words")  # list of (x0, y0, x1, y1, word, block_no, line_no, word_no)
    for b in blocks:
        x0, y0, x1, y1, block_text, block_no, block_type = b
        # only text blocks (block_type == 0)
        if block_type != 0:
            continue

        rect = fitz.Rect(x0, y0, x1, y1)
        # collect words whose bbox falls inside this block rect
        wlist = [w for w in words if fitz.Rect(w[:4]) in rect]
        if not wlist:
            continue

        # sort left→right
        wlist.sort(key=lambda w: w[0])
        # group by rounded y1 to form lines
        lines = {}
        for w in wlist:
            y = round(w[3], 1)
            lines.setdefault(y, []).append(w[4])
        # sort lines top→bottom, join words
        text = "\n".join(" ".join(lines[y]) for y in sorted(lines))
        all_blocks.append(text)

# now split blocks into key/value
key_vals = []
for blk in all_blocks:
    if "\n" in blk:
        key, val = blk.split("\n", 1)
    else:
        parts = blk.split(":", 1)
        key, val = parts[0], parts[1] if len(parts)>1 else ""
    key, val = key.strip(), val.strip()
    # remove unwanted symbols
    for sym in ("*", "#", ":"):
        key = key.replace(sym, "")
        val = val.replace(sym, "")
    # if purely digits, remove internal spaces
    if val and not any(c.isalpha() for c in val):
        val = val.replace(" ", "")
    key_vals.append((key, val))

# build dict
report = {k: v for k, v in key_vals}

# special-key logic
for special in ("LOCALITY", "MANNER OF CRASH COLLISION/IMPACT", "CRASH SEVERITY"):
    if special in report:
        v = report[special]
        segments = []
        last = 0
        for i in range(len(v)-1):
            if v[i+1].isdigit():
                segments.append(v[last:i+1])
                last = i+1
        segments.append(v[last:])
        # choose last segment by default
        choice = segments[-1].strip()
        report[special] = choice

# final DataFrame
df = pd.DataFrame([report])
if "VEHICLE IDENTIFICATION" in df.columns:
    df.at[0, "VEHICLE IDENTIFICATION"] = df.at[0, "VEHICLE IDENTIFICATION"].replace(" ", "")

print(df)





###}}}
Thanks for confirming! I’ll prepare a full, clean end-to-end Python script that uses PyMuPDF to extract text from all annotations in a PDF, process the extracted content, clean it, convert it to a structured dictionary, and finally output it as a pandas DataFrame.

I’ll update you once it’s ready.

# PDF Annotation Extraction and DataFrame Conversion

## Steps Overview

The following Python script performs the requested tasks step by step:

1. **Open the PDF file** using PyMuPDF (`fitz`).
2. **Iterate through all pages** and collect all annotation bounding boxes.
3. **Extract words within each annotation** by filtering words whose coordinates lie inside the annotation rectangle.
4. **Sort and assemble words** into readable text blocks (preserving the correct order of lines and words).
5. **Split text blocks into key-value pairs**, assuming a format like "Key: Value" (the split is done on newline or colon).
6. **Clean the extracted values** by removing unwanted symbols (`*`, `#`, `:`) and extra spaces (especially removing spaces within numeric values).
7. **Special handling** for keys `'LOCALITY'`, `'MANNER OF CRASH COLLISION/IMPACT'`, and `'CRASH SEVERITY'` to extract the last valid value if multiple values are present in the text.
8. **Construct a dictionary** from the cleaned key-value pairs.
9. **Convert the dictionary to a pandas DataFrame** for easy use or export.

## Python Code Implementation

Below is the complete end-to-end Python script with explanatory comments. This script assumes the PDF file **"Mansfield--70-21009048 - ConvertToExcel.pdf"** is present in the same directory. It will output the final pandas DataFrame containing the extracted data.

```python
import fitz  # PyMuPDF for PDF handling
import pandas as pd
import sys

# 1. Open the PDF file
pdf_path = 'Mansfield--70-21009048 - ConvertToExcel.pdf'
try:
    doc = fitz.open(pdf_path)
except Exception as e:
    print(f"Error: Unable to open PDF file '{pdf_path}'. {e}")
    sys.exit(1)

# 2. Prepare a list to collect text from all annotations
all_annots_text = []

# 3. Iterate through each page of the PDF
for page_num in range(len(doc)):
    page = doc[page_num]
    # Skip this page if there are no annotations
    if page.first_annot is None:
        continue
    # Extract all words on the page along with their bounding boxes
    words = page.get_text("words")
    # Iterate through each annotation on the page
    for annot in page.annots():
        if annot is None:
            continue  # safety check (should not be None here)
        rect = annot.rect  # the bounding rectangle of the annotation
        # 3. Collect all words that lie within the annotation's bounding box
        # (Convert word coordinates to a Rect and check if it's inside the annotation rect)
        mywords = [w for w in words if fitz.Rect(w[:4]) in rect]
        if not mywords:
            continue  # skip if no words found inside this annotation (e.g., annotation with no text)
        # 4. Sort words left-to-right and group them by line (using the y-coordinate)
        mywords.sort(key=lambda w: w[0])  # sort by x0 (horizontal position)
        line_dict = {}
        for w in mywords:
            # Round the y1 coordinate (bottom of word bbox) to group words that are on the same line
            y_coord = round(w[3], 1)
            text = w[4]
            line_dict.setdefault(y_coord, []).append(text)
        # Sort the lines by their vertical position (top to bottom) and join words in each line
        sorted_lines = sorted(line_dict.items(), key=lambda item: item[0])
        annot_text = "\n".join(" ".join(line_words) for _, line_words in sorted_lines)
        # Add the assembled text block from this annotation to our list
        all_annots_text.append(annot_text)

# 5. Split each text block into a key and value
key_values = []
for text in all_annots_text:
    # Split at the first newline or colon to separate the key from the value
    if "\n" in text:
        parts = text.split("\n", 1)
    else:
        parts = text.split(":", 1)
    # Ensure we have two parts (key and value); if value is missing, use an empty string
    if len(parts) < 2:
        parts = [parts[0], ""]
    key = parts[0].strip()
    value = parts[1].strip()
    # 6. Remove unwanted symbols and trim whitespace
    for sym in ["*", "#", ":"]:
        key = key.replace(sym, "")
        value = value.replace(sym, "")
    key = key.strip()
    value = value.strip()
    # Remove internal spaces from purely numeric values (e.g., "12 345" -> "12345")
    if not any(ch.isalpha() for ch in value):
        value = value.replace(" ", "")
    key_values.append((key, value))

# 8. Construct a dictionary from the key-value pairs
report_data = {k: v for k, v in key_values}

# 7. Special handling for specific keys to get the last valid value
special_keys = ["LOCALITY", "MANNER OF CRASH COLLISION/IMPACT", "CRASH SEVERITY"]
for k in special_keys:
    if k in report_data:
        val = report_data[k]
        # Split the value string by numeric characters (if present) to separate multiple options
        segments = []
        last_idx = 0
        for i in range(len(val) - 1):
            if val[i+1].isdigit():
                segments.append(val[last_idx:i+1])
                last_idx = i+1
        segments.append(val[last_idx:])  # add the final segment
        # Determine the "selected" segment (we assume it's the last one or marked by a repeating initial character)
        if len(segments) > 1:
            seen_initials = set()
            repeat_initial = None
            for seg in segments:
                if seg and seg[0] in seen_initials:
                    # Found an initial that repeats, mark it and break
                    repeat_initial = seg[0]
                    break
                if seg:
                    seen_initials.add(seg[0])
            chosen_segment = segments[-1].strip()  # default to the last segment
            if repeat_initial:
                # If a repeat initial was found, choose the last segment that starts with that initial
                for seg in segments:
                    if seg and seg[0] == repeat_initial:
                        chosen_segment = seg.strip()
            report_data[k] = chosen_segment

# 9. Convert the final dictionary to a pandas DataFrame
df = pd.DataFrame([report_data])
# Optional: If 'VEHICLE IDENTIFICATION' is a key (e.g., a VIN), remove any spaces in its value
if "VEHICLE IDENTIFICATION" in report_data:
    df.at[0, "VEHICLE IDENTIFICATION"] = report_data["VEHICLE IDENTIFICATION"].replace(" ", "")

# Print the resulting DataFrame (one row of extracted data)
print(df)
```

The script includes comments explaining each step for clarity. After running this code, the variable `df` will contain the extracted data in a pandas DataFrame (with keys as columns and a single row of values), which you can further process or export as needed.
