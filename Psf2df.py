import pdfplumber, re
import pandas as pd

# Open and extract all text from the PDF
pdf_path = "data_dictionary.pdf"  # path to the 141-page PDF
text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text += page.extract_text() + "\n"

records = []  # to collect each attribute as a record (dict)

# Regex to identify the Entity name line (handles "Entity Name: X (Y)" or "Entity Name: X - Y")
entity_pattern = re.compile(r"Entity Name:\s*([^(-]+?)\s*(?:\(|-)\s*([^)]+)\)?")

current_entity = None
current_table = None
in_table = False
current_attr_lines = []

# Iterate through each line of the text
for line in text.splitlines():
    # Check for a new entity line
    match = entity_pattern.match(line)
    if match:
        # Start of a new entity
        current_entity = match.group(1).strip()
        current_table = match.group(2).strip()
        in_table = False
        current_attr_lines = []
        continue

    if current_entity is None:
        continue  # skip any lines before the first entity (if any)

    # Detect table header line
    if not in_table:
        # Look for the header indicating start of attributes table
        if ("Attribute Name" in line and "Attribute Definition" in line and 
            "Column Name" in line):
            in_table = True
        # Skip lines until the table starts (these could be entity description)
        continue

    # If we encounter another header (page break in middle of same entity), skip it
    if "Attribute Name" in line and "Attribute Definition" in line and "Column Name" in line:
        continue
    if line.strip() == "":
        continue  # skip empty lines in table

    # Handle attribute lines (which may span multiple physical lines)
    # If this line is a continuation (starts with whitespace) or if we are already in a multi-line attribute
    if line.startswith(" ") or current_attr_lines:
        # If currently collecting a multi-line attribute, append this line
        current_attr_lines.append(line)
        # Check if this line signals end of the attribute (contains PK at end)
        if line.strip().endswith(("Yes", "No")):
            # Combine lines of this attribute into one string
            full_line = "".join([ln if i==0 else ln.lstrip() 
                                  for i, ln in enumerate(current_attr_lines)])
            # Split into parts by 2+ spaces (columns separation)
            parts = re.split(r"\s{2,}", full_line.strip())
            # Determine PK value and remove it from parts
            pk_val = False
            if parts and parts[-1].strip().lower() in ("yes", "y"):
                pk_val = True
                parts = parts[:-1]
            elif parts and parts[-1].strip().lower() in ("no", "n"):
                pk_val = False
                parts = parts[:-1]
            # Now parts should contain [attributeName, attributeDefinition, columnName, columnDataType]
            attr_name = attr_def = col_name = col_type = ""
            if len(parts) == 4:
                attr_name, attr_def, col_name, col_type = parts
            elif len(parts) == 3:
                # If definition is missing or merged with col_name
                attr_name, middle, col_type = parts
                # Try to split middle into attr_def and col_name
                m = re.search(r"([A-Z0-9_]+)$", middle)
                if m:
                    col_name = m.group(1)
                    attr_def = middle[:m.start()].strip()
                else:
                    # If no match, assume no definition provided
                    col_name = middle.strip()
                    attr_def = ""
            else:
                # If parts length is unexpected, handle generically
                attr_name = parts[0]
                col_type = parts[-1] if len(parts) > 1 else ""
                col_name = parts[-2] if len(parts) > 2 else ""
                # Join any remaining middle parts as definition
                if len(parts) > 3:
                    attr_def = " ".join(parts[1:-2])
                else:
                    attr_def = ""
            # Add the record to list
            records.append({
                "entityName": current_entity,
                "tableName": current_table,
                "attributeName": attr_name.strip(),
                "attributeDefinition": attr_def.strip(),
                "columnName": col_name.strip(),
                "columnDataType": col_type.strip(),
                "PK": pk_val
            })
            # Reset for next attribute
            current_attr_lines = []
    else:
        # New attribute line (not a continuation and not currently in a multi-line attr)
        if line.strip().endswith(("Yes", "No")):
            # Attribute fits in one line
            parts = re.split(r"\s{2,}", line.strip())
            pk_val = False
            if parts and parts[-1].strip().lower() in ("yes", "y"):
                pk_val = True
                parts = parts[:-1]
            elif parts and parts[-1].strip().lower() in ("no", "n"):
                pk_val = False
                parts = parts[:-1]
            # Parse parts similarly to above
            attr_name = attr_def = col_name = col_type = ""
            if len(parts) == 4:
                attr_name, attr_def, col_name, col_type = parts
            elif len(parts) == 3:
                attr_name, middle, col_type = parts
                m = re.search(r"([A-Z0-9_]+)$", middle)
                if m:
                    col_name = m.group(1); attr_def = middle[:m.start()].strip()
                else:
                    col_name = middle; attr_def = ""
            else:
                attr_name = parts[0]
                col_type = parts[-1] if len(parts)>1 else ""
                col_name = parts[-2] if len(parts)>2 else ""
                attr_def = " ".join(parts[1:-2]) if len(parts)>3 else ""
            records.append({
                "entityName": current_entity,
                "tableName": current_table,
                "attributeName": attr_name.strip(),
                "attributeDefinition": attr_def.strip(),
                "columnName": col_name.strip(),
                "columnDataType": col_type.strip(),
                "PK": pk_val
            })
        else:
            # Attribute spans multiple lines (does not end with PK), start collecting
            current_attr_lines = [line]


# Create DataFrame from the records list
df = pd.DataFrame(records, columns=[
    "entityName", "tableName", "attributeName", 
    "attributeDefinition", "columnName", "columnDataType", "PK"
])
df["PK"] = df["PK"].astype(bool)  # ensure PK is boolean True/False

print(df.head(10))  # sample output of first 10 rows
