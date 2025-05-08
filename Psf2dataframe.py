Parsing a Data Dictionary PDF into a Pandas DataFrame

Objective: Extract entities and attributes from a 141-page text-based logical-physical data dictionary PDF, and structure them into a pandas DataFrame. Each entity entry begins with a line like "Entity Name: Customer (CUSTOMER_TABLE)" (or using a hyphen instead of parentheses), optionally followed by a description paragraph, then a table of attributes with columns: Attribute Name, Attribute Definition, Column Name, Column Data Type, and PK? (primary key indicator). The goal is to parse all entities and their attributes into rows of a DataFrame with columns: entityName, tableName, attributeName, attributeDefinition (empty string if none), columnName, columnDataType, PK (boolean).
Approach Overview

PDF Text Extraction: Use a PDF parsing library (such as PyPDF2 or pdfplumber) to read the text from all 141 pages. We iterate through pages and concatenate text. For example, with PyPDF2 you can do: reader = PdfReader("file.pdf"); page_text = page.extract_text()
pypdf2.readthedocs.io
. Similarly, with pdfplumber:
import pdfplumber
text = ""
with pdfplumber.open("data_dictionary.pdf") as pdf:
    for page in pdf.pages:
        text += page.extract_text()
This reads all page contents into the string text
thedataschool.co.uk
.
Identify Entity Sections: Use regular expressions (re) to find each Entity Name line and its associated block of text. For example, a regex pattern like:
pattern = r"Entity Name:\s*([^(-]+?)\s*(?:\(|-)\s*([^)]+)\)?"
can capture the entityName and tableName from lines of the form "Entity Name: Name (TABLE)" or "Entity Name: Name - TABLE". We find all such occurrences and split the text accordingly. Each match marks the start of a new entity’s definition.
Parse Each Entity Block: For each entity, skip any description lines until we reach the attribute table. We detect the table header (e.g., a line containing "Attribute Name", "Attribute Definition", etc.) and skip header lines (these may repeat if the table spans pages). Now, parse the attribute rows:
Multi-line handling: Attribute rows might span multiple lines if the Attribute Definition is long. We accumulate lines until an end-of-row is reached. A key indicator is the PK? field – if a line ends with "Yes" or "No" (the PK indicator), we consider that the end of that attribute entry. Lines that start with whitespace (no new attribute name) are treated as continuation of the previous attribute’s definition.
We reconstruct the full attribute line by merging continued lines. For example, if an attribute definition wraps to the next line, we join them (taking care to preserve or insert spaces as needed) so that the logical columns line up.
Split Columns for Each Attribute: Once we have a complete line for an attribute (either originally one line or merged from multiple lines), we split it into the five expected fields. We can split by columns of whitespace: in the text, columns are separated by at least two spaces (assuming consistent formatting). Using re.split(r'\s{2,}', line) will split the line on runs of 2+ spaces
stackoverflow.com
. This typically yields a list: [Attribute Name, Attribute Definition, Column Name, Column Data Type, PK]. We then clean each part:
Trim whitespace and handle edge cases. If an attribute has no definition, the split might produce an empty string for that part (we then set attributeDefinition to "" as required).
Sometimes, if the spacing was tight, the Attribute Definition and Column Name might end up in one segment. We detect this by checking if the combined segment contains what looks like a column name (e.g., an uppercase identifier with underscores) and split it out. For example, "ID of the customer (FK to Customer) CUSTOMER_ID" can be separated into definition "ID of the customer (FK to Customer)" and column name "CUSTOMER_ID" by finding the trailing uppercase token.
Convert the PK field to a boolean: e.g., "Yes" → True, "No" (or empty) → False.
Build the DataFrame: For each parsed attribute, create a row in the DataFrame with columns:
entityName – e.g. "Customer"
tableName – e.g. "CUSTOMER_TABLE"
attributeName – e.g. "Customer ID"
attributeDefinition – e.g. "Unique identifier for customer" (or "" if none provided)
columnName – e.g. "CUSTOMER_ID"
columnDataType – e.g. "INT" or "VARCHAR(100)"
PK – boolean True/False (based on PK? being Yes/No)
We collect all such rows into a list of dictionaries, then construct a pandas DataFrame from it (pandas allows creating a DataFrame directly from a list of dicts, where each dict is a row
pandas.pydata.org
).
Implementation – Python Script

Below is a Python script that implements the above steps. It uses pdfplumber for text extraction (you can swap in PyPDF2 similarly) and regular expressions for parsing. The final output is a pandas DataFrame df containing all entity-attribute mappings.
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
After running the above parsing logic, we have all attributes collected in the list records. Finally, convert this into a pandas DataFrame and ensure the data types are as desired:
# Create DataFrame from the records list
df = pd.DataFrame(records, columns=[
    "entityName", "tableName", "attributeName", 
    "attributeDefinition", "columnName", "columnDataType", "PK"
])
df["PK"] = df["PK"].astype(bool)  # ensure PK is boolean True/False

print(df.head(10))  # sample output of first 10 rows
Each row of df now represents one attribute of an entity, with the requested columns. The PK column is a Boolean (True for primary key attributes, False otherwise). This DataFrame can be further used for export (e.g., to CSV) or analysis as needed. Sources:
PyPDF2 documentation on extracting text from PDF pages
pypdf2.readthedocs.io
pdfplumber usage example for reading PDF text
thedataschool.co.uk
Regex splitting on variable whitespace (e.g., \s{2,} for 2+ spaces)
stackoverflow.com
pandas DataFrame construction from list of dictionaries
