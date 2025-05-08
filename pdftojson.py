import re
import json
from PyPDF2 import PdfReader  # Alternatively, use pdfplumber.pdfplumber as needed

# Path to the PDF file
pdf_path = "data_dictionary.pdf"

# Read and extract text from the PDF
reader = PdfReader(pdf_path)
full_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# Split the text into blocks, each corresponding to one entity
entity_blocks = re.findall(r"Entity Name:.*?(?=(?:\nEntity Name:)|$)", full_text, flags=re.S)

entities = []  # list to hold each entity's data

for block in entity_blocks:
    # Split block into lines and remove any completely empty lines
    lines = [ln for ln in block.splitlines() if ln.strip() != ""]
    if not lines:
        continue

    # Parse the Entity Name line (first line in the block)
    first_line = lines[0].strip()
    entity_name = ""
    table_name = ""
    # Try format: Entity Name: X (Y)
    m = re.match(r"Entity Name:\s*([^(\r\n]+)\s*\(([^)]+)\)", first_line)
    if m:
        entity_name = m.group(1).strip()
        table_name = m.group(2).strip()
    else:
        # Try format: Entity Name: X - Y
        m2 = re.match(r"Entity Name:\s*([^-]+)-\s*(.+)", first_line)
        if m2:
            entity_name = m2.group(1).strip()
            table_name = m2.group(2).strip()
        else:
            # If no table name format recognized, take the remainder after "Entity Name:"
            if ":" in first_line:
                entity_name = first_line.split(":", 1)[1].strip()
            else:
                entity_name = first_line
            table_name = ""  # Unknown format for table name

    # Determine if there's an Entity Definition section between the name and table header
    entity_def_text = None
    header_index = None
    # Find the index of the header line (which contains "Attribute Name")
    for idx, ln in enumerate(lines):
        if "Attribute Name" in ln and "Column Data Type" in ln:
            header_index = idx
            break
    if header_index is None:
        # No header found (unexpected format), skip this block
        continue

    if header_index > 1:
        # Lines from 1 up to header_index-1 are the entity definition (could be multiple lines)
        def_lines = lines[1:header_index]
        # If the first definition line has the label, remove "Entity Definition:"
        if def_lines:
            first_def = def_lines[0]
            if first_def.strip().lower().startswith("entity definition:"):
                # Remove the label part
                def_lines[0] = first_def.split(":", 1)[1].strip()
        # Join all definition lines into one paragraph
        entity_def_text = " ".join(line.strip() for line in def_lines).strip()
        if entity_def_text == "":
            entity_def_text = None  # treat empty definition as not present

    # Prepare to parse the attribute table rows
    # Skip the header line and any dashed separator line immediately after it
    data_start_idx = header_index + 1
    if data_start_idx < len(lines) and re.match(r"^[-\s]+$", lines[data_start_idx]):
        data_start_idx += 1  # skip separator (e.g., a line of dashes)

    columns = []
    for row in lines[data_start_idx:]:
        if row.strip() == "":
            continue  # skip any remaining empty line (if any)
        # Use regex to split by 2 or more spaces to separate columns
        parts = re.split(r"\s{2,}", row.rstrip())
        # Remove any empty strings at the end (could happen if row has trailing spaces)
        while parts and parts[-1] == "":
            parts.pop()
        # Initialize variables for each field
        attr_name = attr_def = col_name = col_type = ""
        pk_str = ""
        if len(parts) == 5:
            # All five expected parts present
            attr_name, attr_def, col_name, col_type, pk_str = parts
        elif len(parts) == 4:
            # Four parts could mean missing attribute definition or missing PK
            # If the last part is clearly a PK indicator (e.g., 'Y' or 'N'), then attribute definition is missing
            if parts[-1].strip().lower() in ("y", "n", "yes", "no"):
                attr_name = parts[0]
                attr_def = ""  # missing attribute definition
                col_name = parts[1]
                col_type = parts[2]
                pk_str = parts[3]
            else:
                # Otherwise, the PK is missing (last part is data type, no PK given)
                attr_name = parts[0]
                attr_def = parts[1]
                col_name = parts[2]
                col_type = parts[3]
                pk_str = ""  # no PK flag
        elif len(parts) == 3:
            # Only three parts means both attribute definition and PK are missing
            attr_name = parts[0]
            attr_def = ""  # missing attribute definition
            col_name = parts[1]
            col_type = parts[2]
            pk_str = ""  # no PK
        else:
            # If an unexpected number of parts (e.g., more than 5 due to extra spaces in definition),
            # try to reconstruct: assume last 3 are col_name, col_type, pk and the rest (after first) form attr_def.
            if len(parts) > 5:
                attr_name = parts[0]
                # Last part might be PK or not; check if it's a known PK flag
                if parts[-1].strip().lower() in ("y", "n", "yes", "no"):
                    pk_str = parts[-1]
                    col_type = parts[-2]
                    col_name = parts[-3]
                    attr_def = " ".join(parts[1:-3])
                else:
                    # No PK flag at end
                    pk_str = ""
                    col_type = parts[-1]
                    col_name = parts[-2]
                    attr_def = " ".join(parts[1:-2])
            else:
                # If len(parts) is 0 or some other case, skip this row
                continue

        # Clean up whitespace and determine PK boolean
        attr_name = attr_name.strip()
        attr_def = attr_def.strip()
        col_name = col_name.strip()
        col_type = col_type.strip()
        pk_value = True if pk_str and pk_str.strip().lower().startswith("y") else False

        # Ensure attributeDefinition is an empty string if missing
        if attr_def == "":
            attr_def = ""

        columns.append({
            "attributeName": attr_name,
            "attributeDefinition": attr_def,
            "columnName": col_name,
            "columnDataType": col_type,
            "PK": pk_value
        })
    # Build the entity dictionary
    entity_dict = {
        "entityName": entity_name,
        "tableName": table_name,
        "columns": columns
    }
    if entity_def_text is not None:
        entity_dict["entityDefinition"] = entity_def_text

    entities.append(entity_dict)

# Combine all entities into the final JSON structure
output_data = {"entities": entities}

# Save the JSON to a file (indented for readability)
with open("data_dictionary.json", "w") as f:
    json.dump(output_data, f, indent=2)

# (Optional) Print the JSON string to console
print(json.dumps(output_data, indent=2))
