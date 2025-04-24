Got it! I’ll look into the best way to read a PDF with tabular data from Azure Blob Storage in Databricks, extract just the tables (ignoring other content), and convert the data to JSON using OpenAI's LLM service. I’ll walk you through uploading the file, generating a secure access URL, reading the PDF in Databricks, and using LLMs for structured parsing.

I’ll let you know as soon as everything is ready for you to review.

# Extracting Tables from a PDF in Azure Databricks and Converting to JSON

In this guide, we will walk through a complete solution for extracting tabular data from a PDF stored in Azure Blob Storage using Azure Databricks. The PDF contains multiple tables (with identical columns) mixed with other text content. We will cover how to access the PDF via a SAS URL, use different tools to extract only the tables, and convert those tables into structured JSON. We’ll also discuss leveraging Azure OpenAI if needed for complex layouts. The steps are as follows:

## 1. Accessing the PDF from Azure Blob Storage

To work with the PDF in Databricks, you need to retrieve it from Azure Blob Storage. Since Unity Catalog is not enabled (no direct ADLS passthrough), we will use a **Shared Access Signature (SAS)** or a public URL for access:

- **Generate a SAS URL for the blob:** In Azure Portal, you can generate a SAS token for the blob (or container). Alternatively, use Python with the Azure Storage SDK to programmatically generate a SAS token. For example: 

  ```python
  from azure.storage.blob import generate_blob_sas, BlobSasPermissions
  from datetime import datetime, timedelta

  account_name = "<STORAGE_ACCOUNT_NAME>"
  container_name = "<CONTAINER_NAME>"
  blob_name = "<PDF_FILE_NAME>.pdf"
  account_key = "<STORAGE_ACCOUNT_KEY>"  # Use your storage account key

  sas_token = generate_blob_sas(
      account_name=account_name,
      container_name=container_name,
      blob_name=blob_name,
      account_key=account_key,
      permission=BlobSasPermissions(read=True),
      expiry=datetime.utcnow() + timedelta(hours=1)   # SAS valid for 1 hour
  )
  sas_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
  print("SAS URL:", sas_url)
  ``` 

  The above uses `generate_blob_sas` to create a read-only SAS valid for 1 hour ([copying azure blob to azure fileshare, using Python SDK - Stack Overflow](https://stackoverflow.com/questions/62879216/copying-azure-blob-to-azure-fileshare-using-python-sdk#:~:text=,utcnow%28%29%20%2B%20timedelta%28hours%3D1%29)). The `sas_url` can be used to access the PDF.

- **Public access option:** If the blob’s container access level is set to allow anonymous read (e.g. **Container** or **Blob** access level), you could directly use the blob URL without a SAS token ([How to read a blob in Azure databricks with SAS - Stack Overflow](https://stackoverflow.com/questions/60879481/how-to-read-a-blob-in-azure-databricks-with-sas#:~:text=WITH%20CONTAINER%20ACCESS%3A)). For example: `https://<account>.blob.core.windows.net/<container>/<file>.pdf`. This is only advisable for non-sensitive, publicly shareable files.

**In Databricks:** Once you have the URL (SAS or public), you can retrieve the PDF content. One easy method is to use Python’s `requests` library to download the PDF into memory:

```python
import requests
from io import BytesIO

url = sas_url  # or public URL if available
response = requests.get(url)
response.raise_for_status()  # ensure download succeeded

pdf_data = BytesIO(response.content)  # in-memory bytes buffer
```

This will give you a `BytesIO` stream of the PDF, which can be read by PDF processing libraries. (If the PDF is large, you could also save it to DBFS or local disk in Databricks, but in-memory is fine for most cases.)

**Note:** If using `requests` or other libraries not pre-installed, you can install them in a Databricks notebook via `%pip install requests azure-storage-blob pdfplumber tabula-py fitz openai` as needed. Ensure your cluster has internet access to reach Blob storage (or that your storage account is accessible to the Databricks network).

## 2. Extracting Tables from the PDF

With the PDF accessible, the next step is to parse it and **extract only the table data**, ignoring paragraphs and other non-table text. There are multiple approaches and tools for this. We will discuss a few reliable methods that work in Azure Databricks:

### 2.1 Using PDFPlumber (Pure Python library)

**PDFPlumber** is a Python library built on PDFMiner that can extract text and tables from PDFs. It has built-in table extraction capabilities that detect table structures (lines or text alignment) and return table data as lists. 

**How it works:** PDFPlumber’s `Page.extract_tables()` method finds tables by analyzing ruling lines and word alignment ([GitHub - jsvine/pdfplumber: Plumb a PDF for detailed information about each char, rectangle, line, et cetera — and easily extract text and tables.](https://github.com/jsvine/pdfplumber#extracting-tables#:~:text=%60.extract_tables%28table_settings%3D,properties)). It returns a list of tables, each table being a list of rows, and each row a list of cell values (text) ([GitHub - jsvine/pdfplumber: Plumb a PDF for detailed information about each char, rectangle, line, et cetera — and easily extract text and tables.](https://github.com/jsvine/pdfplumber#extracting-tables#:~:text=%60.extract_tables%28table_settings%3D,properties)). This makes it easy to isolate tables from other content.

**Usage in Databricks:**

1. **Install**: Ensure pdfplumber is installed (`%pip install pdfplumber`).  
2. **Open PDF and extract**: 

   ```python
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
   ```
   In this snippet, `extract_tables()` returns all tables on a page. We append each table (as a list of rows) to `tables_data`. If the PDF always has the same columns, you might expect one table per page, or multiple tables – this code collects them all.

   **Ignoring non-table text:** `extract_tables` inherently skips any text that isn’t organized as a table. So paragraphs or stray text will not appear in `page.extract_tables()` output (they would appear in `page.extract_text()`, but we are not using that here).

   **Table structure:** Each `table` here is a list of rows, and each row is a list of cell texts. For example, `tables_data[0][0]` would be the first row (possibly the header row). If the PDF’s tables have clearly defined columns (e.g., drawn lines or consistent alignment), PDFPlumber usually detects them correctly. If needed, you can adjust `table_settings` (such as switching to detection by text alignment instead of lines) to improve accuracy ([python - How to extract table details into rows and columns using pdfplumber - Stack Overflow](https://stackoverflow.com/questions/68872782/how-to-extract-table-details-into-rows-and-columns-using-pdfplumber#:~:text=table%20%3D%20page.extract_table%28table_settings%3D%7B)). See pdfplumber’s documentation for table extraction settings ([python - How to extract table details into rows and columns using pdfplumber - Stack Overflow](https://stackoverflow.com/questions/68872782/how-to-extract-table-details-into-rows-and-columns-using-pdfplumber#:~:text=Morover%2C%20Please%20have%20a%20read,in%20your%20input%20file)) for complex cases (e.g., borderless tables).

3. **Verify tables**: You might want to inspect the first table to ensure it captured correctly:
   ```python
   first_table = tables_data[0]
   for row in first_table[:2]:  # print first 2 rows
       print(row)
   ```
   This helps confirm that the header and data cells are properly separated.

**Pros:** PDFPlumber is pure Python (no external dependencies beyond PDFMiner), and works well for many PDFs. It returns table data in a structured way that’s easy to manipulate in Python.

**Cons:** For extremely complex layouts or if the PDF content is an image (scanned), pdfplumber alone won’t work (OCR would be needed in that case). Also, for tables without any drawn lines or consistent column spacing, you may need to tweak `table_settings` to split columns properly (e.g., use `"vertical_strategy": "text"` to separate columns by spacing instead of lines ([python - How to extract table details into rows and columns using pdfplumber - Stack Overflow](https://stackoverflow.com/questions/68872782/how-to-extract-table-details-into-rows-and-columns-using-pdfplumber#:~:text=table%20%3D%20page.extract_table%28table_settings%3D%7B))).

### 2.2 Using Tabula-py (Java-based library)

**Tabula** is another popular tool for PDF table extraction. Tabula is originally a Java library; **tabula-py** is a Python wrapper that invokes Tabula under the hood. It can also directly output tables as DataFrames or JSON.

**Usage in Databricks:**

1. **Install**: Ensure Java is available on the cluster (Databricks clusters come with JVM, so that’s fine). Install tabula-py via `%pip install tabula-py`. This will also bring in dependencies like `pandas` if not present.
2. **Read tables**: Use `tabula.read_pdf()` to extract tables.
   ```python
   import tabula
   # Read all pages, expecting tables of the same structure on each page
   dfs = tabula.read_pdf(pdf_data, pages="all", multiple_tables=True)
   ```
   Here, `pdf_data` can be a file path or file-like object (for a BytesIO, you might need to use `pdf_data.getvalue()` or save the BytesIO to a temporary file, since tabula might expect a path or bytes). Alternatively, provide the SAS URL directly to `read_pdf` if it can handle URLs (tabula may support file-like objects or URLs as input).

   We used `pages="all"` to process every page and `multiple_tables=True` to allow detection of more than one table per page ([dataframe - How to extract more than one table present in a PDF file with tabula in Python? - Stack Overflow](https://stackoverflow.com/questions/49733576/how-to-extract-more-than-one-table-present-in-a-pdf-file-with-tabula-in-python#:~:text=If%20your%20PDF%20has%20multiple,option)) ([dataframe - How to extract more than one table present in a PDF file with tabula in Python? - Stack Overflow](https://stackoverflow.com/questions/49733576/how-to-extract-more-than-one-table-present-in-a-pdf-file-with-tabula-in-python#:~:text=using%20,will%20solve%20the%20issue)). If each page has one table, Tabula will still return a list of DataFrames (one per page). If tables span multiple pages or if there are multiple tables in one page, the behavior can vary:
   - If the tables have identical structure and appear on every page, Tabula can sometimes concatenate them into one DataFrame when using `pages="all"` ([dataframe - How to extract more than one table present in a PDF file with tabula in Python? - Stack Overflow](https://stackoverflow.com/questions/49733576/how-to-extract-more-than-one-table-present-in-a-pdf-file-with-tabula-in-python#:~:text=0)). If not, you will get a list of DataFrames.
   - If a page has multiple smaller tables, setting `multiple_tables=True` ensures you get each one.

3. **Inspect data**: 
   ```python
   print(len(dfs), "DataFrames extracted")
   print(dfs[0].head())
   ```
   This will show the first few rows of the first table.

   If you expect a single combined table across pages (and got multiple DataFrames), you can concatenate them later (provided they have the same columns).

**Tabula output:** Tabula gives you Pandas DataFrames directly, which is convenient for conversion to JSON or CSV. Tabula can also convert PDF to JSON/CSV without coding (it’s known for that), but in code we can use DataFrame methods ([tabula-py: Read tables in a PDF into DataFrame — tabula-py  documentation](https://tabula-py.readthedocs.io/#:~:text=%60tabula,PDF%20file%20into%20CSV%2FTSV%2FJSON%20file)).

**Pros:** Often very effective for tables with clear column boundaries. It also handles spanning of tables across pages if structured consistently.  
**Cons:** Requires a Java dependency (which is fine on Databricks, but adds a bit of overhead). Also, you might need to adjust parameters (guess vs stream mode) for borderless tables.

### 2.3 Using PyMuPDF (fitz)

**PyMuPDF** (imported as `fitz`) is a fast PDF parsing library. It can extract text and details like text coordinates, which theoretically allows custom table parsing. However, PyMuPDF does **not directly extract tables** as structured data – it gives you text boxes or lines, and you would have to infer table structure from coordinates.

**Possible approach with PyMuPDF:** 
- Extract all text boxes or lines on the page using `page.get_text("blocks")` or `page.get_text("words")`. These methods give positions of text on the page.
- Using the positions, cluster text into rows and columns by their y and x coordinates (e.g., group text with similar y coordinate as one row, and within that group by x ranges as columns).

While this is doable, it is quite complex to implement from scratch. PDFPlumber actually uses a similar approach internally (finding aligned text to define cells) ([GitHub - jsvine/pdfplumber: Plumb a PDF for detailed information about each char, rectangle, line, et cetera — and easily extract text and tables.](https://github.com/jsvine/pdfplumber#extracting-tables#:~:text=1,Group%20contiguous%20cells%20into%20tables)). Therefore, for simplicity, using PDFPlumber or Tabula is recommended over manual PyMuPDF handling.

**When to use PyMuPDF:** If you need to do additional processing (like extract text formatting, or handle PDFs page by page for other content as well), you might use PyMuPDF to get raw text and then use an LLM to parse it (see Section 4). Otherwise, you can generally skip PyMuPDF for direct table extraction, since specialized libraries or services do a better job out-of-the-box.

### 2.4 Using Azure Form Recognizer (Azure AI Document Intelligence)

For a robust and supported solution in Azure, **Azure Form Recognizer** (recently renamed Azure AI Document Intelligence) is a great choice. Form Recognizer’s prebuilt **Layout** or **General Document** models can detect tables in documents and return their structure as part of a JSON result. This approach offloads the heavy lifting to an AI service and can handle messy layouts, spanning tables, or scanned documents (with OCR), etc.

**How to use in Databricks:**

1. **Set up credentials:** You need an Azure Form Recognizer resource (endpoint URL and API key). In Databricks, you might store these in a secret scope or provide directly.
2. **Install SDK:** `%pip install azure-ai-formrecognizer` (and azure-core if not included).
3. **Call the Layout model:** Use the `DocumentAnalysisClient` to analyze the document:
   ```python
   from azure.ai.formrecognizer import DocumentAnalysisClient
   from azure.core.credentials import AzureKeyCredential

   endpoint = "https://<your-form-recognizer-resource>.cognitiveservices.azure.com/"
   key = "<your-form-recognizer-key>"

   client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))

   # We can pass the SAS URL directly to the service to have it fetch the PDF
   poller = client.begin_analyze_document_from_url(model_id="prebuilt-layout", document_url=sas_url)
   result = poller.result()
   for table in result.tables:
       print(f"Table with {table.row_count} rows and {table.column_count} columns")
       for cell in table.cells:
           if cell.row_index == 0:
               # This cell is in the header row (row_index 0). You can identify headers like this because Layout can detect column headers.
               pass
           print(f" Cell[{cell.row_index}, {cell.column_index}]: {cell.content}")
   ```
   In this code, we used `begin_analyze_document_from_url` so the service directly downloads the PDF via our SAS URL (this saves having to send the file over the network from Databricks). We specified the `prebuilt-layout` model, which extracts text lines, tables, and selection marks from documents. The result includes a list of `tables`. Each table has `row_count`, `column_count`, and a list of `cells`. Each `cell` has properties like `row_index`, `column_index`, `content`, and flags for whether it's a header.

   Form Recognizer’s layout model **identifies table regions and cells for us**. It even recognizes header rows: each cell has an attribute `is_header` or similar (depending on API version) indicating if it’s part of the table’s header ([Extract table data and put them into dictionary with azure form recognizer - Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/850819/extract-table-data-and-put-them-into-dictionary-wi#:~:text=Extract%20Column%20Header%20Information%3A%20Layout,make%20up%20the%20table%20header)). This makes it straightforward to separate header vs data.

4. **Parse result:** The `result` from Form Recognizer is already structured (essentially a JSON under the hood). You can construct your own JSON from it by mapping headers to values for each row (we'll cover JSON construction in the next section). 

   Form Recognizer outputs are in JSON form if you use the REST API directly, or Python objects via the SDK. For example, `result.tables` is a list of Table objects, but you could also get the raw JSON via `result.to_dict()` if needed.

**Pros:** Very high accuracy for structured data extraction. Can handle scans (with OCR) and complex layouts better than simple libraries. No need to maintain parsing code; it’s an Azure service.  
**Cons:** It’s a paid service (charges per page analyzed), and using it adds a network call which has some latency. Also, you need the Azure resource set up. But for enterprise scenarios, this is a supported solution and integrates well with Azure pipelines. 

## 3. Converting the Extracted Tables to JSON

After extracting table data using one of the methods above, we need to convert it into a structured JSON format. The JSON should ideally have meaningful keys (e.g., column names) and values (cell entries), rather than just an array of arrays.

The approach will differ slightly depending on which tool you used:

### 3.1 If using PDFPlumber or Tabula (Python Data Structures)

**From PDFPlumber:** We obtained `tables_data` as a list of tables, where each table is a list of rows (which are lists of strings). Commonly, the first row of each table will be the header (column names). Since the problem states all tables have identical column structure, we can assume the header row is the same for each table. We can do one of two things:
- Use the first table’s first row as the master header, and append all subsequent tables’ rows (excluding their header) under it.
- Or, if we want to keep tables separate, produce a JSON array for each table individually (with repeated headers).

**Example – combine all tables into one JSON array:**

```python
import json

all_rows = []
master_headers = None

for table in tables_data:
    if not table:
        continue
    headers = table[0]
    if master_headers is None:
        master_headers = headers  # set master headers from first table
    # Ensure current table has same headers if expected
    # Append each data row as dict
    for row in table[1:]:  # skip header row
        item = {header: value for header, value in zip(master_headers, row)}
        all_rows.append(item)

# Now all_rows is a list of dictionaries (each dict is one row of the tables)
json_output = json.dumps(all_rows, indent=2)
print(json_output[:200], "...")
```

This will produce a JSON string like:
```json
[
  {
    "Column1": "value11",
    "Column2": "value12",
    "Column3": "value13"
  },
  {
    "Column1": "value21",
    "Column2": "value22",
    "Column3": "value23"
  },
  ...
]
```
Each object in the array represents a row. The keys are the column names taken from the header. We combined all tables since they share columns. If instead you want a JSON structure separating tables, you could create a dictionary with each table as a list (or use a list of tables).

**From Tabula (DataFrames):** If you used `tabula.read_pdf`, you likely have a list of Pandas DataFrames (`dfs`). You can handle these similarly:
- If combining: use `pd.concat` to concatenate all DataFrames (assuming same columns), then use `df.to_json(orient='records')`.
- If separate: call `df.to_dict(orient='records')` on each DataFrame.

**Example – using Pandas with tabula output:**

```python
import pandas as pd

# Combine all DataFrames (this will stack them vertically if they have identical columns)
combined_df = pd.concat(dfs, ignore_index=True)

# Convert to JSON (list of records)
json_records = combined_df.to_dict(orient='records')
print(json_records[:2])
# Save to a JSON string if needed
json_text = json.dumps(json_records, indent=2)
```

This uses pandas to do the heavy lifting. Each row of the DataFrame becomes a dict in the list. (Make sure the DataFrames had the correct header row; Tabula sometimes includes the header as a row if it didn’t auto-detect. You might need to set `columns=` manually if the header row wasn’t recognized.)

**From Form Recognizer:** The result is already a structured object. For each table in `result.tables`, you can do something like:
```python
tables_json = []
for table in result.tables:
    # Get column headers (Form Recognizer may give multiple header rows; combine them if so)
    headers = []
    for cell in table.cells:
        if cell.row_index == 0:
            headers.append(cell.content)
    # Now collect each data row under these headers
    table_data = []
    for row_idx in range(1, table.row_count):  # assuming row 0 is header
        row_dict = {}
        for cell in table.cells:
            if cell.row_index == row_idx:
                header_text = headers[cell.column_index]
                row_dict[header_text] = cell.content
        table_data.append(row_dict)
    tables_json.append(table_data)
```
This will produce a list of tables, where each table is a list of row dicts. You can flatten if needed. Note: Form Recognizer cells are not necessarily in reading order, so it’s good to iterate by row_index and column_index as shown. The `headers` derivation assumes the first row is all headers; in practice, FR might mark `cell.column_header` boolean. You could use `if cell.kind == "columnHeader"` in newer SDK versions to identify header cells.

### 3.2 Example JSON Output

Regardless of method, the final **structured JSON** can be an array of row objects. If multiple tables logically form one dataset, combining them is convenient. If they need separation, you might output a JSON with a top-level key per table or an array of tables.

For example, combined output:
```json
[
  { "Name": "Alice", "Age": "30", "City": "New York" },
  { "Name": "Bob", "Age": "25", "City": "Los Angeles" },
  { "Name": "Charlie", "Age": "35", "City": "Chicago" }
]
```
If separate tables:
```json
{
  "Table1": [
    {...row1...},
    {...row2...}
  ],
  "Table2": [
    {...row1...},
    ...
  ]
}
```
Choose the format that suits your downstream use case. Storing as a single JSON array of records is often easiest for further processing (e.g., loading into a DataFrame or database).

## 4. Leveraging Azure OpenAI for Complex Table Extraction (Optional)

In most cases, the above methods will suffice. However, if the PDF’s layout is **very complex or messy** (e.g., irregular column spacing, multi-line cells that confuse the parser, or if the PDF is essentially unstructured text), you might consider using an **Azure OpenAI** large language model to assist in parsing. Azure OpenAI can be used at two possible points:
- **Instead of a traditional parser (for step 2)**: Feed the raw text of the PDF or page to the model and prompt it to extract the table.
- **After using a parser (for step 3)**: If the extracted data is messy (e.g., cells merged or misaligned), have the model clean up or restructure the data into JSON.

Azure OpenAI (with models like GPT-3.5 or GPT-4) can understand text and format output as instructed. For example, you can prompt: *“Extract the following table into JSON with these columns: ...”* and provide the table content.

**Setting up Azure OpenAI in Databricks:**

1. **Credentials**: You need an Azure OpenAI endpoint and an API key. Set environment variables or use a secret scope for these (e.g., `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`).
2. **Install OpenAI client**: `%pip install openai` (the OpenAI Python library supports Azure OpenAI with a few config tweaks).
3. **Configure the client for Azure**:
   ```python
   import openai
   openai.api_type = "azure"
   openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://<resource>.openai.azure.com/
   openai.api_version = "2024-02-01"  # or the API version your endpoint requires
   openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
   ```

4. **Prompt the model**: Suppose we have a string `table_text` that contains a table’s text exactly as it appeared in the PDF (perhaps extracted via `pdfplumber` or even by copying and pasting if simple). We can ask the model to parse it:
   ```python
   deployment_name = "<your-chatgpt-deployment>"  # the model deployment name in Azure OpenAI

   table_text = """\
   Name    | Age | City
   Alice   | 30  | New York
   Bob     | 25  | Los Angeles
   Charlie | 35  | Chicago
   """  # this is an example; in practice you'd build this from PDF content.

   system_prompt = "You are an assistant that extracts tables from text."
   user_prompt = f"""Given the following table data, output it as a JSON array of objects with proper keys:
   ``` 
   {table_text}
   ```
   """

   response = openai.ChatCompletion.create(
       engine=deployment_name,
       messages=[
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}
       ],
       temperature=0,
       max_tokens=500
   )
   output_text = response['choices'][0]['message']['content']
   print(output_text)
   ```
   In the prompt, we wrapped the table in triple backticks for clarity (and described the desired format). Setting `temperature=0` encourages a deterministic, focused response. The model should return something like the JSON snippet shown earlier. We can then parse `output_text` with `json.loads(output_text)` to get a Python object.

**Why/When to use OpenAI:** If traditional parsing struggles (for example, if columns aren’t separated by clear delimiters or if the PDF content is semi-structured), an LLM can use its understanding of language to infer the structure. It’s also useful if the table data needs additional interpretation or cleaning that a raw parser can’t do (like splitting combined fields, fixing OCR errors, etc.). Keep in mind:
- There are token limits – don’t send extremely large text in one go. If the PDF is long, you may need to process page by page or table by table.
- Azure OpenAI has a cost per token. Use it when automation of messy data extraction is worth the cost.
- Always validate the output. LLMs might occasionally hallucinate or format incorrectly. You can mitigate this by giving a very clear instruction and even a schema example in the prompt (few-shot learning).

**Advanced:** Azure OpenAI also offers an image understanding model (e.g., GPT-4 with vision, referred to as GPT-4V or “GPT-4o” in some docs) which could directly interpret a PDF page image ([Using Azure OpenAI GPT-4o to extract structured JSON data from PDF documents - Code Samples | Microsoft Learn](https://learn.microsoft.com/en-us/samples/azure-samples/azure-openai-gpt-4-vision-pdf-extraction-sample/using-azure-openai-gpt-4o-to-extract-structured-json-data-from-pdf-documents/#:~:text=This%20sample%20demonstrates%20how%20to,invoices%2C%20using%20the%20%206)). However, that’s a specialized scenario and requires the model to be available in your region. For our case, using text input to ChatGPT is usually sufficient since we can extract text from PDF with the earlier tools.

## 5. Putting It All Together in Databricks

To summarize and integrate the steps, here’s a high-level flow you can implement in an Azure Databricks notebook:

1. **Configuration**: Set your Azure Blob storage details (and Azure OpenAI/Form Recognizer credentials if using those).  
2. **Download PDF**: Generate SAS (if needed) and fetch the PDF via `requests` (or use Spark APIs to read binary – not covered above, but e.g. `dbutils.fs.cp` with wasbs:// URL and pre-set SAS config could work too).  
3. **Extract Tables**: Use PDFPlumber or Tabula in Python to get table data. Print sample output to verify correctness.  
4. **Convert to JSON**: Transform the extracted table structures into JSON. If using pandas, this is straightforward; otherwise, use Python’s json library as shown.  
5. **(Optional) Post-process with LLM**: If the direct extraction is not clean, feed the result or original text to Azure OpenAI to get a clean JSON. Parse the LLM output.  
6. **Use the JSON**: You can save the JSON string to a file, write to blob storage, or directly load it into a Spark DataFrame for further processing in Databricks (Spark can read JSON into DataFrame using `spark.read.json` if you write the JSON to DBFS, for example).

Throughout this process, prioritize the simpler, tried-and-true solutions (like PDFPlumber/Tabula or Form Recognizer for robust needs). Use Azure OpenAI as a fallback or enhancement for edge cases. By following these steps, you can reliably extract only the tabular data from a PDF and have it in a machine-readable JSON format for downstream analysis or storage.

**Sources:**

- Azure Databricks access to Blob storage via SAS ([copying azure blob to azure fileshare, using Python SDK - Stack Overflow](https://stackoverflow.com/questions/62879216/copying-azure-blob-to-azure-fileshare-using-python-sdk#:~:text=,utcnow%28%29%20%2B%20timedelta%28hours%3D1%29)) ([How to read a blob in Azure databricks with SAS - Stack Overflow](https://stackoverflow.com/questions/60879481/how-to-read-a-blob-in-azure-databricks-with-sas#:~:text=WITH%20CONTAINER%20ACCESS%3A))  
- PDFPlumber documentation on table extraction ([GitHub - jsvine/pdfplumber: Plumb a PDF for detailed information about each char, rectangle, line, et cetera — and easily extract text and tables.](https://github.com/jsvine/pdfplumber#extracting-tables#:~:text=%60.extract_tables%28table_settings%3D,properties))  
- Tabula-py usage and capabilities ([tabula-py: Read tables in a PDF into DataFrame — tabula-py  documentation](https://tabula-py.readthedocs.io/#:~:text=%60tabula,PDF%20file%20into%20CSV%2FTSV%2FJSON%20file)) ([dataframe - How to extract more than one table present in a PDF file with tabula in Python? - Stack Overflow](https://stackoverflow.com/questions/49733576/how-to-extract-more-than-one-table-present-in-a-pdf-file-with-tabula-in-python#:~:text=If%20your%20PDF%20has%20multiple,option))  
- Azure Form Recognizer (Document Intelligence) table handling ([Extract table data and put them into dictionary with azure form recognizer - Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/850819/extract-table-data-and-put-them-into-dictionary-wi#:~:text=Extract%20Column%20Header%20Information%3A%20Layout,make%20up%20the%20table%20header)) (layout model recognizes table headers, returning JSON output to parse)  
- Azure OpenAI Python integration (setting `openai.api_type = "azure"` and using chat completions) ([Quickstart - Get started using chat completions with Azure OpenAI Service - Azure OpenAI Service | Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart#:~:text=openai.api_type%20%3D%20,01))

