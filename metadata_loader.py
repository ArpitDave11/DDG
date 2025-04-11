from azure.storage.blob import BlobServiceClient
import csv, io, logging
from config import config

def load_metadata_from_blob() -> list[dict]:
    """Fetch all metadata files from Azure Blob Storage and return list of attributes."""
    conn_str = config['azure']['blob_connection_string']
    container = config['azure']['blob_container']
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service.get_container_client(container)
    logging.info(f"Connecting to Azure Blob container: {container}")
    attributes = []
    try:
        blob_list = container_client.list_blobs()
    except Exception as e:
        logging.error(f"Failed to list blobs: {e}")
        raise

    for blob in blob_list:
        try:
            # Download blob content
            blob_client = container_client.get_blob_client(blob.name)
            data = blob_client.download_blob().readall()  # get content&#8203;:contentReference[oaicite:3]{index=3}
            text = data.decode('utf-8')  # assuming text data
        except Exception as e:
            logging.error(f"Error reading blob {blob.name}: {e}")
            continue  # skip this blob and continue with others

        # Determine format by extension (assuming .csv or .json for simplicity)
        if blob.name.lower().endswith('.csv'):
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                # Normalize keys
                table = row.get('table_name') or row.get('table') or row.get('Table') or ''
                column = row.get('column_name') or row.get('column') or row.get('Column') or ''
                datatype = row.get('data_type') or row.get('datatype') or row.get('DataType') or ''
                description = row.get('description') or row.get('Description') or ''
                if table and column and datatype:
                    attributes.append({
                        "table": table.strip(),
                        "column": column.strip(),
                        "type": datatype.strip(),
                        "existing_desc": description.strip() if description else ""
                    })
        elif blob.name.lower().endswith('.json'):
            import json
            try:
                meta_json = json.loads(text)
            except Exception as e:
                logging.error(f"Failed to parse JSON in {blob.name}: {e}")
                continue
            # Expecting either a list of entries or a dict of table->columns.
            if isinstance(meta_json, list):
                for entry in meta_json:
                    table = entry.get('table') or entry.get('table_name')
                    column = entry.get('column') or entry.get('column_name')
                    datatype = entry.get('type') or entry.get('data_type')
                    description = entry.get('description') or ""
                    if table and column and datatype:
                        attributes.append({
                            "table": table.strip(),
                            "column": column.strip(),
                            "type": datatype.strip(),
                            "existing_desc": description.strip()
                        })
            elif isinstance(meta_json, dict):
                # If top-level keys are tables
                for table, cols in meta_json.items():
                    for col, meta in cols.items():
                        datatype = meta.get('type') or meta.get('data_type')
                        description = meta.get('description') or ""
                        attributes.append({
                            "table": table.strip(),
                            "column": col.strip(),
                            "type": datatype.strip(),
                            "existing_desc": description.strip()
                        })
    logging.info(f"Loaded {len(attributes)} attributes from metadata.")
    return attributes
