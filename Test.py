#!/usr/bin/env python3
import psycopg2
import json
import logging

def generate_database_metadata(conn):
    """
    Queries the PostgreSQL information_schema to generate nested metadata.
    
    Output format is a dictionary where:
      - keys are schema names;
      - values are dictionaries whose keys are table names;
      - each table entry has a key "columns" which is a list of dictionaries
        containing details about each column.
    """
    metadata = {}
    cur = conn.cursor()

    # Retrieve non-system schemas (filtering out pg_catalog and information_schema)
    cur.execute("""
        SELECT schema_name 
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema');
    """)
    schemas = cur.fetchall()
    logging.info("Found %d user schemas.", len(schemas))
    
    for (schema,) in schemas:
        metadata[schema] = {}

        # Get all base tables for the current schema
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables
            WHERE table_schema = %s AND table_type = 'BASE TABLE';
        """, (schema,))
        tables = cur.fetchall()
        logging.info("Schema '%s' contains %d tables.", schema, len(tables))
        
        for (table,) in tables:
            metadata[schema][table] = {"columns": []}

            # Query column details for each table
            cur.execute("""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable, 
                    column_default 
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s;
            """, (schema, table))
            columns = cur.fetchall()
            
            for col in columns:
                col_name, data_type, is_nullable, default = col
                col_dict = {
                    "column": col_name,
                    "data_type": data_type,
                    "is_nullable": is_nullable,
                    "default": default if default is not None else ""
                }
                metadata[schema][table]["columns"].append(col_dict)
                
    cur.close()
    return metadata


def main():
    # Assume your config module contains something like:
    # config = {
    #    "postgres": {
    #         "connection_string": "postgresql://user:password@host:port/dbname"
    #    }
    # }
    try:
        from config import config
        db_url = config['postgres']['connection_string']
    except Exception as e:
        logging.error("Error loading database configuration: %s", e)
        return

    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(db_url)
        logging.info("Connected to PostgreSQL database.")
    except Exception as e:
        logging.error("Error connecting to PostgreSQL: %s", e)
        return

    # Generate metadata dictionary
    metadata = generate_database_metadata(conn)
    conn.close()

    # Write JSON output to file
    output_file = "database_metadata.json"
    try:
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info("Database metadata JSON file generated: %s", output_file)
    except Exception as e:
        logging.error("Error writing JSON to file: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
