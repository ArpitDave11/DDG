import psycopg2, logging
from config import config

def get_db_connection():
    """Open a new database connection using config credentials."""
    pg_conf = config['postgres']
    conn = psycopg2.connect(
        host=pg_conf['host'],
        port=pg_conf.get('port', 5432),
        dbname=pg_conf['database'],
        user=pg_conf['user'],
        password=pg_conf['password'],
        sslmode=pg_conf.get('sslmode', 'prefer')
    )
    return conn

def ensure_table_exists():
    """Create the output table if it doesn't already exist."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS attribute_descriptions (
        table_name TEXT NOT NULL,
        column_name TEXT NOT NULL,
        data_type TEXT,
        generated_description TEXT,
        existing_description TEXT,
        PRIMARY KEY (table_name, column_name)
    );
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(create_sql)
        conn.commit()
        cur.close()
        logging.info("Ensured target table exists (attribute_descriptions).")
    except Exception as e:
        logging.error(f"Failed to create table: {e}")
        raise
    finally:
        if conn:
            conn.close()

def save_attribute_descriptions(rows: list[tuple]):
    """
    Save multiple attribute description records to the database.
    `rows` should be a list of tuples: (table_name, column_name, data_type, generated_description, existing_description).
    """
    if not rows:
        return 0
    insert_sql = """
    INSERT INTO attribute_descriptions 
        (table_name, column_name, data_type, generated_description, existing_description)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (table_name, column_name)
    DO UPDATE SET 
        data_type = EXCLUDED.data_type,
        generated_description = EXCLUDED.generated_description,
        existing_description = EXCLUDED.existing_description;
    """
    conn = None
    inserted = 0
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.executemany(insert_sql, rows)
        inserted = cur.rowcount
        conn.commit()
        cur.close()
        logging.info(f"Upserted {inserted} rows into attribute_descriptions.")
    except Exception as e:
        logging.error(f"Database insert failed: {e}")
        raise
    finally:
        if conn:
            conn.close()
    return inserted
