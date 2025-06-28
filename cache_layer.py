

config_path = '/dbfs/mnt/dataops/config/dhub_gload_config.txt'
filenames = []
with open(config_path, 'r') as f:
    for line in f:
        name = line.strip()
        if not name:
            continue
        # Remove any trailing ".json" (case-insensitive)
        name = re.sub(r'(?i)\.json$', '', name)
        filenames.append(name)
print("Tables to process:", filenames)

pgf_target_dir = "/dbfs/mnt/registry/schema/pgf"
os.makedirs(pgf_target_dir, exist_ok=True)
for table in filenames:
    # ... build ddl_statement and index_statements ...
    with open(f"{pgf_target_dir}/{table}.sql", 'w') as f:
        f.write(ddl_statement)
    if index_statements:
        with open(f"{pgf_target_dir}/{table}_indexes.sql", 'w') as f:
            for idx_sql in index_statements:
                f.write(idx_sql + ";\n")

