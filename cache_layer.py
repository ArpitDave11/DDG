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
