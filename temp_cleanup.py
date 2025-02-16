import os
import time

def cleanup_temp_files(temp_dir, retention_days=7):
    """
    Remove files in temp_dir older than retention_days.
    For example, delete files starting with "chunk_" that are older than the specified days.
    """
    now = time.time()
    retention_seconds = retention_days * 86400
    if not os.path.exists(temp_dir):
        return
    for filename in os.listdir(temp_dir):
        filepath = os.path.join(temp_dir, filename)
        if os.path.isfile(filepath) and filename.startswith("chunk_"):
            file_age = now - os.path.getmtime(filepath)
            if file_age > retention_seconds:
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Error removing file {filepath}: {e}")
