import json
import os
import time
import logging

logger = logging.getLogger(__name__)

# --- JSON GENERATIONS (HISTORY) ---

def load_generations(file_path):
    """Load generation history."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading generations: {e}")
            return []
    return []

def save_generations(file_path, generations):
    """Save generation history."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(generations, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving generations: {e}")

# --- PROJECT FILES (.kproj) ---

def save_project_file(file_path, project_data):
    """Saves the project structure."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving project: {e}")
        return False

def load_project_file(file_path):
    """Loads a project file."""
    if not os.path.exists(file_path): 
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else None
    except Exception as e:
        logger.error(f"Error loading project: {e}")
        return None

# --- FILE MANAGEMENT & CLEANUP  ---

def cleanup_temp_files(temp_dir, retention_days=7):
    """Remove old chunk files to save space."""
    now = time.time()
    retention_seconds = retention_days * 86400
    if not os.path.exists(temp_dir): 
        return
    
    deleted_count = 0
    for filename in os.listdir(temp_dir):
        filepath = os.path.join(temp_dir, filename)
        if os.path.isfile(filepath) and filename.startswith("chunk_"):
            if now - os.path.getmtime(filepath) > retention_seconds:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error removing temp file {filepath}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleanup: Removed {deleted_count} old temporary files.")

def delete_file(filepath: str) -> bool:
    """Deletes a single file safely."""
    if not filepath: 
        return False
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted file: {filepath}")
            return True
        return True # Already gone is a success
    except Exception as e:
        logger.error(f"Failed to delete {filepath}: {e}")
        return False