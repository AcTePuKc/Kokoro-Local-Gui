import json
import os

def load_generations(file_path):
    """
    Load generations from a JSON file.
    Returns a list; if file doesn't exist or an error occurs, returns an empty list.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading generations: {e}")
            return []
    else:
        return []

def save_generations(file_path, generations):
    """
    Save the given list of generations to a JSON file.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(generations, f, indent=2)
    except Exception as e:
        print(f"Error saving generations: {e}")
