import os

def delete_file(file_path: str) -> None:
    """
    Delete a file if it exists.
    
    Args:
        file_path (str): Path to the file.
    
    Raises:
        OSError: If deletion fails.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError as e:
        raise OSError(f"Error deleting file {file_path}: {str(e)}")

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Cleaned text.
    """
    return " ".join(text.strip().split())