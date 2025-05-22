def chunk_text(text: str, max_chunk_size: int = 250) -> list:
    """
    Split text into chunks of specified size.
    Returns a list of text chunks.
    """
    # Type checking
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be a string, got {type(text)}")

    if not text:
        return []

    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length <= max_chunk_size:
            current_chunk.append(word)
            current_length += word_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks