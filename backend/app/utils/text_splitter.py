from app.core.config import settings

class AdvancedTextSplitter:
    def __init__(self, max_chunk_size=None, overlap=None):
        self.max_chunk_size = max_chunk_size or settings.MAX_CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP

    def split_text(self, text):
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > self.max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-self.overlap:] if self.overlap > 0 else []
                current_length = sum(len(w) + 1 for w in current_chunk)
            current_chunk.append(word)
            current_length += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
