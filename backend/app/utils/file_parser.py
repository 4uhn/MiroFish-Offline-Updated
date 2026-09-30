"""File parsing utilities — extract text from PDF, Markdown, and TXT files."""

from pathlib import Path
from typing import List


def _read_text_with_fallback(file_path: str) -> str:
    """Read a text file, auto-detecting encoding if UTF-8 fails."""
    data = Path(file_path).read_bytes()

    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        pass

    encoding = None
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best and best.encoding:
            encoding = best.encoding
    except Exception:
        pass

    if not encoding:
        try:
            import chardet
            result = chardet.detect(data)
            encoding = result.get('encoding') if result else None
        except Exception:
            pass

    if not encoding:
        encoding = 'utf-8'

    return data.decode(encoding, errors='replace')


class FileParser:
    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.markdown', '.txt'}

    @classmethod
    def extract_text(cls, file_path: str) -> str:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {suffix}")

        if suffix == '.pdf':
            return cls._extract_from_pdf(file_path)
        elif suffix in {'.md', '.markdown'}:
            return cls._extract_from_md(file_path)
        elif suffix == '.txt':
            return cls._extract_from_txt(file_path)

        raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required: pip install PyMuPDF")

        text_parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)

        return "\n\n".join(text_parts)

    @staticmethod
    def _extract_from_md(file_path: str) -> str:
        return _read_text_with_fallback(file_path)

    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        return _read_text_with_fallback(file_path)

def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            for sep in ['.\n', '!\n', '?\n', '\n\n', '. ', '! ', '? ']:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > chunk_size * 0.3:
                    end = start + last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        # Start the overlap at a line/sentence boundary, or skip it. The overlap
        # only repeats text the previous chunk already saw whole, so a partial
        # start just hands NER fragments like "otham Gardens" for "Cotham Gardens".
        window_start = max(end - overlap, start + 1)
        window = text[window_start:end]
        cuts = [window.find(sep) + len(sep) for sep in ('\n', '. ', '! ', '? ')
                if 0 <= window.find(sep) < len(window) - len(sep)]
        start = window_start + min(cuts) if cuts else end

    return chunks
