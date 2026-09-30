"""
Text preprocessing and chunking service.
"""

from typing import List
from ..utils.file_parser import split_text_into_chunks

class TextProcessor:
    """Text processor"""
    
    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        return split_text_into_chunks(text, chunk_size, overlap)
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        import re
        

        text = text.replace('\r\n', '\n').replace('\r', '\n')
        

        text = re.sub(r'\n{3,}', '\n\n', text)
        

        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    