# backend/ir_core/tokenization.py

from abc import ABC, abstractmethod
from typing import List, Dict
import re

class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        ...

TOKEN_PATTERN = re.compile(r"[a-z0-9#+]+")

def _basic_tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())

class SimpleTokenizer(BaseTokenizer):
    """
    Basic tokenizer: lowercase + split on non-alphanumeric (keeps + and # for skills).
    """
    def tokenize(self, text: str) -> List[str]:
        return _basic_tokenize(text)

class ResumeAwareTokenizer(BaseTokenizer):
    """
    Resume-aware tokenizer:
    - uses the same regex tokenizer as SimpleTokenizer
    - merges common job phrases into multiword tokens like 'data_scientist'
    """

    COMMON_PHRASES = {
        "data scientist",
        "machine learning",
        "software engineer",
        "data engineer",
        "business analyst",
        "data analyst",
    }

    def tokenize(self, text: str) -> List[str]:
        base_tokens = _basic_tokenize(text)

        tokens: List[str] = []
        i = 0
        while i < len(base_tokens):
            tok = base_tokens[i]

            if i < len(base_tokens) - 1:
                bigram = tok + " " + base_tokens[i + 1]
                if bigram in self.COMMON_PHRASES:
                    tokens.append(bigram.replace(" ", "_"))
                    i += 2
                    continue

            tokens.append(tok)
            i += 1

        return tokens

TOKENIZER_REGISTRY: Dict[str, BaseTokenizer] = {
    "simple": SimpleTokenizer(),
    "resume_aware": ResumeAwareTokenizer(),
}
