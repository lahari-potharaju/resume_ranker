# backend/ir_core/corpus.py

from typing import List, Any, Dict
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class Document:
    id: Any
    text: str
    meta: Dict

class Corpus:
    def __init__(
        self,
        name: str,
        path: Path,
        text_col: str,
        id_col: str | None = None
    ):
        self.name = name
        self.path = path
        self.text_col = text_col
        self.id_col = id_col
        self.df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if self.df is None:
            self.df = pd.read_csv(self.path)
            if self.id_col is None:
                self.df["doc_id"] = range(len(self.df))
                self.id_col = "doc_id"
        return self.df

    def documents(self) -> List[Document]:
        df = self.load()
        docs: List[Document] = []
        for _, row in df.iterrows():
            docs.append(
                Document(
                    id=row[self.id_col],
                    text=str(row[self.text_col]),
                    meta=row.to_dict(),
                )
            )
        return docs
