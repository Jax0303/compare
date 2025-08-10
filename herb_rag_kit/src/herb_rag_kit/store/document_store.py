from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import os, json, time

@dataclass
class Document:
    doc_id: str
    text: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DocumentStore:
    def __init__(self):
        self.docs: Dict[str, Document] = {}

    def add(self, doc: Document) -> None:
        self.docs[doc.doc_id] = doc

    def get(self, doc_id: str) -> Optional[Document]:
        return self.docs.get(doc_id)

    def all(self) -> List[Document]:
        return list(self.docs.values())
