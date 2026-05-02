from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MemoryItem:
    role: str
    content: str
    type: str = "general"   # fact / instruction / casual / general


class ConversationMemory:
    def __init__(self, max_turns: int = 5, use_embeddings: bool = True):
        self.max_turns = max_turns
        self.history: List[MemoryItem] = []

        self.use_embeddings = use_embeddings
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2") if use_embeddings else None
        self.embeddings = []

    # -----------------------------
    # ADD MESSAGES
    # -----------------------------
    def add_user_message(self, message: str, msg_type: str = "general"):
        self._add("user", message, msg_type)

    def add_assistant_message(self, message: str, msg_type: str = "general"):
        self._add("assistant", message, msg_type)

    def _add(self, role: str, message: str, msg_type: str):
        item = MemoryItem(role=role, content=message.strip(), type=msg_type)
        self.history.append(item)

        if self.use_embeddings:
            emb = self.embedder.encode([item.content])[0]
            self.embeddings.append(emb)

        self._trim()

    # -----------------------------
    # TRIMMING (smart-ish)
    # -----------------------------
    def _trim(self):
        max_messages = self.max_turns * 2

        if len(self.history) <= max_messages:
            return

        # Keep important messages
        important = [i for i, m in enumerate(self.history) if m.type in ["fact", "instruction"]]

        # Always preserve important ones
        keep_indices = set(important)

        # Fill remaining with most recent
        remaining_slots = max_messages - len(keep_indices)
        recent_indices = list(range(len(self.history) - remaining_slots, len(self.history)))

        keep_indices.update(recent_indices)

        # Rebuild
        new_history = []
        new_embeddings = []

        for i in sorted(keep_indices):
            new_history.append(self.history[i])
            if self.use_embeddings:
                new_embeddings.append(self.embeddings[i])

        self.history = new_history
        self.embeddings = new_embeddings

    # -----------------------------
    # RELEVANT MEMORY RETRIEVAL
    # -----------------------------
    def get_relevant_memory(self, query: str, top_k: int = 3) -> List[MemoryItem]:
        if not self.use_embeddings or not self.history:
            return self.history[-top_k:]

        query_emb = self.embedder.encode([query])[0]
        scores = cosine_similarity([query_emb], self.embeddings)[0]

        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [self.history[i] for i in top_indices]

    # -----------------------------
    # FORMAT FOR PROMPT
    # -----------------------------
    def format_memory(self, query: Optional[str] = None, top_k: int = 3) -> str:
        if not self.history:
            return ""

        if query:
            selected = self.get_relevant_memory(query, top_k)
        else:
            selected = self.history[-top_k:]

        formatted = []

        for msg in selected:
            role = msg.role.capitalize()
            formatted.append(f"{role}: {msg.content}")

        return "\n".join(formatted)

    # -----------------------------
    # UTILITIES
    # -----------------------------
    def get_last_user_query(self) -> str:
        for msg in reversed(self.history):
            if msg.role == "user":
                return msg.content
        return ""

    def clear(self):
        self.history = []
        self.embeddings = []

