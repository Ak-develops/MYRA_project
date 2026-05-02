from typing import List, Dict
from dataclasses import dataclass

from src.retriever import retrieve
from src.prompt_builder import build_prompt
from src.llm import generate_response
from evaluation.evaluate import evaluate_answer
from src.memory import ConversationMemory


@dataclass
class Document:
    content: str
    metadata: Dict


class RAGCore:
    def __init__(
        self,
        top_k: int = 5,
        max_context_length: int = 3000,
        use_sources: bool = True,
        confidence_threshold: float = 0.4,
        memory: ConversationMemory = None
    ):
        self.top_k = top_k
        self.max_context_length = max_context_length
        self.use_sources = use_sources
        self.confidence_threshold = confidence_threshold
        self.memory = memory if memory else ConversationMemory()

    # -----------------------------
    # CONTEXT FORMATTING
    # -----------------------------
    def _format_context(self, docs: List) -> str:
        context = ""
        total_len = 0

        for i, doc in enumerate(docs):
            chunk = f"\n[Doc {i+1}]\n{doc.page_content}\n"

            if total_len + len(chunk) > self.max_context_length:
                break

            context += chunk
            total_len += len(chunk)

        return context.strip()
    
    def apply_style(self, answer, style_config):
        return answer
    

    # -----------------------------
    # SOURCE EXTRACTION
    # -----------------------------
    def _extract_sources(self, docs: List) -> List[str]:
        sources = []

        for doc in docs:
            src = doc.metadata.get("source")
            if src:
                sources.append(src)

        return list(set(sources))

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------

    def query(self, query, style_config=None):

        # 🔹 1. Retrieve docs
        docs = retrieve(query)

        # 🔹 2. Format context
        context = self._format_context(docs) if docs else ""

        # 🔹 3. Get memory
        memory_text = self.memory.format_memory(query=query, top_k=3)

        # 🔹 4. Decide mode
        use_memory_only = False if docs else True

        # 🔹 5. Build prompt (CLEAN — no style)
        prompt = build_prompt(
        query=query,
        context=context,
        memory=memory_text,
        style_config=style_config,   
        use_memory_only=use_memory_only
    )

        # 🔹 6. Generate CLEAN answer
        clean_answer = generate_response(prompt)

        if not clean_answer or not isinstance(clean_answer, str):
            return {
                "answer": "Model failed to generate a valid response.",
                "raw_answer": "",
                "confidence": 0.0,
                "verdict": "Generation Failed",
                "sources": []
            }

        # 🔹 7. Apply style (your new layer)
        styled_answer = self.apply_style(clean_answer, style_config)

        # 🔹 8. Evaluate grounding
        if docs:
            evaluation = evaluate_answer(
                query=query,
                answer=clean_answer,   #  evaluate CLEAN answer
                context=context,
                docs=docs
            )

            confidence = evaluation.get("confidence", 0.0)
            verdict = evaluation.get("verdict", "Unknown")

            if confidence < self.confidence_threshold:
                styled_answer = "I don't know based on the provided documents."

        else:
            confidence = 0.3
            verdict = "LLM training Based answer."

        # 🔹 9. Update memory (CLEAN only)
        self.memory.add_user_message(query)

        msg_type = "fact" if len(clean_answer) < 200 else "general"
        self.memory.add_assistant_message(clean_answer, msg_type=msg_type)

        # 🔹 10. Sources
        sources = self._extract_sources(docs) if (docs and self.use_sources) else []

        return {
            "answer": styled_answer,
            "raw_answer": clean_answer,
            "confidence": confidence,
            "verdict": verdict,
            "sources": sources
        }

# ---------- CLI TEST ----------
if __name__ == "__main__":
    rag = RAGCore()

    while True:
        query = input("\nEnter query (or 'exit'): ")

        if query.lower() == "exit":
            break

        result = rag.query(query)

        print("\n--- ANSWER ---")
        print(result["answer"])

        print("\n--- CONFIDENCE ---")
        print(result["confidence"], "|", result["verdict"])

        print("\n--- SOURCES ---")
        for s in result["sources"]:
            print("-", s)

