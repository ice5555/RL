from typing import List, Optional, Dict

class PromptBuilder:
    def __init__(self,
                 system_prompt: str = "You are a helpful assistant.",
                 few_shot: Optional[List[Dict[str, str]]] = None,
                 max_context: int = 3):
        self.system_prompt = system_prompt
        self.few_shot = few_shot or []
        self.max_context = max_context

    def build(self, query: str, contexts: List[str]) -> str:
        parts = [f"SYSTEM:\n{self.system_prompt}\n"]
        for ex in self.few_shot:
            parts.append(f"Q: {ex['q']}\nA: {ex['a']}\n")
        parts.append(f"USER QUERY:\n{query}\n")
        for i, ctx in enumerate(contexts[:self.max_context]):
            parts.append(f"CONTEXT {i+1}:\n{ctx}\n")
        # parts.append("PLEASE ANSWER BASED ON THE ABOVE CONTEXT.")
        parts.append("Answer in one short sentence based ONLY on the context above.\nAnswer:")
        return "\n".join(parts)