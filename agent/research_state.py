# agent/research_state.py

from typing import List, Dict


class ResearchState:
    def __init__(self, query: str):
        self.query = query

        # Generated planning questions
        self.sub_questions: List[str] = []

        # Iteration counter
        self.iteration: int = 0

        # ✅ NEW: stores iteration-level research notes
        self.iteration_summaries: List[str] = []

        # (Optional) store validated evidence if needed later
        self.knowledge: List[Dict] = []

    def add_sub_questions(self, questions: List[str]):
        self.sub_questions = questions

    def add_knowledge(self, item: Dict):
        """
        Optional: store validated evidence objects
        """
        if isinstance(item, dict):
            self.knowledge.append(item)

    def get_knowledge(self) -> List[Dict]:
        return self.knowledge
