"""The context is numbered ([1], [2], ...) exactly as the module promises, and truncation stays inside the budget."""
import unittest

from services.rag.prompt_builder import build_context_block, build_messages
from services.rag.schemas import ContextDocument


def document(text, doc_id="d1"):
    return ContextDocument(doc_id=doc_id, text=text, score=1.0)


class NumberingTest(unittest.TestCase):
    def test_documents_carry_their_original_position(self):
        block = build_context_block([document("first"), document("second")], 1000)
        self.assertEqual(block, "[1] first\n\n[2] second")

    def test_a_skipped_document_keeps_the_numbering_of_the_rest(self):
        block = build_context_block([document("   "), document("only")], 1000)
        self.assertEqual(block, "[2] only")

    def test_the_prompt_asks_for_bracketed_citations(self):
        messages = build_messages("question?", [document("first")], max_context_chars=1000)
        self.assertIn("[1]", messages[0]["content"])
        self.assertIn("[1] first", messages[1]["content"])


class BudgetTest(unittest.TestCase):
    def test_the_ellipsis_is_inside_the_budget_so_the_last_document_survives(self):
        long_text = "word " * 400  # 2000 characters
        block = build_context_block([document(long_text, "a"), document(long_text, "b")], 1000)
        self.assertEqual(block.count("..."), 2, "the second document was dropped for its ellipsis")
        self.assertLessEqual(len(block), 1000)

    def test_a_short_document_is_not_truncated(self):
        self.assertEqual(build_context_block([document("short text")], 1000), "[1] short text")


if __name__ == "__main__":
    unittest.main()
