import unittest
from pathlib import Path

try:
    from src import inference_checkeval as ic
except ModuleNotFoundError:  # pragma: no cover
    ic = None


class TestInferenceChecklistLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ic is None:
            raise unittest.SkipTest("Required dependencies are not installed for this test.")

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_resolve_question_path_seed(self):
        path = ic.resolve_question_path("summeval", "coherence", "seed")
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "coherence_seed.yaml")

    def test_extract_all_questions_with_sub_aspect(self):
        payload = {
            "definition": {"summeval": "Test definition"},
            "sub_aspect": {
                "Logical Flow": [
                    "Does the summary maintain order?",
                    "Is the structure coherent?",
                ]
            },
        }
        questions = ic.extract_all_questions(payload)
        self.assertEqual(
            questions,
            ["Does the summary maintain order?", "Is the structure coherent?"],
        )

    def test_extract_all_questions_with_sub_dimension(self):
        payload = {
            "definition": "테스트 정의",
            "sub_dimension": {
                "Predicate Presence": "요청 동사가 명확히 제시되어 있는가?"
            },
        }
        questions = ic.extract_all_questions(payload)
        self.assertEqual(questions, ["요청 동사가 명확히 제시되어 있는가?"])

    def test_load_definition_fallback(self):
        payload = {"definition": {"coherence": "Fallback definition text"}}
        definition = ic.load_definition(payload, "summeval")
        self.assertEqual(definition, "Fallback definition text")

    def test_make_question_list_from_list(self):
        question_block = ic.make_question_list(
            ["Does A hold?", "Is B supported?"]
        )
        self.assertIn("Q1: Does A hold?", question_block)
        self.assertIn("Q2: Is B supported?", question_block)


if __name__ == "__main__":
    unittest.main()
