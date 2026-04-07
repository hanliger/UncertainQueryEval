import asyncio
import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

try:
    from src import generate_checklists as gc
except ModuleNotFoundError:  # pragma: no cover
    gc = None


class TestGenerateChecklistsParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if yaml is None or gc is None:
            raise unittest.SkipTest("Required dependencies are not installed for this test.")

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_parse_single_dimension_seed(self):
        seed_path = self.repo_root / "prompt" / "summeval_questions" / "coherence_seed.yaml"
        dimensions = gc.parse_seed_dimensions(seed_path)

        self.assertEqual(len(dimensions), 1)
        dim = dimensions[0]
        self.assertEqual(dim.name, "coherence")
        self.assertEqual(dim.question_key, "sub_aspect")
        self.assertIn("Logical Flow", dim.sub_questions)
        self.assertGreaterEqual(len(dim.sub_questions["Logical Flow"]), 1)

    def test_parse_multi_dimension_seed(self):
        seed_path = self.repo_root / "prompt" / "ambiguity_questions" / "ambiguity_seed.yaml"
        dimensions = gc.parse_seed_dimensions(seed_path)
        names = {d.name for d in dimensions}

        self.assertEqual(len(dimensions), 8)
        self.assertIn("information_sufficiency", names)
        self.assertTrue(all(d.question_key == "sub_dimension" for d in dimensions))

    def test_normalize_generated_questions_yes_no_validation(self):
        parsed = {
            "Logical Flow": [
                "Does the response follow the prior context?",
                "What is the main idea of this response?",
                "Is the response coherent",
            ]
        }
        normalized = gc.normalize_generated_questions(parsed, ["Logical Flow"])
        self.assertIn(
            "Does the response follow the prior context?",
            normalized["Logical Flow"],
        )
        self.assertIn("Is the response coherent?", normalized["Logical Flow"])
        self.assertNotIn(
            "What is the main idea of this response?",
            normalized["Logical Flow"],
        )

    def test_filter_is_remove_only(self):
        combined_pool = {
            "Logical Flow": [
                "Does the response follow the prior context?",
                "Is the response coherent?",
            ]
        }
        filtered_output = {
            "Logical Flow": [
                "Does the response follow the prior context?",
                "Is the response coherent?",
                "Does the answer add a new unrelated point?",
            ]
        }

        normalized = gc.normalize_filtered_questions(filtered_output, combined_pool)
        self.assertEqual(
            normalized["Logical Flow"],
            [
                "Does the response follow the prior context?",
                "Is the response coherent?",
            ],
        )


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = responses

    async def create(self, **kwargs):
        if not self._responses:
            raise RuntimeError("No fake responses left")
        payload = self._responses.pop(0)
        return _FakeResponse(json.dumps(payload, ensure_ascii=False))


class _FakeChat:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)


class _FakeAsyncOpenAI:
    responses_queue = []

    def __init__(self, **kwargs):
        self.chat = _FakeChat(self.responses_queue)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class TestGenerateChecklistsE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if yaml is None or gc is None:
            raise unittest.SkipTest("Required dependencies are not installed for this test.")

    def test_e2e_smoke_with_mocked_llm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            seed_path = tmp_path / "demo_seed.yaml"
            output_dir = tmp_path / "out"

            seed_payload = {
                "definition": {"summeval": "Demo definition for coherence quality."},
                "sub_aspect": {
                    "Logical Flow": [
                        "Does the summary present information in a logical order?"
                    ]
                },
            }
            with seed_path.open("w", encoding="utf-8") as fp:
                yaml.safe_dump(seed_payload, fp, allow_unicode=True, sort_keys=False)

            _FakeAsyncOpenAI.responses_queue = [
                {
                    "Logical Flow": [
                        "Does each sentence connect naturally to the next sentence?"
                    ]
                },
                {
                    "Logical Flow": [
                        "Does the summary keep cause-and-effect links explicit?"
                    ]
                },
                {
                    "Logical Flow": [
                        "Does the summary present information in a logical order?",
                        "Does each sentence connect naturally to the next sentence?",
                    ]
                },
            ]

            args = Namespace(
                seed_input=str(seed_path),
                output_dir=str(output_dir),
                benchmark_name="summeval",
                backend="openai",
                model="fake-model",
                base_url="",
                api_key_env="OPENAI_API_KEY",
            )

            os.environ["OPENAI_API_KEY"] = "fake-key"
            with patch("src.generate_checklists.AsyncOpenAI", _FakeAsyncOpenAI):
                asyncio.run(gc.run_pipeline(args))

            self.assertTrue((output_dir / "demo_seed.yaml").exists())
            self.assertTrue((output_dir / "demo_diversification.yaml").exists())
            self.assertTrue((output_dir / "demo_elaboration.yaml").exists())
            self.assertTrue((output_dir / "demo_filtered.yaml").exists())


if __name__ == "__main__":
    unittest.main()
