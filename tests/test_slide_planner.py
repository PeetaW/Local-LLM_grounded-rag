import unittest
from rag.research_tools.slide_planner import ResearchFinding, plan_deck_from_discussion
from rag.research_tools.registry import run_named_tool


class SlidePlanningTests(unittest.TestCase):
    def test_tentative_is_labeled_and_questions_preserved(self):
        deck = plan_deck_from_discussion(
            "Fe3O4 progress",
            [ResearchFinding("Fe3O4 Raman data", "A candidate band at 670 cm^-1",
                             "May reflect magnetite", ("sample1.asc",), "tentative")],
            next_steps=("Check XRD reference standards",),
            unresolved_questions=("Does laser power change the spectrum?",)
        )
        self.assertIn("Tentative interpretation", deck.slides[0].bullets[1])
        self.assertEqual(deck.slides[1].title, "Current questions")
        self.assertEqual(len(deck.conclusion), 0)

    def test_validated_claim_requires_evidence(self):
        with self.assertRaises(ValueError):
            ResearchFinding("Test", "Claim", status="validated")

    def test_tool_allowlist(self):
        with self.assertRaises(ValueError):
            run_named_tool("shell.exec", {"command": "echo no"})


if __name__ == "__main__":
    unittest.main()
