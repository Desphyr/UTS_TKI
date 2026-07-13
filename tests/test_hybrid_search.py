import unittest

from SemanticSearchAndReranking import HybridGreenIndustrySearch, can_run_search


class HybridSearchTests(unittest.TestCase):
    def test_can_run_search_requires_documents_and_query(self):
        self.assertFalse(can_run_search([], ""))
        self.assertFalse(can_run_search([], "energi hijau"))
        self.assertFalse(can_run_search(["dokumen contoh"], "   "))
        self.assertTrue(can_run_search(["dokumen contoh"], "energi hijau"))

    def test_falls_back_to_lexical_relevance_when_dense_models_are_disabled(self):
        documents = [
            "Program daur ulang limbah industri membantu menjaga lingkungan hidup.",
            "Energi hijau sangat penting untuk keberlangsungan lingkungan hidup dan masa depan negara.",
            "Teknologi pintar mempercepat pemantauan kualitas udara.",
        ]

        engine = HybridGreenIndustrySearch(documents=documents, use_bert=False, use_reranking=False)
        results = engine.search(
            "energi hijau untuk keberlangsungan lingkungan hidup bagi masa depan negara",
            top_k=3,
        )

        self.assertTrue(results)
        self.assertGreater(results[0]["score"], 0.0)
        self.assertEqual(results[0]["doc_id"], 1)


if __name__ == "__main__":
    unittest.main()
