import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class FAISSSimilarityChecker:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.8):
        """
        Initialize the SimilarityChecker with a SentenceTransformer model and similarity threshold.
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold

    def _compute_similarity(self, ground_truth: list, model_output: str) -> bool:
        """
        Compute similarity between ground truth (list) and model output (string).

        Args:
            ground_truth (list): A list of correct answers.
            model_output (str): A string output by the model.

        Returns:
            bool: True if the model output has high similarity with any ground truth answer, else False.
        """
        # Ensure ground truth is unique
        unique_answers = list(set(ground_truth))
        
        # Substring matching
        for answer in unique_answers:
            if answer.lower() in model_output.lower():
                return True

        # Generate embeddings for ground truth and model output
        ground_truth_embeddings = self.model.encode(unique_answers)
        model_output_embedding = self.model.encode([model_output])

        # Normalize embeddings
        faiss.normalize_L2(ground_truth_embeddings)
        faiss.normalize_L2(model_output_embedding)

        # Create FAISS index for similarity search
        index = faiss.IndexFlatIP(ground_truth_embeddings.shape[1])
        index.add(ground_truth_embeddings)

        # Perform similarity search
        similarities, _ = index.search(model_output_embedding, len(unique_answers))

        # Check if any similarity is above the threshold
        return (similarities >= self.similarity_threshold).any()

#     def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.8):
#         self.model = SentenceTransformer(model_name)
#         self.similarity_threshold = similarity_threshold
 
#     def _compute_similarity(self, ground_truth: list, model_output: list) -> bool:
#         ground_truth_embeddings = self.model.encode(ground_truth)
#         model_output_embeddings = self.model.encode(model_output)
 
#         faiss.normalize_L2(ground_truth_embeddings)
#         faiss.normalize_L2(model_output_embeddings)
 
#         index = faiss.IndexFlatIP(ground_truth_embeddings.shape[1])
#         index.add(ground_truth_embeddings)
 
#         k = len(ground_truth)
#         similarities, _ = index.search(model_output_embeddings, k)
 
#         max_similarity = np.max(similarities)
#         print("Max Similarity is:",max_similarity)
#         return max_similarity >= self.similarity_threshold
 
#     def are_answers_similar(self,model_output: list, ground_truth: list) -> bool:
#         return self._compute_similarity(ground_truth, model_output)
    