import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from typing import Dict, List, Tuple, Optional
import re
from slang_preprocessor import IndonesianSlangPreprocessor

class FlexibleSimilaritySearch:
    """
    Sistem similarity search yang lebih fleksibel dengan:
    - Threshold yang dapat disesuaikan
    - Reprocessing jawaban berdasarkan konteks
    - Multiple similarity metrics
    - Preprocessing bahasa gaul
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.3,
                 max_results: int = 5,
                 use_svd: bool = True,
                 svd_components: int = 100):
        
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self.use_svd = use_svd
        self.svd_components = svd_components
        
        # Initialize components
        self.slang_preprocessor = IndonesianSlangPreprocessor()
        self.vectorizer = None
        self.svd = None
        self.dataset = []
        self.question_vectors = None
        self.processed_questions = []
        
        # Similarity weights for different metrics
        self.similarity_weights = {
            'cosine': 0.4,
            'keyword': 0.3,
            'semantic': 0.3
        }
        
        # Answer reprocessing templates
        self.reprocessing_templates = {
            'similar_problem': "Berdasarkan masalah serupa, {original_answer}. Namun untuk kasus Anda yang spesifik: {specific_advice}",
            'related_component': "Masalah ini berkaitan dengan {component}. {original_answer}. Perlu diperhatikan juga: {additional_info}",
            'general_guidance': "Secara umum, {original_answer}. Untuk situasi spesifik Anda, disarankan: {custom_advice}"
        }
    
    def load_dataset(self, dataset_path: str) -> None:
        """
        Load dan preprocess dataset untuk similarity search
        
        Args:
            dataset_path: Path ke file dataset JSON
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            
            print(f"Loaded {len(self.dataset)} entries from dataset")
            
            # Preprocess questions
            self._preprocess_dataset()
            
            # Build similarity index
            self._build_similarity_index()
            
            print("Similarity search index built successfully")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def _preprocess_dataset(self) -> None:
        """
        Preprocess semua questions dalam dataset
        """
        self.processed_questions = []
        
        for entry in self.dataset:
            if 'question' in entry:
                # Normalize slang
                processed = self.slang_preprocessor.normalize_text(entry['question'])
                self.processed_questions.append(processed)
            else:
                self.processed_questions.append("")
        
        print(f"Preprocessed {len(self.processed_questions)} questions")
    
    def _build_similarity_index(self) -> None:
        """
        Build TF-IDF vectors dan optional SVD untuk similarity search
        """
        if not self.processed_questions:
            raise ValueError("No processed questions available")
        
        # Initialize TF-IDF vectorizer dengan parameter yang dioptimasi
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),  # Unigram, bigram, trigram
            stop_words=self._get_indonesian_stopwords(),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Fit dan transform questions
        question_tfidf = self.vectorizer.fit_transform(self.processed_questions)
        
        # Optional: Apply SVD untuk dimensionality reduction
        if self.use_svd and question_tfidf.shape[1] > self.svd_components:
            self.svd = TruncatedSVD(n_components=self.svd_components, random_state=42)
            self.question_vectors = self.svd.fit_transform(question_tfidf)
            print(f"Applied SVD: {question_tfidf.shape[1]} -> {self.svd_components} dimensions")
        else:
            self.question_vectors = question_tfidf.toarray()
            print(f"Using full TF-IDF vectors: {self.question_vectors.shape[1]} dimensions")
    
    def _get_indonesian_stopwords(self) -> List[str]:
        """
        Get Indonesian stopwords untuk TF-IDF
        """
        return [
            'yang', 'dan', 'di', 'ke', 'dari', 'dalam', 'untuk', 'pada', 'dengan',
            'adalah', 'ini', 'itu', 'atau', 'juga', 'akan', 'dapat', 'telah',
            'sudah', 'belum', 'masih', 'bisa', 'harus', 'tidak', 'ada', 'saya',
            'anda', 'kita', 'mereka', 'dia', 'ia', 'nya', 'mu', 'ku', 'se',
            'ter', 'ber', 'me', 'pe', 'an', 'kan', 'lah', 'kah', 'pun'
        ]
    
    def _calculate_keyword_similarity(self, query_keywords: List[str], target_text: str) -> float:
        """
        Calculate keyword-based similarity
        
        Args:
            query_keywords: Keywords dari query
            target_text: Target text untuk comparison
            
        Returns:
            float: Keyword similarity score (0-1)
        """
        if not query_keywords:
            return 0.0
        
        target_words = set(target_text.lower().split())
        matched_keywords = sum(1 for keyword in query_keywords if keyword.lower() in target_words)
        
        return matched_keywords / len(query_keywords)
    
    def _calculate_semantic_similarity(self, query: str, target: str) -> float:
        """
        Calculate semantic similarity berdasarkan context
        
        Args:
            query: Query text
            target: Target text
            
        Returns:
            float: Semantic similarity score (0-1)
        """
        # Extract domain-specific terms
        motor_terms = {
            'mesin': ['piston', 'silinder', 'klep', 'timing', 'kompresi'],
            'kelistrikan': ['aki', 'spul', 'cdi', 'busi', 'koil', 'starter'],
            'transmisi': ['kopling', 'gigi', 'rantai', 'sprocket', 'gear'],
            'rangka': ['shock', 'fork', 'suspensi', 'rem', 'ban', 'velg'],
            'bahan_bakar': ['karburator', 'injeksi', 'bensin', 'filter', 'pompa']
        }
        
        query_lower = query.lower()
        target_lower = target.lower()
        
        semantic_score = 0.0
        total_categories = len(motor_terms)
        
        for category, terms in motor_terms.items():
            query_has_category = any(term in query_lower for term in terms)
            target_has_category = any(term in target_lower for term in terms)
            
            if query_has_category and target_has_category:
                # Both have terms from same category
                semantic_score += 1.0 / total_categories
            elif query_has_category or target_has_category:
                # Only one has terms from this category
                semantic_score += 0.3 / total_categories
        
        return semantic_score
    
    def _combine_similarity_scores(self, cosine_sim: float, keyword_sim: float, semantic_sim: float) -> float:
        """
        Combine multiple similarity scores dengan weighted average
        
        Args:
            cosine_sim: Cosine similarity score
            keyword_sim: Keyword similarity score
            semantic_sim: Semantic similarity score
            
        Returns:
            float: Combined similarity score
        """
        combined = (
            cosine_sim * self.similarity_weights['cosine'] +
            keyword_sim * self.similarity_weights['keyword'] +
            semantic_sim * self.similarity_weights['semantic']
        )
        
        return combined
    
    def search_similar_questions(self, query: str, custom_threshold: Optional[float] = None) -> List[Dict[str, any]]:
        """
        Search untuk questions yang similar dengan query
        
        Args:
            query: User query
            custom_threshold: Custom similarity threshold untuk search ini
            
        Returns:
            List[Dict]: List of similar questions dengan metadata
        """
        if not self.vectorizer or self.question_vectors is None:
            raise ValueError("Similarity index not built. Call load_dataset first.")
        
        threshold = custom_threshold if custom_threshold is not None else self.similarity_threshold
        
        # Preprocess query
        preprocessed_query = self.slang_preprocessor.normalize_text(query)
        query_keywords = self.slang_preprocessor.extract_keywords(query)
        
        # Transform query ke vector space
        query_tfidf = self.vectorizer.transform([preprocessed_query])
        
        if self.use_svd and self.svd:
            query_vector = self.svd.transform(query_tfidf)
        else:
            query_vector = query_tfidf.toarray()
        
        # Calculate cosine similarities
        cosine_similarities = cosine_similarity(query_vector, self.question_vectors)[0]
        
        # Calculate additional similarities
        results = []
        for i, (cosine_sim, entry, processed_q) in enumerate(zip(cosine_similarities, self.dataset, self.processed_questions)):
            if cosine_sim < 0.1:  # Skip very low cosine similarities
                continue
            
            # Calculate keyword similarity
            keyword_sim = self._calculate_keyword_similarity(query_keywords, processed_q)
            
            # Calculate semantic similarity
            semantic_sim = self._calculate_semantic_similarity(preprocessed_query, processed_q)
            
            # Combine similarities
            combined_sim = self._combine_similarity_scores(cosine_sim, keyword_sim, semantic_sim)
            
            if combined_sim >= threshold:
                results.append({
                    'index': i,
                    'question': entry.get('question', ''),
                    'answer': entry.get('answer', ''),
                    'cosine_similarity': float(cosine_sim),
                    'keyword_similarity': float(keyword_sim),
                    'semantic_similarity': float(semantic_sim),
                    'combined_similarity': float(combined_sim),
                    'processed_question': processed_q
                })
        
        # Sort by combined similarity
        results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        
        return results[:self.max_results]
    
    def _extract_key_components(self, text: str) -> List[str]:
        """
        Extract key components dari text untuk reprocessing
        
        Args:
            text: Input text
            
        Returns:
            List[str]: List of key components
        """
        component_patterns = {
            'mesin': r'(piston|silinder|klep|timing|kompresi|oli|radiator|thermostat)',
            'kelistrikan': r'(aki|spul|cdi|busi|koil|starter|lampu|klakson|kiprok)',
            'transmisi': r'(kopling|gigi|rantai|sprocket|gear|transmisi)',
            'rangka': r'(shock|fork|suspensi|rem|ban|velg|bearing)',
            'bahan_bakar': r'(karburator|injeksi|bensin|filter|pompa|throttle)'
        }
        
        components = []
        text_lower = text.lower()
        
        for category, pattern in component_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                components.extend(matches)
        
        return list(set(components))  # Remove duplicates
    
    def _generate_specific_advice(self, query: str, similar_result: Dict[str, any]) -> str:
        """
        Generate specific advice berdasarkan query dan similar result
        
        Args:
            query: Original user query
            similar_result: Similar question result
            
        Returns:
            str: Specific advice
        """
        query_components = self._extract_key_components(query)
        similar_components = self._extract_key_components(similar_result['question'])
        
        # Find different components
        different_components = set(query_components) - set(similar_components)
        
        if different_components:
            return f"Perhatikan juga komponen {', '.join(different_components)} yang mungkin terkait dengan masalah Anda."
        else:
            return "Pastikan untuk memeriksa kondisi komponen terkait lainnya."
    
    def reprocess_answer(self, query: str, similar_results: List[Dict[str, any]]) -> str:
        """
        Reprocess jawaban berdasarkan similarity results
        
        Args:
            query: Original user query
            similar_results: List of similar question results
            
        Returns:
            str: Reprocessed answer
        """
        if not similar_results:
            return "Maaf, tidak ditemukan informasi yang relevan untuk pertanyaan Anda."
        
        best_result = similar_results[0]
        similarity_score = best_result['combined_similarity']
        
        # Determine reprocessing strategy based on similarity score
        if similarity_score >= 0.8:
            # High similarity - use answer directly with minor adaptation
            return f"Berdasarkan analisis, {best_result['answer']}"
        
        elif similarity_score >= 0.5:
            # Medium similarity - reprocess with specific advice
            specific_advice = self._generate_specific_advice(query, best_result)
            template = self.reprocessing_templates['similar_problem']
            
            return template.format(
                original_answer=best_result['answer'],
                specific_advice=specific_advice
            )
        
        else:
            # Lower similarity - provide general guidance
            components = self._extract_key_components(query)
            component_str = ', '.join(components) if components else "komponen yang disebutkan"
            
            template = self.reprocessing_templates['related_component']
            additional_info = "Disarankan untuk melakukan pemeriksaan menyeluruh."
            
            return template.format(
                component=component_str,
                original_answer=best_result['answer'],
                additional_info=additional_info
            )
    
    def get_enhanced_response(self, query: str, custom_threshold: Optional[float] = None) -> Dict[str, any]:
        """
        Get enhanced response dengan similarity search dan reprocessing
        
        Args:
            query: User query
            custom_threshold: Custom similarity threshold
            
        Returns:
            Dict: Enhanced response dengan metadata
        """
        # Preprocess query
        preprocessed = self.slang_preprocessor.preprocess_for_ai(query)
        
        # Search similar questions
        similar_results = self.search_similar_questions(query, custom_threshold)
        
        # Reprocess answer
        reprocessed_answer = self.reprocess_answer(query, similar_results)
        
        return {
            'original_query': query,
            'preprocessed_query': preprocessed['normalized'],
            'keywords': preprocessed['keywords'],
            'slang_confidence': preprocessed['confidence'],
            'similar_results': similar_results,
            'reprocessed_answer': reprocessed_answer,
            'similarity_threshold_used': custom_threshold or self.similarity_threshold,
            'total_similar_found': len(similar_results)
        }
    
    def update_similarity_weights(self, cosine: float, keyword: float, semantic: float) -> None:
        """
        Update similarity weights
        
        Args:
            cosine: Weight untuk cosine similarity
            keyword: Weight untuk keyword similarity  
            semantic: Weight untuk semantic similarity
        """
        total = cosine + keyword + semantic
        if total <= 0:
            raise ValueError("Total weights must be positive")
        
        self.similarity_weights = {
            'cosine': cosine / total,
            'keyword': keyword / total,
            'semantic': semantic / total
        }
        
        print(f"Updated similarity weights: {self.similarity_weights}")
    
    def get_search_statistics(self) -> Dict[str, any]:
        """
        Get statistics tentang similarity search system
        
        Returns:
            Dict: Search statistics
        """
        return {
            'dataset_size': len(self.dataset),
            'processed_questions': len(self.processed_questions),
            'vector_dimensions': self.question_vectors.shape[1] if self.question_vectors is not None else 0,
            'similarity_threshold': self.similarity_threshold,
            'max_results': self.max_results,
            'use_svd': self.use_svd,
            'svd_components': self.svd_components if self.use_svd else None,
            'similarity_weights': self.similarity_weights
        }

# Test function
if __name__ == "__main__":
    # Initialize flexible similarity search
    search_engine = FlexibleSimilaritySearch(
        similarity_threshold=0.3,
        max_results=5,
        use_svd=True,
        svd_components=100
    )
    
    # Load dataset
    try:
        search_engine.load_dataset("new_motorcycle_qa_dataset.json")
        
        # Test queries dengan bahasa gaul
        test_queries = [
            "motor gue mogok terus nih, kenapa ya?",
            "rem motor ane blong banget, gimana benerin?",
            "oli motor udah item, hrs ganti ga?",
            "busi motor kotor, bs bersihin sendiri?",
            "shock motor bunyi ngorok parah"
        ]
        
        print("\n=== Testing Flexible Similarity Search ===")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}: {query}")
            print("-" * 50)
            
            response = search_engine.get_enhanced_response(query)
            
            print(f"Preprocessed: {response['preprocessed_query']}")
            print(f"Keywords: {response['keywords']}")
            print(f"Slang confidence: {response['slang_confidence']:.2f}")
            print(f"Similar results found: {response['total_similar_found']}")
            
            if response['similar_results']:
                best_match = response['similar_results'][0]
                print(f"Best match similarity: {best_match['combined_similarity']:.3f}")
                print(f"Best match question: {best_match['question'][:100]}...")
            
            print(f"\nReprocessed Answer: {response['reprocessed_answer'][:200]}...")
            print("=" * 70)
        
        # Print statistics
        stats = search_engine.get_search_statistics()
        print("\n=== Search Engine Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error during testing: {e}")