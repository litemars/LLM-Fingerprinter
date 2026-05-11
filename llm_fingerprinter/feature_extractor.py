import numpy as np
import logging
import re

logger = logging.getLogger(__name__)

def _setup_nltk():
    """Ensure NLTK data packages are present, downloading only if missing.

    Data is stored in llm_fingerprinter/nltk_data/ — next to this file —
    keeping everything self-contained within the project directory.
    """
    import nltk
    from pathlib import Path

    # Store NLTK data alongside this file: llm_fingerprinter/nltk_data/
    nltk_data_dir = str(Path(__file__).parent / "nltk_data")

    # Prepend so it's checked first, before any system/home locations
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)

    packages_to_try = ['punkt_tab', 'punkt', 'stopwords']
    for package in packages_to_try:
        try:
            # Check if already downloaded — avoids a network call on every import
            nltk.data.find(f"tokenizers/{package}" if package != 'stopwords'
                           else "corpora/stopwords")
        except LookupError:
            try:
                nltk.download(package, download_dir=nltk_data_dir, quiet=True)
            except Exception as e:
                logger.debug(f"NLTK {package} download failed: {e}")

    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        sent_tokenize("Test sentence.")
        word_tokenize("Test words.")
    except Exception as e:
        logger.warning(f"NLTK tokenizer unavailable, will use fallback: {e}")

_setup_nltk()

from sentence_transformers import SentenceTransformer

# Safe NLTK imports with fallbacks
try:
    from nltk import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using basic tokenization")


def _safe_sent_tokenize(text):
    """Tokenize text into sentences with fallback."""
    if _NLTK_AVAILABLE:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    # Fallback: split on sentence-ending punctuation
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _safe_word_tokenize(text: str):
    """Tokenize text into words with fallback."""
    if _NLTK_AVAILABLE:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    # Fallback: simple split with punctuation handling
    import re
    return re.findall(r'\b\w+\b', text.lower())


def _get_stopwords():
    """Get English stopwords with fallback."""
    if _NLTK_AVAILABLE:
        try:
            return set(stopwords.words('english'))
        except Exception:
            pass
    # Fallback: basic English stopwords
    return {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'this',
            'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our',
            'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'he', 'him',
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
            'which', 'who', 'whom'}


class FeatureExtractor:
    
    LINGUISTIC_DIM = 12
    BEHAVIORAL_DIM = 6 
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize feature extractor.
        
        Args:
            model_name: SentenceTransformer model name. 
                       Note: all-MiniLM-L6-v2 outputs 384-dim embeddings.
        """
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.stop_words = _get_stopwords()
            
        logger.info(f"Initialized FeatureExtractor with {model_name} "
                   f"(embedding dim: {self.embedding_dim})")
    
    def extract_batch(self, prompt_response_pairs: list) -> list:
        """Extract features for multiple (prompt, response) pairs efficiently.

        Batches the embedding step (the expensive GPU/CPU forward pass) so that
        N responses are encoded in one call instead of N separate calls.
        Linguistic and behavioral features are still computed per-response
        (they are pure Python and not the bottleneck).

        Args:
            prompt_response_pairs: list of (prompt_str, response_str) tuples

        Returns:
            List of np.ndarray, one per pair, in the same order.
            Zero-vectors are returned for empty/failed responses.
        """
        if not prompt_response_pairs:
            return []

        total_dim = self.embedding_dim + self.LINGUISTIC_DIM + self.BEHAVIORAL_DIM
        responses = [r for _, r in prompt_response_pairs]

        # --- Batch encode all responses in one forward pass ---
        try:
            embeddings = self.embedding_model.encode(
                responses,
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
        except Exception as e:
            logger.error(f"Batch embedding failed, falling back per-response: {e}")
            embeddings = np.stack([
                self._embedding_features(r) for r in responses
            ])

        results = []
        for (prompt, response), embedding in zip(prompt_response_pairs, embeddings):
            if not response or not response.strip():
                results.append(np.zeros(total_dim, dtype=np.float32))
                continue
            ling = self._linguistic_features(response)
            beh  = self._behavioral_features(prompt, response)
            results.append(np.concatenate([embedding, ling, beh]))

        return results

    def extract(self, prompt: str, response: str):

        if not response or not response.strip():
            logger.warning("Empty response, returning zero features")
            total_dim = self.embedding_dim + self.LINGUISTIC_DIM + self.BEHAVIORAL_DIM
            return np.zeros(total_dim, dtype=np.float32)
        
        embedding_features = self._embedding_features(response)
        linguistic_features = self._linguistic_features(response)
        behavioral_features = self._behavioral_features(prompt, response)
        
        all_features = np.concatenate([
            embedding_features,
            linguistic_features,
            behavioral_features
        ])
        
        return all_features
    
    def get_feature_dim(self):

        return self.embedding_dim + self.LINGUISTIC_DIM + self.BEHAVIORAL_DIM
    
    def _embedding_features(self, response: str):
        try:
            embedding = self.embedding_model.encode(response, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _linguistic_features(self, response: str):
        """
        Extract linguistic features (12-dim).
        
        Features:
            0: Total characters
            1: Total words
            2: Type-token ratio (vocabulary diversity)
            3: Average word length
            4: Number of sentences
            5: Average sentence length
            6: Punctuation ratio
            7: Code block ratio
            8: Structural markers ratio
            9: Token entropy
            10: AI marker count
            11: Capital letter ratio
        """
        features = []
        
        # Length features
        features.append(len(response))  # total chars
        
        # Tokenize safely using our fallback-enabled functions
        words = _safe_word_tokenize(response.lower())
        
        features.append(len(words))  # total words
        
        # Vocabulary features
        unique_words = len(set(words))
        type_token_ratio = unique_words / max(len(words), 1)
        features.append(type_token_ratio)
        
        # Complexity
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        features.append(avg_word_len)
        
        # Sentence stats using safe tokenizer
        sentences = _safe_sent_tokenize(response)
        num_sentences = max(len(sentences), 1)
        
        avg_sent_len = len(words) / max(num_sentences, 1)
        features.append(num_sentences)
        features.append(avg_sent_len)
        
        # Punctuation ratio
        punctuation_count = sum(1 for c in response if c in '.,!?;:')
        features.append(punctuation_count / max(len(response), 1))
        
        # Code blocks (using ``` markers)
        code_block_count = response.count("```")
        code_ratio = code_block_count / max(len(response), 1) * 100  # Scale up for visibility
        features.append(code_ratio)
        
        # Structural markers (bullet points, numbered lists)
        bullet_patterns = [
            r'^[-*•]\s',  # Bullet points (-, *, •)
            r'^\d+\.\s',  # Numbered lists (1., 2., etc.)
        ]
        struct_count = 0
        for line in response.split('\n'):
            line = line.strip()
            for pattern in bullet_patterns:
                if re.match(pattern, line):
                    struct_count += 1
                    break
        features.append(struct_count / max(len(response.split('\n')), 1))
        
        # Token entropy
        if words:
            word_freq = {}
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1
            probs = np.array(list(word_freq.values())) / len(words)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
        else:
            entropy = 0
        features.append(entropy)
        
        # Characteristic AI phrases
        ai_markers = sum([
            response.lower().count("as an ai"),
            response.lower().count("as a language model"),
            response.lower().count("as an artificial"),
            response.lower().count("i cannot"),
            response.lower().count("i can't"),
            response.lower().count("i'm not able"),
            response.lower().count("i am not able"),
        ])
        features.append(ai_markers)
        
        # Capital ratio
        capital_count = sum(1 for c in response if c.isupper())
        capital_ratio = capital_count / max(len(response), 1)
        features.append(capital_ratio)
        
        return np.array(features, dtype=np.float32)
    
    def _behavioral_features(self, prompt: str, response: str):
        """
        Extract behavioral features (6-dim).
        
        Features:
            0: Refusal score
            1: Format adherence score
            2: Reasoning presence score
            3: Instruction compliance score
            4: Length normalization score
            5: Formality score
        """
        features = []
        
        response_lower = response.lower()
        prompt_lower = prompt.lower()
        word_count = max(len(response.split()), 1)
        
        refusal_keywords = [
            "cannot", "can't", "unable", "not able", "refuse", 
            "apologize", "sorry", "won't", "will not", "inappropriate"
        ]
        refusal_score = sum(1 for kw in refusal_keywords if kw in response_lower) / word_count
        features.append(refusal_score)
        
        format_markers = (
            response.count("1.") + response.count("2.") + response.count("3.") +
            response.count("- ") + response.count("* ") +
            response.count("**") + response.count("##")
        )
        format_score = min(format_markers / word_count, 1.0)
        features.append(format_score)
        
        reasoning_keywords = [
            "therefore", "step", "reason", "because", "thus",
            "first", "second", "third", "finally", "however",
            "consequently", "as a result", "let me", "let's"
        ]
        reasoning_score = sum(1 for kw in reasoning_keywords if kw in response_lower) / word_count
        features.append(reasoning_score)
        
        compliance = 0.5
        
        if "list" in prompt_lower or "enumerate" in prompt_lower:
            # Check for list structure
            has_list = (
                response.count("\n") > 2 and 
                (response.count("- ") > 0 or response.count("1.") > 0 or response.count("* ") > 0)
            )
            compliance = 1.0 if has_list else 0.0
        elif "code" in prompt_lower or "implement" in prompt_lower or "write a function" in prompt_lower:
            # Check for code block
            has_code = "```" in response or response.count("def ") > 0 or response.count("function") > 0
            compliance = 1.0 if has_code else 0.0
        elif "brief" in prompt_lower or "short" in prompt_lower:
            # Check for brevity
            compliance = 1.0 if word_count < 100 else 0.5 if word_count < 200 else 0.0
        elif "detailed" in prompt_lower or "explain" in prompt_lower:
            # Check for detail
            compliance = 1.0 if word_count > 100 else 0.5 if word_count > 50 else 0.0
            
        features.append(compliance)
        
        length_score = 1.0 - min(abs(word_count - 150) / 300, 1.0)
        features.append(length_score)
        
        formal_words = [
            "indeed", "furthermore", "thus", "moreover", "however",
            "nevertheless", "consequently", "therefore", "accordingly",
            "hence", "subsequently", "nonetheless"
        ]
        formal_score = sum(1 for w in formal_words if w in response_lower) / word_count
        features.append(formal_score)
        
        return np.array(features, dtype=np.float32)
