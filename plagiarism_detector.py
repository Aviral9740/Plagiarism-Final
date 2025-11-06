import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from typing import List, Dict, Tuple, Optional
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# pip install scikit-learn requests beautifulsoup4 python-dotenv lxml
try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    print("Warning: beautifulsoup4 not installed. Web scraping disabled.")
    print("Install with: pip install beautifulsoup4 lxml")
    BS4_AVAILABLE = False


class Config:
    """Configuration class to manage environment variables"""

    def __init__(self):
        # API Keys
        self.SEARCHAPI_KEY = os.getenv('SEARCHAPI_KEY')
        # Search Limits
        self.MAX_RESULTS_WEB = int(os.getenv('MAX_RESULTS_WEB', 5))
        self.MAX_RESULTS_SCHOLAR = int(os.getenv('MAX_RESULTS_SCHOLAR', 5))
        self.MAX_RESULTS_ARXIV = int(os.getenv('MAX_RESULTS_ARXIV', 3))

        # Similarity Thresholds
        self.SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.15))

        # Timeouts
        self.API_TIMEOUT = int(os.getenv('API_TIMEOUT', 10))

    def validate(self):
        """Validate that required configuration is present"""
        if not self.SEARCHAPI_KEY:
            print("Warning: SEARCHAPI_KEY not found in environment variables.")
            print("Web search and Google Scholar features will be disabled.")
            print("Please add SEARCHAPI_KEY to your .env file.")


class PlagiarismDetector:
    def __init__(self, config: Optional[Config] = None):
        print("Initializing PlagiarismDetector...")

        # Load configuration
        self.config = config or Config()
        self.config.validate()

        self.searchapi_key = self.config.SEARCHAPI_KEY

        print("Creating TF-IDF vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            stop_words='english'
        )

        # Knowledge base of famous papers
        print("Loading knowledge base...")
        self.knowledge_base = [
            {
                'title': 'Attention Is All You Need',
                'content': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.',
                'url': 'https://arxiv.org/abs/1706.03762',
                'authors': 'Vaswani et al.',
                'year': 2017,
                'venue': 'NeurIPS 2017'
            },
            {
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
                'content': 'We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks.',
                'url': 'https://arxiv.org/abs/1810.04805',
                'authors': 'Devlin et al.',
                'year': 2018,
                'venue': 'NAACL 2019'
            },
            {
                'title': 'ImageNet Classification with Deep Convolutional Neural Networks',
                'content': 'We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers.',
                'url': 'https://papers.nips.cc/paper/4824',
                'authors': 'Krizhevsky, Sutskever, Hinton',
                'year': 2012,
                'venue': 'NeurIPS 2012'
            },
            {
                'title': 'Generative Adversarial Networks',
                'content': 'We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake.',
                'url': 'https://arxiv.org/abs/1406.2661',
                'authors': 'Goodfellow et al.',
                'year': 2014,
                'venue': 'NeurIPS 2014'
            }
        ]
        print(f"✓ PlagiarismDetector initialized successfully ({len(self.knowledge_base)} papers in knowledge base)")

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 100) -> List[str]:
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - 20):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) >= 20:
                chunks.append(chunk)

        return chunks

    def calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        try:
            vectors = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0

    def calculate_ngram_similarity(self, text1: str, text2: str, n: int = 5) -> float:
        words1 = self.preprocess_text(text1).split()
        words2 = self.preprocess_text(text2).split()

        if len(words1) < n or len(words2) < n:
            return 0.0

        ngrams1 = set([' '.join(words1[i:i + n]) for i in range(len(words1) - n + 1)])
        ngrams2 = set([' '.join(words2[i:i + n]) for i in range(len(words2) - n + 1)])

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)

        return len(intersection) / len(union) if union else 0.0

    def calculate_lcs_similarity(self, text1: str, text2: str) -> float:
        words1 = self.preprocess_text(text1).split()
        words2 = self.preprocess_text(text2).split()

        m, n = len(words1), len(words2)
        if m == 0 or n == 0:
            return 0.0

        # Dynamic programming LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i - 1] == words2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        return lcs_length / max(m, n)

    def calculate_combined_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Calculate similarity using TF-IDF, N-gram, and LCS methods"""
        # Calculate individual similarities
        tfidf_sim = self.calculate_tfidf_similarity(text1, text2)
        ngram_sim = self.calculate_ngram_similarity(text1, text2, n=5)
        lcs_sim = self.calculate_lcs_similarity(text1, text2)

        # Weighted combination
        combined = (
                tfidf_sim * 0.45 +
                ngram_sim * 0.35 +
                lcs_sim * 0.20
        )

        return {
            'tfidf': tfidf_sim,
            'ngram': ngram_sim,
            'lcs': lcs_sim,
            'combined': combined
        }

    def search_web(self, query: str, num_results: Optional[int] = None) -> List[Dict]:
        if not self.searchapi_key:
            print("SearchAPI.io API key not provided. Skipping web search.")
            return []

        if num_results is None:
            num_results = self.config.MAX_RESULTS_WEB

        url = 'https://www.searchapi.io/api/v1/search'
        params = {
            'api_key': self.searchapi_key,
            'engine': 'google',
            'q': query,
            'num': num_results
        }

        try:
            response = requests.get(url, params=params, timeout=self.config.API_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            results = []
            organic_results = data.get('organic_results', [])

            for item in organic_results[:num_results]:
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'Web Search'
                })

            return results

        except Exception as e:
            print(f"SearchAPI.io web search error: {e}")
            return []

    def search_scholar(self, query: str, num_results: Optional[int] = None) -> List[Dict]:
        if not self.searchapi_key:
            return []

        if num_results is None:
            num_results = self.config.MAX_RESULTS_SCHOLAR

        url = 'https://www.searchapi.io/api/v1/search'
        params = {
            'api_key': self.searchapi_key,
            'engine': 'google_scholar',
            'q': query,
            'num': num_results
        }

        try:
            response = requests.get(url, params=params, timeout=self.config.API_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            results = []
            organic_results = data.get('organic_results', [])

            for item in organic_results[:num_results]:
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'Google Scholar'
                })

            return results

        except Exception as e:
            print(f"SearchAPI.io scholar search error: {e}")
            return []

    def search_arxiv(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        if max_results is None:
            max_results = self.config.MAX_RESULTS_ARXIV

        base_url = 'http://export.arxiv.org/api/query'
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results
        }

        try:
            response = requests.get(base_url, params=params, timeout=self.config.API_TIMEOUT)
            response.raise_for_status()

            if not BS4_AVAILABLE:
                return []

            soup = BeautifulSoup(response.text, 'xml')
            entries = soup.find_all('entry')

            results = []
            for entry in entries:
                title = entry.find('title').text.strip() if entry.find('title') else ''
                summary = entry.find('summary').text.strip() if entry.find('summary') else ''
                link = entry.find('id').text.strip() if entry.find('id') else ''

                results.append({
                    'title': title,
                    'url': link,
                    'snippet': summary[:500],
                    'source': 'ArXiv'
                })

            return results

        except Exception as e:
            print(f"ArXiv search error: {e}")
            return []

    def detect_plagiarism(self, text: str, use_web_search: bool = False) -> Dict:
        if not text or len(text.strip()) < 50:
            return {
                'error': 'Text too short. Please provide at least 50 characters.',
                'matches': []
            }

        print(f"\n{'=' * 60}")
        print("PLAGIARISM DETECTION STARTED")
        print(f"{'=' * 60}")
        print(f"Input text length: {len(text)} characters")
        print(f"Word count: {len(text.split())} words")

        all_matches = []

        # 1. Search knowledge base
        print("\n[1/4] Searching knowledge base...")
        for doc in self.knowledge_base:
            similarities = self.calculate_combined_similarity(text, doc['content'])

            if similarities['combined'] > self.config.SIMILARITY_THRESHOLD:
                all_matches.append({
                    'title': doc['title'],
                    'authors': doc['authors'],
                    'year': doc['year'],
                    'venue': doc['venue'],
                    'url': doc['url'],
                    'snippet': doc['content'][:400],
                    'similarity': similarities['combined'],
                    'details': similarities,
                    'source': 'Knowledge Base'
                })
                print(f"  ✓ Match found: {doc['title']} ({similarities['combined'] * 100:.1f}%)")

        # 2. Search ArXiv
        print("\n[2/4] Searching ArXiv...")
        search_query = ' '.join(text.split()[:15])
        arxiv_results = self.search_arxiv(search_query)

        for result in arxiv_results:
            similarities = self.calculate_combined_similarity(text, result['snippet'])

            if similarities['combined'] > self.config.SIMILARITY_THRESHOLD:
                all_matches.append({
                    'title': result['title'],
                    'url': result['url'],
                    'snippet': result['snippet'][:400],
                    'similarity': similarities['combined'],
                    'details': similarities,
                    'source': 'ArXiv'
                })
                print(f"  ✓ Match found: {result['title'][:60]}... ({similarities['combined'] * 100:.1f}%)")

        # 3. Search Web (if enabled)
        if use_web_search and self.searchapi_key:
            print("\n[3/4] Searching the web...")
            web_results = self.search_web(search_query)

            for result in web_results:
                similarities = self.calculate_combined_similarity(text, result['snippet'])

                if similarities['combined'] > self.config.SIMILARITY_THRESHOLD:
                    all_matches.append({
                        'title': result['title'],
                        'url': result['url'],
                        'snippet': result['snippet'],
                        'similarity': similarities['combined'],
                        'details': similarities,
                        'source': 'Web Search'
                    })
                    print(f"  ✓ Match found: {result['title'][:60]}... ({similarities['combined'] * 100:.1f}%)")
        else:
            print("\n[3/4] Web search skipped (disabled or no API key)")

        # 4. Search Google Scholar (if enabled)
        if use_web_search and self.searchapi_key:
            print("\n[4/4] Searching Google Scholar...")
            scholar_results = self.search_scholar(search_query)

            for result in scholar_results:
                similarities = self.calculate_combined_similarity(text, result['snippet'])

                if similarities['combined'] > self.config.SIMILARITY_THRESHOLD:
                    all_matches.append({
                        'title': result['title'],
                        'url': result['url'],
                        'snippet': result['snippet'],
                        'similarity': similarities['combined'],
                        'details': similarities,
                        'source': 'Google Scholar'
                    })
                    print(f"  ✓ Match found: {result['title'][:60]}... ({similarities['combined'] * 100:.1f}%)")
        else:
            print("\n[4/4] Google Scholar search skipped (disabled or no API key)")

        # Sort by similarity
        all_matches.sort(key=lambda x: x['similarity'], reverse=True)

        # Calculate overall score
        if all_matches:
            top_3_avg = np.mean([m['similarity'] for m in all_matches[:3]])
            overall_score = all_matches[0]['similarity'] * 0.7 + top_3_avg * 0.3
        else:
            overall_score = 0.0

        # Generate report
        print(f"\n{'=' * 60}")
        print("DETECTION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Overall Plagiarism Score: {overall_score * 100:.1f}%")
        print(f"Total Matches Found: {len(all_matches)}")

        if overall_score >= 0.6:
            verdict = "CRITICAL - Likely Plagiarized"
        elif overall_score >= 0.4:
            verdict = "HIGH RISK - Review Required"
        elif overall_score >= 0.2:
            verdict = "MODERATE RISK - Some Similarities"
        else:
            verdict = "LOW RISK - Appears Original"

        print(f"Verdict: {verdict}")
        print(f"{'=' * 60}\n")

        return {
            'overall_score': overall_score,
            'verdict': verdict,
            'matches': all_matches[:10],
            'total_matches': len(all_matches),
            'word_count': len(text.split())
        }

    def print_detailed_report(self, results: Dict):
        """Print a detailed plagiarism report"""
        if 'error' in results:
            print(f"\nError: {results['error']}")
            return

        print("\n" + "=" * 80)
        print("DETAILED PLAGIARISM REPORT")
        print("=" * 80)

        print(f"\nOverall Score: {results['overall_score'] * 100:.1f}%")
        print(f"Verdict: {results['verdict']}")
        print(f"Total Matches: {results['total_matches']}")
        print(f"Input Word Count: {results['word_count']}")

        if results['matches']:
            print(f"\n{'=' * 80}")
            print("TOP MATCHES")
            print(f"{'=' * 80}\n")

            for i, match in enumerate(results['matches'], 1):
                print(f"[{i}] {match['title']}")
                print(f"    Similarity: {match['similarity'] * 100:.1f}%")
                print(f"    Source: {match['source']}")
                if 'authors' in match:
                    print(f"    Authors: {match['authors']} ({match.get('year', 'N/A')})")
                print(f"    URL: {match['url']}")

                # Print similarity breakdown
                details = match['details']
                print(f"    Details:")
                print(f"      - TF-IDF: {details['tfidf'] * 100:.1f}%")
                print(f"      - N-gram: {details['ngram'] * 100:.1f}%")
                print(f"      - LCS: {details['lcs'] * 100:.1f}%")

                print(f"    Snippet: {match['snippet'][:200]}...")
                print()
        else:
            print("\n✓ No significant matches found. Text appears original.")

        print("=" * 80)


def main():
    # Initialize detector with configuration from .env
    config = Config()
    detector = PlagiarismDetector(config)

    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("\nEnter your text to check for plagiarism (or 'quit' to exit):")
    print("-" * 80)

    user_text = input("\nYour text: ").strip()

    if user_text and user_text.lower() != 'quit':
        results = detector.detect_plagiarism(user_text, use_web_search=True)
        detector.print_detailed_report(results)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()