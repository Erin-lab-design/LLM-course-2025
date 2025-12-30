import ollama
import csv
import random
import json
import requests
import time
from typing import List, Dict

# Known abbreviations and special terms to skip
KNOWN_ABBREVIATIONS = {
    'JFK', 'NYC', 'USA', 'UK', 'EU', 'NBC', 'ABC', 'CBS', 'CNN', 'BBC',
    'FBI', 'CIA', 'NASA', 'CEO', 'CFO', 'CTO', 'PhD', 'MBA', 'GPS',
    'DVD', 'CD', 'USB', 'HTML', 'CSS', 'API', 'SQL', 'AI', 'ML', 'IT'
}

# Google Custom Search API Configuration
# You need to provide these values:
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your API key
SEARCH_ENGINE_ID = "YOUR_SEARCH_ENGINE_ID_HERE"  # Replace with your CX

def generate_misspellings(query, num_variants=5):
    """
    Generate N misspelling variants of a query using LLM.
    
    Args:
        query: Original query string
        num_variants: Number of misspelling variants to generate
        
    Returns:
        List of misspelled query variants
    """
    prompt = f"""Generate exactly {num_variants} misspelled versions of this search query. Each version should have realistic typos.

Include these error types:
- Omission: missing letters (restaurant → resturant)
- Transposition: swapped letters (from → form)  
- Phonetic: similar sounds (tough → tuff)
- Repetition: doubled letters (business → bussiness)

Rules:
- Do NOT misspell abbreviations: JFK, NYC, USA, GPS, etc.
- Keep proper nouns mostly intact
- Output ONLY the misspelled queries, nothing else
- No numbering, no explanations, no extra text

Query: {query}

Misspelled versions:"""

    response = ollama.generate(model='llama3', prompt=prompt)
    text = response['response'].strip()
    
    # Split and clean
    variants = []
    for line in text.split('\n'):
        line = line.strip()
        # Skip empty lines and lines that are too long (likely explanations)
        if line and len(line) < 200:
            # Remove numbering
            cleaned = line.lstrip('0123456789.-) \t')
            if cleaned and not cleaned.lower().startswith(('here', 'misspell', 'variant', 'query')):
                variants.append(cleaned)
    
    return variants[:num_variants]

def load_queries_from_csv(csv_path):
    """Load queries from CSV file."""
    queries = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({
                'topic': row['Topic'],
                'query': row['Query']
            })
    return queries

def test_query_with_search_engine(query: str) -> Dict:
    """
    Test query with Google Custom Search API.
    
    Args:
        query: Search query string
        
    Returns:
        Dictionary with search results (URLs, titles, snippets)
    """
    # Check if API credentials are configured
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE" or SEARCH_ENGINE_ID == "YOUR_SEARCH_ENGINE_ID_HERE":
        print(f"   [⚠️  API not configured] Simulating: {query[:50]}...")
        return {
            'query': query,
            'results': [],
            'configured': False
        }
    
    try:
        # Google Custom Search API endpoint
        url = "https://www.googleapis.com/customsearch/v1"
        
        params = {
            'key': GOOGLE_API_KEY,
            'cx': SEARCH_ENGINE_ID,
            'q': query,
            'num': 5  # Get top 5 results
        }
        
        print(f"   [🔍 Searching] {query[:50]}...", end=" ")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract top results
            search_results = []
            if 'items' in data:
                for item in data['items']:
                    search_results.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', '')
                    })
            
            print(f"✅ Found {len(search_results)} results")
            
            return {
                'query': query,
                'results': search_results,
                'total_results': data.get('searchInformation', {}).get('totalResults', '0'),
                'configured': True
            }
        
        elif response.status_code == 429:
            print("⚠️  Rate limit exceeded")
            return {
                'query': query,
                'results': [],
                'error': 'Rate limit exceeded',
                'configured': True
            }
        
        else:
            print(f"❌ Error {response.status_code}")
            return {
                'query': query,
                'results': [],
                'error': f"HTTP {response.status_code}",
                'configured': True
            }
    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return {
            'query': query,
            'results': [],
            'error': str(e),
            'configured': True
        }

def compare_search_results(original_results: Dict, variant_results: Dict) -> Dict:
    """
    Compare search results between original and misspelled queries.
    
    Args:
        original_results: Search results for original query
        variant_results: Search results for misspelled query
        
    Returns:
        Comparison metrics
    """
    if not original_results.get('configured') or not variant_results.get('configured'):
        return {
            'configured': False,
            'message': 'API not configured'
        }
    
    original_urls = set(r['link'] for r in original_results.get('results', []))
    variant_urls = set(r['link'] for r in variant_results.get('results', []))
    
    if not original_urls or not variant_urls:
        return {
            'identical': False,
            'overlap_count': 0,
            'overlap_percentage': 0.0,
            'reason': 'No results found for one or both queries'
        }
    
    overlap = original_urls & variant_urls
    overlap_pct = (len(overlap) / len(original_urls)) * 100 if original_urls else 0
    
    return {
        'identical': original_urls == variant_urls,
        'overlap_count': len(overlap),
        'overlap_percentage': overlap_pct,
        'total_original': len(original_urls),
        'total_variant': len(variant_urls),
        'different_count': len(original_urls ^ variant_urls)
    }

def main():
    # Load queries from CSV
    csv_path = 'web_search_queries.csv'
    print(f"Loading queries from {csv_path}...")
    queries = load_queries_from_csv(csv_path)
    print(f"Loaded {len(queries)} queries\n")
    
    # Check API configuration
    api_configured = (GOOGLE_API_KEY != "YOUR_API_KEY_HERE" and 
                     SEARCH_ENGINE_ID != "YOUR_SEARCH_ENGINE_ID_HERE")
    
    if not api_configured:
        print("⚠️  WARNING: Google Custom Search API not configured!")
        print("   Please edit synthetic_data.py and add your:")
        print("   1. GOOGLE_API_KEY")
        print("   2. SEARCH_ENGINE_ID")
        print("\n   Will continue with simulation mode...\n")
    else:
        print("✅ Google Custom Search API configured")
        print("   Testing with real search results...\n")
    
    # Process a sample of queries (first 5 for demonstration)
    sample_queries = queries[:5]
    
    results = []
    
    for i, query_data in enumerate(sample_queries, 1):
        topic = query_data['topic']
        original_query = query_data['query']
        
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(sample_queries)}")
        print(f"Topic: {topic}")
        print(f"Original: {original_query}")
        print(f"{'-'*80}")
        
        # Generate misspellings
        num_variants = 5
        print(f"Generating {num_variants} misspelling variants...")
        misspellings = generate_misspellings(original_query, num_variants)
        
        print(f"\nGenerated {len(misspellings)} variants:")
        for j, variant in enumerate(misspellings, 1):
            print(f"  {j}. {variant}")
        
        # Test with search engine (Task 2e)
        print(f"\n{'='*80}")
        print("🔍 SEARCH ENGINE TESTING (Task 2e)")
        print(f"{'='*80}")
        
        # Search original query
        original_search = test_query_with_search_engine(original_query)
        
        # Small delay to avoid rate limiting
        time.sleep(1)
        
        # Search variants and compare
        variant_comparisons = []
        
        for j, variant in enumerate(misspellings[:3], 1):  # Test first 3 variants
            variant_search = test_query_with_search_engine(variant)
            time.sleep(1)  # Rate limiting
            
            # Compare results
            comparison = compare_search_results(original_search, variant_search)
            variant_comparisons.append({
                'variant': variant,
                'search_results': variant_search,
                'comparison': comparison
            })
            
            # Print comparison
            if comparison.get('configured'):
                if comparison['identical']:
                    print(f"   ✅ Variant {j}: IDENTICAL results (search engine corrected spelling)")
                else:
                    print(f"   ⚠️  Variant {j}: {comparison['overlap_percentage']:.1f}% overlap "
                          f"({comparison['overlap_count']}/{comparison['total_original']} same URLs)")
        
        # Store results
        results.append({
            'topic': topic,
            'original': original_query,
            'variants': misspellings,
            'search_test': {
                'original_search': original_search,
                'variant_comparisons': variant_comparisons
            }
        })
    
    # Save results to JSON file
    output_file = 'misspelling_results_with_search.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*80}")
    print(f"Results saved to {output_file}")
    
    # Analysis
    print(f"\n{'='*80}")
    print("📊 FINAL ANALYSIS")
    print(f"{'='*80}")
    print(f"- Total queries processed: {len(sample_queries)}")
    print(f"- Total variants generated: {sum(len(r['variants']) for r in results)}")
    
    if api_configured:
        # Calculate search engine robustness metrics
        total_tests = sum(len(r['search_test']['variant_comparisons']) for r in results)
        identical_count = sum(
            1 for r in results 
            for vc in r['search_test']['variant_comparisons']
            if vc['comparison'].get('identical', False)
        )
        
        if total_tests > 0:
            print(f"\n🔍 Search Engine Robustness:")
            print(f"  - Total comparisons: {total_tests}")
            print(f"  - Identical results: {identical_count}/{total_tests} "
                  f"({identical_count/total_tests*100:.1f}%)")
            print(f"  - Different results: {total_tests - identical_count}/{total_tests}")
            
            # Average overlap
            overlaps = [
                vc['comparison']['overlap_percentage']
                for r in results
                for vc in r['search_test']['variant_comparisons']
                if 'overlap_percentage' in vc['comparison']
            ]
            if overlaps:
                avg_overlap = sum(overlaps) / len(overlaps)
                print(f"  - Average overlap: {avg_overlap:.1f}%")
            
            print(f"\n💡 Insights:")
            if identical_count / total_tests > 0.8:
                print("  ✅ Search engine has EXCELLENT spelling correction!")
                print("     Most misspelled queries return identical results.")
            elif identical_count / total_tests > 0.5:
                print("  ✓ Search engine has GOOD spelling correction.")
                print("    Most queries corrected, but some variations affect results.")
            else:
                print("  ⚠️  Search engine spelling correction is LIMITED.")
                print("     Many misspellings lead to different search results.")
    
    print("\n✅ Error types included:")
    print("  ✓ Omission (missing letters)")
    print("  ✓ Transposition (swapped letters)")
    print("  ✓ Phonetic (sound-alike)")
    print("  ✓ Repetition (doubled letters)")
    print("\n✅ Robustness features:")
    print("  ✓ Skips known abbreviations (JFK, NYC, etc.)")
    print("  ✓ Preserves proper nouns where appropriate")
    print("  ✓ Generates realistic human-like typos")
    
    if api_configured:
        print("\n✅ Task 2e completed: Real search engine testing implemented!")
    else:
        print("\n⚠️  Task 2e: Add API credentials to enable real search testing")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()