import chromadb
import os

client = chromadb.PersistentClient(path='../knowledge_base/vector_db')
collection = client.get_collection("langchain")

# Get all documents
print("Searching for 'Sanborn' mentions across all documents...")
results = collection.get(
    limit=8518,  # All docs
    include=['metadatas', 'documents']
)

# Find documents mentioning Sanborn
sanborn_mentions = []
for i, doc_content in enumerate(results['documents']):
    if 'sanborn' in doc_content.lower():
        metadata = results['metadatas'][i]
        source = metadata.get('source', 'Unknown')
        source_name = os.path.basename(source)
        
        # Count how many times Sanborn appears
        count = doc_content.lower().count('sanborn')
        
        # Extract snippet around first mention
        lower_content = doc_content.lower()
        idx = lower_content.find('sanborn')
        snippet_start = max(0, idx - 100)
        snippet_end = min(len(doc_content), idx + 200)
        snippet = doc_content[snippet_start:snippet_end]
        
        sanborn_mentions.append({
            'source': source_name,
            'count': count,
            'snippet': snippet,
            'chunk_length': len(doc_content)
        })

print(f"\nFound {len(sanborn_mentions)} chunks mentioning 'Sanborn'")
print("="*80)

# Group by source
from collections import defaultdict
by_source = defaultdict(list)
for mention in sanborn_mentions:
    by_source[mention['source']].append(mention)

# Show results grouped by source
for source_name, mentions in sorted(by_source.items()):
    total_mentions = sum(m['count'] for m in mentions)
    print(f"\n{source_name}:")
    print(f"  Chunks with 'Sanborn': {len(mentions)}")
    print(f"  Total 'Sanborn' mentions: {total_mentions}")
    
    # Show first snippet from this source
    if mentions:
        print(f"  Sample snippet:")
        print(f"    ...{mentions[0]['snippet']}...")
    print("-" * 80)

print("\n" + "="*80)
print("DETAILED VIEW - First 5 Sanborn chunks:")
print("="*80)

for i, mention in enumerate(sanborn_mentions[:5], 1):
    print(f"\nChunk {i}:")
    print(f"  Source: {mention['source']}")
    print(f"  Sanborn count: {mention['count']}")
    print(f"  Chunk size: {mention['chunk_length']} chars")
    print(f"  Snippet:")
    print(f"    {mention['snippet']}")
    print("-" * 80)