import chromadb

client = chromadb.PersistentClient(path='../knowledge_base/vector_db')
collection = client.get_collection("langchain")

# Get all documents
print("Searching for 1962 documents...")
results = collection.get(
    limit=8518,  # All docs
    include=['metadatas', 'documents']
)

# Find documents with 1962 in source
found_1962 = []
for i, metadata in enumerate(results['metadatas']):
    source = metadata.get('source', '')
    if '1962' in source:
        found_1962.append({
            'index': i,
            'metadata': metadata,
            'preview': results['documents'][i][:150]
        })

print(f"\nFound {len(found_1962)} documents with '1962' in source path")

if found_1962:
    print("\n" + "="*70)
    print("SAMPLE 1962 DOCUMENTS (First 5):")
    print("="*70)
    
    for i, doc in enumerate(found_1962[:5], 1):
        print(f"\nDocument {i}:")
        print(f"  Full metadata: {doc['metadata']}")
        print(f"  Source field: '{doc['metadata'].get('source', 'NO SOURCE FIELD')}'")
        print(f"  Preview: {doc['preview']}...")
        print("-" * 70)
    
    # Check what the exact source path looks like
    first_source = found_1962[0]['metadata'].get('source', '')
    print(f"\n" + "="*70)
    print("METADATA ANALYSIS:")
    print("="*70)
    print(f"Exact source string: '{first_source}'")
    print(f"Length: {len(first_source)}")
    print(f"Contains '1962': {'1962' in first_source}")
    print(f"Contains 'Partnership': {'Partnership' in first_source}")
    
    # Show path components
    import os
    print(f"\nPath components:")
    print(f"  Basename: {os.path.basename(first_source)}")
    print(f"  Dirname: {os.path.dirname(first_source)}")
    
else:
    print("\n❌ NO DOCUMENTS WITH '1962' FOUND IN DATABASE!")
    print("\nThis means the metadata doesn't contain '1962' in the source field.")
    print("Let's check a sample of what IS in the database:")
    
    print("\nFirst 10 document metadata samples:")
    for i in range(min(10, len(results['metadatas']))):
        print(f"\n{i+1}. {results['metadatas'][i]}")