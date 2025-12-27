import sys

print("Starting comprehensive database check...")

try:
    import chromadb
    print("✓ ChromaDB imported successfully")
except Exception as e:
    print(f"✗ Error importing ChromaDB: {e}")
    sys.exit(1)

try:
    client = chromadb.PersistentClient(path='../knowledge_base/vector_db')
    collection = client.get_collection("langchain")
    print("✓ Found 'langchain' collection")
except Exception as e:
    print(f"✗ Error getting collection: {e}")
    sys.exit(1)

try:
    count = collection.count()
    print(f"✓ Total documents in database: {count}")
except Exception as e:
    print(f"✗ Error counting documents: {e}")
    sys.exit(1)

try:
    # Get ALL documents to find all unique sources
    print("\nRetrieving ALL documents to catalog sources...")
    print("(This may take a moment...)")
    
    results = collection.get(
        limit=count,  # Get ALL documents, not just 100!
        include=['metadatas']
    )
    
    print(f"✓ Retrieved {len(results['metadatas'])} documents")
    
    # Count ALL sources
    sources = {}
    for metadata in results['metadatas']:
        source = metadata.get('source', 'Unknown')
        sources[source] = sources.get(source, 0) + 1
    
    print("\n" + "="*70)
    print("ALL SOURCES IN DATABASE:")
    print("="*70)
    
    # Group by folder
    import os
    from collections import defaultdict
    
    folders = defaultdict(list)
    for source, count in sources.items():
        folder = os.path.dirname(source)
        basename = os.path.basename(source)
        folders[folder].append((basename, count))
    
    # Display by folder
    for folder in sorted(folders.keys()):
        folder_name = folder if folder else "Root"
        total_in_folder = sum(count for _, count in folders[folder])
        print(f"\n{folder_name}/ ({total_in_folder} chunks):")
        for filename, count in sorted(folders[folder]):
            print(f"  {filename}: {count} chunks")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total unique PDF files: {len(sources)}")
    print(f"Total chunks: {count}")
    
    # Check for Partnership Letters specifically
    partnership_sources = [s for s in sources.keys() if 'Partnership' in s or 'partnership' in s.lower()]
    
    print("\n" + "="*70)
    if partnership_sources:
        print("✓ PARTNERSHIP LETTERS FOUND!")
        print("="*70)
        for source in sorted(partnership_sources):
            basename = os.path.basename(source)
            chunk_count = sources[source]
            print(f"  {basename}: {chunk_count} chunks")
    else:
        print("✗ PARTNERSHIP LETTERS NOT FOUND")
        print("="*70)
        print("\nLikely issue: Files exist but weren't processed into database.")
        print("Check if process_documents.py completed successfully.")
    
except Exception as e:
    print(f"✗ Error retrieving documents: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Comprehensive database check complete!")