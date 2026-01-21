import argparse
from thesorimed_matcher import ThesorimedMatcher

def main():
    parser = argparse.ArgumentParser(description="Interactive Thesorimed Matcher Inspector")
    parser.add_argument("--db", default="../../data/thesorimed/THESORIMED_SQ3", help="Path to DB")
    args = parser.parse_args()

    print(f"Loading Matcher with DB: {args.db} ...")
    try:
        matcher = ThesorimedMatcher(args.db)
    except Exception as e:
        print(f"Failed to load matcher: {e}")
        return

    print("\n--- Thesorimed Fuzzy Matcher Interactive Mode ---")
    print("Type a drug name (or 'q' to quit). Example: 'Doliprane', 'Paracytamol'")
    
    while True:
        query = input("\nQuery > ").strip()
        if query.lower() in ['q', 'quit', 'exit']:
            break
        if not query:
            continue
            
        results = matcher.match(query, top_k=5)
        
        if not results:
            print("  No matches found (score < 60%)")
        else:
            print(f"  Found {len(results)} candidates:")
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['score']:.1f}%] {r['name']} (CIS: {r['cis']})")

if __name__ == "__main__":
    main()
