import json
from collections import Counter

try:
    with open('expanded_motorcycle_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("✅ JSON structure is valid!")
    print(f"📊 Total entries: {len(data)}")
    
    # Count by category
    categories = [item.get('category', 'Unknown') for item in data]
    category_counts = Counter(categories)
    
    print("\n📋 Breakdown by category:")
    for category, count in sorted(category_counts.items()):
        print(f"   {category}: {count} problems")
    
    # Count by ID prefix
    id_prefixes = [item.get('id', '').split('_')[0] for item in data if item.get('id')]
    prefix_counts = Counter(id_prefixes)
    
    print("\n🔍 Breakdown by ID prefix:")
    for prefix, count in sorted(prefix_counts.items()):
        print(f"   {prefix}: {count} entries")
    
    print(f"\n🎯 Total problems in dataset: {len(data)}")
    
except json.JSONDecodeError as e:
    print(f"❌ JSON validation failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")