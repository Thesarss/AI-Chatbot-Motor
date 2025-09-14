import json
import logging
from datetime import datetime
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveDatasetMerger:
    """Merge and clean multiple datasets for better training quality"""
    
    def __init__(self):
        self.merged_data = []
        self.seen_questions = set()
        self.quality_filters = {
            'min_question_length': 10,
            'min_answer_length': 20,
            'max_question_length': 200,
            'max_answer_length': 1000
        }
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove weird characters and symbols
        text = re.sub(r'[^\w\s\.,\?\!\-\(\)\:]+', '', text)
        
        # Fix common Indonesian text issues
        text = re.sub(r'\b(ga|gak|gk)\b', 'tidak', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(gimana|gmn)\b', 'bagaimana', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(kenapa|knp)\b', 'mengapa', text, flags=re.IGNORECASE)
        
        return text
    
    def is_quality_entry(self, entry):
        """Check if entry meets quality standards"""
        question = entry.get('question', '').strip()
        answer = entry.get('answer', '').strip()
        
        # Length checks
        if (len(question) < self.quality_filters['min_question_length'] or 
            len(question) > self.quality_filters['max_question_length']):
            return False
            
        if (len(answer) < self.quality_filters['min_answer_length'] or 
            len(answer) > self.quality_filters['max_answer_length']):
            return False
        
        # Content quality checks
        if not re.search(r'[a-zA-Z]', question) or not re.search(r'[a-zA-Z]', answer):
            return False
            
        # Check for nonsensical patterns
        nonsense_patterns = [
            r'\b[a-zA-Z]{1,2}\b.*\b[a-zA-Z]{1,2}\b.*\b[a-zA-Z]{1,2}\b',  # Too many short words
            r'[a-zA-Z]{20,}',  # Very long words (likely corrupted)
            r'(.)\1{4,}',  # Repeated characters
            r'\d{10,}',  # Long numbers
        ]
        
        for pattern in nonsense_patterns:
            if re.search(pattern, answer):
                return False
        
        return True
    
    def normalize_question(self, question):
        """Normalize question for duplicate detection"""
        # Convert to lowercase and remove punctuation
        normalized = re.sub(r'[^\w\s]', '', question.lower())
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        return normalized
    
    def load_dataset(self, file_path):
        """Load and process a dataset file"""
        logger.info(f"Loading dataset from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            valid_entries = 0
            duplicate_entries = 0
            
            for entry in data:
                # Clean the entry
                cleaned_entry = {
                    'question': self.clean_text(entry.get('question', '')),
                    'answer': self.clean_text(entry.get('answer', '')),
                    'category': entry.get('category', 'general'),
                    'subcategory': entry.get('subcategory', ''),
                    'type': entry.get('type', 'qa'),
                    'source': file_path
                }
                
                # Quality check
                if not self.is_quality_entry(cleaned_entry):
                    continue
                
                # Duplicate check
                normalized_q = self.normalize_question(cleaned_entry['question'])
                if normalized_q in self.seen_questions:
                    duplicate_entries += 1
                    continue
                
                self.seen_questions.add(normalized_q)
                self.merged_data.append(cleaned_entry)
                valid_entries += 1
            
            logger.info(f"Loaded {valid_entries} valid entries from {file_path}")
            logger.info(f"Skipped {duplicate_entries} duplicate entries")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    def balance_categories(self):
        """Balance dataset by categories"""
        category_counts = defaultdict(list)
        
        # Group by category
        for entry in self.merged_data:
            category_counts[entry['category']].append(entry)
        
        # Log category distribution
        logger.info("Category distribution:")
        for category, entries in category_counts.items():
            logger.info(f"  {category}: {len(entries)} entries")
        
        # Optional: Limit categories with too many entries
        max_per_category = 200
        balanced_data = []
        
        for category, entries in category_counts.items():
            if len(entries) > max_per_category:
                # Take the first max_per_category entries (could be randomized)
                balanced_data.extend(entries[:max_per_category])
                logger.info(f"Limited {category} to {max_per_category} entries")
            else:
                balanced_data.extend(entries)
        
        self.merged_data = balanced_data
        logger.info(f"Final balanced dataset: {len(self.merged_data)} entries")
    
    def save_merged_dataset(self, output_path):
        """Save the merged and cleaned dataset"""
        logger.info(f"Saving merged dataset to {output_path}")
        
        # Add metadata
        dataset_info = {
            'total_entries': len(self.merged_data),
            'created_at': datetime.now().isoformat(),
            'quality_filters': self.quality_filters,
            'sources': list(set(entry['source'] for entry in self.merged_data))
        }
        
        # Save dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.merged_data, f, ensure_ascii=False, indent=2)
        
        # Save metadata
        metadata_path = output_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset saved: {len(self.merged_data)} entries")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return output_path, dataset_info

def main():
    """Main function to merge datasets"""
    print("=== COMPREHENSIVE DATASET MERGER ===")
    
    # Initialize merger
    merger = ComprehensiveDatasetMerger()
    
    # Dataset files to merge
    datasets_to_merge = [
        'adaptive_motorcycle_dataset.json',
        'merged_motorcycle_dataset.json',
        'ultra_clean_motorcycle_dataset_fixed.json'
    ]
    
    # Load and merge datasets
    for dataset_file in datasets_to_merge:
        try:
            merger.load_dataset(dataset_file)
        except Exception as e:
            logger.warning(f"Could not load {dataset_file}: {e}")
    
    # Balance categories
    merger.balance_categories()
    
    # Save merged dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"comprehensive_motorcycle_dataset_{timestamp}.json"
    
    dataset_path, metadata = merger.save_merged_dataset(output_path)
    
    print(f"\n✅ DATASET MERGER COMPLETED!")
    print(f"📁 Output: {dataset_path}")
    print(f"📊 Total Entries: {metadata['total_entries']}")
    print(f"📋 Sources: {len(metadata['sources'])}")
    print(f"🔧 Quality Filters Applied: Yes")
    print(f"🚫 Duplicates Removed: Yes")
    print(f"⚖️ Categories Balanced: Yes")
    
    return dataset_path

if __name__ == "__main__":
    main()