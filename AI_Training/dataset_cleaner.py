import json
import re
from difflib import SequenceMatcher
from collections import defaultdict
from datetime import datetime
import hashlib

class DatasetCleaner:
    def __init__(self):
        self.similarity_threshold = 0.85  # Threshold untuk mendeteksi duplikasi
        self.min_question_length = 10
        self.min_answer_length = 20
        self.max_question_length = 200
        self.max_answer_length = 1000
        
    def normalize_text(self, text):
        """Normalize text untuk perbandingan"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation for comparison
        text = re.sub(r'[^\w\s]', '', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def generate_content_hash(self, question, answer):
        """Generate hash untuk content unik"""
        content = self.normalize_text(question + " " + answer)
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_valid_entry(self, entry):
        """Validate entry quality"""
        if not isinstance(entry, dict):
            return False, "Entry bukan dictionary"
        
        # Check required fields
        required_fields = ['question', 'answer']
        for field in required_fields:
            if field not in entry or not entry[field]:
                return False, f"Field '{field}' missing atau kosong"
        
        question = entry['question'].strip()
        answer = entry['answer'].strip()
        
        # Check length constraints
        if len(question) < self.min_question_length:
            return False, f"Pertanyaan terlalu pendek ({len(question)} < {self.min_question_length})"
        
        if len(question) > self.max_question_length:
            return False, f"Pertanyaan terlalu panjang ({len(question)} > {self.max_question_length})"
        
        if len(answer) < self.min_answer_length:
            return False, f"Jawaban terlalu pendek ({len(answer)} < {self.min_answer_length})"
        
        if len(answer) > self.max_answer_length:
            return False, f"Jawaban terlalu panjang ({len(answer)} > {self.max_answer_length})"
        
        # Check for placeholder text
        placeholders = ['{', '}', 'lorem ipsum', 'placeholder', 'example']
        for placeholder in placeholders:
            if placeholder in question.lower() or placeholder in answer.lower():
                return False, f"Mengandung placeholder: {placeholder}"
        
        # Check for repetitive content
        if self.is_repetitive(question) or self.is_repetitive(answer):
            return False, "Content terlalu repetitif"
        
        return True, "Valid"
    
    def is_repetitive(self, text, max_repeat=3):
        """Check if text is too repetitive"""
        words = text.lower().split()
        if len(words) < 5:
            return False
        
        # Check for repeated words
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1
            if word_count[word] > max_repeat and len(word) > 3:
                return True
        
        # Check for repeated phrases
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if text.lower().count(phrase) > 2:
                return True
        
        return False
    
    def find_duplicates(self, dataset):
        """Find duplicate entries"""
        print("🔍 Mencari duplikasi...")
        
        duplicates = []
        seen_hashes = set()
        similar_groups = []
        
        for i, entry in enumerate(dataset):
            # Check exact duplicates by hash
            content_hash = self.generate_content_hash(entry['question'], entry['answer'])
            if content_hash in seen_hashes:
                duplicates.append((i, "exact_duplicate", content_hash))
                continue
            seen_hashes.add(content_hash)
            
            # Check similarity with previous entries
            for j in range(max(0, i-100), i):  # Check last 100 entries for efficiency
                if j >= len(dataset):
                    continue
                    
                similarity = self.calculate_similarity(
                    entry['question'], 
                    dataset[j]['question']
                )
                
                if similarity >= self.similarity_threshold:
                    duplicates.append((i, "similar_duplicate", f"similar to index {j} (similarity: {similarity:.3f})"))
                    break
        
        return duplicates
    
    def clean_dataset(self, dataset):
        """Clean dataset by removing invalid and duplicate entries"""
        print(f"🧹 Membersihkan dataset dengan {len(dataset)} entries...")
        
        cleaned_dataset = []
        removed_entries = []
        
        # Step 1: Validate entries
        print("📋 Step 1: Validating entries...")
        valid_entries = []
        for i, entry in enumerate(dataset):
            is_valid, reason = self.is_valid_entry(entry)
            if is_valid:
                valid_entries.append(entry)
            else:
                removed_entries.append((i, "invalid", reason))
        
        print(f"✅ Valid entries: {len(valid_entries)}")
        print(f"❌ Invalid entries removed: {len(removed_entries)}")
        
        # Step 2: Remove duplicates
        print("📋 Step 2: Removing duplicates...")
        duplicates = self.find_duplicates(valid_entries)
        
        # Create set of indices to remove
        indices_to_remove = set(dup[0] for dup in duplicates)
        
        # Keep only non-duplicate entries
        for i, entry in enumerate(valid_entries):
            if i not in indices_to_remove:
                cleaned_dataset.append(entry)
        
        print(f"🔄 Duplicate entries removed: {len(duplicates)}")
        print(f"✨ Final clean dataset: {len(cleaned_dataset)} entries")
        
        return cleaned_dataset, removed_entries + duplicates
    
    def merge_with_existing(self, new_dataset, existing_file=None):
        """Merge new dataset with existing one"""
        if existing_file and existing_file != "merged_motorcycle_dataset.json":
            try:
                with open(existing_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                print(f"📂 Loaded existing dataset: {len(existing_data)} entries")
                
                # Combine datasets
                combined = existing_data + new_dataset
                print(f"🔗 Combined dataset: {len(combined)} entries")
                
                return combined
            except FileNotFoundError:
                print(f"⚠️ Existing file not found: {existing_file}")
                return new_dataset
        else:
            return new_dataset
    
    def save_cleaned_dataset(self, dataset, filename=None, save_report=True):
        """Save cleaned dataset and report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"clean_motorcycle_dataset_{timestamp}.json"
        
        # Add IDs to entries
        for i, entry in enumerate(dataset):
            entry['id'] = i + 1
        
        # Save dataset
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Clean dataset saved: {filename}")
        
        if save_report:
            self.generate_report(dataset, filename)
        
        return filename
    
    def generate_report(self, dataset, dataset_filename):
        """Generate cleaning report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"cleaning_report_{timestamp}.json"
        
        # Analyze dataset
        categories = defaultdict(int)
        types = defaultdict(int)
        avg_question_length = 0
        avg_answer_length = 0
        
        for entry in dataset:
            if 'category' in entry:
                categories[entry['category']] += 1
            if 'type' in entry:
                types[entry['type']] += 1
            avg_question_length += len(entry['question'])
            avg_answer_length += len(entry['answer'])
        
        avg_question_length /= len(dataset)
        avg_answer_length /= len(dataset)
        
        report = {
            "timestamp": timestamp,
            "dataset_file": dataset_filename,
            "total_entries": len(dataset),
            "statistics": {
                "avg_question_length": round(avg_question_length, 2),
                "avg_answer_length": round(avg_answer_length, 2),
                "categories": dict(categories),
                "types": dict(types)
            },
            "quality_metrics": {
                "similarity_threshold": self.similarity_threshold,
                "min_question_length": self.min_question_length,
                "min_answer_length": self.min_answer_length,
                "max_question_length": self.max_question_length,
                "max_answer_length": self.max_answer_length
            }
        }
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Cleaning report saved: {report_filename}")
        return report_filename

def main():
    """Main function to clean dataset"""
    cleaner = DatasetCleaner()
    
    print("🧹 Starting Dataset Cleaning Process...")
    
    # Load new generated dataset
    new_dataset_file = "expanded_motorcycle_dataset_20250914_074916.json"
    try:
        with open(new_dataset_file, 'r', encoding='utf-8') as f:
            new_dataset = json.load(f)
        print(f"📂 Loaded new dataset: {len(new_dataset)} entries")
    except FileNotFoundError:
        print(f"❌ File not found: {new_dataset_file}")
        return
    
    # Merge with existing dataset
    existing_file = "merged_motorcycle_dataset.json"
    combined_dataset = cleaner.merge_with_existing(new_dataset, existing_file)
    
    # Clean dataset
    cleaned_dataset, removed_entries = cleaner.clean_dataset(combined_dataset)
    
    # Save cleaned dataset
    output_filename = "ultra_clean_motorcycle_dataset.json"
    cleaner.save_cleaned_dataset(cleaned_dataset, output_filename)
    
    # Print summary
    print("\n📈 Cleaning Summary:")
    print(f"Original entries: {len(combined_dataset)}")
    print(f"Removed entries: {len(removed_entries)}")
    print(f"Final clean entries: {len(cleaned_dataset)}")
    print(f"Cleaning efficiency: {(len(cleaned_dataset)/len(combined_dataset)*100):.1f}%")
    
    # Show some removed entries for review
    if removed_entries:
        print("\n🗑️ Sample removed entries:")
        for i, (idx, reason_type, reason) in enumerate(removed_entries[:5]):
            print(f"  {i+1}. Index {idx}: {reason_type} - {reason}")
        if len(removed_entries) > 5:
            print(f"  ... and {len(removed_entries)-5} more")
    
    print(f"\n✅ Dataset cleaning completed!")
    print(f"📁 Clean dataset: {output_filename}")
    print(f"🎯 Ready for adaptive learning system!")
    
    return output_filename

if __name__ == "__main__":
    main()