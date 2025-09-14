import json
import re
from collections import defaultdict
from typing import List, Dict, Tuple

class DatasetAnalyzer:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = []
        self.inconsistencies = []
        self.load_dataset()
    
    def load_dataset(self):
        """Load dataset from JSON file"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Dataset loaded: {len(self.data)} entries")
        except Exception as e:
            print(f"Error loading dataset: {e}")
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split into words and filter out common words
        stop_words = {'dan', 'atau', 'yang', 'pada', 'di', 'ke', 'dari', 'untuk', 'dengan', 'adalah', 'apa', 'kenapa', 'bagaimana', 'cara', 'tips', 'motor', 'saya'}
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        return words
    
    def check_question_answer_relevance(self, question: str, answer: str, category: str) -> Dict:
        """Check if question and answer are relevant"""
        q_keywords = set(self.extract_keywords(question))
        a_keywords = set(self.extract_keywords(answer))
        
        # Calculate keyword overlap
        overlap = q_keywords.intersection(a_keywords)
        overlap_ratio = len(overlap) / max(len(q_keywords), 1)
        
        # Check if answer mentions different components than question
        motor_components = {
            'rem': ['rem', 'kaliper', 'kampas', 'cakram', 'tromol', 'minyak rem'],
            'mesin': ['mesin', 'piston', 'silinder', 'klep', 'timing', 'kompresi', 'busi'],
            'kelistrikan': ['aki', 'coil', 'kabel', 'lampu', 'klakson', 'starter', 'alternator'],
            'transmisi': ['kopling', 'gigi', 'rantai', 'belt', 'final drive', 'cvt'],
            'ban_velg': ['ban', 'velg', 'pentil', 'tekanan', 'gundul'],
            'body_rangka': ['fairing', 'rangka', 'shock', 'fork', 'bearing']
        }
        
        q_components = set()
        a_components = set()
        
        for cat, components in motor_components.items():
            for comp in components:
                if comp in question.lower():
                    q_components.add(comp)
                if comp in answer.lower():
                    a_components.add(comp)
        
        component_mismatch = len(q_components) > 0 and len(a_components) > 0 and not q_components.intersection(a_components)
        
        return {
            'overlap_ratio': overlap_ratio,
            'component_mismatch': component_mismatch,
            'q_keywords': list(q_keywords),
            'a_keywords': list(a_keywords),
            'overlap': list(overlap),
            'q_components': list(q_components),
            'a_components': list(a_components)
        }
    
    def analyze_inconsistencies(self) -> List[Dict]:
        """Analyze dataset for inconsistencies"""
        inconsistencies = []
        
        for i, entry in enumerate(self.data):
            question = entry.get('question', '')
            answer = entry.get('answer', '')
            category = entry.get('category', '')
            
            relevance = self.check_question_answer_relevance(question, answer, category)
            
            # Flag as inconsistent if:
            # 1. Very low keyword overlap (< 0.1)
            # 2. Component mismatch
            # 3. Answer is too generic or doesn't address the question
            
            is_inconsistent = (
                relevance['overlap_ratio'] < 0.1 or
                relevance['component_mismatch'] or
                len(relevance['a_keywords']) < 3
            )
            
            if is_inconsistent:
                inconsistencies.append({
                    'id': entry.get('id', i),
                    'question': question,
                    'answer': answer,
                    'category': category,
                    'relevance_score': relevance['overlap_ratio'],
                    'component_mismatch': relevance['component_mismatch'],
                    'issues': self.identify_issues(relevance)
                })
        
        return inconsistencies
    
    def identify_issues(self, relevance: Dict) -> List[str]:
        """Identify specific issues with the entry"""
        issues = []
        
        if relevance['overlap_ratio'] < 0.1:
            issues.append("Low keyword overlap between question and answer")
        
        if relevance['component_mismatch']:
            issues.append("Question and answer discuss different motor components")
        
        if len(relevance['a_keywords']) < 3:
            issues.append("Answer is too generic or short")
        
        return issues
    
    def generate_report(self) -> Dict:
        """Generate analysis report"""
        inconsistencies = self.analyze_inconsistencies()
        
        # Category breakdown
        category_issues = defaultdict(int)
        for inc in inconsistencies:
            category_issues[inc['category']] += 1
        
        # Issue type breakdown
        issue_types = defaultdict(int)
        for inc in inconsistencies:
            for issue in inc['issues']:
                issue_types[issue] += 1
        
        report = {
            'total_entries': len(self.data),
            'inconsistent_entries': len(inconsistencies),
            'consistency_rate': (len(self.data) - len(inconsistencies)) / len(self.data) * 100,
            'category_issues': dict(category_issues),
            'issue_types': dict(issue_types),
            'sample_inconsistencies': inconsistencies[:10]  # First 10 examples
        }
        
        return report
    
    def save_inconsistencies(self, output_path: str):
        """Save inconsistent entries to file"""
        inconsistencies = self.analyze_inconsistencies()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(inconsistencies, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(inconsistencies)} inconsistent entries to {output_path}")

def main():
    # Analyze the dataset
    analyzer = DatasetAnalyzer('ultra_clean_motorcycle_dataset.json')
    
    print("\n=== DATASET ANALYSIS REPORT ===")
    report = analyzer.generate_report()
    
    print(f"Total entries: {report['total_entries']}")
    print(f"Inconsistent entries: {report['inconsistent_entries']}")
    print(f"Consistency rate: {report['consistency_rate']:.1f}%")
    
    print("\n=== ISSUES BY CATEGORY ===")
    for category, count in report['category_issues'].items():
        print(f"{category}: {count} issues")
    
    print("\n=== ISSUE TYPES ===")
    for issue_type, count in report['issue_types'].items():
        print(f"{issue_type}: {count} occurrences")
    
    print("\n=== SAMPLE INCONSISTENCIES ===")
    for i, inc in enumerate(report['sample_inconsistencies'][:5]):
        print(f"\n[{i+1}] ID: {inc['id']}")
        print(f"Question: {inc['question']}")
        print(f"Answer: {inc['answer']}")
        print(f"Issues: {', '.join(inc['issues'])}")
    
    # Save inconsistencies for manual review
    analyzer.save_inconsistencies('dataset_inconsistencies.json')
    
    # Save full report
    with open('dataset_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n=== FILES CREATED ===")
    print("- dataset_inconsistencies.json: List of problematic entries")
    print("- dataset_analysis_report.json: Full analysis report")

if __name__ == "__main__":
    main()