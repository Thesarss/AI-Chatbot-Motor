import json
from typing import Dict, List

class DatasetFixer:
    def __init__(self, dataset_path: str, inconsistencies_path: str):
        self.dataset_path = dataset_path
        self.inconsistencies_path = inconsistencies_path
        self.dataset = []
        self.inconsistencies = []
        self.load_data()
    
    def load_data(self):
        """Load dataset and inconsistencies"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        with open(self.inconsistencies_path, 'r', encoding='utf-8') as f:
            self.inconsistencies = json.load(f)
        
        print(f"Loaded {len(self.dataset)} dataset entries")
        print(f"Found {len(self.inconsistencies)} inconsistencies to fix")
    
    def generate_relevant_answer(self, question: str, category: str) -> str:
        """Generate relevant answer based on question and category"""
        
        # Define answer templates for different problems and categories
        answer_templates = {
            'goyang': {
                'suspensi': "Masalah goyang pada motor biasanya disebabkan oleh shock absorber yang aus atau rusak. Solusi: 1) Periksa kondisi shock depan dan belakang, 2) Ganti shock yang sudah aus, 3) Cek bearing roda yang mungkin longgar, 4) Periksa tekanan ban yang tidak seimbang.",
                'default': "Masalah goyang pada motor umumnya terkait suspensi. Periksa shock absorber, bearing roda, dan tekanan ban."
            },
            'getaran': {
                'mesin': "Getaran berlebih pada motor biasanya disebabkan oleh: 1) Engine mounting yang aus atau longgar, 2) Ketidakseimbangan pada crankshaft, 3) Busi yang tidak berfungsi optimal, 4) Timing yang tidak tepat. Solusi: Periksa dan ganti engine mounting, tune-up mesin, dan cek kondisi busi.",
                'default': "Getaran berlebih umumnya disebabkan masalah pada engine mounting atau ketidakseimbangan mesin. Lakukan tune-up dan periksa mounting."
            },
            'drop': {
                'kelistrikan': "Motor yang sering drop (mati mendadak) biasanya disebabkan oleh: 1) Masalah pada sistem pengapian (CDI, coil), 2) Koneksi kabel yang longgar, 3) Aki yang lemah, 4) Sensor yang rusak. Solusi: Periksa sistem kelistrikan secara menyeluruh, ganti komponen yang rusak.",
                'default': "Motor drop umumnya masalah kelistrikan. Periksa CDI, coil, aki, dan koneksi kabel."
            },
            'aus': {
                'transmisi': "Komponen motor yang cepat aus biasanya disebabkan oleh: 1) Kurangnya pelumasan, 2) Pemakaian yang berlebihan, 3) Kualitas oli yang buruk, 4) Rantai atau belt yang kendor. Solusi: Ganti oli secara rutin, setel rantai/belt, gunakan oli berkualitas baik.",
                'default': "Keausan komponen umumnya karena kurang pelumasan atau perawatan. Ganti oli rutin dan periksa komponen yang aus."
            },
            'bocor': {
                'body_rangka': "Kebocoran pada motor bisa terjadi di berbagai tempat: 1) Kebocoran oli dari gasket atau seal, 2) Kebocoran bensin dari tangki atau selang, 3) Kebocoran air radiator. Solusi: Identifikasi sumber kebocoran, ganti gasket/seal yang rusak, periksa kondisi tangki dan selang.",
                'default': "Kebocoran motor bisa dari oli, bensin, atau air radiator. Identifikasi sumber dan ganti seal/gasket yang rusak."
            }
        }
        
        # Extract problem type from question
        question_lower = question.lower()
        problem_type = None
        
        for problem in answer_templates.keys():
            if problem in question_lower:
                problem_type = problem
                break
        
        if problem_type and category in answer_templates[problem_type]:
            return answer_templates[problem_type][category]
        elif problem_type:
            return answer_templates[problem_type]['default']
        else:
            # Generic answer based on category
            generic_answers = {
                'mesin': "Masalah pada mesin motor memerlukan pemeriksaan komponen seperti busi, filter udara, oli mesin, dan sistem pembakaran. Lakukan tune-up rutin untuk menjaga performa mesin.",
                'kelistrikan': "Masalah kelistrikan motor biasanya terkait aki, sistem pengapian, atau koneksi kabel. Periksa tegangan aki dan kondisi kabel secara berkala.",
                'transmisi': "Masalah transmisi memerlukan pemeriksaan kopling, rantai/belt, dan oli transmisi. Pastikan pelumasan adequate dan setel rantai sesuai spesifikasi.",
                'rem': "Sistem rem motor perlu perawatan rutin. Periksa kampas rem, minyak rem, dan kondisi cakram/tromol secara berkala untuk keamanan berkendara.",
                'suspensi': "Sistem suspensi yang baik penting untuk kenyamanan dan keamanan. Periksa shock absorber, per, dan bearing secara rutin.",
                'ban_velg': "Perawatan ban dan velg meliputi pemeriksaan tekanan ban, kondisi tapak, dan keseimbangan velg untuk performa dan keamanan optimal.",
                'body_rangka': "Perawatan body dan rangka meliputi pemeriksaan struktur, cat, dan komponen pengikat untuk menjaga integritas dan penampilan motor."
            }
            return generic_answers.get(category, "Lakukan pemeriksaan dan perawatan rutin sesuai manual motor untuk menjaga performa optimal.")
    
    def fix_inconsistencies(self) -> List[Dict]:
        """Fix inconsistent entries in dataset"""
        fixed_entries = []
        inconsistent_ids = {inc['id'] for inc in self.inconsistencies}
        
        for entry in self.dataset:
            if entry['id'] in inconsistent_ids:
                # Find the inconsistency details
                inc_detail = next(inc for inc in self.inconsistencies if inc['id'] == entry['id'])
                
                # Generate new relevant answer
                new_answer = self.generate_relevant_answer(entry['question'], entry['category'])
                
                # Create fixed entry
                fixed_entry = entry.copy()
                fixed_entry['answer'] = new_answer
                fixed_entry['fixed'] = True
                fixed_entry['original_answer'] = entry['answer']
                
                fixed_entries.append({
                    'id': entry['id'],
                    'question': entry['question'],
                    'category': entry['category'],
                    'original_answer': entry['answer'],
                    'new_answer': new_answer,
                    'issues_fixed': inc_detail['issues']
                })
                
                # Update in dataset
                entry['answer'] = new_answer
        
        return fixed_entries
    
    def save_fixed_dataset(self, output_path: str):
        """Save the fixed dataset"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)
        print(f"Fixed dataset saved to {output_path}")
    
    def save_fix_report(self, fixed_entries: List[Dict], output_path: str):
        """Save report of fixes made"""
        report = {
            'total_fixes': len(fixed_entries),
            'fixes': fixed_entries
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Fix report saved to {output_path}")

def main():
    print("=== DATASET FIXER ===")
    
    # Initialize fixer
    fixer = DatasetFixer('ultra_clean_motorcycle_dataset.json', 'dataset_inconsistencies.json')
    
    # Fix inconsistencies
    print("\nFixing inconsistencies...")
    fixed_entries = fixer.fix_inconsistencies()
    
    print(f"\nFixed {len(fixed_entries)} entries:")
    for i, fix in enumerate(fixed_entries[:3]):  # Show first 3 examples
        print(f"\n[{i+1}] ID: {fix['id']}")
        print(f"Question: {fix['question']}")
        print(f"Original: {fix['original_answer'][:100]}...")
        print(f"Fixed: {fix['new_answer'][:100]}...")
    
    # Save fixed dataset
    fixer.save_fixed_dataset('ultra_clean_motorcycle_dataset_fixed.json')
    
    # Save fix report
    fixer.save_fix_report(fixed_entries, 'dataset_fix_report.json')
    
    print("\n=== SUMMARY ===")
    print(f"Total entries processed: {len(fixer.dataset)}")
    print(f"Entries fixed: {len(fixed_entries)}")
    print(f"Fix rate: {len(fixed_entries)/len(fixer.dataset)*100:.1f}%")
    
    print("\n=== FILES CREATED ===")
    print("- ultra_clean_motorcycle_dataset_fixed.json: Fixed dataset")
    print("- dataset_fix_report.json: Report of all fixes made")

if __name__ == "__main__":
    main()