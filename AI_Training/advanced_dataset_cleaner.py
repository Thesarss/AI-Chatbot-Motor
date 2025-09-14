import json
import re
from datetime import datetime

class AdvancedDatasetCleaner:
    def __init__(self):
        self.problematic_patterns = [
            # Pola kalimat yang tidak masuk akal
            r'\b(musim hujan|cara perawatan|aus tidak rata)\b.*penyebabnya',
            r'Masalah (cara perawatan|musim hujan) di',
            r'penyebab (cara perawatan|musim hujan) pada',
            
            # Pola grammar yang buruk
            r'periksa kondisi \w+ setiap secara berkala',
            r'untuk \w+ awet dan berfungsi baik\.$',
            r'^\w+ pada motor \w+ umumnya disebabkan oleh masalah pada \w+\. Untuk mengatasinya, periksa dan perbaiki \w+\.$',
            
            # Pola jawaban template yang terlalu repetitif
            r'^Masalah \w+ di \w+ bisa jadi karena masalah pada \w+\. Coba periksa dan perbaiki \w+\.$',
            r'^Penyebab \w+ pada \w+ biasanya karena masalah pada \w+\. Solusinya adalah periksa dan perbaiki \w+\.$'
        ]
        
        self.invalid_questions = [
            # Pertanyaan yang tidak masuk akal
            r'Motor \w+ musim hujan',
            r'Mengapa \w+ saya cara perawatan',
            r'Kenapa \w+ saya (gundul|lepas|pecah)',
            r'\w+ (putus|pecah) terus'
        ]
        
        self.quality_checks = {
            'min_answer_length': 20,
            'max_answer_length': 500,
            'min_question_length': 10,
            'max_question_length': 200
        }
    
    def is_problematic_answer(self, answer):
        """Cek apakah jawaban memiliki pola yang bermasalah"""
        for pattern in self.problematic_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return True
        return False
    
    def is_invalid_question(self, question):
        """Cek apakah pertanyaan tidak valid"""
        for pattern in self.invalid_questions:
            if re.search(pattern, question, re.IGNORECASE):
                return True
        return False
    
    def check_grammar_quality(self, text):
        """Cek kualitas grammar dasar"""
        # Cek kata-kata yang tidak masuk akal atau typo
        weird_words = ['masalem', 'masek', 'masala', 'periksas', 'tidakar', 'nyebabnya']
        for word in weird_words:
            if word in text.lower():
                return False
        
        # Cek struktur kalimat yang aneh
        if re.search(r'\b(untuk|dari|pada)\s*,\s*\w+', text):
            return False
            
        # Cek kalimat yang terlalu pendek atau tidak informatif
        if len(text.split()) < 5:
            return False
            
        return True
    
    def is_meaningful_content(self, question, answer):
        """Cek apakah konten memiliki makna yang jelas"""
        # Cek apakah jawaban relevan dengan pertanyaan
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Minimal ada beberapa kata yang berkaitan
        common_words = question_words.intersection(answer_words)
        if len(common_words) < 2:
            return False
        
        # Cek apakah jawaban memberikan solusi atau informasi yang berguna
        useful_indicators = ['periksa', 'ganti', 'bersihkan', 'setel', 'perbaiki', 'rawat', 'cek']
        has_useful_info = any(indicator in answer.lower() for indicator in useful_indicators)
        
        return has_useful_info
    
    def clean_dataset(self, input_file, output_file):
        """Bersihkan dataset dari entri yang bermasalah"""
        print(f"Memuat dataset dari {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Dataset asli: {len(data)} entri")
        
        cleaned_data = []
        removed_count = 0
        removal_reasons = {
            'problematic_answer': 0,
            'invalid_question': 0,
            'poor_grammar': 0,
            'length_issues': 0,
            'meaningless_content': 0
        }
        
        for entry in data:
            question = entry.get('question', '')
            answer = entry.get('answer', '')
            
            # Cek berbagai kriteria kualitas
            if self.is_problematic_answer(answer):
                removal_reasons['problematic_answer'] += 1
                removed_count += 1
                continue
            
            if self.is_invalid_question(question):
                removal_reasons['invalid_question'] += 1
                removed_count += 1
                continue
            
            if not self.check_grammar_quality(answer) or not self.check_grammar_quality(question):
                removal_reasons['poor_grammar'] += 1
                removed_count += 1
                continue
            
            # Cek panjang teks
            if (len(answer) < self.quality_checks['min_answer_length'] or 
                len(answer) > self.quality_checks['max_answer_length'] or
                len(question) < self.quality_checks['min_question_length'] or
                len(question) > self.quality_checks['max_question_length']):
                removal_reasons['length_issues'] += 1
                removed_count += 1
                continue
            
            if not self.is_meaningful_content(question, answer):
                removal_reasons['meaningless_content'] += 1
                removed_count += 1
                continue
            
            # Jika lolos semua tes, tambahkan ke dataset bersih
            cleaned_data.append(entry)
        
        # Simpan dataset yang sudah dibersihkan
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        # Buat laporan pembersihan
        report = {
            'timestamp': datetime.now().isoformat(),
            'original_count': len(data),
            'cleaned_count': len(cleaned_data),
            'removed_count': removed_count,
            'removal_percentage': round((removed_count / len(data)) * 100, 2),
            'removal_reasons': removal_reasons,
            'input_file': input_file,
            'output_file': output_file
        }
        
        report_file = f"dataset_cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== LAPORAN PEMBERSIHAN DATASET ===")
        print(f"Dataset asli: {len(data)} entri")
        print(f"Dataset bersih: {len(cleaned_data)} entri")
        print(f"Entri dihapus: {removed_count} ({report['removal_percentage']}%)")
        print(f"\nAlasan penghapusan:")
        for reason, count in removal_reasons.items():
            print(f"  - {reason}: {count} entri")
        print(f"\nDataset bersih disimpan ke: {output_file}")
        print(f"Laporan disimpan ke: {report_file}")
        
        return cleaned_data, report

def main():
    cleaner = AdvancedDatasetCleaner()
    
    input_file = "ultra_clean_motorcycle_dataset_fixed.json"
    output_file = f"super_clean_motorcycle_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        cleaned_data, report = cleaner.clean_dataset(input_file, output_file)
        print(f"\nPembersihan dataset selesai!")
        print(f"Dataset berkualitas tinggi siap untuk training.")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()