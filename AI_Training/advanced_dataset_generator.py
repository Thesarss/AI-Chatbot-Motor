import json
import random
from datetime import datetime
import itertools

class AdvancedMotorcycleDatasetGenerator:
    def __init__(self):
        # Kategori utama motor
        self.categories = {
            "mesin": {
                "subcategories": ["karburator", "injeksi", "oli", "filter", "busi", "kompresi", "timing", "valve"],
                "problems": ["susah hidup", "brebet", "overheat", "ngebul", "boros bensin", "tenaga kurang", "suara kasar", "getaran berlebih"]
            },
            "transmisi": {
                "subcategories": ["kopling", "gigi", "rantai", "belt", "CVT", "final drive"],
                "problems": ["slip", "keras", "bunyi", "aus", "macet", "tidak responsif", "getaran"]
            },
            "kelistrikan": {
                "subcategories": ["aki", "alternator", "starter", "lampu", "klakson", "CDI", "coil", "kabel"],
                "problems": ["mati", "redup", "tidak nyala", "konslet", "tekor", "drop", "putus"]
            },
            "suspensi": {
                "subcategories": ["shock depan", "shock belakang", "per", "bushing", "bearing"],
                "problems": ["keras", "empuk", "bocor", "bunyi", "tidak stabil", "goyang"]
            },
            "rem": {
                "subcategories": ["kampas", "cakram", "tromol", "minyak rem", "master rem", "kaliper"],
                "problems": ["blong", "keras", "bunyi", "aus", "bocor", "tidak pakem", "getaran"]
            },
            "ban_velg": {
                "subcategories": ["ban", "velg", "jari-jari", "bearing roda", "pentil"],
                "problems": ["gundul", "pecah", "kempes", "peyang", "goyang", "aus tidak rata"]
            },
            "body_rangka": {
                "subcategories": ["rangka", "fairing", "spakbor", "jok", "tangki", "knalpot"],
                "problems": ["retak", "penyok", "karat", "lepas", "longgar", "bocor"]
            },
            "perawatan": {
                "subcategories": ["service rutin", "tune up", "ganti oli", "cuci motor", "penyimpanan"],
                "problems": ["jadwal service", "cara perawatan", "tips awet", "musim hujan", "motor jarang pakai"]
            }
        }
        
        # Jenis motor
        self.motor_types = ["matic", "bebek", "sport", "naked bike", "touring", "trail", "cruiser"]
        
        # Brand motor populer
        self.brands = ["Honda", "Yamaha", "Suzuki", "Kawasaki", "TVS", "Benelli", "KTM"]
        
        # Kapasitas mesin
        self.engine_sizes = ["110cc", "125cc", "150cc", "160cc", "200cc", "250cc", "300cc", "400cc"]
        
        # Template pertanyaan
        self.question_templates = {
            "problem": [
                "Kenapa {motor} saya {problem}?",
                "Motor {brand} {problem}, apa penyebabnya?",
                "{motor} {problem} terus, gimana solusinya?",
                "Apa yang menyebabkan {motor} {problem}?",
                "Motor saya {problem}, kira-kira kenapa ya?",
                "Mengapa {motor} {brand} saya {problem}?"
            ],
            "maintenance": [
                "Bagaimana cara merawat {component} motor {motor}?",
                "Kapan harus ganti {component} motor {brand}?",
                "Tips merawat {component} motor {motor}?",
                "Berapa lama umur {component} motor?",
                "Cara cek kondisi {component} motor gimana?"
            ],
            "troubleshooting": [
                "Cara memperbaiki {component} motor yang {problem}?",
                "Solusi {motor} {problem} di {component}?",
                "Gimana cara benerin {component} yang {problem}?",
                "Langkah-langkah perbaikan {component} motor?"
            ],
            "general": [
                "Apa perbedaan motor {type1} dan {type2}?",
                "Mana yang lebih baik {brand1} atau {brand2}?",
                "Rekomendasi motor {type} untuk pemula?",
                "Motor {engine_size} cocok untuk harian?"
            ]
        }
        
        # Template jawaban
        self.answer_templates = {
            "diagnosis": [
                "Penyebab {problem} pada {motor} biasanya karena {cause}. Solusinya adalah {solution}.",
                "{problem} pada motor {brand} umumnya disebabkan oleh {cause}. Untuk mengatasinya, {solution}.",
                "Masalah {problem} di {motor} bisa jadi karena {cause}. Coba {solution}."
            ],
            "maintenance": [
                "Untuk merawat {component}, sebaiknya {maintenance_action} setiap {interval}. Hal ini penting untuk {benefit}.",
                "{component} motor perlu {maintenance_action} secara rutin {interval} agar {benefit}.",
                "Perawatan {component} yang baik adalah {maintenance_action} setiap {interval} untuk menjaga {benefit}."
            ],
            "step_by_step": [
                "Langkah-langkah {action}: 1) {step1}, 2) {step2}, 3) {step3}. Pastikan {safety_note}.",
                "Cara {action}: Pertama {step1}, kemudian {step2}, terakhir {step3}. Jangan lupa {safety_note}."
            ]
        }
        
        # Data detail untuk setiap komponen
        self.component_details = {
            "karburator": {
                "causes": ["setelan angin tidak tepat", "spuyer kotor", "pelampung rusak", "jarum skep aus"],
                "solutions": ["setel ulang angin karburator", "bersihkan spuyer", "ganti pelampung", "ganti jarum skep"],
                "maintenance": ["bersihkan karburator", "setel angin karburator", "ganti filter udara"],
                "intervals": ["3 bulan", "5000 km", "setiap service"]
            },
            "oli": {
                "causes": ["oli kotor", "oli habis", "oli tidak sesuai spek", "kebocoran oli"],
                "solutions": ["ganti oli baru", "tambah oli", "gunakan oli sesuai spek", "perbaiki kebocoran"],
                "maintenance": ["ganti oli mesin", "cek level oli", "gunakan oli berkualitas"],
                "intervals": ["2000 km", "3 bulan", "setiap bulan"]
            },
            "busi": {
                "causes": ["busi kotor", "celah busi tidak tepat", "busi aus", "busi basah"],
                "solutions": ["bersihkan busi", "setel celah busi", "ganti busi baru", "keringkan busi"],
                "maintenance": ["bersihkan busi", "cek celah busi", "ganti busi"],
                "intervals": ["5000 km", "6 bulan", "setiap service"]
            }
        }
        
    def generate_problem_qa(self, category, subcategory, problem):
        """Generate Q&A untuk masalah spesifik"""
        motor_type = random.choice(self.motor_types)
        brand = random.choice(self.brands)
        
        # Generate question
        template = random.choice(self.question_templates["problem"])
        question = template.format(
            motor=motor_type,
            brand=brand,
            problem=problem
        )
        
        # Generate answer
        if subcategory in self.component_details:
            details = self.component_details[subcategory]
            cause = random.choice(details["causes"])
            solution = random.choice(details["solutions"])
        else:
            cause = f"masalah pada {subcategory}"
            solution = f"periksa dan perbaiki {subcategory}"
        
        answer_template = random.choice(self.answer_templates["diagnosis"])
        answer = answer_template.format(
            problem=problem,
            motor=motor_type,
            brand=brand,
            cause=cause,
            solution=solution
        )
        
        return {
            "question": question,
            "answer": answer,
            "category": category,
            "subcategory": subcategory,
            "type": "problem_diagnosis"
        }
    
    def generate_maintenance_qa(self, category, subcategory):
        """Generate Q&A untuk perawatan"""
        motor_type = random.choice(self.motor_types)
        brand = random.choice(self.brands)
        
        # Generate question
        template = random.choice(self.question_templates["maintenance"])
        question = template.format(
            component=subcategory,
            motor=motor_type,
            brand=brand
        )
        
        # Generate answer
        if subcategory in self.component_details:
            details = self.component_details[subcategory]
            maintenance_action = random.choice(details["maintenance"])
            interval = random.choice(details["intervals"])
            benefit = f"performa {subcategory} tetap optimal"
        else:
            maintenance_action = f"periksa kondisi {subcategory}"
            interval = "secara berkala"
            benefit = f"{subcategory} awet dan berfungsi baik"
        
        answer_template = random.choice(self.answer_templates["maintenance"])
        answer = answer_template.format(
            component=subcategory,
            maintenance_action=maintenance_action,
            interval=interval,
            benefit=benefit
        )
        
        return {
            "question": question,
            "answer": answer,
            "category": category,
            "subcategory": subcategory,
            "type": "maintenance"
        }
    
    def generate_comparison_qa(self):
        """Generate Q&A untuk perbandingan"""
        comparison_types = [
            ("motor_types", self.motor_types),
            ("brands", self.brands),
            ("engine_sizes", self.engine_sizes)
        ]
        
        comp_type, items = random.choice(comparison_types)
        item1, item2 = random.sample(items, 2)
        
        if comp_type == "motor_types":
            question = f"Apa perbedaan motor {item1} dan {item2}?"
            answer = f"Motor {item1} lebih cocok untuk {self.get_usage(item1)}, sedangkan {item2} lebih baik untuk {self.get_usage(item2)}. Dari segi perawatan, {item1} {self.get_maintenance_note(item1)}, sementara {item2} {self.get_maintenance_note(item2)}."
        elif comp_type == "brands":
            question = f"Mana yang lebih baik motor {item1} atau {item2}?"
            answer = f"Kedua brand memiliki keunggulan masing-masing. {item1} dikenal karena {self.get_brand_strength(item1)}, sedangkan {item2} unggul dalam {self.get_brand_strength(item2)}. Pilihan tergantung kebutuhan dan budget Anda."
        else:
            question = f"Motor {item1} vs {item2}, mana yang lebih baik?"
            answer = f"Motor {item1} cocok untuk {self.get_engine_usage(item1)}, sedangkan {item2} lebih sesuai untuk {self.get_engine_usage(item2)}. Konsumsi BBM {item1} lebih {self.get_fuel_efficiency(item1)} dibanding {item2}."
        
        return {
            "question": question,
            "answer": answer,
            "category": "comparison",
            "subcategory": comp_type,
            "type": "comparison"
        }
    
    def get_usage(self, motor_type):
        usage_map = {
            "matic": "penggunaan harian di kota",
            "bebek": "efisiensi dan ekonomis",
            "sport": "performa dan kecepatan",
            "naked bike": "gaya dan handling",
            "touring": "perjalanan jauh",
            "trail": "medan off-road",
            "cruiser": "kenyamanan berkendara"
        }
        return usage_map.get(motor_type, "penggunaan umum")
    
    def get_maintenance_note(self, motor_type):
        maintenance_map = {
            "matic": "memerlukan perawatan CVT",
            "bebek": "perawatan relatif mudah",
            "sport": "butuh perawatan lebih intensif",
            "naked bike": "perawatan standar",
            "touring": "perawatan berkala penting",
            "trail": "perlu pembersihan ekstra",
            "cruiser": "perawatan rutin standar"
        }
        return maintenance_map.get(motor_type, "perawatan standar")
    
    def get_brand_strength(self, brand):
        strength_map = {
            "Honda": "keandalan dan efisiensi BBM",
            "Yamaha": "performa mesin dan handling",
            "Suzuki": "teknologi dan inovasi",
            "Kawasaki": "power dan performa sport",
            "TVS": "value for money",
            "Benelli": "desain dan styling",
            "KTM": "performa off-road dan adventure"
        }
        return strength_map.get(brand, "kualitas produk")
    
    def get_engine_usage(self, engine_size):
        usage_map = {
            "110cc": "penggunaan dalam kota",
            "125cc": "harian ekonomis",
            "150cc": "balance performa dan efisiensi",
            "160cc": "touring ringan",
            "200cc": "performa menengah",
            "250cc": "touring dan sport",
            "300cc": "performa tinggi",
            "400cc": "touring jarak jauh"
        }
        return usage_map.get(engine_size, "penggunaan umum")
    
    def get_fuel_efficiency(self, engine_size):
        cc_value = int(engine_size.replace("cc", ""))
        if cc_value <= 125:
            return "irit"
        elif cc_value <= 200:
            return "sedang"
        else:
            return "boros"
    
    def generate_dataset(self, target_size=1000):
        """Generate dataset dengan target size tertentu"""
        dataset = []
        generated_questions = set()  # Untuk menghindari duplikasi
        
        print(f"Generating {target_size} unique Q&A entries...")
        
        while len(dataset) < target_size:
            # Pilih jenis Q&A yang akan digenerate
            qa_types = ["problem", "maintenance", "comparison"]
            weights = [0.5, 0.3, 0.2]  # 50% problem, 30% maintenance, 20% comparison
            qa_type = random.choices(qa_types, weights=weights)[0]
            
            if qa_type == "comparison":
                qa_entry = self.generate_comparison_qa()
            else:
                # Pilih kategori dan subkategori
                category = random.choice(list(self.categories.keys()))
                subcategory = random.choice(self.categories[category]["subcategories"])
                
                if qa_type == "problem":
                    problem = random.choice(self.categories[category]["problems"])
                    qa_entry = self.generate_problem_qa(category, subcategory, problem)
                else:  # maintenance
                    qa_entry = self.generate_maintenance_qa(category, subcategory)
            
            # Check for duplicates
            question_key = qa_entry["question"].lower().strip()
            if question_key not in generated_questions:
                generated_questions.add(question_key)
                qa_entry["id"] = len(dataset) + 1
                dataset.append(qa_entry)
                
                if len(dataset) % 100 == 0:
                    print(f"Generated {len(dataset)} entries...")
        
        print(f"Successfully generated {len(dataset)} unique Q&A entries!")
        return dataset
    
    def save_dataset(self, dataset, filename=None):
        """Save dataset to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"expanded_motorcycle_dataset_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to: {filename}")
        return filename

def main():
    """Main function to generate expanded dataset"""
    generator = AdvancedMotorcycleDatasetGenerator()
    
    print("🚀 Starting Advanced Motorcycle Dataset Generation...")
    print("📊 Target: 1000+ unique, clean, non-redundant entries")
    
    # Generate dataset
    dataset = generator.generate_dataset(target_size=1200)  # Generate extra untuk filtering
    
    # Save dataset
    filename = generator.save_dataset(dataset)
    
    # Statistics
    categories = {}
    types = {}
    for entry in dataset:
        cat = entry.get("category", "unknown")
        typ = entry.get("type", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        types[typ] = types.get(typ, 0) + 1
    
    print("\n📈 Dataset Statistics:")
    print(f"Total entries: {len(dataset)}")
    print("\nBy Category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print("\nBy Type:")
    for typ, count in sorted(types.items()):
        print(f"  {typ}: {count}")
    
    print(f"\n✅ Dataset generation completed!")
    print(f"📁 File: {filename}")
    print(f"🎯 Ready for deduplication and training!")
    
    return filename

if __name__ == "__main__":
    main()