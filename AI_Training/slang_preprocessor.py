import re
import json
from typing import Dict, List, Tuple

class IndonesianSlangPreprocessor:
    """
    Preprocessor untuk menormalisasi bahasa gaul Indonesia ke bahasa formal
    untuk meningkatkan pemahaman AI terhadap input user
    """
    
    def __init__(self):
        self.slang_dict = self._load_slang_dictionary()
        self.abbreviation_dict = self._load_abbreviation_dictionary()
        self.motorcycle_slang = self._load_motorcycle_slang()
        
    def _load_slang_dictionary(self) -> Dict[str, str]:
        """Load dictionary bahasa gaul umum ke bahasa formal"""
        return {
            # Kata ganti dan sapaan
            "gue": "saya",
            "gw": "saya", 
            "ane": "saya",
            "ente": "anda",
            "lu": "anda",
            "loe": "anda",
            "bro": "saudara",
            "sis": "saudara",
            "gan": "saudara",
            "om": "bapak",
            "tante": "ibu",
            
            # Kata kerja gaul
            "ngapain": "sedang apa",
            "gimana": "bagaimana",
            "kenapa": "mengapa",
            "ngga": "tidak",
            "nggak": "tidak",
            "ga": "tidak",
            "gak": "tidak",
            "udah": "sudah",
            "belom": "belum",
            "udahan": "sudah selesai",
            "nyoba": "mencoba",
            "nyari": "mencari",
            "ngeliat": "melihat",
            "ngelihat": "melihat",
            "ngedenger": "mendengar",
            "ngomong": "berbicara",
            "bilang": "mengatakan",
            "nyerah": "menyerah",
            "nyalain": "menyalakan",
            "matiin": "mematikan",
            "benerin": "memperbaiki",
            "rusak": "rusak",
            "ancur": "rusak parah",
            "jebol": "rusak",
            "mogok": "tidak bisa hidup",
            "mbrebet": "tersendat",
            "ngebul": "mengeluarkan asap",
            "ngorok": "berbunyi kasar",
            "ngeden": "susah hidup",
            "lemot": "lambat",
            "ngelag": "tersendat",
            "ngadat": "bermasalah",
            
            # Kata sifat gaul
            "keren": "bagus",
            "mantap": "bagus",
            "mantul": "bagus",
            "oke": "baik",
            "okeh": "baik",
            "parah": "buruk",
            "ancur": "rusak parah",
            "jelek": "buruk",
            "bagus": "baik",
            "top": "terbaik",
            "jos": "bagus",
            "maknyus": "sangat bagus",
            "juara": "terbaik",
            "beres": "baik",
            "normal": "baik",
            "wajar": "normal",
            "aneh": "tidak normal",
            "ribet": "rumit",
            "susah": "sulit",
            "gampang": "mudah",
            "enteng": "mudah",
            "berat": "sulit",
            
            # Kata tanya gaul
            "apaan": "apa",
            "kenapa": "mengapa",
            "gimana": "bagaimana",
            "dimana": "di mana",
            "kapan": "kapan",
            "siapa": "siapa",
            
            # Kata sambung dan partikel
            "sih": "",
            "dong": "",
            "deh": "",
            "nih": "ini",
            "tuh": "itu",
            "gitu": "begitu",
            "gini": "begini",
            "kayak": "seperti",
            "kaya": "seperti",
            "macam": "seperti",
            "banget": "sekali",
            "bgt": "sekali",
            "abis": "habis",
            "terus": "kemudian",
            "trus": "kemudian",
            "soalnya": "karena",
            "makanya": "oleh karena itu",
            "jadinya": "sehingga",
            "padahal": "padahal",
            "walau": "walaupun",
            "meski": "meskipun",
            
            # Kata waktu gaul
            "kemaren": "kemarin",
            "kmrn": "kemarin",
            "besok": "besok",
            "nanti": "nanti",
            "sekarang": "sekarang",
            "skrg": "sekarang",
            "tadi": "tadi",
            "barusan": "baru saja",
            "lama": "lama",
            "bentar": "sebentar",
            "cepet": "cepat",
            "lambat": "lambat",
        }
    
    def _load_abbreviation_dictionary(self) -> Dict[str, str]:
        """Load dictionary singkatan ke bentuk lengkap"""
        return {
            "yg": "yang",
            "dgn": "dengan",
            "utk": "untuk",
            "krn": "karena",
            "tp": "tapi",
            "tpi": "tapi",
            "klo": "kalau",
            "kalo": "kalau",
            "jd": "jadi",
            "jdi": "jadi",
            "hrs": "harus",
            "hrus": "harus",
            "bs": "bisa",
            "bsa": "bisa",
            "dr": "dari",
            "dri": "dari",
            "ke": "ke",
            "sm": "sama",
            "sma": "sama",
            "pd": "pada",
            "pda": "pada",
            "dl": "dalam",
            "dlm": "dalam",
            "spt": "seperti",
            "sprt": "seperti",
            "krg": "kurang",
            "lbh": "lebih",
            "sdh": "sudah",
            "blm": "belum",
            "tdk": "tidak",
            "tdk": "tidak",
            "dgr": "dengar",
            "lht": "lihat",
            "mnrt": "menurut",
            "sblm": "sebelum",
            "stlh": "setelah",
            "krn": "karena",
            "shg": "sehingga",
            "bhw": "bahwa",
            "thd": "terhadap",
            "ttg": "tentang",
            "dpt": "dapat",
            "msh": "masih",
            "jg": "juga",
            "jga": "juga",
            "aja": "saja",
            "aj": "saja",
            "doang": "saja",
            "kok": "",
            "emg": "memang",
            "emang": "memang",
            "bkn": "bukan",
            "mkn": "makan",
            "mnm": "minum",
            "tdr": "tidur",
            "bnr": "benar",
            "slh": "salah",
            "bgs": "bagus",
            "jlk": "jelek",
            "mhl": "mahal",
            "mrh": "murah",
            "bsr": "besar",
            "kcl": "kecil",
            "tgg": "tinggi",
            "rndh": "rendah",
            "pjg": "panjang",
            "pdk": "pendek",
            "lbr": "lebar",
            "smp": "sempit",
            "tbl": "tebal",
            "tps": "tipis",
            "krs": "keras",
            "lmk": "lemah",
            "kuat": "kuat",
            "lmh": "lemah",
        }
    
    def _load_motorcycle_slang(self) -> Dict[str, str]:
        """Load dictionary bahasa gaul khusus motor"""
        return {
            # Istilah motor gaul
            "motor": "motor",
            "moge": "motor gede",
            "bebek": "motor bebek",
            "matic": "motor matic",
            "kopling": "motor kopling",
            "sport": "motor sport",
            "trail": "motor trail",
            "touring": "motor touring",
            "naked": "motor naked",
            "cruiser": "motor cruiser",
            "scooter": "motor scooter",
            
            # Komponen motor gaul
            "mesin": "mesin",
            "engine": "mesin",
            "karbu": "karburator",
            "injeksi": "injeksi",
            "rem": "rem",
            "brake": "rem",
            "kampas": "kampas rem",
            "cakram": "cakram rem",
            "tromol": "tromol rem",
            "ban": "ban",
            "tire": "ban",
            "velg": "velg",
            "pelek": "velg",
            "shock": "shock absorber",
            "suspensi": "suspensi",
            "fork": "fork depan",
            "swing arm": "swing arm",
            "rantai": "rantai",
            "chain": "rantai",
            "gear": "gigi transmisi",
            "transmisi": "transmisi",
            "kopling": "kopling",
            "clutch": "kopling",
            "starter": "starter",
            "kick starter": "kick starter",
            "electric starter": "starter elektrik",
            "aki": "aki",
            "battery": "aki",
            "spul": "spul",
            "kiprok": "kiprok",
            "cdi": "cdi",
            "ecu": "ecu",
            "busi": "busi",
            "spark plug": "busi",
            "koil": "koil",
            "ignition coil": "koil",
            "filter": "filter",
            "saringan": "filter",
            "oli": "oli",
            "oil": "oli",
            "bensin": "bensin",
            "fuel": "bensin",
            "tangki": "tangki bensin",
            "tank": "tangki",
            "knalpot": "knalpot",
            "exhaust": "knalpot",
            "muffler": "knalpot",
            "lampu": "lampu",
            "light": "lampu",
            "headlight": "lampu depan",
            "taillight": "lampu belakang",
            "sein": "lampu sein",
            "indicator": "lampu sein",
            "klakson": "klakson",
            "horn": "klakson",
            "speedometer": "speedometer",
            "spido": "speedometer",
            "rpm": "rpm",
            "tachometer": "tachometer",
            "fuel gauge": "indikator bensin",
            "temperature gauge": "indikator suhu",
            
            # Masalah motor gaul
            "mogok": "tidak bisa hidup",
            "brebet": "tersendat",
            "mbrebet": "tersendat",
            "ngebul": "mengeluarkan asap",
            "ngorok": "berbunyi kasar",
            "ngeden": "susah hidup",
            "overheat": "overheat",
            "panas": "overheat",
            "bocor": "bocor",
            "leak": "bocor",
            "aus": "aus",
            "wear": "aus",
            "patah": "patah",
            "break": "patah",
            "bengkok": "bengkok",
            "bent": "bengkok",
            "macet": "macet",
            "stuck": "macet",
            "longgar": "longgar",
            "loose": "longgar",
            "kencang": "kencang",
            "tight": "kencang",
            "kotor": "kotor",
            "dirty": "kotor",
            "bersih": "bersih",
            "clean": "bersih",
            "rusak": "rusak",
            "damage": "rusak",
            "broken": "rusak",
            "normal": "normal",
            "ok": "baik",
            "oke": "baik",
            "bagus": "baik",
            "good": "baik",
            "jelek": "buruk",
            "bad": "buruk",
            "parah": "buruk",
            "terrible": "buruk",
            
            # Aksi perbaikan gaul
            "benerin": "memperbaiki",
            "repair": "memperbaiki",
            "fix": "memperbaiki",
            "ganti": "mengganti",
            "replace": "mengganti",
            "change": "mengganti",
            "setel": "menyetel",
            "adjust": "menyetel",
            "tune": "menyetel",
            "bersihkan": "membersihkan",
            "clean": "membersihkan",
            "wash": "mencuci",
            "cuci": "mencuci",
            "isi": "mengisi",
            "fill": "mengisi",
            "tambah": "menambah",
            "add": "menambah",
            "kurangi": "mengurangi",
            "reduce": "mengurangi",
            "cek": "memeriksa",
            "check": "memeriksa",
            "periksa": "memeriksa",
            "inspect": "memeriksa",
            "test": "menguji",
            "tes": "menguji",
            "uji": "menguji",
            "service": "servis",
            "servis": "servis",
            "maintenance": "perawatan",
            "perawatan": "perawatan",
            "overhaul": "overhaul",
            "rebuild": "rebuild",
            "restore": "restore",
            "upgrade": "upgrade",
            "modif": "modifikasi",
            "modify": "modifikasi",
            "custom": "kustomisasi",
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Normalisasi teks dari bahasa gaul ke bahasa formal
        
        Args:
            text: Input teks yang mungkin mengandung bahasa gaul
            
        Returns:
            str: Teks yang sudah dinormalisasi ke bahasa formal
        """
        if not text:
            return text
            
        # Convert to lowercase untuk matching
        normalized = text.lower()
        
        # Remove extra whitespaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Normalize motorcycle specific slang first (lebih spesifik)
        for slang, formal in self.motorcycle_slang.items():
            # Word boundary matching untuk menghindari partial replacement
            pattern = r'\b' + re.escape(slang) + r'\b'
            normalized = re.sub(pattern, formal, normalized)
        
        # Normalize general slang
        for slang, formal in self.slang_dict.items():
            pattern = r'\b' + re.escape(slang) + r'\b'
            normalized = re.sub(pattern, formal, normalized)
        
        # Normalize abbreviations
        for abbrev, full in self.abbreviation_dict.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            normalized = re.sub(pattern, full, normalized)
        
        # Clean up multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove empty words
        words = [word for word in normalized.split() if word.strip()]
        normalized = ' '.join(words)
        
        return normalized
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords penting dari teks yang sudah dinormalisasi
        
        Args:
            text: Teks yang sudah dinormalisasi
            
        Returns:
            List[str]: List keywords penting
        """
        normalized = self.normalize_text(text)
        
        # Keywords penting untuk motor
        important_keywords = {
            'masalah', 'rusak', 'bocor', 'aus', 'patah', 'bengkok', 'macet',
            'overheat', 'mogok', 'tersendat', 'berbunyi', 'susah', 'tidak',
            'mesin', 'rem', 'ban', 'oli', 'bensin', 'aki', 'busi', 'koil',
            'karburator', 'injeksi', 'transmisi', 'kopling', 'shock', 'fork',
            'rantai', 'velg', 'kampas', 'cakram', 'filter', 'spul', 'kiprok',
            'starter', 'lampu', 'klakson', 'speedometer', 'knalpot',
            'memperbaiki', 'mengganti', 'menyetel', 'membersihkan', 'memeriksa',
            'servis', 'perawatan', 'biaya', 'harga', 'estimasi', 'tools',
            'alat', 'waktu', 'jam', 'hari', 'mudah', 'sulit', 'sedang'
        }
        
        words = normalized.split()
        keywords = [word for word in words if word in important_keywords]
        
        return keywords
    
    def get_confidence_score(self, original: str, normalized: str) -> float:
        """
        Hitung confidence score normalisasi
        
        Args:
            original: Teks asli
            normalized: Teks yang sudah dinormalisasi
            
        Returns:
            float: Confidence score (0.0 - 1.0)
        """
        if not original or not normalized:
            return 0.0
            
        original_words = set(original.lower().split())
        normalized_words = set(normalized.split())
        
        # Hitung berapa banyak kata yang berubah
        changed_words = len(original_words - normalized_words)
        total_words = len(original_words)
        
        if total_words == 0:
            return 1.0
            
        # Confidence tinggi jika banyak kata yang berhasil dinormalisasi
        change_ratio = changed_words / total_words
        confidence = min(1.0, 0.5 + (change_ratio * 0.5))
        
        return confidence
    
    def preprocess_for_ai(self, user_input: str) -> Dict[str, any]:
        """
        Preprocess input user untuk AI dengan informasi lengkap
        
        Args:
            user_input: Input asli dari user
            
        Returns:
            Dict: Dictionary berisi normalized text, keywords, confidence, dll
        """
        normalized = self.normalize_text(user_input)
        keywords = self.extract_keywords(normalized)
        confidence = self.get_confidence_score(user_input, normalized)
        
        return {
            'original': user_input,
            'normalized': normalized,
            'keywords': keywords,
            'confidence': confidence,
            'has_slang': confidence > 0.5,
            'word_count': len(normalized.split()),
            'keyword_count': len(keywords)
        }

# Test function
if __name__ == "__main__":
    preprocessor = IndonesianSlangPreprocessor()
    
    # Test cases
    test_cases = [
        "gue punya motor matic, tapi kok mogok terus ya?",
        "motor gw ngebul banget nih, kenapa ya?",
        "rem motor ane blong, gimana benerin nya?",
        "oli motor udah item, hrs ganti ga?",
        "busi motor kotor banget, bs bersihin sendiri ga?",
        "motor brebet kalo pagi, knp ya?",
        "shock motor bunyi ngorok, parah ga?",
        "rantai motor kendor, bahaya ga sih?",
        "aki motor soak, estimasi biaya ganti brp?",
        "karbu motor banjir terus, solusinya apa?"
    ]
    
    print("=== Test Indonesian Slang Preprocessor ===")
    for i, test in enumerate(test_cases, 1):
        result = preprocessor.preprocess_for_ai(test)
        print(f"\nTest {i}:")
        print(f"Original: {result['original']}")
        print(f"Normalized: {result['normalized']}")
        print(f"Keywords: {result['keywords']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Has Slang: {result['has_slang']}")
        print("-" * 50)