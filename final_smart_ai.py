import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import difflib

@dataclass
class SmartContext:
    """Konteks percakapan yang sangat pintar"""
    session_id: str
    conversation_flow: List[Dict[str, Any]]
    problem_context: Dict[str, Any]
    user_knowledge: Dict[str, Any]
    current_diagnosis: Optional[Dict[str, Any]]
    confidence_history: List[float]
    
class FinalSmartMotorcycleAI:
    """AI Motor Pintar dengan Context Awareness Terbaik"""
    
    def __init__(self, dataset_path: str = "ultra_clean_motorcycle_dataset.json"):
        self.dataset = self._load_dataset(dataset_path)
        self.sessions = {}  # session_id -> SmartContext
        self.component_map = self._build_comprehensive_component_map()
        self.context_patterns = self._build_context_patterns()
        self.follow_up_handlers = self._build_follow_up_handlers()
        
    def _load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load dataset dengan error handling yang robust"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Dataset {dataset_path} tidak ditemukan, menggunakan fallback...")
            return self._create_comprehensive_fallback()
    
    def _create_comprehensive_fallback(self) -> List[Dict[str, Any]]:
        """Dataset fallback yang lebih komprehensif"""
        return [
            {
                "id": "smart_001",
                "category": "Starter",
                "problem": "Motor susah hidup",
                "symptoms": ["susah hidup", "ga nyala", "tidak nyala", "ga idup", "susah start"],
                "severity": "sedang",
                "possible_causes": [
                    {
                        "cause": "Sistem pengapian bermasalah",
                        "probability": "tinggi",
                        "difficulty": "sedang",
                        "cost_min": 50000,
                        "cost_max": 300000,
                        "solution": "Periksa komponen pengapian: CDI, koil, kabel busi. Jika aki dan busi sudah normal, kemungkinan masalah di CDI atau koil pengapian.",
                        "tools_needed": ["multimeter", "kunci set"],
                        "time_estimate": "45-90 menit"
                    },
                    {
                        "cause": "Sistem bahan bakar tersumbat",
                        "probability": "sedang",
                        "difficulty": "sedang",
                        "cost_min": 25000,
                        "cost_max": 150000,
                        "solution": "Bersihkan karburator atau cek filter bensin. Pastikan aliran bensin lancar dari tangki ke karburator.",
                        "tools_needed": ["obeng set", "carburetor cleaner"],
                        "time_estimate": "60-120 menit"
                    }
                ],
                "keywords": ["motor", "susah", "hidup", "nyala", "start", "idup"]
            }
        ]
    
    def _build_comprehensive_component_map(self) -> Dict[str, Dict[str, Any]]:
        """Mapping komponen yang sangat lengkap"""
        return {
            'aki': {
                'keywords': ['aki', 'battery', 'accu', 'baterai', 'listrik'],
                'related_problems': ['susah hidup', 'lampu redup', 'starter lemah'],
                'typical_solutions': ['charge aki', 'ganti aki', 'cek terminal']
            },
            'busi': {
                'keywords': ['busi', 'spark plug', 'pengapian', 'api'],
                'related_problems': ['susah hidup', 'brebet', 'tenaga kurang'],
                'typical_solutions': ['ganti busi', 'bersihkan busi', 'setel gap']
            },
            'cdi': {
                'keywords': ['cdi', 'ecu', 'modul', 'pengapian'],
                'related_problems': ['susah hidup', 'mati mendadak', 'tidak ada api'],
                'typical_solutions': ['ganti cdi', 'cek kabel cdi']
            },
            'koil': {
                'keywords': ['koil', 'coil', 'ignition', 'pengapian'],
                'related_problems': ['tidak ada api', 'api lemah', 'susah hidup'],
                'typical_solutions': ['ganti koil', 'cek resistansi koil']
            },
            'karburator': {
                'keywords': ['karbu', 'karburator', 'carburetor', 'bensin'],
                'related_problems': ['susah hidup', 'boros bensin', 'brebet'],
                'typical_solutions': ['bersihkan karbu', 'setel karbu', 'ganti jet']
            },
            'filter_udara': {
                'keywords': ['filter', 'saringan', 'udara', 'air filter'],
                'related_problems': ['tenaga kurang', 'boros bensin', 'suara kasar'],
                'typical_solutions': ['bersihkan filter', 'ganti filter']
            }
        }
    
    def _build_context_patterns(self) -> Dict[str, List[str]]:
        """Pattern untuk mendeteksi konteks percakapan"""
        return {
            'follow_up_questions': [
                r'terus\s+(gimana|bagaimana|apa|dong)',
                r'lalu\s+(gimana|bagaimana|apa)',
                r'kalau\s+gitu',
                r'setelah\s+itu',
                r'masih\s+(gimana|bagaimana)',
                r'udah\s+gitu\s+masih',
                r'abis\s+itu',
                r'habis\s+itu',
                r'trus\s+(gimana|apa)',
                r'selanjutnya\s+(gimana|apa)'
            ],
            'confirmation_patterns': [
                r'udah\s+(cek|ganti|bersih|service)',
                r'sudah\s+(cek|ganti|bersih|service)',
                r'baru\s+(ganti|beli|service)',
                r'masih\s+(bagus|oke|normal)',
                r'kondisi\s+(baik|bagus|normal)'
            ],
            'problem_indicators': [
                r'(susah|sulit|ga|gak|tidak)\s+(hidup|nyala|start|idup)',
                r'(brebet|ngempos|lemah|loyo)',
                r'(berisik|berisik|kasar|aneh)',
                r'(panas|overheat|mendidih)',
                r'(boros|banyak|habis)\s+(bensin|bbm)'
            ]
        }
    
    def _build_follow_up_handlers(self) -> Dict[str, callable]:
        """Handler untuk berbagai jenis follow-up"""
        return {
            'component_confirmation': self._handle_component_confirmation,
            'follow_up_question': self._handle_follow_up_question,
            'problem_description': self._handle_problem_description
        }
    
    def _get_or_create_context(self, session_id: str) -> SmartContext:
        """Dapatkan atau buat konteks yang pintar"""
        if session_id not in self.sessions:
            self.sessions[session_id] = SmartContext(
                session_id=session_id,
                conversation_flow=[],
                problem_context={
                    'main_problem': None,
                    'symptoms': [],
                    'checked_components': [],
                    'excluded_causes': [],
                    'current_focus': None
                },
                user_knowledge={
                    'technical_level': 'beginner',
                    'tools_available': [],
                    'previous_actions': []
                },
                current_diagnosis=None,
                confidence_history=[]
            )
        return self.sessions[session_id]
    
    def _analyze_input_context(self, user_input: str, context: SmartContext) -> Dict[str, Any]:
        """Analisis konteks input yang sangat pintar"""
        user_lower = user_input.lower().strip()
        analysis = {
            'input_type': 'unknown',
            'confidence': 0.0,
            'extracted_info': {},
            'suggested_response_type': 'diagnosis',
            'context_clues': []
        }
        
        # Deteksi jenis input
        if len(user_input.strip()) < 15 and len(context.conversation_flow) > 0:
            analysis['input_type'] = 'short_follow_up'
            analysis['context_clues'].append('Input pendek setelah percakapan')
        
        # Cek pattern follow-up
        for pattern in self.context_patterns['follow_up_questions']:
            if re.search(pattern, user_lower):
                analysis['input_type'] = 'follow_up_question'
                analysis['context_clues'].append(f'Pattern follow-up: {pattern}')
                break
        
        # Cek pattern konfirmasi
        for pattern in self.context_patterns['confirmation_patterns']:
            if re.search(pattern, user_lower):
                analysis['input_type'] = 'component_confirmation'
                analysis['context_clues'].append(f'Konfirmasi komponen: {pattern}')
                # Extract komponen yang dikonfirmasi
                self._extract_confirmed_components(user_input, context, analysis)
                break
        
        # Cek indikator masalah baru
        for pattern in self.context_patterns['problem_indicators']:
            if re.search(pattern, user_lower):
                analysis['input_type'] = 'problem_description'
                analysis['context_clues'].append(f'Deskripsi masalah: {pattern}')
                break
        
        # Jika input sangat pendek dan ada konteks sebelumnya
        if analysis['input_type'] == 'short_follow_up':
            analysis = self._enhance_short_input_analysis(user_input, context, analysis)
        
        return analysis
    
    def _extract_confirmed_components(self, user_input: str, context: SmartContext, analysis: Dict[str, Any]):
        """Extract komponen yang sudah dikonfirmasi user"""
        user_lower = user_input.lower()
        
        for component, info in self.component_map.items():
            for keyword in info['keywords']:
                if keyword in user_lower:
                    if component not in context.problem_context['checked_components']:
                        context.problem_context['checked_components'].append(component)
                    
                    # Tentukan status komponen
                    if any(word in user_lower for word in ['bagus', 'oke', 'normal', 'baik']):
                        analysis['extracted_info'][component] = 'good'
                    elif any(word in user_lower for word in ['ganti', 'baru']):
                        analysis['extracted_info'][component] = 'replaced'
                    else:
                        analysis['extracted_info'][component] = 'checked'
    
    def _enhance_short_input_analysis(self, user_input: str, context: SmartContext, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance analisis untuk input pendek dengan konteks"""
        # Jika ada diagnosis sebelumnya
        if context.current_diagnosis:
            analysis['suggested_response_type'] = 'contextual_next_steps'
            analysis['context_clues'].append('Ada diagnosis sebelumnya')
            analysis['confidence'] = 0.8
        
        # Jika ada komponen yang sudah dicek
        if context.problem_context['checked_components']:
            analysis['suggested_response_type'] = 'progressive_diagnosis'
            analysis['context_clues'].append(f"Komponen dicek: {context.problem_context['checked_components']}")
            analysis['confidence'] = 0.7
        
        # Jika ada masalah utama yang teridentifikasi
        if context.problem_context['main_problem']:
            analysis['suggested_response_type'] = 'focused_troubleshooting'
            analysis['context_clues'].append(f"Masalah utama: {context.problem_context['main_problem']}")
            analysis['confidence'] = 0.9
        
        return analysis
    
    def _find_smart_matches(self, user_input: str, context: SmartContext, analysis: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        """Cari matches dengan algoritma yang sangat pintar"""
        matches = []
        
        for item in self.dataset:
            similarity = self._calculate_smart_similarity(user_input, item, context, analysis)
            if similarity > 0.05:  # Threshold minimum diturunkan
                matches.append((item, similarity))
        
        # Sort berdasarkan similarity dan konteks
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:3]
    
    def _calculate_smart_similarity(self, user_input: str, item: Dict[str, Any], 
                                  context: SmartContext, analysis: Dict[str, Any]) -> float:
        """Hitung similarity dengan algoritma pintar"""
        base_score = 0.0
        user_lower = user_input.lower()
        
        # Jika ini follow-up atau input pendek, gunakan konteks
        if analysis['input_type'] in ['short_follow_up', 'follow_up_question']:
            base_score = self._calculate_contextual_similarity(item, context)
        else:
            # Hitung similarity normal
            base_score = self._calculate_normal_similarity(user_lower, item)
        
        # Boost berdasarkan komponen yang sudah dicek
        if context.problem_context['checked_components']:
            base_score = self._apply_component_boost(base_score, item, context)
        
        # Penalty untuk penyebab yang sudah di-exclude
        if context.problem_context['excluded_causes']:
            base_score = self._apply_exclusion_penalty(base_score, item, context)
        
        # Boost berdasarkan confidence history
        if context.confidence_history:
            avg_confidence = sum(context.confidence_history) / len(context.confidence_history)
            if avg_confidence < 0.5:  # Jika confidence rendah, coba pendekatan berbeda
                base_score *= 1.2
        
        return min(base_score, 1.0)
    
    def _calculate_contextual_similarity(self, item: Dict[str, Any], context: SmartContext) -> float:
        """Hitung similarity berdasarkan konteks percakapan"""
        score = 0.0
        
        # Jika ada masalah utama yang sama
        if context.problem_context['main_problem']:
            if context.problem_context['main_problem'].lower() in item.get('problem', '').lower():
                score += 0.5
        
        # Jika kategori sesuai dengan komponen yang dicek
        checked_components = context.problem_context['checked_components']
        for component in checked_components:
            component_info = self.component_map.get(component, {})
            related_problems = component_info.get('related_problems', [])
            
            for related_problem in related_problems:
                if related_problem.lower() in item.get('problem', '').lower():
                    score += 0.3
        
        # Boost jika symptoms match dengan konteks
        context_symptoms = context.problem_context.get('symptoms', [])
        item_symptoms = item.get('symptoms', [])
        
        for ctx_symptom in context_symptoms:
            for item_symptom in item_symptoms:
                if ctx_symptom.lower() in item_symptom.lower():
                    score += 0.2
        
        return score
    
    def _calculate_normal_similarity(self, user_lower: str, item: Dict[str, Any]) -> float:
        """Hitung similarity normal dengan prioritas konteks dan keyword matching yang lebih akurat"""
        score = 0.0
        
        # PRIORITAS TINGGI: Deteksi kata kunci spesifik komponen dengan exact matching
        component_keywords = {
            'rem': ['rem blong', 'blong', 'rem pakem', 'pakem', 'kampas rem', 'minyak rem', 'rem', 'brake', 'kampas', 'cakram', 'master'],
            'mesin': ['piston macet', 'bunyi mesin', 'suara mesin', 'mesin', 'engine', 'piston', 'silinder'],
            'transmisi': ['rantai', 'chain', 'transmisi', 'gear', 'cvt', 'belt', 'roller', 'v-belt', 'pulley', 'kopling'],
            'kelistrikan': ['aki', 'battery', 'listrik', 'sekring', 'starter', 'spul', 'stator', 'kabel'],
            'pengapian': ['busi', 'spark', 'pengapian', 'koil', 'cdi', 'ecu', 'pulser', 'api', 'percikan'],
            'pelumasan': ['oli mesin bocor', 'filter oli', 'bocor oli', 'pelumasan', 'oli', 'oil', 'pelumas'],
            'pendinginan': ['radiator', 'coolant', 'overheat', 'panas', 'kipas', 'thermostat', 'pendingin']
        }
        
        # Cek apakah user menyebutkan komponen spesifik dengan prioritas keyword yang lebih spesifik
        mentioned_component = None
        best_keyword_specificity = 0
        matched_keyword = None
        
        for component, keywords in component_keywords.items():
            for keyword in keywords:
                if keyword in user_lower:
                    # Prioritaskan keyword yang lebih spesifik (lebih panjang)
                    keyword_specificity = len(keyword)
                    if keyword_specificity > best_keyword_specificity:
                        mentioned_component = component
                        best_keyword_specificity = keyword_specificity
                        matched_keyword = keyword
        
        # Jika user menyebutkan komponen spesifik, prioritaskan kategori yang sesuai
        if mentioned_component:
            item_category = item.get('category', '').lower()
            item_keywords = [k.lower() for k in item.get('keywords', [])]
            item_symptoms = [s.lower() for s in item.get('symptoms', [])]
            
            # Boost tinggi untuk exact category match
            if mentioned_component == 'rem' and ('rem' in item_category or 'pengereman' in item_category):
                score += 0.8  # Boost sangat tinggi untuk rem
                # Extra boost jika keyword spesifik rem juga match
                if any(rem_word in ' '.join(item_keywords + item_symptoms) for rem_word in ['blong', 'pakem', 'kampas']):
                    score += 0.2
            elif mentioned_component == 'mesin' and 'mesin' in item_category:
                score += 0.6
            elif mentioned_component == 'transmisi' and ('transmisi' in item_category or 'cvt' in item_category):
                score += 0.6
            elif mentioned_component == 'kelistrikan' and 'kelistrikan' in item_category:
                score += 0.6
            elif mentioned_component == 'pengapian' and ('pengapian' in item_category or 'starter' in item_category):
                score += 0.6
            elif mentioned_component == 'pelumasan' and 'pelumasan' in item_category:
                score += 0.6
            elif mentioned_component == 'pendinginan' and 'pendinginan' in item_category:
                score += 0.6
        
        # Check symptoms dengan bobot yang disesuaikan
        symptoms = item.get('symptoms', [])
        if symptoms:
            symptom_matches = sum(1 for symptom in symptoms if symptom.lower() in user_lower)
            score += (symptom_matches / len(symptoms)) * 0.25
        
        # Check keywords dengan bobot yang disesuaikan dan exact match boost
        keywords = item.get('keywords', [])
        if keywords:
            keyword_matches = 0
            for keyword in keywords:
                if keyword.lower() in user_lower:
                    keyword_matches += 1
                    # Extra boost untuk keyword rem yang spesifik
                    if keyword.lower() in ['blong', 'pakem', 'kampas'] and any(rem_word in user_lower for rem_word in ['blong', 'pakem', 'makan', 'rem']):
                        keyword_matches += 1  # Double count untuk rem keywords
            
            score += (keyword_matches / len(keywords)) * 0.2
        
        # Check problem similarity dengan bobot yang disesuaikan
        problem = item.get('problem', '').lower()
        problem_similarity = difflib.SequenceMatcher(None, user_lower, problem).ratio()
        score += problem_similarity * 0.15
        
        return score
    
    def _apply_component_boost(self, base_score: float, item: Dict[str, Any], context: SmartContext) -> float:
        """Apply boost berdasarkan komponen yang sudah dicek"""
        checked_components = context.problem_context['checked_components']
        
        for component in checked_components:
            component_info = self.component_map.get(component, {})
            
            # Jika item ini terkait dengan komponen yang sudah dicek
            for keyword in component_info.get('keywords', []):
                if keyword in item.get('problem', '').lower():
                    # Jika komponen sudah dicek dan bagus, kurangi score untuk masalah terkait komponen itu
                    base_score *= 0.8
                    break
        
        return base_score
    
    def _apply_exclusion_penalty(self, base_score: float, item: Dict[str, Any], context: SmartContext) -> float:
        """Apply penalty untuk penyebab yang sudah di-exclude"""
        excluded_causes = context.problem_context['excluded_causes']
        
        for cause_info in item.get('possible_causes', []):
            cause_text = cause_info.get('cause', '').lower()
            
            for excluded in excluded_causes:
                if excluded.lower() in cause_text:
                    base_score *= 0.6
                    break
        
        return base_score
    
    def _generate_smart_response(self, user_input: str, context: SmartContext, 
                               analysis: Dict[str, Any], matches: List[Tuple[Dict[str, Any], float]]) -> str:
        """Generate respons yang sangat pintar berdasarkan konteks"""
        
        if analysis['input_type'] == 'component_confirmation':
            return self._handle_component_confirmation(user_input, context, analysis)
        
        elif analysis['input_type'] in ['short_follow_up', 'follow_up_question']:
            return self._handle_follow_up_question(context, matches)
        
        elif analysis['input_type'] == 'problem_description':
            return self._handle_problem_description(matches, context)
        
        else:
            return self._handle_general_diagnosis(matches, context, analysis)
    
    def _handle_component_confirmation(self, user_input: str, context: SmartContext, analysis: Dict[str, Any]) -> str:
        """Handle konfirmasi komponen dengan bahasa gaul dan natural"""
        response = []
        
        response.append("✅ **Oke, gue catat nih info komponennya**")
        response.append("")
        
        for component, status in analysis['extracted_info'].items():
            component_info = self.component_map.get(component, {})
            
            if status == 'good':
                response.append(f"• {component.upper()}: Kondisi bagus ya ✓")
                # Exclude penyebab terkait komponen ini
                for related_problem in component_info.get('related_problems', []):
                    if related_problem not in context.problem_context['excluded_causes']:
                        context.problem_context['excluded_causes'].append(related_problem)
            
            elif status == 'replaced':
                response.append(f"• {component.upper()}: Udah ganti baru ya ✓")
                # Exclude penyebab terkait komponen ini
                for related_problem in component_info.get('related_problems', []):
                    if related_problem not in context.problem_context['excluded_causes']:
                        context.problem_context['excluded_causes'].append(related_problem)
        
        response.append("")
        response.append("🎯 **Berdasarkan info ini, fokus ke:**")
        
        # Suggest next components to check
        remaining_components = [comp for comp in self.component_map.keys() 
                              if comp not in context.problem_context['checked_components']]
        
        if remaining_components:
            for i, comp in enumerate(remaining_components[:3], 1):
                comp_info = self.component_map[comp]
                response.append(f"{i}. **{comp.upper()}** - {', '.join(comp_info['typical_solutions'][:2])}")
        
        return "\n".join(response)
    
    def _handle_follow_up_question(self, context: SmartContext, matches: List[Tuple[Dict[str, Any], float]]) -> str:
        """Handle pertanyaan follow-up"""
        response = []
        
        response.append("🔄 **LANGKAH SELANJUTNYA BERDASARKAN KONTEKS**")
        response.append("")
        
        if context.problem_context['checked_components']:
            response.append("✅ **YANG SUDAH DICEK:**")
            for comp in context.problem_context['checked_components']:
                response.append(f"• {comp.upper()}")
            response.append("")
        
        if matches:
            best_match, confidence = matches[0]
            
            # Filter causes berdasarkan yang sudah di-exclude
            remaining_causes = []
            for cause in best_match.get('possible_causes', []):
                cause_text = cause.get('cause', '').lower()
                is_excluded = any(excluded.lower() in cause_text 
                                for excluded in context.problem_context['excluded_causes'])
                if not is_excluded:
                    remaining_causes.append(cause)
            
            if remaining_causes:
                response.append("🎯 **KEMUNGKINAN PENYEBAB YANG TERSISA:**")
                response.append("")
                
                for i, cause in enumerate(remaining_causes[:2], 1):
                    response.append(f"**{i}. {cause['cause']}**")
                    
                    # Parse estimated_cost
                    estimated_cost = cause.get('estimated_cost', '0-0')
                    if '-' in estimated_cost:
                        cost_parts = estimated_cost.split('-')
                        cost_min = int(cost_parts[0])
                        cost_max = int(cost_parts[1])
                    else:
                        cost_min = cost_max = int(estimated_cost)
                    
                    # Ambil solutions dari best_match, bukan dari cause
                    solutions = best_match.get('solutions', [])
                    if solutions:
                        response.append(f"   🔧 Solusi: {', '.join(solutions)}")
                    else:
                        response.append(f"   🔧 Solusi: Perlu diagnosis lebih lanjut")
                    
                    response.append(f"   💰 Biaya: Rp {cost_min:,} - Rp {cost_max:,}")
                    response.append(f"   ⏱️ Waktu: {cause.get('repair_time', 'N/A')}")
                    response.append("")
            else:
                response.append("🏥 **REKOMENDASI: BAWA KE BENGKEL**")
                response.append("Berdasarkan yang sudah dicek, kemungkinan perlu diagnosis profesional.")
        
        return "\n".join(response)
    
    def _handle_problem_description(self, matches: List[Tuple[Dict[str, Any], float]], context: SmartContext) -> str:
        """Handle deskripsi masalah baru"""
        if not matches:
            return self._generate_clarification_request(context)
        
        best_match, confidence = matches[0]
        
        # Update context dengan masalah utama
        context.problem_context['main_problem'] = best_match['problem']
        
        return self._format_comprehensive_diagnosis(best_match, confidence, context)
    
    def _handle_general_diagnosis(self, matches: List[Tuple[Dict[str, Any], float]], 
                                context: SmartContext, analysis: Dict[str, Any]) -> str:
        """Handle diagnosis umum dengan konteks pintar dan bahasa gaul"""
        if not matches:
            return self._generate_clarification_request(context)
        
        best_match, confidence = matches[0]
        
        # Cek apakah masih dalam konteks yang sama
        is_same_context = self._is_same_problem_context(best_match, context)
        
        # Update main problem context jika belum ada atau beda konteks
        if not context.problem_context.get('main_problem') or not is_same_context:
            context.problem_context['main_problem'] = best_match.get('problem', '')
            context.problem_context['main_category'] = best_match.get('category', '')
            context.problem_context['main_keywords'] = best_match.get('keywords', [])
        
        return self._format_comprehensive_diagnosis(best_match, confidence, context)
    
    def _format_comprehensive_diagnosis(self, match: Dict[str, Any], confidence: float, context: SmartContext) -> str:
        """Format diagnosis dengan bahasa gaul yang natural dan detail komprehensif"""
        response = []
        
        # Header dengan emoji berdasarkan severity
        severity_emoji = {'ringan': '🟢', 'sedang': '🟡', 'berat': '🔴'}
        emoji = severity_emoji.get(match.get('severity', 'sedang'), '🔵')
        
        # Cek apakah ini masih dalam konteks yang sama
        is_same_context = self._is_same_problem_context(match, context)
        
        if is_same_context and len(context.conversation_flow) > 0:
            # Respons follow-up yang natural
            if confidence < 0.6:
                response.append("🤔 Hmm, masih agak bingung nih tapi coba deh...")
            else:
                response.append("💡 Oke bro, gue yakin banget nih masalahnya!")
        else:
            # Respons diagnosis baru
            if confidence > 0.8:
                response.append(f"{emoji} **Wah, gue yakin banget nih masalahnya!**")
            elif confidence > 0.6:
                response.append(f"{emoji} **Kayaknya sih ini masalahnya...**")
            else:
                response.append(f"{emoji} **Agak ragu sih, tapi coba aja dulu...**")
        
        # Detail masalah yang lebih komprehensif
        response.append(f"📂 **Kategori:** {match['category']}")
        response.append(f"🔧 **Masalah:** {match.get('problem', 'Tidak diketahui')}")
        
        # Gejala yang dialami
        symptoms = match.get('symptoms', [])
        if symptoms:
            response.append(f"🩺 **Gejala yang kamu alami:** {', '.join(symptoms)}")
        
        # Penjelasan severity yang lebih detail
        if match.get('severity') == 'berat':
            response.append("⚠️ **BAHAYA TINGGI!** Ini masalah serius yang bisa bikin celaka. Stop pake motor sekarang juga!")
            response.append("🚨 Dampak: Bisa rusak parah, kecelakaan, atau bahaya nyawa")
        elif match.get('severity') == 'sedang':
            response.append("⚠️ **Perlu Perhatian** - Lumayan serius, tapi masih bisa dihandle dengan hati-hati")
            response.append("⚡ Dampak: Performa menurun, bisa jadi masalah besar kalau diabaikan")
        else:
            response.append("⚠️ **Masalah Ringan** - Santai aja, ga terlalu parah tapi tetep perlu diperbaiki")
            response.append("✅ Dampak: Gangguan kecil, masih aman dipake dengan hati-hati")
        
        # Konteks percakapan
        if len(context.conversation_flow) > 0:
            response.append(f"💬 Udah ngobrol {len(context.conversation_flow) + 1} kali nih")
        
        response.append("")
        
        # Possible causes dengan filtering pintar
        possible_causes = match.get('possible_causes', [])
        filtered_causes = self._filter_causes_by_context(possible_causes, context)
        
        if is_same_context and len(context.conversation_flow) > 0:
            response.append("🔍 **Oke, jadi gini nih kemungkinannya:**")
        else:
            response.append("🔍 **Analisis masalahnya:**")
        response.append("")
        
        for i, cause in enumerate(filtered_causes, 1):
            prob_emoji = {'tinggi': '🔴', 'sedang': '🟡', 'rendah': '🟢'}
            diff_emoji = {'mudah': '✅', 'sedang': '⚠️', 'sulit': '❌'}
            
            prob = cause.get('probability', 'sedang')
            difficulty = cause.get('difficulty', 'sedang')
            
            # Parse estimated_cost yang dalam format "min-max"
            estimated_cost = cause.get('estimated_cost', '0-0')
            if '-' in estimated_cost:
                cost_parts = estimated_cost.split('-')
                cost_min = int(cost_parts[0])
                cost_max = int(cost_parts[1])
            else:
                cost_min = cost_max = int(estimated_cost)
            
            response.append(f"**{i}. {cause['cause']}**")
            
            # Tambahkan note jika sudah dicek
            if cause.get('already_checked'):
                response.append(f"   ℹ️ {cause['already_checked']}")
            
            # Natural probability text
            if prob == 'tinggi':
                response.append(f"   {prob_emoji.get(prob, '🔵')} Hmm ini kemungkinan besar sih bro")
            elif prob == 'sedang':
                response.append(f"   {prob_emoji.get(prob, '🔵')} Bisa jadi nih, lumayan mungkin")
            elif prob == 'rendah':
                response.append(f"   {prob_emoji.get(prob, '🔵')} Kecil kemungkinannya sih")
            else:
                response.append(f"   {prob_emoji.get(prob, '🔵')} Ga yakin juga nih probabilitasnya")
            
            # Natural difficulty text
            if difficulty == 'mudah':
                response.append(f"   {diff_emoji.get(difficulty, '⚠️')} Gampang kok ini mah")
            elif difficulty == 'sedang':
                response.append(f"   {diff_emoji.get(difficulty, '⚠️')} Lumayan ribet dikit")
            elif difficulty == 'sulit':
                response.append(f"   {diff_emoji.get(difficulty, '⚠️')} Waduh susah nih, mending ke bengkel")
            else:
                response.append(f"   {diff_emoji.get(difficulty, '⚠️')} Ga tau seberapa susahnya")
            
            if cost_max > 0:
                if cost_min == cost_max:
                    response.append(f"   💰 Kira-kira abis: Rp {cost_min:,}")
                else:
                    response.append(f"   💰 Kira-kira abis: Rp {cost_min:,} - Rp {cost_max:,}")
            else:
                response.append(f"   💰 Gratis kok ini")
            
            # Ambil solutions dari item utama, bukan dari cause
            solutions = match.get('solutions', [])
            if solutions:
                response.append(f"   🔧 Yang perlu dilakuin: {', '.join(solutions)}")
            else:
                response.append(f"   🔧 Perlu diagnosis lebih lanjut nih")
            
            # Ambil tools dari item utama
            tools = match.get('tools_needed', [])
            if tools:
                response.append(f"   🛠️ Alat yang dibutuhin: {', '.join(tools)}")
            
            # Parse repair_time
            time_est = cause.get('repair_time', 'N/A')
            response.append(f"   ⏱️ Kira-kira butuh waktu: {time_est}")
            response.append("")
        
        # Tips dengan bahasa gaul yang lebih natural
        response.append("💡 **Saran gue:**")
        
        if len(context.conversation_flow) > 3:
            response.append("• 🔄 Udah lama ngobrolnya nih, mending langsung bawa ke bengkel aja deh biar pasti")
        
        if len(context.problem_context['checked_components']) > 2:
            response.append("• 🔧 Udah banyak yang dicek tapi masih bermasalah, kayaknya emang ribet nih masalahnya")
        
        if match.get('severity') == 'berat':
            response.append("• ⚠️ Serius nih bahaya banget, jangan dipake dulu motornya! Bisa celaka nanti")
        
        if confidence < 0.5:
            response.append("• 🤔 Gue masih ragu-ragu nih, coba kasih info lebih detail lagi dong biar gue bisa bantu lebih akurat")
        
        response.append("")
        
        # Penjelasan teknis yang lebih mendalam
        response.append("🔬 **Penjelasan Teknis:**")
        technical_explanation = self._get_technical_explanation(match)
        if technical_explanation:
            response.append(technical_explanation)
        
        response.append("")
        
        # Tips pencegahan untuk masa depan
        response.append("🛡️ **Tips Pencegahan Biar Ga Kejadian Lagi:**")
        prevention_tips = self._get_prevention_tips(match)
        for tip in prevention_tips:
            response.append(f"• {tip}")
        
        response.append("")
        
        # Kapan harus ke bengkel
        response.append("🏪 **Kapan Harus ke Bengkel:**")
        if match.get('severity') == 'berat':
            response.append("• 🚨 SEKARANG JUGA! Jangan tunda lagi, bahaya!")
        elif match.get('severity') == 'sedang':
            response.append("• ⏰ Dalam 1-2 hari ini, jangan sampai lebih parah")
        else:
            response.append("• 📅 Kalau ada waktu luang, tapi jangan lama-lama")
        
        if confidence < 0.6:
            response.append("• 🤷‍♂️ Kalau masih bingung, mending langsung konsultasi ke mekanik")
        
        return "\n".join(response)
    
    def _get_technical_explanation(self, match: Dict[str, Any]) -> str:
        """Memberikan penjelasan teknis yang mudah dipahami"""
        category = match.get('category', '').lower()
        problem = match.get('problem', '').lower()
        
        explanations = {
            'mesin': "Mesin motor itu jantungnya kendaraan bro. Kalau ada masalah di sini, bisa ganggu performa keseluruhan. Biasanya karena komponen aus, pelumasan kurang, atau pembakaran ga sempurna.",
            'bahan bakar': "Sistem bahan bakar tugasnya nyupply bensin ke mesin dengan takaran yang pas. Kalau bermasalah, motor bisa boros, tenaga kurang, atau susah hidup.",
            'kelistrikan': "Sistem kelistrikan ngatur semua komponen elektronik motor. Dari pengapian, lampu, sampai ECU. Kalau ada yang short atau putus, bisa bikin motor mogok total.",
            'rem': "Sistem rem adalah safety utama motor. Kalau bermasalah, bisa bahaya banget karena ga bisa berhenti dengan baik. Jangan main-main sama rem!",
            'transmisi': "Transmisi tugasnya nyalurin tenaga dari mesin ke roda. Kalau bermasalah, motor bisa ga mau jalan atau tenaga hilang.",
            'suspensi': "Suspensi bikin motor nyaman dan stabil. Kalau rusak, motor jadi ga nyaman dan susah dikontrol, terutama di jalan jelek.",
            'pendingin': "Sistem pendingin jaga suhu mesin biar ga overheat. Kalau ga kerja, mesin bisa panas berlebihan dan rusak parah."
        }
        
        for key, explanation in explanations.items():
            if key in category:
                return explanation
        
        return "Ini masalah yang butuh perhatian khusus. Setiap komponen motor punya fungsi penting, jadi kalau ada yang bermasalah harus segera ditangani."
    
    def _get_prevention_tips(self, match: Dict[str, Any]) -> List[str]:
        """Memberikan tips pencegahan berdasarkan kategori masalah"""
        category = match.get('category', '').lower()
        severity = match.get('severity', 'sedang')
        
        base_tips = [
            "🔧 Service rutin setiap 3000-5000 km",
            "🛢️ Ganti oli mesin secara teratur",
            "🧽 Bersihin motor minimal seminggu sekali",
            "🌡️ Jangan biarkan mesin overheat",
            "⛽ Pake bensin yang berkualitas baik"
        ]
        
        category_tips = {
            'mesin': [
                "🔥 Panaskan mesin sebelum berkendara",
                "⚡ Jangan gas pol dari awal",
                "🛑 Matikan mesin kalau macet lama"
            ],
            'bahan bakar': [
                "⛽ Jangan sampai tangki kosong total",
                "🧪 Bersihin filter bensin berkala",
                "🚫 Hindari bensin oplosan"
            ],
            'kelistrikan': [
                "🔋 Cek aki secara rutin",
                "💡 Matikan lampu kalau ga dipake",
                "🌧️ Hindari terendam air"
            ],
            'rem': [
                "🛑 Cek ketebalan kampas rem",
                "💧 Ganti minyak rem berkala",
                "🚫 Jangan rem mendadak kalau ga darurat"
            ]
        }
        
        tips = base_tips.copy()
        for key, specific_tips in category_tips.items():
            if key in category:
                tips.extend(specific_tips)
                break
        
        if severity == 'berat':
            tips.append("🚨 Segera bawa ke bengkel resmi untuk penanganan profesional")
        
        return tips[:6]  # Batasi maksimal 6 tips biar ga terlalu panjang
    
    def _extract_corrected_context(self, user_input: str) -> Dict[str, Any]:
        """Extract konteks yang benar dari clarification user"""
        user_lower = user_input.lower()
        
        # Mapping kata kunci ke kategori yang benar
        context_mapping = {
            'lampu': {'main_category': 'Kelistrikan', 'main_keywords': ['lampu', 'bohlam', 'sein', 'reflektor', 'mika']},
            'rem': {'main_category': 'Pengereman', 'main_keywords': ['rem', 'kampas', 'blong', 'pakem', 'cakram']},
            'mesin': {'main_category': 'Mesin', 'main_keywords': ['mesin', 'piston', 'silinder', 'kompresi', 'oli']},
            'aki': {'main_category': 'Kelistrikan', 'main_keywords': ['aki', 'battery', 'setrum', 'listrik', 'starter']},
            'rantai': {'main_category': 'Transmisi', 'main_keywords': ['rantai', 'gir', 'sprocket', 'chain']},
            'ban': {'main_category': 'Ban', 'main_keywords': ['ban', 'velg', 'pentil', 'angin']},
            'karbu': {'main_category': 'Bahan Bakar', 'main_keywords': ['karburator', 'karbu', 'spuyer', 'bensin']}
        }
        
        # Cari kata kunci yang disebutkan user
        for keyword, context in context_mapping.items():
            if keyword in user_lower:
                return context
        
        return {}
    
    def _extract_main_problem_from_clarification(self, user_input: str) -> str:
        """Extract masalah utama dari clarification user"""
        user_lower = user_input.lower()
        
        # Cari kata kunci komponen yang disebutkan
        if 'lampu' in user_lower:
            return 'lampu motor kadang idup kadang mati'
        elif 'rem' in user_lower:
            return 'rem motor ga makan'
        elif 'mesin' in user_lower:
            return 'mesin motor bermasalah'
        elif 'aki' in user_lower:
            return 'aki motor bermasalah'
        elif 'rantai' in user_lower:
            return 'rantai motor bermasalah'
        elif 'ban' in user_lower:
            return 'ban motor bermasalah'
        elif 'karbu' in user_lower:
            return 'karburator motor bermasalah'
        
        return user_input
    
    def _is_same_problem_context(self, match: Dict[str, Any], context: SmartContext) -> bool:
        """Cek apakah masih dalam konteks masalah yang sama dengan improved context tracking"""
        if not context.problem_context.get('main_problem'):
            return False
        
        current_problem = match.get('problem', '').lower()
        main_problem = context.problem_context['main_problem'].lower()
        
        # Cek kesamaan kategori dan keywords
        current_category = match.get('category', '').lower()
        main_category = context.problem_context.get('main_category', '').lower()
        
        # Jika kategori sama atau ada overlap keywords yang signifikan
        if current_category == main_category:
            return True
        
        # Cek overlap keywords dengan threshold yang lebih ketat
        current_keywords = set(match.get('keywords', []))
        main_keywords = set(context.problem_context.get('main_keywords', []))
        overlap = len(current_keywords.intersection(main_keywords))
        total_keywords = len(current_keywords.union(main_keywords))
        
        # Minimal 30% overlap dan minimal 2 keyword yang sama
        if total_keywords > 0 and overlap >= 2 and (overlap / total_keywords) >= 0.3:
            return True
        
        return False
    
    def _filter_causes_by_context(self, causes: List[Dict[str, Any]], context: SmartContext) -> List[Dict[str, Any]]:
        """Filter penyebab berdasarkan konteks yang sudah ada"""
        filtered = []
        
        for cause in causes:
            cause_copy = cause.copy()
            cause_text = cause.get('cause', '').lower()
            
            # Cek apakah penyebab ini terkait dengan komponen yang sudah dicek
            is_related_to_checked = False
            for component in context.problem_context['checked_components']:
                component_info = self.component_map.get(component, {})
                for keyword in component_info.get('keywords', []):
                    if keyword in cause_text:
                        cause_copy['already_checked'] = f"(Udah dicek {component}nya)"
                        is_related_to_checked = True
                        break
                if is_related_to_checked:
                    break
            
            # Cek apakah sudah di-exclude
            is_excluded = any(excluded.lower() in cause_text 
                            for excluded in context.problem_context['excluded_causes'])
            
            if not is_excluded:
                filtered.append(cause_copy)
            elif is_related_to_checked:
                # Tetap tampilkan tapi dengan note
                cause_copy['probability'] = 'rendah'  # Turunkan probabilitas
                filtered.append(cause_copy)
        
        return filtered
    
    def _is_gibberish_input(self, user_input):
        """Deteksi input yang tidak jelas atau ngawur"""
        # Cek panjang kata rata-rata
        words = user_input.split()
        if not words:
            return True
            
        # Cek jika input terlalu pendek dan tidak mengandung kata kunci motor
        if len(user_input.strip()) < 3:
            return True
            
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Jika rata-rata panjang kata > 10 karakter, kemungkinan ngawur
        if avg_word_length > 10:
            return True
            
        # Cek rasio huruf vs angka/simbol
        letters = sum(1 for c in user_input if c.isalpha())
        total_chars = len(user_input.replace(' ', ''))
        
        if total_chars > 0:
            letter_ratio = letters / total_chars
            # Jika rasio huruf < 60%, kemungkinan ngawur
            if letter_ratio < 0.6:
                return True
        
        # Cek pola keyboard mashing (huruf berulang atau pola aneh)
        # Contoh: asdfgh, qwerty, aaaaaaa, dll
        keyboard_patterns = ['qwerty', 'asdfgh', 'zxcvbn', 'qwertyuiop', 'asdfghjkl', 'zxcvbnm']
        input_lower = user_input.lower().replace(' ', '')
        
        for pattern in keyboard_patterns:
            if pattern in input_lower:
                return True
                
        # Cek huruf berulang berlebihan (lebih dari 3 huruf sama berturut-turut)
        import re
        if re.search(r'(.)\1{3,}', user_input):
            return True
            
        # Cek jika tidak ada kata yang masuk akal (semua kata > 8 karakter tanpa vokal)
        meaningful_words = 0
        for word in words:
            if len(word) <= 8 and any(vowel in word.lower() for vowel in 'aiueo'):
                meaningful_words += 1
                
        if len(words) > 2 and meaningful_words == 0:
            return True
                
        return False
    
    def _generate_clarification_request(self, context: SmartContext) -> str:
        """Generate permintaan klarifikasi yang natural dan gaul"""
        response = []
        
        if len(context.conversation_flow) > 0:
            response.append("🤔 **Eh, gue butuh klarifikasi nih**")
            response.append("")
            response.append("Dari obrolan kita tadi:")
            
            if context.problem_context['checked_components']:
                response.append(f"• Yang udah dibahas: {', '.join(context.problem_context['checked_components'])}")
            
            if context.problem_context['main_problem']:
                response.append(f"• Masalah utamanya: {context.problem_context['main_problem']}")
            
            response.append("")
            response.append("Coba jelasin lagi dong, lebih spesifik apa yang mau ditanyain?")
        else:
            response.append("🤔 **Gue butuh info lebih nih**")
            response.append("")
            response.append("Biar gue bisa bantu lo dengan akurat, kasih tau dong:")
            response.append("")
            response.append("📝 **Info yang gue butuhin:**")
            response.append("• Gejalanya gimana sih detailnya?")
            response.append("• Kapan mulai bermasalah gini?")
            response.append("• Sering kejadian atau cuma kadang-kadang?")
            response.append("• Motor apa dan tahun berapa?")
            response.append("• Baru-baru ini ada ganti part atau servis ga?")
        
        return "\n".join(response)
    
    def _is_follow_up_question(self, user_input):
        """Deteksi apakah input adalah follow-up question"""
        follow_up_patterns = [
            r'\b(bahaya|urgent|penting|serius)\b.*\b(banget|sekali|ga|tidak|kah)\b',
            r'\b(gimana|bagaimana)\b.*\b(dong|sih|nih)\b',
            r'\b(terus|lalu|abis itu)\b.*\b(gimana|apa)\b',
            r'\b(masih|udah|sudah)\b.*\b(bisa|boleh|aman)\b',
            r'\b(berapa|kapan)\b.*\b(lama|waktu|hari)\b',
            r'\b(iya|ya|oh|wah|waduh)\b.*\b(banget|sekali|parah)\b',
            r'\b(emang|memang|beneran)\b.*\b(segitu|separah|seburuk)\b',
            r'\b(masih bisa|aman ga|boleh)\b.*\b(dipake|dipakai|jalan)\b'
        ]
        
        import re
        for pattern in follow_up_patterns:
            if re.search(pattern, user_input.lower()):
                return True
        return False
    
    def _is_thank_you_message(self, user_input):
        """Deteksi ucapan terima kasih dengan lebih akurat"""
        # Pola ucapan terima kasih yang jelas
        explicit_thanks = [
            r'\b(terima kasih|makasih|thanks|thx|tengkyu|thank you)\b',
            r'\b(makasih|thanks)\b.*\b(banget|banyak|ya|bro|gan)\b',
            r'\b(terima kasih|makasih)\b.*\b(banyak|banget|ya|bro)\b'
        ]
        
        # Pola apresiasi yang menunjukkan akhir percakapan
        appreciation_endings = [
            r'^(oke|ok|baik)\s+(makasih|terima kasih|thanks)\b',
            r'^(mantap|keren|bagus)\s+(banget|sekali)\s*(makasih|thanks)?\s*(bro|gan)?$',
            r'\b(udah|sudah)\s+(jelas|paham|ngerti)\s*(makasih|thanks|terima kasih)\b',
            r'\b(siap|oke|ok)\s+(bro|gan)?\s*(makasih|thanks|terima kasih)\b'
        ]
        
        # Hindari false positive untuk pertanyaan atau keluhan
        question_indicators = [
            r'\?',  # ada tanda tanya
            r'\b(gimana|bagaimana|kenapa|mengapa|kapan|berapa)\b',
            r'\b(masih|belum|tidak|ga|gak)\b.*\b(bisa|boleh|jalan|hidup)\b',
            r'\b(susah|sulit|masalah|rusak|error)\b'
        ]
        
        import re
        user_lower = user_input.lower().strip()
        
        # Cek apakah ada indikator pertanyaan atau keluhan
        for pattern in question_indicators:
            if re.search(pattern, user_lower):
                return False
        
        # Cek pola ucapan terima kasih eksplisit
        for pattern in explicit_thanks:
            if re.search(pattern, user_lower):
                return True
                
        # Cek pola apresiasi yang menunjukkan akhir percakapan
        for pattern in appreciation_endings:
            if re.search(pattern, user_lower):
                return True
                
        return False
    
    def _handle_follow_up_natural(self, user_input, context: SmartContext):
        """Handle follow-up questions dengan natural"""
        if not context.current_diagnosis:
            return "Hmm, belum ada diagnosis sebelumnya nih bro. Coba jelasin masalahnya dulu dong!"
            
        # Analisis jenis follow-up
        if any(word in user_input.lower() for word in ['bahaya', 'urgent', 'penting', 'serius']):
            severity = context.current_diagnosis.get('severity', 'sedang')
            # Mapping severity yang lebih akurat
            if severity in ['berat', 'sangat tinggi', 'tinggi']:
                responses = [
                    "Iya bro, ini serius banget! 😰 Jangan dipake dulu motornya, bisa bahaya. Langsung ke bengkel aja ya!",
                    "Waduh iya dong, bahaya banget ini! 🚨 Motor jangan dipake dulu, nanti malah tambah rusak atau bahkan celaka.",
                    "Serius banget nih bro! ⚠️ Ini bukan main-main, safety first ya. Langsung bawa ke bengkel yang terpercaya."
                ]
            elif severity == 'sedang':
                responses = [
                    "Lumayan serius sih bro, tapi masih bisa ditangani. 🤔 Tapi jangan dibiarkan lama-lama ya!",
                    "Iya agak serius nih, tapi ga separah yang gue kira. 😅 Tetep harus segera diperbaiki sih.",
                    "Serius sih, tapi masih dalam batas wajar. 👍 Yang penting jangan ditunda-tunda perbaikannya."
                ]
            else:  # ringan
                responses = [
                    "Santai aja bro, ga terlalu bahaya kok. 😊 Tapi tetep harus diperbaiki ya biar ga tambah parah.",
                    "Ga bahaya-bahaya amat sih, masih aman. 👌 Cuma ya tetep harus dibenerin biar motor tetep prima.",
                    "Tenang bro, ini masih kategori ringan. ✅ Tapi jangan diabaikan ya, nanti malah jadi masalah besar."
                ]
            
            import random
            return random.choice(responses)
            
        elif any(word in user_input.lower() for word in ['masih bisa', 'aman ga', 'boleh']):
            severity = context.current_diagnosis.get('severity', 'sedang')
            if severity in ['berat', 'sangat tinggi', 'tinggi']:
                return "Waduh jangan bro! 🛑 Bahaya banget kalo dipake. Mending jalan kaki atau naik ojek dulu deh."
            elif severity == 'sedang':
                return "Hmm, kalo buat jarak deket sih masih bisa. 🤏 Tapi pelan-pelan ya, dan segera ke bengkel!"
            else:  # ringan
                return "Masih bisa dipake kok bro, tapi hati-hati ya. 👍 Jangan lupa segera diperbaiki."
                
        elif any(word in user_input.lower() for word in ['berapa', 'kapan', 'lama']):
            if context.current_diagnosis and 'possible_causes' in context.current_diagnosis:
                return "Dari diagnosis tadi, estimasi biaya dan waktunya udah gue kasih tau kok bro. 💰 Tapi ya tergantung bengkelnya juga sih."
            else:
                return "Untuk waktu dan biaya, tergantung tingkat kerusakannya bro. 🕐 Dari yang gue analisis tadi, kira-kira segitu deh."
                
        return "Hmm, bisa jelasin lebih spesifik ga bro? 🤔 Gue mau bantu tapi kurang ngerti maksudnya."
    
    def _handle_thank_you_natural(self, user_input):
        """Handle ucapan terima kasih dengan natural"""
        responses = [
            "Sama-sama bro! 😊 Senang banget bisa bantu. Semoga motornya cepet sembuh ya!",
            "Siap bro! 🤝 Gue seneng bisa bantu. Jangan lupa ke bengkel yang terpercaya ya!",
            "Sama-sama dong! 👍 Kapan-kapan kalo ada masalah motor lagi, langsung tanya aja ke gue.",
            "Gue seneng banget bisa bantu! 🏍️ Semoga motornya jadi kenceng lagi setelah diperbaiki.",
            "Siap siap! 😄 Senang hati bisa bantu sesama bikers. Ride safe ya bro!",
            "Sama-sama bro! 🔧 Gue harap diagnosis gue bisa membantu. Jaga motornya baik-baik ya!"
        ]
        
        import random
        return random.choice(responses)

    def diagnose(self, user_input: str, session_id: str = "default") -> str:
        """Fungsi utama diagnosis dengan AI yang sangat pintar"""
        context = self._get_or_create_context(session_id)
        
        # Cek input ngawur
        if self._is_gibberish_input(user_input):
            gibberish_responses = [
                "Waduh kenapa nih bos, kepencet ya keyboardnya? heheh soalnya ga jelas ni maksudnya 😅",
                "Eh bro, kayaknya ada yang salah deh sama inputnya. Keyboard error kah? 🤔",
                "Wah ini mah kayak kucing jalan di atas keyboard ya bro 😸 Coba ketik ulang dong",
                "Hmm... ini bahasa planet mana ya? 🛸 Bisa pake bahasa bumi ga bro?",
                "Kayaknya lagi stress ya sampe ngetik ngawur gini 😂 Santai aja, coba jelasin pelan-pelan"
            ]
            
            import random
            random_response = random.choice(gibberish_responses)
            
            return f"{random_response}\n\n💡 **Tips:** Coba jelasin masalah motornya dengan kata-kata yang jelas ya bro 😊"
        
        # Cek ucapan terima kasih
        if self._is_thank_you_message(user_input):
            return self._handle_thank_you_natural(user_input)
        
        # Cek follow-up question
        if self._is_follow_up_question(user_input) and len(context.conversation_flow) > 0:
            return self._handle_follow_up_natural(user_input, context)
        
        # Deteksi clarification/correction dari user
        clarification_keywords = [
            'bukan', 'salah', 'kok jadi', 'kan gw bahas', 'yang gw maksud', 
            'maksud gw', 'harusnya', 'sebenarnya', 'tapi kan', 'lho kok'
        ]
        
        is_clarification = any(keyword in user_input.lower() for keyword in clarification_keywords)
        
        if is_clarification and len(context.conversation_flow) > 0:
            # Handle clarification
            corrected_context = self._extract_corrected_context(user_input)
            if corrected_context:
                # Re-analyze dengan konteks yang benar
                corrected_input = self._extract_main_problem_from_clarification(user_input)
                analysis = self._analyze_input_context(corrected_input, context)
                matches = self._find_smart_matches(corrected_input, context, analysis)
                
                if matches:
                    context.problem_context.update(corrected_context)
                    response = f"🤦‍♂️ **Waduh maaf bro, salah paham gue!**\n\n" + self._generate_smart_response(corrected_input, context, analysis, matches)
                    self._update_conversation_context(user_input, response, context, analysis, matches)
                    return response
        
        # Analisis konteks input
        analysis = self._analyze_input_context(user_input, context)
        
        # Cari matches dengan algoritma pintar
        matches = self._find_smart_matches(user_input, context, analysis)
        
        # Generate respons pintar
        response = self._generate_smart_response(user_input, context, analysis, matches)
        
        # Update konteks percakapan
        self._update_conversation_context(user_input, response, context, analysis, matches)
        
        return response
    
    def _update_conversation_context(self, user_input: str, response: str, context: SmartContext, 
                                   analysis: Dict[str, Any], matches: List[Tuple[Dict[str, Any], float]]):
        """Update konteks percakapan dengan context correction dan clarification handling"""
        # Deteksi clarification/correction dari user
        clarification_keywords = [
            'bukan', 'salah', 'kok jadi', 'kan gw bahas', 'yang gw maksud', 
            'maksud gw', 'harusnya', 'sebenarnya', 'tapi kan', 'lho kok'
        ]
        
        is_clarification = any(keyword in user_input.lower() for keyword in clarification_keywords)
        
        if is_clarification and len(context.conversation_flow) > 0:
            # User sedang klarifikasi, extract konteks yang benar
            corrected_context = self._extract_corrected_context(user_input)
            if corrected_context:
                # Update main context dengan konteks yang dikoreksi
                context.problem_context.update(corrected_context)
                # Mark diagnosis terakhir sebagai corrected
                if context.conversation_flow:
                    context.conversation_flow[-1]['corrected'] = True
        
        # Tambah ke conversation flow
        context.conversation_flow.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'ai_response': response,
            'analysis': analysis,
            'confidence': matches[0][1] if matches else 0.0,
            'is_clarification': is_clarification
        })
        
        # Update confidence history
        if matches:
            context.confidence_history.append(matches[0][1])
        
        # Update current diagnosis jika bukan clarification dan ada matches
        if not is_clarification and matches:
            context.current_diagnosis = matches[0][0]
            # Tambahkan severity untuk follow-up questions
            context.current_diagnosis['severity'] = matches[0][0].get('severity', 'sedang')
        
        # Extract symptoms dari input
        user_lower = user_input.lower()
        for pattern in self.context_patterns['problem_indicators']:
            if re.search(pattern, user_lower):
                symptom = re.search(pattern, user_lower).group()
                if symptom not in context.problem_context['symptoms']:
                    context.problem_context['symptoms'].append(symptom)
    
    def get_smart_summary(self, session_id: str = "default") -> str:
        """Dapatkan ringkasan percakapan yang sangat pintar"""
        if session_id not in self.sessions:
            return "Belum ada percakapan."
        
        context = self.sessions[session_id]
        summary = []
        
        summary.append(f"📊 **RINGKASAN PERCAKAPAN PINTAR** (Session: {session_id})")
        summary.append(f"🕐 Total interaksi: {len(context.conversation_flow)}")
        
        if context.confidence_history:
            avg_confidence = sum(context.confidence_history) / len(context.confidence_history)
            summary.append(f"🎯 Rata-rata confidence: {avg_confidence:.1%}")
        
        summary.append("")
        
        if context.problem_context['main_problem']:
            summary.append(f"🔍 **MASALAH UTAMA:** {context.problem_context['main_problem']}")
            summary.append("")
        
        if context.problem_context['checked_components']:
            summary.append("🔧 **KOMPONEN YANG SUDAH DICEK:**")
            for comp in context.problem_context['checked_components']:
                summary.append(f"• {comp.upper()}")
            summary.append("")
        
        if context.problem_context['excluded_causes']:
            summary.append("❌ **PENYEBAB YANG SUDAH DI-EXCLUDE:**")
            for cause in context.problem_context['excluded_causes']:
                summary.append(f"• {cause}")
            summary.append("")
        
        if context.current_diagnosis:
            summary.append(f"🎯 **DIAGNOSIS TERKINI:** {context.current_diagnosis['problem']}")
            summary.append(f"📂 Kategori: {context.current_diagnosis['category']}")
            summary.append("")
        
        summary.append("💬 **RIWAYAT PERCAKAPAN TERAKHIR:**")
        for i, flow in enumerate(context.conversation_flow[-3:], 1):  # Show last 3
            summary.append(f"**{i}. {flow['timestamp'][:19]}**")
            summary.append(f"   User: {flow['user_input'][:50]}...")
            summary.append(f"   Confidence: {flow['confidence']:.1%}")
            summary.append("")
        
        return "\n".join(summary)

def main():
    """Testing dengan kasus terminal yang sama"""
    print("🚀 Final Smart Motorcycle AI - Ultimate Version")
    print("=" * 70)
    
    ai = FinalSmartMotorcycleAI()
    session_id = "terminal_final_test"
    
    # Test case yang sama seperti di terminal
    test_inputs = [
        "motor gua kok ga idup yak?",
        "iyaa, motornya susah dihidupin gituu",
        "aki udah dicek, masih bagus kok",
        "busi juga udah ganti baru",
        "terus gimana dong?",
        "masih ga bisa juga nih"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {user_input}")
        print(f"{'='*60}")
        
        response = ai.diagnose(user_input, session_id)
        print(response)
        
        if i == 3:  # Show summary after 3rd interaction
            print("\n" + "="*60)
            print("MID-CONVERSATION SUMMARY")
            print("="*60)
            print(ai.get_smart_summary(session_id))
    
    print("\n" + "="*60)
    print("FINAL SMART SUMMARY")
    print("="*60)
    print(ai.get_smart_summary(session_id))

if __name__ == "__main__":
    main()