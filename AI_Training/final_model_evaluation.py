import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation for motorcycle AI models"""
    
    def __init__(self):
        self.test_questions = [
            "Motor saya susah hidup di pagi hari, apa penyebabnya?",
            "Bagaimana cara merawat rem cakram motor?",
            "Motor bergetar saat idle, solusinya?",
            "Oli motor cepat habis, apa yang harus dicek?",
            "Cara setting karburator yang benar?",
            "Motor tidak bisa distarter elektrik?",
            "Tips agar motor irit bensin?",
            "Shock motor bunyi, berbahaya tidak?",
            "Rantai motor kendor, cara mengencangkan?",
            "Busi motor kotor, bisa dibersihkan?"
        ]
        
        self.evaluation_criteria = {
            'coherence': 'Apakah jawaban koheren dan mudah dipahami?',
            'relevance': 'Apakah jawaban relevan dengan pertanyaan?',
            'indonesian': 'Apakah menggunakan bahasa Indonesia yang baik?',
            'technical': 'Apakah informasi teknis akurat?',
            'completeness': 'Apakah jawaban lengkap dan informatif?'
        }
    
    def load_model(self, model_path):
        """Load model and tokenizer"""
        try:
            logger.info(f"Loading model from {model_path}")
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            model = GPT2LMHeadModel.from_pretrained(model_path)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None, None
    
    def generate_response(self, model, tokenizer, question, max_length=80):
        """Generate response for a question"""
        try:
            prompt = f"Pertanyaan: {question}\nJawaban:"
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.4,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.replace(prompt, "").strip()
            
            # Clean up answer
            if answer:
                # Remove incomplete sentences at the end
                sentences = answer.split('.')
                if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
                    answer = '.'.join(sentences[:-1]) + '.'
            
            return answer
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "[Error generating response]"
    
    def evaluate_response_quality(self, question, answer):
        """Evaluate response quality with scoring"""
        scores = {}
        
        # Coherence (0-10)
        coherence_score = 0
        if answer and len(answer) > 10:
            coherence_score += 2
        if not any(char in answer for char in ['<|', '|>', '[Error']):
            coherence_score += 2
        if len([word for word in answer.split() if len(word) > 15]) < 2:
            coherence_score += 2
        if answer.count(' ') > 5:  # Has multiple words
            coherence_score += 2
        if not any(pattern in answer.lower() for pattern in ['aaaa', 'bbbb', 'cccc']):
            coherence_score += 2
        
        scores['coherence'] = min(coherence_score, 10)
        
        # Relevance (0-10)
        relevance_score = 0
        question_keywords = ['motor', 'mesin', 'oli', 'rem', 'karbu', 'busi', 'shock', 'rantai']
        if any(keyword in answer.lower() for keyword in question_keywords):
            relevance_score += 5
        if any(word in question.lower() for word in answer.lower().split()[:5]):
            relevance_score += 3
        if len(answer) > 20:
            relevance_score += 2
        
        scores['relevance'] = min(relevance_score, 10)
        
        # Indonesian quality (0-10)
        indonesian_score = 0
        if answer:
            # Check for proper Indonesian words
            good_words = ['adalah', 'untuk', 'dengan', 'pada', 'yang', 'dan', 'atau', 'jika', 'karena']
            if any(word in answer.lower() for word in good_words):
                indonesian_score += 3
            
            # Check for bad patterns
            bad_patterns = ['gue', 'gw', 'lu', 'lo', 'wkwk', 'hehe']
            if not any(pattern in answer.lower() for pattern in bad_patterns):
                indonesian_score += 3
            
            # Check capitalization
            if answer and answer[0].isupper():
                indonesian_score += 2
            
            # Check for reasonable length
            if 15 <= len(answer) <= 200:
                indonesian_score += 2
        
        scores['indonesian'] = min(indonesian_score, 10)
        
        # Technical accuracy (0-10) - basic check
        technical_score = 5  # Default neutral score
        if 'periksa' in answer.lower() or 'cek' in answer.lower():
            technical_score += 2
        if 'ganti' in answer.lower() or 'service' in answer.lower():
            technical_score += 2
        if len(answer) > 30:
            technical_score += 1
        
        scores['technical'] = min(technical_score, 10)
        
        # Completeness (0-10)
        completeness_score = 0
        if len(answer) > 30:
            completeness_score += 3
        if len(answer) > 50:
            completeness_score += 2
        if '.' in answer or ',' in answer:
            completeness_score += 2
        if len(answer.split()) > 8:
            completeness_score += 3
        
        scores['completeness'] = min(completeness_score, 10)
        
        # Overall score
        scores['overall'] = sum(scores.values()) / len(scores)
        
        return scores
    
    def evaluate_model(self, model_path, model_name):
        """Evaluate a single model"""
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"Path: {model_path}")
        print(f"{'='*60}")
        
        model, tokenizer = self.load_model(model_path)
        if model is None:
            return None
        
        results = {
            'model_name': model_name,
            'model_path': model_path,
            'responses': [],
            'scores': [],
            'average_scores': {}
        }
        
        total_scores = {criterion: 0 for criterion in self.evaluation_criteria.keys()}
        total_scores['overall'] = 0
        
        for i, question in enumerate(self.test_questions, 1):
            print(f"\n{i}. {question}")
            
            answer = self.generate_response(model, tokenizer, question)
            print(f"Jawaban: {answer}")
            
            scores = self.evaluate_response_quality(question, answer)
            
            print(f"Scores: Coherence={scores['coherence']}/10, Relevance={scores['relevance']}/10, "
                  f"Indonesian={scores['indonesian']}/10, Technical={scores['technical']}/10, "
                  f"Completeness={scores['completeness']}/10, Overall={scores['overall']:.1f}/10")
            
            results['responses'].append({
                'question': question,
                'answer': answer,
                'scores': scores
            })
            
            for criterion in scores:
                total_scores[criterion] += scores[criterion]
        
        # Calculate averages
        num_questions = len(self.test_questions)
        for criterion in total_scores:
            results['average_scores'][criterion] = total_scores[criterion] / num_questions
        
        print(f"\n{'='*60}")
        print(f"AVERAGE SCORES FOR {model_name}:")
        for criterion, score in results['average_scores'].items():
            print(f"{criterion.capitalize()}: {score:.1f}/10")
        print(f"{'='*60}")
        
        return results
    
    def compare_models(self, model_results):
        """Compare multiple models"""
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Sort by overall score
        sorted_models = sorted(model_results, key=lambda x: x['average_scores']['overall'], reverse=True)
        
        print(f"\n{'Rank':<4} {'Model Name':<35} {'Overall':<8} {'Coherence':<10} {'Relevance':<10} {'Indonesian':<11} {'Technical':<10} {'Complete':<10}")
        print("-" * 100)
        
        for rank, result in enumerate(sorted_models, 1):
            scores = result['average_scores']
            print(f"{rank:<4} {result['model_name']:<35} {scores['overall']:<8.1f} {scores['coherence']:<10.1f} "
                  f"{scores['relevance']:<10.1f} {scores['indonesian']:<11.1f} {scores['technical']:<10.1f} "
                  f"{scores['completeness']:<10.1f}")
        
        # Best model recommendation
        best_model = sorted_models[0]
        print(f"\n🏆 BEST MODEL: {best_model['model_name']}")
        print(f"📍 Path: {best_model['model_path']}")
        print(f"📊 Overall Score: {best_model['average_scores']['overall']:.1f}/10")
        
        return best_model

def main():
    """Main evaluation function"""
    print("=== COMPREHENSIVE MOTORCYCLE AI MODEL EVALUATION ===")
    
    evaluator = ModelEvaluator()
    
    # Models to evaluate
    models_to_evaluate = [
        {
            'name': 'Original Fixed Model (72%)',
            'path': 'fixed_motorcycle_model_20250914_090726'
        },
        {
            'name': 'Enhanced Fixed Model',
            'path': 'enhanced_fixed_motorcycle_model_20250914_094739'
        },
        {
            'name': 'Final Enhanced Model',
            'path': 'final_enhanced_motorcycle_model_20250914_095635'
        },
        {
            'name': 'Simple Instruction Model',
            'path': 'simple_instruction_motorcycle_model_20250914_100127'
        }
    ]
    
    # Evaluate each model
    results = []
    for model_info in models_to_evaluate:
        if os.path.exists(model_info['path']):
            result = evaluator.evaluate_model(model_info['path'], model_info['name'])
            if result:
                results.append(result)
        else:
            print(f"\n⚠️  Model not found: {model_info['path']}")
    
    if results:
        # Compare models
        best_model = evaluator.compare_models(results)
        
        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_file = f"model_evaluation_results_{timestamp}.json"
        
        evaluation_data = {
            'timestamp': timestamp,
            'evaluation_criteria': evaluator.evaluation_criteria,
            'test_questions': evaluator.test_questions,
            'model_results': results,
            'best_model': {
                'name': best_model['model_name'],
                'path': best_model['model_path'],
                'overall_score': best_model['average_scores']['overall']
            },
            'recommendations': [
                "Model dengan skor tertinggi direkomendasikan untuk penggunaan",
                "Perhatikan skor coherence untuk kualitas output",
                "Model dengan skor Indonesian tinggi lebih cocok untuk user Indonesia",
                "Evaluasi ulang diperlukan jika menambah data training baru"
            ]
        }
        
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Evaluation results saved to: {evaluation_file}")
        print(f"\n✅ EVALUATION COMPLETED!")
        print(f"🎯 Best performing model: {best_model['model_name']}")
        print(f"📁 Location: {best_model['model_path']}")
        
    else:
        print("\n❌ No models could be evaluated")

if __name__ == "__main__":
    main()