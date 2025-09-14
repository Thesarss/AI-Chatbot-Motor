import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from datetime import datetime
import re

class ModelEvaluator:
    def __init__(self):
        self.test_questions = [
            "Motor saya susah hidup, apa penyebabnya?",
            "Bagaimana cara merawat rem motor?",
            "Motor bergetar saat idle, kenapa ya?",
            "Oli motor cepat habis, apa masalahnya?",
            "Tips agar motor irit bensin?",
            "Motor keluar asap putih dari knalpot",
            "Kenapa motor saya boros bensin?",
            "Cara mengatasi motor yang overheat?",
            "Motor tidak bisa distarter, solusinya?",
            "Suara mesin kasar, apa penyebabnya?",
            "Rem motor blong, bagaimana mengatasinya?",
            "Motor brebet saat gas ditarik, kenapa?",
            "Cara merawat rantai motor yang benar?",
            "Motor sering mati mendadak, apa sebabnya?",
            "Shock motor keras, perlu diganti?"
        ]
        
    def load_model(self, model_path):
        """Load model dan tokenizer"""
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            model = GPT2LMHeadModel.from_pretrained(model_path)
            model.eval()
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model {model_path}: {str(e)}")
            return None, None
    
    def generate_answer(self, model, tokenizer, question):
        """Generate jawaban untuk pertanyaan"""
        prompt = f"Pertanyaan: {question}\nJawaban:"
        
        try:
            # Tokenize
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            
            # Generate dengan parameter konservatif
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 80,
                    num_return_sequences=1,
                    temperature=0.6,
                    top_p=0.8,
                    top_k=40,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text.replace(prompt, "").strip()
            
            # Clean answer
            answer = self.clean_answer(answer)
            
            return answer
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def clean_answer(self, answer):
        """Clean dan format jawaban"""
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer)
        
        # Truncate at first sentence end if too long
        sentences = re.split(r'[.!?]', answer)
        if len(sentences) > 1 and len(answer) > 150:
            answer = sentences[0] + '.'
        
        # Limit length
        if len(answer) > 200:
            answer = answer[:200] + '...'
            
        return answer.strip()
    
    def evaluate_answer_quality(self, question, answer):
        """Evaluate kualitas jawaban"""
        score = 0
        feedback = []
        
        # Check if answer is relevant to motorcycles
        motorcycle_keywords = [
            'motor', 'mesin', 'oli', 'rem', 'bensin', 'karburator', 'aki', 'busi',
            'rantai', 'shock', 'knalpot', 'starter', 'idle', 'overheat', 'service',
            'ganti', 'perawatan', 'perbaikan', 'masalah', 'rusak', 'aus'
        ]
        
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        # Relevance check (0-4 points)
        relevant_words = sum(1 for word in motorcycle_keywords if word in answer_lower)
        if relevant_words >= 3:
            score += 4
            feedback.append("Sangat relevan dengan topik motor")
        elif relevant_words >= 2:
            score += 3
            feedback.append("Relevan dengan topik motor")
        elif relevant_words >= 1:
            score += 2
            feedback.append("Cukup relevan")
        else:
            score += 1
            feedback.append("Kurang relevan")
        
        # Coherence check (0-3 points)
        if len(answer) < 10:
            score += 1
            feedback.append("Jawaban terlalu singkat")
        elif any(char in answer for char in ['@', '#', '$', '%', '^', '&', '*']):
            score += 1
            feedback.append("Mengandung karakter tidak valid")
        elif len(re.findall(r'[a-zA-Z]{15,}', answer)) > 0:
            score += 1
            feedback.append("Mengandung kata yang terlalu panjang")
        else:
            # Check for Indonesian language patterns
            indonesian_patterns = ['yang', 'dan', 'atau', 'untuk', 'pada', 'dengan', 'adalah', 'bisa', 'dapat']
            if any(pattern in answer_lower for pattern in indonesian_patterns):
                score += 3
                feedback.append("Respons koheren dalam bahasa Indonesia")
            else:
                score += 2
                feedback.append("Respons cukup koheren")
        
        # Completeness check (0-3 points)
        if len(answer) >= 50 and '.' in answer:
            score += 3
            feedback.append("Jawaban lengkap")
        elif len(answer) >= 30:
            score += 2
            feedback.append("Jawaban cukup lengkap")
        else:
            score += 1
            feedback.append("Jawaban kurang lengkap")
        
        return min(score, 10), feedback
    
    def evaluate_model(self, model_path):
        """Evaluate satu model"""
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL: {model_path}")
        print(f"{'='*80}")
        
        model, tokenizer = self.load_model(model_path)
        if model is None:
            return None
        
        results = []
        total_score = 0
        
        for i, question in enumerate(self.test_questions, 1):
            print(f"\n[Test {i}/{len(self.test_questions)}]")
            print(f"❓ Q: {question}")
            
            answer = self.generate_answer(model, tokenizer, question)
            score, feedback = self.evaluate_answer_quality(question, answer)
            
            total_score += score
            
            result = {
                "question": question,
                "answer": answer,
                "score": score,
                "feedback": feedback
            }
            results.append(result)
            
            print(f"🤖 A: {answer}")
            print(f"📊 Score: {score}/10")
            print(f"💬 Feedback: {', '.join(feedback)}")
            print("-" * 60)
        
        # Calculate final metrics
        max_score = len(self.test_questions) * 10
        percentage = (total_score / max_score) * 100
        average_score = total_score / len(self.test_questions)
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Total Score: {total_score}/{max_score} ({average_score:.1f}/10)")
        print(f"Percentage: {percentage:.1f}%")
        
        return {
            "model_path": model_path,
            "total_score": total_score,
            "max_score": max_score,
            "average_score": average_score,
            "percentage": percentage,
            "results": results
        }
    
    def compare_models(self, model_paths):
        """Compare multiple models"""
        all_results = []
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                result = self.evaluate_model(model_path)
                if result:
                    all_results.append(result)
            else:
                print(f"⚠️ Model not found: {model_path}")
        
        if not all_results:
            print("❌ No valid models found for comparison")
            return
        
        # Sort by percentage
        all_results.sort(key=lambda x: x['percentage'], reverse=True)
        
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        for i, result in enumerate(all_results, 1):
            model_name = os.path.basename(result['model_path'])
            print(f"{i}. {model_name}:")
            print(f"   Score: {result['total_score']}/{result['max_score']} ({result['percentage']:.1f}%)")
            print(f"   Average: {result['average_score']:.1f}/10")
            print()
        
        # Save comparison report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"model_comparison_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "comparison_results": all_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Comparison report saved to: {report_file}")
        
        # Recommend best model
        best_model = all_results[0]
        print(f"\n🏆 BEST MODEL: {os.path.basename(best_model['model_path'])}")
        print(f"   Performance: {best_model['percentage']:.1f}%")
        
        return all_results

def main():
    evaluator = ModelEvaluator()
    
    # List of models to compare
    model_paths = [
        "improved_motorcycle_model_20250914_084736",
        "fixed_motorcycle_model_20250914_090726",
        "optimized_motorcycle_model_20250914_092147"
    ]
    
    print("🔍 Starting comprehensive model evaluation...")
    print(f"📊 Testing {len(evaluator.test_questions)} questions per model")
    
    # Compare all models
    results = evaluator.compare_models(model_paths)
    
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    main()