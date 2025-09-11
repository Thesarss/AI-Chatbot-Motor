# Enhanced Motorcycle AI Assistant

## 🚀 Overview

Proyek ini mengembangkan AI Assistant untuk diagnosis masalah motor dengan kemampuan **context awareness** yang ditingkatkan. AI dapat memahami konteks percakapan, mengingat informasi sebelumnya, dan memberikan respons yang lebih relevan dan personal.

## 🎯 Fitur Utama

### 1. **Context-Aware Conversation**
- AI dapat mengingat percakapan sebelumnya dalam sesi yang sama
- Memahami referensi ke topik yang sudah dibahas
- Dapat kembali ke konteks sebelumnya jika topik berubah

### 2. **Conversation Flow Management**
- Mendeteksi state percakapan (greeting, problem_description, diagnosis, dll)
- Memberikan respons yang sesuai dengan tahap percakapan
- Mengarahkan percakapan ke arah yang produktif

### 3. **Context Scoring System**
- Menilai relevansi pesan dengan konteks percakapan
- Memberikan skor berdasarkan keyword match, topic consistency, dll
- Membantu AI memutuskan kapan harus menggunakan konteks vs informasi baru

### 4. **Session Management**
- Mengelola multiple user sessions secara bersamaan
- Menyimpan dan memuat session data
- Automatic cleanup untuk session yang sudah lama

### 5. **Enhanced Dataset**
- Dataset diperkaya dengan informasi konteks
- Keyword mapping untuk pencarian yang lebih baik
- Follow-up questions dan prevention tips

## 📁 Struktur File

```
BengkelAI/
├── context_manager.py          # Mengelola memori percakapan
├── enhanced_ai_model.py         # Model AI utama dengan context awareness
├── context_scorer.py            # Sistem penilaian relevansi konteks
├── conversation_flow.py         # Manajemen alur percakapan
├── enhanced_dataset_example.json # Dataset diperkaya dengan konteks
├── main_enhanced_ai.py          # Integrasi semua komponen
├── demo_enhanced_ai.py          # Demo dan testing
├── requirements_enhanced.txt    # Dependencies yang diperlukan
└── README_enhanced_ai.md        # Dokumentasi ini
```

## 🛠️ Installation

### 1. Install Dependencies

```bash
# Install dependencies dasar
pip install -r requirements.txt

# Install dependencies tambahan untuk enhanced features
pip install -r requirements_enhanced.txt
```

### 2. Setup Environment

```bash
# Pastikan semua file ada di direktori yang sama
# Tidak perlu setup tambahan, semua komponen sudah terintegrasi
```

## 🚀 Quick Start

### 1. Jalankan Demo

```bash
python demo_enhanced_ai.py
```

Demo akan menunjukkan semua fitur enhanced AI:
- Basic conversation
- Context awareness
- Conversation flow
- Context scoring
- Session management
- Enhanced dataset
- Full integration

### 2. Gunakan AI Assistant

```python
from main_enhanced_ai import EnhancedMotorcycleAssistant

# Initialize assistant
assistant = EnhancedMotorcycleAssistant()

# Start conversation
session_id = "user_123"
response = assistant.process_message("Motor saya susah dihidupkan", session_id)
print(response)

# Continue conversation with context
response = assistant.process_message("Motor Honda Beat 2020", session_id)
print(response)

# AI akan mengingat informasi sebelumnya
response = assistant.process_message("Berapa biaya perbaikannya?", session_id)
print(response)
```

### 3. Gunakan Komponen Individual

```python
# Context Manager
from context_manager import ContextManager
manager = ContextManager()
manager.add_message("session_1", "user", "Motor bermasalah")

# Context Scorer
from context_scorer import ContextScorer
scorer = ContextScorer()
score = scorer.calculate_relevance_score("Kampas rem perlu diganti?", 
                                       ["Motor rem bunyi", "Rem depan bermasalah"])

# Conversation Flow
from conversation_flow import ConversationFlow
flow = ConversationFlow()
state = flow.determine_state("Halo, motor saya bermasalah")
```

## 📊 Cara Kerja Context Awareness

### 1. **Conversation Memory**
```python
# AI mengingat percakapan dalam session
User: "Motor saya susah dihidupkan"
AI: "Bisa ceritakan lebih detail tentang masalahnya?"

User: "Motor Honda Beat"
AI: "Baik, untuk Honda Beat yang susah dihidupkan, kemungkinan penyebabnya..."
# AI mengingat masalah "susah dihidupkan" dari pesan sebelumnya
```

### 2. **Topic Tracking**
```python
# AI melacak topik yang sedang dibahas
User: "Rem motor bunyi mencicit"
AI: "Masalah rem mencicit biasanya karena..."
# Current topic: "rem"

User: "Berapa biayanya?"
AI: "Untuk perbaikan rem mencicit, biayanya sekitar..."
# AI tahu "biayanya" merujuk ke perbaikan rem
```

### 3. **Context Switching**
```python
# AI dapat beralih konteks dengan smooth
User: "Rem sudah diperbaiki, sekarang oli mesin kapan diganti?"
AI: "Bagus rem sudah diperbaiki. Untuk oli mesin Honda Beat..."
# AI acknowledge konteks lama dan beralih ke topik baru
```

## 🔧 Konfigurasi

### Context Manager Settings
```python
# Di context_manager.py
MAX_CONTEXT_MESSAGES = 20  # Maksimal pesan yang diingat
CONTEXT_DECAY_HOURS = 24   # Konteks expire setelah 24 jam
AUTO_CLEANUP_INTERVAL = 3600  # Cleanup otomatis setiap jam
```

### Context Scoring Weights
```python
# Di context_scorer.py
KEYWORD_WEIGHT = 0.3      # Bobot keyword matching
TOPIC_WEIGHT = 0.25       # Bobot topic consistency
TEMPORAL_WEIGHT = 0.2     # Bobot temporal relevance
SEMANTIC_WEIGHT = 0.15    # Bobot semantic similarity
INTENT_WEIGHT = 0.1       # Bobot user intent
```

## 📈 Performance Metrics

### Context Relevance Score
- **0.8-1.0**: Sangat relevan (same topic, recent)
- **0.6-0.8**: Relevan (related topic)
- **0.4-0.6**: Cukup relevan (general context)
- **0.0-0.4**: Kurang relevan (different topic)

### Response Quality Indicators
- **Context Usage**: Persentase respons yang menggunakan konteks
- **Topic Consistency**: Konsistensi topik dalam percakapan
- **User Satisfaction**: Feedback dari user tentang relevansi respons

## 🧪 Testing

### Unit Tests
```bash
# Test individual components
python -m pytest test_context_manager.py
python -m pytest test_context_scorer.py
python -m pytest test_conversation_flow.py
```

### Integration Tests
```bash
# Test full system
python -m pytest test_enhanced_ai_integration.py
```

### Manual Testing
```bash
# Interactive testing
python demo_enhanced_ai.py
```

## 🔍 Troubleshooting

### Common Issues

1. **Context tidak tersimpan**
   ```python
   # Pastikan session_id konsisten
   session_id = "user_123"  # Gunakan ID yang sama
   ```

2. **Respons tidak kontekstual**
   ```python
   # Check context score
   scorer = ContextScorer()
   score = scorer.calculate_relevance_score(message, history)
   print(f"Context score: {score}")  # Should be > 0.5 for good context
   ```

3. **Memory usage tinggi**
   ```python
   # Cleanup old sessions
   manager.cleanup_old_sessions(max_age_hours=24)
   ```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Atau set debug flag
assistant = EnhancedMotorcycleAssistant(debug=True)
```

## 📚 Advanced Usage

### Custom Context Scoring
```python
class CustomContextScorer(ContextScorer):
    def calculate_custom_score(self, message, context):
        # Implement custom scoring logic
        return score

scorer = CustomContextScorer()
```

### Custom Conversation Flow
```python
class CustomConversationFlow(ConversationFlow):
    def determine_custom_state(self, message):
        # Implement custom state detection
        return state

flow = CustomConversationFlow()
```

### Integration dengan Database
```python
# Save context to database
class DatabaseContextManager(ContextManager):
    def save_to_db(self, session_id, context):
        # Implement database saving
        pass

manager = DatabaseContextManager()
```

## 🚀 Deployment

### Production Setup
```python
# main_production.py
from main_enhanced_ai import EnhancedMotorcycleAssistant
import redis

# Use Redis for session storage
class ProductionAssistant(EnhancedMotorcycleAssistant):
    def __init__(self):
        super().__init__()
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def save_session(self, session_id, context):
        self.redis_client.set(f"session:{session_id}", 
                             json.dumps(context.__dict__))

assistant = ProductionAssistant()
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_enhanced.txt .
RUN pip install -r requirements_enhanced.txt

COPY . .
EXPOSE 8000

CMD ["python", "main_enhanced_ai.py"]
```

### Kubernetes Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: enhanced-motorcycle-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: enhanced-motorcycle-ai
  template:
    metadata:
      labels:
        app: enhanced-motorcycle-ai
    spec:
      containers:
      - name: ai-assistant
        image: enhanced-motorcycle-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-service"
```

## 📊 Monitoring

### Metrics to Track
```python
# metrics.py
class AIMetrics:
    def __init__(self):
        self.context_usage_rate = 0
        self.average_context_score = 0
        self.session_duration = 0
        self.user_satisfaction = 0
    
    def track_context_usage(self, used_context):
        # Track context usage
        pass
    
    def track_response_quality(self, score):
        # Track response quality
        pass
```

### Logging
```python
# Enhanced logging
import logging
from datetime import datetime

class ContextAwareLogger:
    def __init__(self):
        self.logger = logging.getLogger('enhanced_ai')
    
    def log_conversation(self, session_id, message, response, context_score):
        self.logger.info(f"Session: {session_id}, "
                        f"Context Score: {context_score:.2f}, "
                        f"Message: {message[:50]}...")
```

## 🔮 Future Enhancements

### Planned Features
1. **Multi-language Support**
   - Support untuk bahasa Indonesia dan Inggris
   - Context awareness lintas bahasa

2. **Voice Integration**
   - Speech-to-text input
   - Text-to-speech output
   - Voice context understanding

3. **Visual Context**
   - Image analysis untuk diagnosis
   - Visual context dalam percakapan

4. **Predictive Context**
   - Prediksi kebutuhan user berdasarkan konteks
   - Proactive suggestions

5. **Learning from Feedback**
   - Adaptive context scoring
   - Personalized conversation flow

### Research Areas
1. **Advanced NLP**
   - Transformer-based context understanding
   - Attention mechanisms untuk context

2. **Memory Networks**
   - Long-term memory untuk user preferences
   - Episodic memory untuk conversation history

3. **Reinforcement Learning**
   - RL untuk optimasi conversation flow
   - Reward dari user feedback

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd BengkelAI

# Install development dependencies
pip install -r requirements_enhanced.txt
pip install -r requirements_dev.txt

# Run tests
python -m pytest

# Run demo
python demo_enhanced_ai.py
```

### Code Style
```bash
# Format code
black *.py

# Check style
flake8 *.py

# Type checking
mypy *.py
```

### Pull Request Process
1. Fork repository
2. Create feature branch
3. Add tests untuk fitur baru
4. Ensure all tests pass
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 📞 Support

Untuk pertanyaan atau bantuan:
- Create issue di GitHub repository
- Email: support@bengkelai.com
- Documentation: https://docs.bengkelai.com

## 🙏 Acknowledgments

- Google Gemma untuk base model
- Hugging Face untuk transformers library
- OpenAI untuk inspiration dalam conversation AI
- Community contributors

---

**Happy Coding! 🚀**

*Enhanced Motorcycle AI Assistant - Making motorcycle diagnosis smarter with context awareness*