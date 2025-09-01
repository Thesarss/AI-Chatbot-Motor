# BengkelAi - Digital Motorcycle Health & Service Ecosystem with AI Analysis

🏍️ **AI Platform for Digital Motorcycle Health & Service Ecosystem**

[![Development Status](https://img.shields.io/badge/Status-Development%20v1-orange)](https://bengkel-ai.id)
[![Live Demo](https://img.shields.io/badge/Demo-Live-green)](https://bengkel-ai.id)

## 👥 Development Team

- **Widya** - Business and Product Lead
- **Aidil Baihaqi** - Full Stack Developer  
- **Muhammad Thesar** - AI Engineer

## 🤖 AI Technology Overview
**Project Theme**: Digital Motorcycle Health & Service Ecosystem with AI Analysis  
**Project Name**: BengkelAi

An application that can help motorcycle owners diagnose problems, provide cost estimates, give recommendations, and directly connect them to the nearest workshop. This platform is specifically designed for 120 million motorcycle users in Indonesia who need practical and reliable solutions for their vehicle maintenance.

### Large Language Model (LLM) Integration
BengkelAi uses LLM technology to understand and respond to user questions about motorcycle problems in a natural and contextual way. LLM enables the system to:
- Process diverse Indonesian language inputs
- Provide easy-to-understand explanations about motorcycle problems
- Generate personalized recommendations based on conversation context
- Handle variations in how users describe motorcycle symptoms

### Natural Language Processing (NLP)
Our NLP system is specifically designed for the automotive domain with capabilities:

#### Text Processing
- **Tokenization**: Breaking down user input into analyzable components
- **Named Entity Recognition (NER)**: Identifying motorcycle components, symptoms, and technical terms
- **Sentiment Analysis**: Detecting user concern levels from how they describe problems

#### Domain-Specific Understanding
- **Automotive Vocabulary**: Database of motorcycle terms in Indonesian and technical language
- **Symptom Mapping**: Connecting symptom descriptions with possible causes
- **Severity Classification**: Determining problem urgency level (urgent/can be postponed)

### Intent Recognition System
The intent recognition system classifies user purposes into categories:

#### Primary Intents
- **Diagnosis Request**: "My motorcycle won't start", "Exhaust emits white smoke"
- **Cost Estimation**: "How much does an oil change cost?", "Regular service price?"
- **Workshop Finding**: "Nearest workshop", "Find a good workshop"
- **Booking Intent**: "Want to book service", "When is the schedule available?"
- **Maintenance Reminder**: "When to change oil?", "Regular service schedule"

#### Intent Classification Pipeline
```
User Input → Preprocessing → Feature Extraction → Intent Classification → Response Generation
```

#### Technical Implementation
- **Model Architecture**: Transformer-based model fine-tuned for automotive domain
- **Training Data**: Motorcycle symptom dataset and service conversations in Indonesian
- **Confidence Scoring**: System provides confidence scores for each prediction
- **Fallback Mechanism**: If confidence is low, system will ask for clarification

### AI-Powered Features

#### 1. Symptom-to-Diagnosis Mapping
```
Input: "My motorcycle exhaust keeps emitting white smoke"
AI Processing:
├── Intent: Diagnosis Request
├── Entities: [exhaust, white smoke]
├── Possible Causes: [head gasket leak, oil entering combustion chamber]
└── Urgency: Medium-High
```

#### 2. Cost Estimation AI
- Analyzes damage types and provides cost estimates
- Compares prices between workshops based on location
- Predicts additional costs that might be needed

#### 3. Intelligent Service Reminders
- Machine learning to predict optimal maintenance schedules
- Personalization based on motorcycle usage patterns
- Adaptation to weather and road conditions

### Dataset & Training

#### Automotive Dataset
- **Symptom Database**: 1000+ motorcycle symptoms with causes and solutions
- **Conversation Logs**: Real conversation data between mechanics and customers
- **Technical Manuals**: Knowledge extraction from motorcycle service manuals
- **Regional Variations**: Local terms and Indonesian dialects

#### Model Training Process
1. **Data Collection**: Gathering data from partner workshops
2. **Data Annotation**: Intent and entity labeling by experienced mechanics
3. **Model Fine-tuning**: Adapting pre-trained models for automotive domain
4. **Validation**: Testing with real user scenarios
5. **Continuous Learning**: Model updates based on user feedback

### Performance Metrics
- **Intent Accuracy**: >95% for primary intents
- **Entity Recognition**: >90% for motorcycle components
- **Response Time**: <2 seconds for simple diagnosis
- **User Satisfaction**: Target >4.5/5 rating

### Future AI Enhancements
- **Computer Vision**: Photo/video analysis of motorcycle damage
- **Voice Recognition**: Voice input for abnormal engine sound detection
- **Predictive Maintenance**: AI prediction of damage before it occurs
- **Multi-modal AI**: Combination of text, image, and audio analysis

---

*This project was developed for the Developer Day Competition with a focus on practical and beneficial AI implementation for Indonesia's automotive ecosystem.*