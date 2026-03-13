# CallGuard Sentinel

A comprehensive mobile dialer app with real-time scam detection capabilities. The system uses advanced machine learning models to analyze phone calls in real-time and alert users to potential scams.

## 🚀 Features

### Frontend (React Web App)
- **Modern Dialer Interface**: Clean, intuitive dialer with red/black theme
- **Contact Synchronization**: Sync and search through device contacts
- **Real-time Scam Alerts**: Live notifications when scams are detected
- **Call History**: Track all calls with scam detection results
- **Responsive Design**: Works on desktop and mobile devices

### Backend (Microservices Architecture)
- **Audio Ingestion Service**: Receives live audio streams from CPaaS providers
- **Transcription Service**: Converts audio to text using OpenAI Whisper
- **Feature Extraction Service**: Extracts linguistic, conversational, and agnostic features
- **Analysis Service**: ML-powered scam detection using extracted features
- **Alerting Service**: Real-time WebSocket-based notifications
- **Twilio Integration**: CPaaS functionality for making/receiving calls

### ML Pipeline
- **Linguistic Features**: Authority, urgency, threat, bait, sensitivity, repetition, language switching
- **Conversational Features**: Turn taking, pause length, speech rate
- **Agnostic Features**: Background noise, energy spikes, pitch raising
- **Hybrid Model**: Combines rule-based and ML-based detection
- **Real-time Processing**: Sub-second response times

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   CPaaS Provider │    │   Backend       │
│   (React App)   │◄──►│   (Twilio)       │◄──►│   (Microservices)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                              │
         │              WebSocket                       │
         └──────────────────────────────────────────────┘
```

### Service Flow
1. **Call Initiation**: User makes call through dialer
2. **Audio Streaming**: Twilio streams audio to Audio Ingestion Service
3. **Transcription**: Audio converted to text using Whisper
4. **Feature Extraction**: Extract linguistic, conversational, and agnostic features
5. **ML Analysis**: Scam detection using trained model
6. **Real-time Alerts**: WebSocket notifications to frontend

## 🛠️ Installation

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker and Docker Compose (optional)

### Frontend Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

3. **Build for production**:
   ```bash
   npm run build
   ```

### Backend Setup

1. **Create virtual environment**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize database**:
   ```bash
   python -c "from services.database import init_database; import asyncio; asyncio.run(init_database())"
   ```

5. **Start services**:
   ```bash
   # Start all services
   python main.py
   
   # Or start individual services
   python -m services.audio_ingestion
   python -m services.transcription
   python -m services.feature_extraction
   python -m services.analysis
   python -m services.alerting
   ```

### Docker Setup

1. **Start all services with Docker Compose**:
   ```bash
   cd backend
   docker-compose up -d
   ```

2. **View logs**:
   ```bash
   docker-compose logs -f
   ```

## 🔧 Configuration

### Environment Variables

#### Frontend (.env)
```env
VITE_API_URL=http://localhost:8000
VITE_ALERT_SERVICE_URL=ws://localhost:8005
```

#### Backend (backend/.env)
```env
# Database
DATABASE_URL=postgresql://user:password@localhost/callguard

# Redis
REDIS_URL=redis://localhost:6379

# Twilio (for CPaaS)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_phone_number

# Service URLs
AUDIO_SERVICE_URL=http://localhost:8000/api/audio
TRANSCRIPTION_SERVICE_URL=http://localhost:8001/api/transcription
FEATURE_SERVICE_URL=http://localhost:8002/api/features
ANALYSIS_SERVICE_URL=http://localhost:8003/api/analysis
ALERT_SERVICE_URL=http://localhost:8004/api/alerts
```

## 📱 Usage

### Making Calls
1. Open the dialer interface
2. Enter phone number or search contacts
3. Click "Call" to initiate
4. Real-time scam detection runs during the call
5. Receive alerts if suspicious activity is detected

### Contact Management
1. Click the contacts icon to sync device contacts
2. Search through contacts using the search bar
3. View scam status for each contact
4. Add notes and mark contacts as safe/scam

### Monitoring
1. View call history with scam detection results
2. Monitor real-time alerts during calls
3. Review detailed feature analysis
4. Track model performance and accuracy

## 🔬 ML Model Details

### Feature Categories

#### Linguistic Features
- **Authority**: Detection of authority claims (bank, government, etc.)
- **Urgency**: Urgent language patterns
- **Threat**: Threatening language detection
- **Bait**: Baiting tactics and promises
- **Sensitivity**: Requests for sensitive information
- **Repetition**: Excessive repetition patterns
- **Language Switching**: Code-switching detection

#### Conversational Features
- **Turn Taking**: Speaker change patterns
- **Pause Length**: Silence duration analysis
- **Speech Rate**: Words per minute analysis

#### Agnostic Features
- **Background Noise**: Audio quality assessment
- **Energy Spikes**: Sudden volume changes
- **Pitch Raising**: Voice pitch analysis

### Model Architecture
- **Primary Model**: Random Forest Classifier
- **Fallback**: Rule-based detection system
- **Feature Engineering**: 25+ extracted features
- **Real-time Processing**: <1 second response time

## 🚀 Deployment

### Production Deployment

1. **Frontend**:
   ```bash
   npm run build
   # Deploy dist/ folder to your web server
   ```

2. **Backend**:
   ```bash
   # Use Docker Compose for production
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Database**:
   - Set up PostgreSQL with proper security
   - Configure Redis for caching
   - Set up monitoring and backups

### Scaling Considerations
- Use load balancers for multiple service instances
- Implement Redis clustering for high availability
- Set up database replication
- Use message queues (RabbitMQ/Kafka) for high-volume processing

## 🧪 Testing

### Frontend Tests
```bash
npm run test
```

### Backend Tests
```bash
cd backend
python -m pytest tests/
```

### Integration Tests
```bash
# Test full pipeline
python tests/integration_test.py
```

## 📊 Monitoring

### Health Checks
- Frontend: `http://localhost:3000/health`
- Backend: `http://localhost:8000/health`
- Individual services: `http://localhost:800X/health`

### Metrics
- Call volume and duration
- Scam detection accuracy
- Response times
- Error rates
- User engagement

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

## 🔮 Roadmap

### Phase 1 (Current)
- ✅ Basic dialer functionality
- ✅ Contact synchronization
- ✅ Real-time scam detection
- ✅ WebSocket alerts

### Phase 2 (Next)
- [ ] Mobile app (React Native)
- [ ] Advanced ML models
- [ ] Multi-language support
- [ ] Call recording and playback

### Phase 3 (Future)
- [ ] AI-powered call analysis
- [ ] Integration with other CPaaS providers
- [ ] Advanced analytics dashboard
- [ ] Enterprise features

---

**CallGuard Sentinel** - Protecting users from phone scams with cutting-edge technology.