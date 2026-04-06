# Trading Management System

Advanced crypto trading scanner and management system with GPU-accelerated AI ranking and autonomous workflows.

## Quick Start

```bash
# 1. Configure environment
cp backend/.env.example backend/.env  # Add your Binance API keys

# 2. Launch services
docker-compose up -d --build

# 3. Access the dashboard
# Frontend: http://localhost:5173
# Backend API: http://localhost:8001
# API Documentation: http://localhost:8001/docs
```

## Key Features

- **🚀 Turbo-Scanner**: Multithreaded analysis of 20+ pairs across 7 timeframes in ~3 seconds
- **🤖 AI Ranking**: Local Qwen 2.5 14B model identifies top trading setups with technical reasoning
- **🛡️ Smart Trading**: Set-and-forget OCO orders with fee-aware clipping and automatic reconciliation
- **⚡ Lead (Futures) Trading**: Leveraged positions with panic-sell and atomic rollback safety
- **📊 Professional Dashboard**: Real-time charts, portfolio monitoring, and comprehensive trade history

## Architecture

```
├── backend/           # FastAPI with SQLite, TA-Lib, Binance SDK
├── frontend/          # React 19, TypeScript, TailwindCSS
├── langgraph/         # Autonomous agent workflows
└── ollama/            # GPU-accelerated Qwen 2.5 14B model
```

## Documentation

- **[GEMINI.md](GEMINI.md)** - Complete project documentation, features, and workflows
- **[WORKFLOWS.md](WORKFLOWS.md)** - Detailed application workflows and architecture
- **API Docs**: Available at `http://localhost:8001/docs` when backend is running

## Prerequisites

1. **NVIDIA GPU** with [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
2. **Binance API Keys** with trading permissions
3. **Docker & Docker Compose** (latest versions)

## Development

```bash
# Backend development
cd backend
pip install -r requirements.txt
python -m app.main

# Frontend development  
cd frontend
npm install
npm run dev
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 5173 | React dashboard UI |
| Backend API | 8001 | FastAPI trading engine |
| LangGraph | 2024 | Autonomous agent workflows |
| Ollama | 11434 | Local LLM for AI ranking |

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Production Ready • April 2026 • [Detailed Documentation](GEMINI.md)*