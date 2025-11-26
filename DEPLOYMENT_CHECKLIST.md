# 🎯 FCD Cloud Server - File Tree

```
fcd-cloud-server/
│
├── 📄 README.md                         # Complete documentation
├── 📄 QUICK_DEPLOY.md                   # 5-minute deployment guide
├── 📄 DEPLOYMENT_CHECKLIST.md           # This file
│
├── 🐍 main.py                           # FastAPI webhook server (CORE)
├── 🧪 test_server.py                    # Local testing script
│
├── 📦 requirements.txt                  # Python dependencies
├── 🐍 runtime.txt                       # Python 3.11.0
├── 🚀 Procfile                          # Railway startup command
├── ⚙️  railway.json                      # Railway configuration
├── 🙈 .gitignore                        # Git ignore rules
│
├── 📊 trades.csv                        # Auto-generated (DO NOT COMMIT)
│
├── fcd/                                 # FCD Model Package
│   ├── __init__.py
│   │
│   ├── core/                            # ⚠️  DO NOT MODIFY THESE FILES
│   │   ├── __init__.py
│   │   ├── fcd_indicator.py            # Complete FCD-PSE indicator
│   │   ├── fcd_state.py                # State transformation (A→B→C→X→A')
│   │   ├── probabilistic.py            # Monte Carlo predictions
│   │   ├── kalman.py                   # Kalman filtering
│   │   ├── monte_carlo.py              # Path generation
│   │   ├── primitives.py               # Math primitives
│   │   ├── multi_scale.py              # Multi-timeframe analysis
│   │   ├── visualization.py            # Plotting utilities
│   │   └── btc_mode_config.py          # Regime configuration
│   │
│   ├── signal/                          # Signal Generation
│   │   ├── __init__.py
│   │   └── fcd_signal_generator.py     # Signal logic + bar caching
│   │
│   └── rankings/                        # BecomingScore Data
│       └── consolidated_futures.csv     # Pre-calculated rankings
│
└── utils/                               # Utilities
    ├── __init__.py
    └── paper_trader.py                  # Paper trading engine
```

## ✅ File Purpose Summary

### Core Application Files

| File | Purpose | Modify? |
|------|---------|---------|
| `main.py` | FastAPI server, webhook endpoint | ✅ Yes |
| `requirements.txt` | Python dependencies | ✅ Yes |
| `Procfile` | Railway startup command | ⚠️  Rarely |
| `railway.json` | Railway config | ⚠️  Rarely |
| `runtime.txt` | Python version | ⚠️  Rarely |

### FCD Core (Original Model)

| File | Purpose | Modify? |
|------|---------|---------|
| `fcd/core/fcd_indicator.py` | Main FCD-PSE indicator | ❌ NO |
| `fcd/core/fcd_state.py` | State transformation | ❌ NO |
| `fcd/core/probabilistic.py` | Monte Carlo engine | ❌ NO |
| `fcd/core/kalman.py` | Kalman filtering | ❌ NO |
| `fcd/core/monte_carlo.py` | Path generation | ❌ NO |
| `fcd/core/primitives.py` | Math functions | ❌ NO |
| `fcd/core/multi_scale.py` | Multi-timeframe | ❌ NO |
| `fcd/core/btc_mode_config.py` | Regime config | ❌ NO |

### Custom Components

| File | Purpose | Modify? |
|------|---------|---------|
| `fcd/signal/fcd_signal_generator.py` | Signal generation wrapper | ✅ Yes |
| `utils/paper_trader.py` | Paper trading logic | ✅ Yes |
| `fcd/rankings/consolidated_futures.csv` | BecomingScore data | ✅ Update |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Complete documentation |
| `QUICK_DEPLOY.md` | Fast deployment guide |
| `DEPLOYMENT_CHECKLIST.md` | This file |

## 📊 Generated Files (Not in Git)

These are created automatically:

- `trades.csv` - Trade log (persists on Railway)
- `__pycache__/` - Python bytecode
- `*.pyc` - Compiled Python files

## 🔒 What's in `.gitignore`

```
__pycache__/
*.pyc
.env
.venv
venv/
.DS_Store
*.log
```

## 📏 File Sizes

Approximate sizes for reference:

```
main.py                    ~10 KB
fcd_signal_generator.py    ~12 KB
paper_trader.py            ~8 KB
fcd_indicator.py           ~25 KB
fcd_state.py               ~20 KB
README.md                  ~15 KB
```

Total project size: **~500 KB** (small enough for fast deploys)

## 🔄 Update Workflow

### To Update FCD Core:
```bash
# Copy from original project
cp ../src/core/*.py fcd/core/
git add fcd/core/
git commit -m "Update FCD core"
git push
```

### To Update Rankings:
```bash
# Copy new rankings
cp ../outputs/rankings/consolidated_futures.csv fcd/rankings/
git add fcd/rankings/
git commit -m "Update BecomingScore rankings"
git push
```

### To Modify Server Logic:
```bash
# Edit main.py or signal generator
git add main.py fcd/signal/
git commit -m "Update server logic"
git push
```

Railway auto-redeploys on every push to main branch.

---

**Last Updated:** November 2025  
**Total Files:** 25  
**Lines of Code:** ~3,500
