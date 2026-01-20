# Food Classification MLOps Pipeline

Production-ready machine learning pipeline for food image classification with complete MLOps practices.

## ��� Project Overview

This project demonstrates a complete ML engineering workflow from development to production deployment, focusing on:
- **Infrastructure & MLOps** (80% focus)
- **Model Development** (20% focus)

### What This Project Covers

- ✅ Professional development workflow (Git, PRs, code review)
- ✅ Experiment tracking (Weights & Biases)
- ✅ Cloud training (AWS)
- ✅ Containerization (Docker)
- ✅ CI/CD pipelines (GitHub Actions)
- ✅ Model serving (FastAPI)
- ✅ Production deployment (AWS ECS/EKS)
- ✅ Monitoring & observability (Prometheus, Grafana)
- ✅ Model versioning (MLflow)
- ✅ Infrastructure as Code (Terraform)

## ��� Quick Start
```bash
# Clone repository
git clone git@github.com:tanmaynaik11/food-classification-mlops.git
cd food-classification-mlops

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run tests
make test
```

## ��� Project Structure
```
food-classification-mlops/
├── src/                    # Source code
│   ├── data/              # Data loading and processing
│   ├── models/            # Model definitions
│   ├── training/          # Training loops and utilities
│   ├── serving/           # API and inference
│   └── monitoring/        # Monitoring and logging
├── tests/                 # Test suite
├── configs/               # Configuration files
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks for exploration
├── docker/                # Docker configurations
├── .github/workflows/     # CI/CD pipelines
└── docs/                  # Documentation
```

## ���️ Technology Stack

**ML Framework:** PyTorch  
**Experiment Tracking:** Weights & Biases  
**Model Registry:** MLflow  
**API Framework:** FastAPI  
**Containerization:** Docker  
**Cloud Platform:** AWS  
**CI/CD:** GitHub Actions  
**Monitoring:** Prometheus + Grafana  
**IaC:** Terraform  

## ��� Current Status

- [x] Phase 1: Project setup
- [ ] Phase 2: Development workflow
- [ ] Phase 3: Cloud training
- [ ] Phase 4: CI/CD pipeline
- [ ] Phase 5: API development
- [ ] Phase 6: AWS deployment
- [ ] Phase 7: Model registry
- [ ] Phase 8: Production hardening

## ��� Author

**Tanmay Naik** ([@tanmaynaik11](https://github.com/tanmaynaik11))

## ��� License

MIT License
