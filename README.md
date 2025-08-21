# A Data-Driven AI System for Credit Scoring of Underbanked SMEs in Rwanda Based on Sales Transactions and Tax Declarations

An AI-driven credit scoring system for Small and Medium Enterprises (SMEs) in Rwanda, integrating tax compliance data with traditional financial metrics to enhance credit risk assessment accuracy.

## Overview

This research project addresses the critical challenge of credit assessment for underbanked SMEs in Rwanda, where traditional scoring methods fail due to limited formal financial records. The system integrates tax declaration data with sales transactions using machine learning to provide accurate, interpretable credit risk assessments for underbanked enterprises.

## Key Features

- **Compliance-Enhanced Scoring**: Integrates tax declaration data with sales/purchase records
- **High Accuracy**: Achieves 99.25% classification accuracy with 0.9999 ROC-AUC score
- **Explainable AI**: Provides detailed reasoning for each credit decision
- **Real-time API**: Flask-based REST API for instant credit assessments
- **Synthetic Data Generation**: Controlled experimental environment with realistic SME patterns
- **Feature Engineering**: 64 sophisticated predictive variables from 51 raw features

## Project Structure

```
uel-credit-scoring/
├── dataset/
│   └── credit_data.csv          # Generated synthetic dataset
├── functions/
│   ├── features_engineering.py      # Feature engineering logic
│   ├── prediction_reasons.py        # AI explanation generation
│   └── plot_confusion_matrix.py     # Visualization utilities
├── results/
│   ├── data/
│   │   └── preprocessed_data.csv
│   ├── models/
│   │   └── credit_model.pkl         # Trained model
│   └── plots/                       # Generated visualizations
├── generate_dataset.py              # Synthetic data generation
├── pipeline.py                      # Main training pipeline
├── predict_api.py                   # Flask API server
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/sergekarim/uel-credit-scoring.git
cd uel-credit-scoring

# Install required packages
pip install pandas numpy scikit-learn flask matplotlib seaborn joblib

# Create necessary directories
mkdir -p results/data results/models results/plots dataset
```

## Usage

### 1. Generate Synthetic Dataset
```bash
python generate_dataset.py
```
Creates a balanced dataset of 2,000 SME profiles across four credit grades (A, B, C, D).

### 2. Train the Model
```bash
python pipeline.py
```
Runs the complete ML pipeline:
- Data validation and preprocessing
- Feature engineering (51 → 64 features)
- 15-fold cross-validation training
- Model evaluation and visualization
- Model persistence

### 3. Start the API Server
```bash
python predict_api.py
```
Launches Flask API on `http://localhost:5001`

## API Endpoints

### POST /predict
Predicts credit grade for an SME
```json
{
  "Sales_M1": 358337,
  "Sales_M2": 342030,
  // ... (48 monthly features required)
  "Decl_Purchases_M12": 156116
}
```

**Response:**
```json
{
  "status": "success",
  "predicted_grade": "A",
  "confidence": 0.9905,
  "derived_features": {
    "Compliance_Sales": 0.9905,
    "Sales_Stability": 0.0287
  },
  "reasons": [
    "Excellent credit profile with minimal risk",
    "Excellent sales compliance"
  ]
}
```

### GET /health
Returns API health status

### GET /models-info
Returns model metadata and required features

## Model Performance

| Metric | Score |
|--------|--------|
| Accuracy | 99.25% |
| ROC-AUC | 0.9999 |
| F1-Score (Macro) | 0.99 |
| Training Time | 90.64s |

### Feature Importance
1. **Compliance_Sales** (16.92%) - Primary risk indicator
2. **Compliance_Adjusted** (14.55%) - Sophisticated compliance measure
3. **Purchase_to_Sales_Ratio** (14.26%) - Operational efficiency
4. **Profitability** (13.46%) - Financial health
5. **Compliance_Purchases** (7.99%) - Secondary compliance measure
<img width="435" height="215" alt="image" src="https://github.com/user-attachments/assets/a844fa96-917f-48fa-977f-eaf481a2e5ba" />


## Key Insights

- **Compliance Dominance**: Tax compliance features contribute 39.46% of model decisions
- **Conservative Bias**: Misclassifications favor underestimating rather than overestimating risk
- **Balanced Performance**: Excellent precision across all credit grades (A: 100%, B: 99%, C: 98%, D: 100%)

## Research Context

This project serves as a proof-of-concept for academic research on AI-driven credit scoring for underbanked SMEs in emerging markets. The synthetic data approach addresses:
- Institutional access barriers
- Data privacy regulations  
- Research timeline constraints
- Ethical considerations

## Limitations

- **Synthetic Data**: Requires real-world validation with actual SME and RRA data
- **Single Algorithm**: Currently implements Random Forest only
- **Prototype Status**: Additional development needed for production deployment

## Future Work

- Real-world data validation with Rwanda Revenue Authority
- Multi-algorithm comparison study
- Production-ready deployment architecture
- Cross-market adaptation research

## Contributing

This is an academic research project. For collaboration opportunities:
1. Fork the repository
2. Create feature branches
3. Submit pull requests with detailed descriptions
4. Follow existing code style and documentation standards

## Citation

If you use this work in academic research, please cite:
```
Serge, P. M. (2025). A Data-Driven AI System for Credit Scoring of 
Underbanked SMEs in Rwanda Based on Sales Transactions and Tax Declarations. 
University of East London (via Unicaf).
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Rwanda Revenue Authority (conceptual framework)
- SME sector stakeholders (domain expertise)
- University of East London and Unicaf (academic support)
- Academic supervisors and reviewers

## Contact

**SERGE PALUKU MULWAHALI**
- Email: pm.serge@gmail.com
- Phone: +250787283185
- Institution: University of East London (via Unicaf)

Project Link: https://github.com/sergekarim/uel-credit-scoring
