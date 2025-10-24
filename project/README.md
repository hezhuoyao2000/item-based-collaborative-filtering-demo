# E-commerce Recommendation System - Item-Based Collaborative Filtering

## Project Overview

This is an e-commerce recommendation system based on Item-Based Collaborative Filtering, integrating real classification data, user preference modeling, hybrid recommendation strategies, and a comprehensive evaluation system. The system supports a complete process from data generation and model training to recommendation generation and effect evaluation.

## Core Features

- **Real Classification Data Integration**: Uses a real product classification system, supporting multi-level category structures
- **Intelligent User Preference Modeling**: Generates personalized preference data based on user interest intensity
- **Hybrid Recommendation Strategy**: Combines similarity-based recommendations and category preference recommendations to improve recommendation quality
- **Cold Start Handling**: Provides effective cold start solutions for new users and low-interaction users
- **Comprehensive Evaluation System**: Supports dual evaluation with traditional accuracy metrics and interest grasp metrics
- **Scalable Architecture**: Modular design, supporting various similarity algorithms and recommendation strategies

## System Architecture

```
Recommendation System Architecture
├── Data Layer
│   ├── UserGenerator
│   ├── ItemGenerator 
│   └── InteractionGenerator
├── Processing Layer
│   └── EnhancedDataProcessor
├── Model Layer
│   └── ModelTrainer
├── Recommendation Layer
│   └── Recommender
└── Application Layer
    └── Main Controller
```

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Example

1. **Generate Complete Dataset**
```bash
python main.py generate_data
```

2. **Train Recommendation Model**
```bash
python main.py train_model
```

3. **Evaluate Model Performance**
```bash
python main.py evaluate
```

4. **Generate Recommendations for User**
```bash
python main.py recommend --user_id u123
```

5. **View System Status**
```bash
python main.py status
```

## Configuration Instructions

System configurations are centralized in the `config.py` file, mainly including:

- **Data Configuration**: Number of users, number of items, number of interactions, etc.
- **Model Configuration**: Similarity algorithm, number of recommendations, etc.
- **File Configuration**: Data file paths, model file paths, etc.
- **Category Configuration**: Category weights, popular category settings, etc.

## Module Details

For detailed design descriptions of each module, please refer to the design documents in the `doc/` directory:

- [Data Generation Module Design](data_generation_design_en.md)
- [Data Processing Module Design](data_processing_design.md) 
- [Model Training Module Design](model_training_design.md)
- [Recommendation Generation Module Design](recommendation_design_en.md)
- [System Architecture Design](system_architecture.md)

## Performance Metrics

The system supports various evaluation metrics:

- **Traditional Accuracy Metrics**: Precision@K, Recall@K, F1-Score@K
- **Interest Grasp Metrics**: Category matching rate, interest alignment, diversity score
- **User Experience Metrics**: Recommendation coverage, personalization degree, cold start effectiveness

## Extensibility

The system design has good extensibility:

- Supports various similarity algorithms (Cosine, Pearson, Adjusted Cosine)
- Easily add new recommendation strategies
- Supports custom classification systems and user preference models
- Modular design facilitates functional expansion and maintenance

## Technical Support

For technical questions or suggestions, please contact the project maintenance team.