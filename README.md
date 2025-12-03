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


***

### 📋 Experimental Results

#### 1. Performance Metrics

The model was evaluated on a sparse **implicit-feedback dataset** (matrix density ≈ 0.65%). The results demonstrate strong retrieval capabilities, particularly in the Top-20 recommendation lists.

**Table 1: Accuracy Metrics**

| Metric | Top-5 | Top-10 | Top-20 |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.1440 ± 0.1627 | 0.1040 ± 0.1029 | 0.1225 ± 0.0726 |
| **Recall** | 0.1356 ± 0.1677 | 0.1855 ± 0.2028 | **0.4587 ± 0.3011** |
| **F1-Score** | 0.1318 ± 0.1497 | 0.1262 ± 0.1260 | 0.1858 ± 0.1084 |

**Table 2: User Experience & Diversity Metrics**

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Personalization** | **0.9988** | Extremely high distinctiveness; users receive almost entirely unique lists. |
| **Diversity Score** | **0.9347** | High intra-list variety, effectively preventing item homogenization. |
| **Interest Alignment** | 0.6091 | ~61% of recommendations align with the user's core historical categories. |
| **Category Match** | 0.3386 | ~34% of items match exact sub-categories, indicating cross-category exploration. |
| **Coverage** | 0.2413 | 24% catalogue coverage (limited by the long-tail distribution of the training data). |

---

#### 2. Experimental Analysis

Based on the metrics above, the **Item-KNN Collaborative Filtering** model demonstrates the following strengths:

*   **Robust Recall Capability**:
    Achieving a **Recall@20 of 45.87%** is significant given the high sparsity of the dataset (users average only 4.8 positive interactions). This confirms the model effectively retrieves nearly half of the potential items of interest, serving as a highly competent **Retrieval Layer** in a recommendation funnel.

*   **Hyper-Personalization**:
    With a **Personalization Degree of 99.88%**, the system avoids the common pitfall of reverting to "Popularity-based" recommendations. It successfully captures unique user preferences, ensuring that different users see distinct content ("千人千面").

*   **Strategic Precision Trade-off**:
    While Precision@20 hovers around 12.25%, this is consistent with implicit feedback settings where "negatives" are not explicitly defined. The metrics show that as $K$ increases, Recall improves significantly while Precision degrades gracefully, validating the strategy of prioritizing **candidate broadening** over immediate precision.

*   **Balanced Exploration**:
    An **Interest Alignment of 0.61** ensures the dominant user interests are respected, while a lower Category Match Rate (0.34) indicates the system is capable of **cross-category recommendations** (e.g., suggesting accessories based on main product purchases), mimicking realistic e-commerce discovery patterns.

**Conclusion**:
The model performs robustly in **sparse data** scenarios, successfully balancing core interest alignment with high diversity and personalization. It effectively solves the retrieval problem for active users while maintaining a realistic level of serendipity.

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