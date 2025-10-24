# System Architecture Design Document

## Architecture Overview

This e-commerce recommendation system adopts a modular, layered architecture design, based on item-based collaborative filtering algorithms, integrating real classification data and intelligent user preference modeling. The system has good scalability, maintainability, and performance.

## Overall Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Main Controller │  │ Command Line │  │    Web API        │  │
│  │  (Main)     │  │  (CLI)      │  │    (Web API)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Recommendation Layer                      │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 Recommender                           │  │
│  │  - Personalized Recs    - Hybrid Strategy    - Cold Start Handling   │  │
│  │  - Recommendation Expl. - User Profiling     - Quality Evaluation     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Layer                               │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 ModelTrainer                          │  │
│  │  - Similarity Calc    - Matrix Opt.      - Model Persist.     │  │
│  │  - Quality Eval.      - Algorithm Sel.   - Statistics       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Processing Layer                          │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │             EnhancedDataProcessor                     │  │
│  │  - Data Loading       - Data Cleaning    - Data Partitioning      │  │
│  │  - Matrix Building    - Category Proc.   - Statistical Monitoring      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ User Generator │  │ Item Generator │  │ Interaction Generator │  │
│  │ (UserGenerator)│ (ItemGenerator)│ (InteractionGenerator)│  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Storage Layer                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Data Files    │  │ Model Files    │  │ Configuration Files │  │
│  │  (CSV/JSON) │  │  (NPY/PKL)  │  │    (JSON)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Module Design

### 1. Data Layer

#### Responsibilities
- Generate simulated user preference data
- Assign items to real classification systems
- Generate user-item interaction data

#### Key Features
- **Real Classification Integration**: Based on real e-commerce classification systems
- **Intelligent Preference Modeling**: Simulates different types of user behavior
- **Interaction Weight System**: Differentiates weights for clicks, add-to-cart, and favorites

### 2. Processing Layer

#### Responsibilities
- Data loading and cleaning
- Training set/test set splitting
- User-item matrix construction
- Category data processing and statistics

#### Key Features
- **Data Quality Control**: Comprehensive data validation and cleaning
- **Time-Aware Splitting**: Splits data chronologically to avoid leakage
- **Sparse Matrix Optimization**: Efficiently handles large-scale sparse data

### 3. Model Layer

#### Responsibilities
- Calculate item similarity matrix
- Model training and optimization
- Model persistence and loading
- Quality evaluation and validation

#### Key Features
- **Multi-Algorithm Support**: Cosine, Pearson, Adjusted Cosine similarity
- **Memory Optimization**: Supports large-scale matrix computation
- **Quality Monitoring**: Comprehensive model statistics and validation

### 4. Recommendation Layer

#### Responsibilities
- Generate personalized recommendation lists
- User profile construction and analysis
- Recommendation explanation generation
- Recommendation quality evaluation

#### Key Features
- **Hybrid Strategy**: Combines similarity and category preference for hybrid recommendations
- **Cold Start Handling**: Intelligent recommendation strategies for new users
- **Explainability**: Provides detailed explanations for recommendation results
- **Comprehensive Evaluation**: Multi-dimensional recommendation quality evaluation system

### 5. Application Layer

#### Responsibilities
- System flow control and coordination
- Command-line interface provision
- Web API service exposure
- System status monitoring

#### Key Features
- **Modular Design**: Independent and replaceable modules
- **Flexible Interfaces**: Supports multiple usage methods
- **Status Monitoring**: Real-time system status viewing

## Data Flow Design

### Training Phase Data Flow
```
1. User Generator → User Preference Data
2. Item Generator → Item Classification Data  
3. Interaction Generator → User-Item Interaction Data
4. Data Processor → Cleaned Training Data
5. Data Processor → User-Item Matrix
6. Model Trainer → Item Similarity Matrix
7. Model Trainer → Persistent Model Files
```

### Recommendation Phase Data Flow
```
1. Recommender → Load Model Files
2. Recommender → Get User Historical Interactions
3. Recommender → Generate Recommendation Results
4. Recommender → Provide Recommendation Explanations
5. Recommender → Evaluate Recommendation Quality
```

## Configuration Management System

### 1. Core Configuration File (`config.py`)
```python
# Data Configuration
DATA_CONFIG = {
    'num_users': 1000,           # Number of users
    'num_items': 3000,           # Number of items
    'num_interactions': 150000,  # Number of interactions
    # ... Other data parameters
}

# Model Configuration  
MODEL_CONFIG = {
    'similarity_metric': 'pearson',  # Similarity algorithm
    'min_similarity': 0.1,          # Minimum similarity threshold
    'top_n': 10,                    # Number of recommendations
}

# File Configuration
FILE_CONFIG = {
    'interactions_file': 'interactions.csv',
    'items_file': 'items.csv', 
    'similarity_matrix_file': 'similarity_matrix.npy',
    # ... Other file paths
}
```

### 2. Category Mapping File (`category_mapping.json`)
```json
{
    "category_id_1": "Category Name 1",
    "category_id_2": "Category Name 2",
    # ... Other category mappings
}
```

## Performance Architecture Design

### 1. Memory Optimization Strategies
- **Sparse Matrix Storage**: Uses scipy.sparse matrix format
- **Batch Processing**: Processes large-scale data in batches
- **Caching Mechanism**: Caches frequently used data in memory

### 2. Computation Optimization Strategies
- **Vectorized Operations**: Uses numpy vectorized computations
- **Algorithm Selection**: Selects optimal algorithms based on data characteristics
- **Parallel Computing**: Supports multi-process parallel processing

### 3. IO Optimization Strategies
- **Binary Format**: Uses npy, pkl and other binary formats
- **Batch Read/Write**: Reduces number of IO operations
- **Compressed Storage**: Supports data compression storage

## Extensibility Design

### 1. Algorithm Extensibility
- **Plugin Architecture**: Supports custom similarity algorithms
- **Parameter Configuration**: All algorithm parameters are configurable
- **Algorithm Combination**: Supports combining multiple algorithms

### 2. Data Extensibility
- **Multiple Data Sources**: Supports files, databases, real-time streams
- **Format Support**: Supports CSV, JSON, Parquet and other formats
- **Scale Expansion**: Supports data scales from thousands to millions

### 3. Feature Extensibility
- **Modular Design**: Each functional module is independent and replaceable
- **Standardized Interfaces**: Unified module interface specifications
- **Plugin Mechanism**: Supports functional plugin extensions

## Reliability Design

### 1. Error Handling
- **Exception Catching**: Comprehensive exception catching and handling
- **Retry Mechanism**: Critical operations support retries
- **Degradation Strategy**: Automatic degradation in case of failure

### 2. Data Security
- **Data Validation**: Comprehensive input data validation
- **Backup Mechanism**: Regular backup of important data
- **Recovery Strategy**: Rapid recovery after failure

### 3. Monitoring and Alerting
- **Performance Monitoring**: Monitoring of key performance indicators
- **Error Alerting**: Timely alerts for abnormal situations
- **Logging System**: Complete operation log recording

## Deployment Architecture

### 1. Standalone Deployment
```
Suitable for development and testing environments
All modules run on the same machine
Uses local file system for storage
```

### 2. Distributed Deployment
```
Suitable for production environments
Modules can be deployed distributively
Uses distributed file system
Supports load balancing
```

### 3. Cloud-Native Deployment
```
Suitable for cloud environments
Containerized deployment
Automatic scaling
High availability design
```

## Development Guidelines

### 1. Code Specification
- PEP8 code style
- Type annotation support
- Complete docstrings

### 2. Testing Specification
- Unit test coverage
- Comprehensive integration tests
- Regular performance testing

### 3. Documentation Specification
- Complete module documentation
- Comprehensive API documentation
- Detailed deployment documentation

This architecture design ensures high performance, high availability, and easy extensibility of the system, meeting the personalized recommendation needs of e-commerce at different scales.