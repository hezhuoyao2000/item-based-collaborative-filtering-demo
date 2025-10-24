# Data Processing Module Design Document

## Module Overview

The data processing module is the core infrastructure of the recommendation system, responsible for data loading, cleaning, transformation, and feature engineering. This module integrates real-world category data, supporting the complete processing flow from raw data to the format required for model training.

## Core Components

### EnhancedDataProcessor - Enhanced Data Processor

#### Core Functions
- **Data Loading**: Supports CSV file loading and real-time data generation
- **Data Cleaning**: Deduplication, invalid value handling, cold start filtering
- **Data Splitting**: Splits training and test sets by time
- **Matrix Construction**: Creates user-item interaction matrices
- **Category Processing**: Integrates real-world category metadata

## Data Processing Flow

```
Complete data processing flow
1. Load category mapping data
   ↓
2. Load/generate product data
   ↓
3. Load/generate interaction data  
   ↓
4. Data cleaning and validation
   ↓
5. Training set/test set splitting
   ↓
6. User-item matrix construction
   ↓
7. Statistical information generation
   ↓
8. Data persistence storage
```

## Key Technical Implementation

### 1. Category Data Processing

#### Category Structure
```python
def _extract_categories_from_mapping(self):
    # Extract hierarchical information from category mapping
    # Automatically infer parent category relationships
    # Build complete category metadata
```

#### Category Weight Calculation
- Based on the configured popular category list
- Supports custom category weights
- Automatic normalization processing

### 2. Data Cleaning Strategy

#### Deduplication Handling
```python
def clean_data(self):
    # Remove duplicate user-item-time records
    # Filter invalid user IDs and item IDs
    # Remove users and items with too few interactions (cold start handling)
```

#### Cold Start Handling
- Users with at least 2 interactions are retained
- Items with at least 2 interactions are retained  
- Ensures the validity of model training

### 3. Time Series Splitting

#### Time-aware Splitting
```python
def split_train_test(self):
    # Sort interaction data in time order
    # First 80% as training set, last 20% as test set
    # Maintain time continuity, avoid data leakage
```

#### Time Range Statistics
- Training set time range display
- Test set time range display
- Time span analysis

### 4. Sparse Matrix Construction

#### Interaction Matrix Construction
```python
def create_enhanced_user_item_matrix(self, data_type='train'):
    # Create user and item index mappings
    # Calculate user-item weights based on interaction type weights
    # Build CSR format sparse matrix
```

#### Weight Calculation Strategy
- Click: Weight 1
- Add to cart: Weight 3
- Favorite: Weight 3
- Accumulate weights of multiple interactions

## Data Structure Design

### 1. Interaction Data Format
```python
interactions_df: DataFrame
- user_id: User ID
- item_id: Item ID  
- interaction_type: Interaction type (click/cart/favorite)
- category: Item category
- timestamp: Timestamp
```

### 2. Item Data Format
```python
items_df: DataFrame  
- item_id: Item ID
- category: Second-level category name
- price: Item price
- popularity_score: Popularity score
- category_id: Category ID
- parent_category: First-level category name
- parent_category_id: Parent category ID
```

### 3. Matrix Data Structure
```python
user_item_matrix: csr_matrix
- Row: User index
- Column: Item index
- Value: Sum of interaction weights

user_index: Dict[str, int]  # User ID to row index mapping
item_index: Dict[str, int]  # Item ID to column index mapping
```

## Statistical Monitoring System

### 1. Basic Statistics
```python
def get_statistics(self):
    # Number of users, items, interactions
    # Interaction type distribution statistics
    # User interaction count distribution (mean, median, extremes)
    # Item interaction count distribution (mean, median, extremes)
```

### 2. Category Statistics
```python
def get_category_statistics(self):
    # Second-level category distribution (number of items)
    # First-level category distribution (number of items)
    # Total category count statistics
```

### 3. Matrix Statistics
```python
def get_matrix_statistics(self):
    # Matrix shape and density
    # Number of non-zero elements
    # Sparsity analysis
```

## Performance Optimization Strategies

### 1. Memory Optimization
- Use sparse matrices to store large-scale data
- Batch processing to avoid memory overflow
- Timely release of temporary variables

### 2. Computational Optimization
- Vectorized operations instead of loops
- Use efficient data structures
- Parallel processing of independent tasks

### 3. IO Optimization
- Batch file read/write operations
- Use binary format to save models
- Cache frequently used data

## Extensibility Design

### 1. Data Source Extension
- Supports multiple file formats (CSV, JSON, Parquet)
- Supports database connections
- Supports real-time data streams

### 2. Processing Flow Extension
- Pluggable data cleaning steps
- Custom feature engineering pipelines
- Flexible data splitting strategies

### 3. Category System Extension
- Supports multi-level category structures
- Dynamic loading of category mappings
- Custom category weights

## Quality Control

### 1. Data Validation
- Data type checks
- Value range validation
- Consistency checks

### 2. Exception Handling
- File not found handling
- Data format error handling
- Insufficient memory handling

### 3. Log Monitoring
- Processing progress logs
- Error warning logs
- Performance statistics logs

## Usage Examples

### Basic Usage
```python
# Create processor instance
processor = EnhancedDataProcessor()

# Load category data
processor.load_category_data('category_mapping.json')

# Generate complete dataset
results = processor.generate_complete_dataset()

# Or load existing data
processor.load_data('interactions.csv', 'items.csv')
processor.clean_data()
processor.split_train_test()
matrix = processor.create_enhanced_user_item_matrix('train')
```

### Statistical Information Retrieval
```python
# Get basic statistics
stats = processor.get_statistics()

# Get category statistics  
category_stats = processor.get_category_statistics()

# Get user interaction history
user_history = processor.get_user_interactions('u123', 'train')
```

## Best Practices

1. **Data Scale Control**: Adjust data scale according to memory capacity
2. **Regular Cleanup**: Clean up intermediate data promptly after processing
3. **Backup Mechanism**: Regularly back up important data
4. **Monitoring and Alerts**: Set processing time threshold alerts
5. **Version Management**: Version management of data processing flows

This module provides stable, efficient, and scalable data processing capabilities for the recommendation system, ensuring reliable system operation.