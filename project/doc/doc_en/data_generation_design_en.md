# Data Generation Module Design Document

## Module Overview

The data generation module is responsible for creating simulated e-commerce recommendation system data, including user preferences, product information, and user-item interaction data. This module generates simulated data that aligns with real-world business scenarios based on real classification systems.

## Core Components

### 1. UserGenerator - User Preference Generator

#### Core Functionality
- Generates user preference types (single category, multi-category, explorer)
- Allocates user interest intensity based on category popularity
- Provides a user interaction probability calculation interface

#### Design Principles
- **Preference Type Distribution**: 30% single category users, 50% multi-category users, 20% explorer users
- **Interest Intensity**: High interest (30%), Medium interest (50%), Low interest (20%)
- **Category Weighting**: Weighted random selection based on category popularity from the configuration file

#### Key Algorithms
```python
def generate_category_interests(self, user_id, preference_type, categories):
    # Generate category interests based on preference type
    if preference_type == 'single_category':
        # Select only one high-interest category
    elif preference_type == 'multi_category':
        # Select 2-5 categories with varying interest intensities
    else:  # explorer
        # Have interest in all categories, intensity based on popularity
```

### 2. ItemGenerator - Product Allocation Generator

#### Core Functionality
- Allocates products to real classification systems
- Generates product attributes (price, popularity, etc.)
- Maintains category-to-product mapping relationships

#### Design Principles
- **Category Weight Allocation**: Popular categories receive more product resources
- **Attribute Generation**: Price and popularity are adjusted based on category characteristics
- **Quantity Control**: Each category has at least 5 products, with a maximum of 20 products

#### Key Features
- Supports real classification mapping
- Attribute adjustment based on category characteristics
- Intelligent product quantity allocation

### 3. InteractionGenerator - Interaction Generator

#### Core Functionality
- Generates user-item interactions based on user interests
- Supports multiple interaction types (click, add to cart, favorite)
- Avoids duplicate interactions, simulating real user behavior

#### Design Principles
- **Interest-Driven**: Users have a higher interaction probability with products in their preferred categories
- **Exploration Behavior**: 20% probability of random exploration
- **Interaction Types**: Click (70%), Add to Cart (20%), Favorite (10%)

#### Interaction Weights
- Click: Weight 1
- Add to Cart: Weight 3  
- Favorite: Weight 3

## Data Flow

```
Data Generation Process
1. Load category mapping data
2. Generate user preference data
   ↓
3. Allocate products to categories
   ↓  
4. Generate user-item interactions
   ↓
5. Data cleaning and preprocessing
   ↓
6. Save to file system
```

## Configuration Parameters

### User Preference Configuration
```python
user_preference: {
    'preference_types': {
        'single_category': 0.3,      # 30% single category users
        'multi_category': 0.5,       # 50% multi-category users
        'explorer': 0.2              # 20% explorer users
    },
    'max_categories_per_user': 5,    # Maximum 5 interested categories per user
    'min_categories_per_user': 1,    # Minimum 1 interested category per user
}
```

### Category Interest Configuration
```python
category_interest: {
    'category_popularity': {
        'Baby & Toddler Food': 0.15,  # Baby food is the most popular
        'Fresh Foods & Bakery': 0.12,
        'Health & Body': 0.10,
        # ... other categories
    },
    'interest_based_probability': {
        'high_interest': 0.8,        # 80% interaction probability for high-interest categories
        'medium_interest': 0.4,      # 40% for medium interest
        'low_interest': 0.1          # 10% for low interest
    }
}
```

### Product Allocation Configuration
```python
enhanced_processor: {
    'popular_categories': [           # List of popular categories
        'Baby & Toddler Food',
        'Health & Body', 
        'Household & Cleaning',
        'Pantry',
        'Fresh Foods & Bakery',
        'Hot & Cold Drinks'
    ],
    'category_weights': {
        'popular': 3.0,              # Popular category weight
        'food_baby': 2.0,            # Food and baby products weight
        'default': 1.0               # Default weight
    },
    'price_range': (10, 500),        # Price range
    'items_per_category': {
        'min': 5,                    # Minimum 5 products per category
        'max': 20,                   # Maximum 20 products per category
        'popular_multiplier': 2      # Double the number of products for popular categories
    }
}
```

## Data Quality Assurance

### 1. Data Cleaning
- Removes duplicate user-item-time interaction records
- Filters invalid user IDs and product IDs
- Removes users and products with too few interactions (cold start handling)

### 2. Data Splitting
- Splits training and test sets in chronological order
- Default 80% training set, 20% test set
- Ensures temporal continuity, avoiding data leakage

### 3. Statistical Validation
- Validates user interaction frequency distribution
- Validates product interaction frequency distribution  
- Validates category distribution statistics
- Validates interaction type distribution

## Extensibility Design

### Supports Custom Classification Systems
- By modifying the `category_mapping.json` file
- Supports any level of category structure
- Automatically extracts second-level categories for recommendations

### Flexible Parameter Configuration
- All parameters are managed through configuration files
- Supports dynamic adjustment of data scale
- Easy to perform A/B testing

### Data Export Formats
- CSV format for easy use by other systems
- JSON format for saving metadata information
- Complete statistical information report

## Performance Considerations

### Memory Optimization
- Uses generator pattern to generate data incrementally
- Processes large-scale data in batches
- Sparse matrix storage for user-item interactions

### Computational Efficiency
- Vectorized operations replace loops
- Weight-based random sampling algorithm
- Parallel processing of user data generation

## Usage Examples

```python
# Generate complete dataset
processor = EnhancedDataProcessor()
results = processor.generate_complete_dataset()

# Get statistics
stats = processor.get_statistics()
category_stats = processor.get_category_statistics()
```

This module provides high-quality, realistic simulated data for recommendation systems, laying a solid foundation for subsequent model training and recommendation generation.