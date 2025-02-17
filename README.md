# Meteorite Landings Predictive Analysis Application

## Project Overview
This application implements advanced data mining techniques to analyze, predict, and classify meteorite landings using historical data. The project combines machine learning algorithms with an interactive GUI to provide insights into meteorite characteristics and landing patterns.

## Technical Implementation

### Core Technologies
- **Python 3.x** - Primary programming language
- **NumPy** - Numerical computations and array operations
- **Pandas** - Data manipulation and analysis
- **Tkinter** - GUI framework
- **Matplotlib** - Data visualization and plotting
- **Dill/Pickle** - Model serialization

### Machine Learning Algorithms

#### 1. Custom Decision Tree Classification
- Implementation: `algorithms/classification.py`
- Purpose: Predicts whether a meteorite was "Fell" or "Found"
- Key Features:
  - Custom entropy calculation
  - Information gain optimization
  - Feature importance analysis
  - Model persistence using pickle
  - Configurable max depth parameter

#### 2. K-Means Clustering
- Implementation: `algorithms/clustering.py`
- Purpose: Groups meteorites based on mass and geographical location
- Features:
  - Dynamic K selection
  - Optimization reporting
  - Interactive cluster visualization
  - Geographical plotting

#### 3. Regression Analysis
- Implementation: `algorithms/regression1.py`, `algorithms/regression2.py`
- Purpose: Predicts meteorite landing locations and characteristics
- Methods:
  - Latitude/Longitude prediction
  - Mass prediction
  - Feature correlation analysis

### Architecture

#### GUI Implementation (`GUI.py`)
- **Framework**: Tkinter with ttk widgets
- **Features**:
  - Multi-tabbed interface
  - Dynamic graph generation
  - Interactive map visualization
  - Real-time predictions
  - Input validation
  - Responsive layout

#### Data Management
- Dataset: `dataset/meteorites.csv`
- Format: CSV with standardized fields
- Features tracked:
  - Mass (g)
  - Year
  - Geographical coordinates
  - Fall/Found classification
  - Name and composition

## Project Structure
```
├── GUI.py                 # Main application interface
├── algorithms/
│   ├── classification.py  # Decision tree implementation
│   ├── clustering.py      # K-means clustering
│   ├── regression1.py     # Location prediction
│   ├── regression2.py     # Mass prediction
│   └── *.pkl             # Serialized model files
├── dataset/
│   └── meteorites.csv    # Primary dataset
└── App_Icon.ico          # Application icon
```

## Setup and Execution

### Prerequisites
1. Python 3.x environment
2. Required libraries:
   ```
   numpy
   pandas
   tkinter
   matplotlib
   dill
   pickle
   ```

### Running the Application
1. Clone the repository maintaining the directory structure
2. Ensure all dependencies are installed
3. Generate model files (if not present):
   - Run each algorithm file in the `/algorithms` folder
   - Wait for `.pkl` files to be generated
4. Execute `GUI.py` to launch the application

### Model Training
- Each algorithm file can be run independently to retrain models
- Models are automatically saved as `.pkl` files in the `/algorithms` directory
- Existing models can be replaced by running the respective algorithm files

## Technical Features
- Custom implementation of decision tree algorithm
- Dynamic feature importance calculation
- Real-time data visualization
- Interactive geographical plotting
- Model persistence and loading
- Input validation and error handling
- Responsive GUI with multiple visualization options

## Performance Considerations
- Models are serialized to avoid retraining
- Efficient data structures for large dataset handling
- Optimized algorithms for real-time predictions
- Memory-efficient data loading and processing
