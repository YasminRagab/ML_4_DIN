# DIN Classification Flask App

This Flask application classifies walls as per DIN 276 standards using machine learning models. The models are trained on data extracted from a Building Information Modeling (BIM) system and can classify walls based on various features such as location, function, levels, and materials.

## Features

- Upload JSON files containing wall data for classification.
- Classify walls using pre-trained LightGBM and CatBoost models.
- View F1 scores and confusion matrices for each model.
- Models automatically improve with each file uploaded and classified.

## Setup Instructions

### Prerequisites

- Python 3.6+
- pip (Python package installer)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/YasminRagab/ML_4_DIN.git
   cd your-repository

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt

### Running the App

1. **Set up the Flask environment**:

   ```bash
   set FLASK_APP=app
   set FLASK_ENV=development

2. **Run the Flask app**:

   ```bash
   flask run

3. **Open the app in your browser**:

   Navigate to http://127.0.0.1:5000 in your web browser to access the app.

