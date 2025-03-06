# DETECTION-OF-CYBER-ATTACKS-IN-NETWORK-USING-MACHIN-LEARNIG-TECHNIQUES

## Features
- Algorithms Used: Random Forest, SVM, CNN, ANN.
- Dataset: CICIDS2017 dataset.
- Flask API: Deploy the trained model as a REST API for predictions.

---

## Installation

1. Clone the Repository:
   ```bash
   git clone https://github.com/your-username/cyber-attack-detection.git
   cd cyber-attack-detection
   ```

2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Train the Model:
   - Place your dataset (`Train.txt` and `test.txt`) in the `data/` folder.
   - Run the training script:
     ```bash
     python train_model.py
     ```

2. Run the Flask API:
   - Start the Flask server:
     ```bash
     python app.py
     ```
   - The API will be available at `http://127.0.0.1:5000/`.

3. Make Predictions:
   - Send a POST request to the `/predict` endpoint with input features:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"features": [16,6,324,0,0,0,22,0,0,0,0,0]}' http://127.0.0.1:5000/predict
     ```

---

## Project Structure
```
cyber-attack-detection/
├── data/                   # Dataset files
├── train_model.py          # Data preprocessing and model training
├── app.py                  # Flask API for predictions
├── model.pkl               # Trained model (generated after training)
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

---

## Requirements
- Python 3.7+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `flask`, `pickle`
