import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import UnidentifiedImageError

class FruitPredictor:

    def __init__(self, classifier_path="models/tf_ripeness_model.keras", regressor_path="models/shelflife_regressor.pkl"):

        print("Loading tensorflow ripeness classifier...")
        self.classifier = tf.keras.models.load_model(classifier_path)
        self.class_names = ['Overripe', 'Ripe', 'Unripe']

        print("Loading sklearn shelflife regressor...")
        self.regressor = joblib.load(regressor_path)
        
    def predict(self, image_path, storage_temp_c=22.0, humidity_pct=60.0):

        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.classifier.predict(img_array, verbose=0)
        confidence = float(np.max(predictions[0]))
        predicted_idx = int(np.argmax(predictions[0]))
        ripeness_stage = self.class_names[predicted_idx]

        input_data = pd.DataFrame([{
            'ripeness_stage': ripeness_stage,
            'storage_temp_c': float(storage_temp_c),
            'humidity_pct': float(humidity_pct)
        }])

        estimated_days = float(self.regressor.predict(input_data)[0])

        return {
            "status": "success",
            "ripenss_analysis": {
                "ripeness_stage": ripeness_stage,
                "confidence": round(confidence, 4),
            },
            "preservation_estimate": {
                "storage_temp_c": storage_temp_c,
                "humidity_pct": humidity_pct,
                "estimated_shelf_life_days": round(max(0.0, estimated_days), 1)
            }
        }

if __name__ == "__main__":
    import sys
    predictor = FruitPredictor()

    sample_dir = "data/Train/Ripe"
    if os.path.exists(sample_dir) and os.listdir(sample_dir):
        sample_img = os.path.join(sample_dir, os.listdir(sample_dir)[0])
        print(f"\nTesting pipeline on sample image: {sample_img}")
        
        result = predictor.predict(sample_img, storage_temp_c=20.0, humidity_pct=65.0)
        print("\nUnified Prediction Output:")
        print(json.dumps(result, indent=2))
    else:
        print("Please provide an image path to test.")
        