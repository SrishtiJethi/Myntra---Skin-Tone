from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import random

app = Flask(__name__)
CORS(app)

best_model_dress = joblib.load('model_dress.pkl')
label_encoders = joblib.load('label_encoders.pkl')
def get_compatible_lip_shade(dress_shade, skin_tone, hair_color, lip_color):
    if dress_shade in ['pastel', 'neutral']:
        if skin_tone in ['fair', 'light', 'porcelain', 'ivory']:
            return random.choice(['nude', 'coral', 'peach'])
        elif skin_tone in ['medium', 'olive', 'tan']:
            return random.choice(['nude', 'coral', 'peach', 'rose'])
        elif skin_tone in ['dark', 'deep', 'ebony']:
            return random.choice(['nude', 'coral', 'peach', 'berry'])
    elif dress_shade in ['earth tones', 'jewel tones']:
        if skin_tone in ['fair', 'light', 'porcelain', 'ivory']:
            return random.choice(['rose', 'berry', 'mauve'])
        elif skin_tone in ['medium', 'olive', 'tan']:
            return random.choice(['rose', 'berry', 'red'])
        elif skin_tone in ['dark', 'deep', 'ebony']:
            return random.choice(['rose', 'berry', 'plum'])
    elif dress_shade in ['vibrant colors', 'bright']:
        if skin_tone in ['fair', 'light', 'porcelain', 'ivory']:
            return random.choice(['red', 'berry', 'cherry'])
        elif skin_tone in ['medium', 'olive', 'tan']:
            return random.choice(['red', 'berry', 'cherry'])
        elif skin_tone in ['dark', 'deep', 'ebony']:
            return random.choice(['red', 'berry', 'plum'])
    elif dress_shade in ['metallic', 'muted']:
        if skin_tone in ['fair', 'light', 'porcelain', 'ivory']:
            return random.choice(['mauve', 'plum', 'bronze'])
        elif skin_tone in ['medium', 'olive', 'tan']:
            return random.choice(['mauve', 'plum', 'bronze'])
        elif skin_tone in ['dark', 'deep', 'ebony']:
            return random.choice(['mauve', 'plum', 'bronze'])
    return random.choice(['nude', 'coral', 'rose', 'berry', 'red', 'mauve', 'plum', 'peach', 'bronze', 'cherry'])

@app.route('/suggest_shade', methods=['POST'])
def suggest_shade():
    data = request.get_json()
    print("Received data:", data)  
    skin_tone = data['skin_tone']
    hair_color = data['hair_color']
    lip_color = data['lip_color']
    eye_color = data['eye_color']
    age = data['age']
    season = data['season']

    input_data = {
        'skin_tone': label_encoders['skin_tone'].transform([skin_tone])[0],
        'hair_color': label_encoders['hair_color'].transform([hair_color])[0],
        'lip_color': label_encoders['lip_color'].transform([lip_color])[0],
        'eye_color': label_encoders['eye_color'].transform([eye_color])[0],
        'age': age,
        'season': label_encoders['season'].transform([season])[0]
    }

    input_df = pd.DataFrame([input_data])
    dress_shade_encoded = best_model_dress.predict(input_df)[0]
    dress_shade = label_encoders['dress_shade'].inverse_transform([dress_shade_encoded])[0]

    lip_shade = get_compatible_lip_shade(dress_shade, skin_tone, hair_color, lip_color)
    print("Sending response:", {'dress_shade': dress_shade, 'lip_shade': lip_shade})  

    return jsonify({'dress_shade': dress_shade, 'lip_shade': lip_shade})

if __name__ == '__main__':
    app.run(debug=True)
