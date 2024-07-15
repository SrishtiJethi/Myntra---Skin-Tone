import joblib
import pandas as pd

model_dress_loaded = joblib.load('model_dress.pkl')
model_lip_loaded = joblib.load('model_lip.pkl')

label_encoders = joblib.load('label_encoders.pkl')  

def suggest_shade(skin_tone, hair_color, lip_color, eye_color):
    input_data = {
        'skin_tone': label_encoders['skin_tone'].transform([skin_tone])[0],
        'hair_color': label_encoders['hair_color'].transform([hair_color])[0],
        'lip_color': label_encoders['lip_color'].transform([lip_color])[0],
        'eye_color': label_encoders['eye_color'].transform([eye_color])[0]
    }
    input_df = pd.DataFrame([input_data])
    dress_shade_encoded = model_dress_loaded.predict(input_df)[0]
    lip_shade_encoded = model_lip_loaded.predict(input_df)[0]
    dress_shade = label_encoders['dress_shade'].inverse_transform([dress_shade_encoded])[0]
    lip_shade = label_encoders['lip_shade'].inverse_transform([lip_shade_encoded])[0]
    
    return dress_shade, lip_shade

def get_user_input():
    skin_tone = input("Enter your skin tone: ")
    hair_color = input("Enter your hair color: ")
    lip_color = input("Enter your lip color: ")
    eye_color = input("Enter your eye color: ")    
    suggested_dress_shade, suggested_lip_shade = suggest_shade(skin_tone, hair_color, lip_color, eye_color)
    print(f'Suggested Dress Shade: {suggested_dress_shade}')
    print(f'Suggested Lip Shade: {suggested_lip_shade}')

if __name__ == "__main__":
    get_user_input()
