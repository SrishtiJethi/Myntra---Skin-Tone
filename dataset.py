import pandas as pd
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

skin_tones = ['fair', 'light', 'medium', 'olive', 'tan', 'dark', 'deep', 'porcelain', 'ebony', 'ivory']
hair_colors = ['blonde', 'light brown', 'brown', 'dark brown', 'black', 'red', 'auburn', 'chestnut', 'gray', 'platinum']
lip_colors = ['pink', 'peach', 'nude', 'red', 'purple', 'burgundy', 'mauve', 'coral', 'brown', 'plum']
eye_colors = ['blue', 'green', 'hazel', 'brown', 'gray', 'amber', 'violet', 'black', 'teal', 'turquoise']
dress_shades = ['pastel', 'earth tones', 'vibrant colors', 'neutral', 'metallic', 'jewel tones', 'monochrome', 'bright', 'muted', 'neon']
lip_shades = ['nude', 'coral', 'rose', 'berry', 'red', 'mauve', 'plum', 'peach', 'bronze', 'cherry']

data = []

for _ in range(10000):  
    skin_tone = random.choice(skin_tones)
    hair_color = random.choice(hair_colors)
    lip_color = random.choice(lip_colors)
    eye_color = random.choice(eye_colors)
    
    if skin_tone in ['fair', 'light', 'porcelain', 'ivory']:
        if hair_color in ['blonde', 'light brown', 'platinum']:
            dress_shade = 'pastel'
            lip_shade = random.choice(['nude', 'coral', 'peach'])
        elif hair_color in ['brown', 'dark brown', 'auburn']:
            dress_shade = 'pastel'
            lip_shade = random.choice(['rose', 'berry', 'mauve'])
    elif skin_tone in ['medium', 'olive', 'tan']:
        if hair_color in ['blonde', 'light brown', 'chestnut']:
            dress_shade = 'earth tones'
            lip_shade = random.choice(['nude', 'coral', 'peach'])
        elif hair_color in ['brown', 'dark brown', 'auburn']:
            dress_shade = 'earth tones'
            lip_shade = random.choice(['rose', 'berry', 'mauve'])
    elif skin_tone in ['dark', 'deep', 'ebony']:
        if hair_color in ['black', 'dark brown']:
            dress_shade = 'vibrant colors'
            lip_shade = random.choice(['nude', 'coral', 'berry', 'plum'])
        elif hair_color in ['brown', 'red', 'auburn']:
            dress_shade = 'vibrant colors'
            lip_shade = random.choice(['rose', 'berry', 'red', 'plum'])

    data.append({
        'skin_tone': skin_tone,
        'hair_color': hair_color,
        'lip_color': lip_color,
        'eye_color': eye_color,
        'dress_shade': dress_shade,
        'lip_shade': lip_shade
    })

df = pd.DataFrame(data)


for column in df.columns:
    df[column] = df[column].astype(str)

label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le


X = df[['skin_tone', 'hair_color', 'lip_color', 'eye_color']]
y_dress = df['dress_shade']
y_lip = df['lip_shade']

X_train, X_test, y_train_dress, y_test_dress = train_test_split(X, y_dress, test_size=0.2, random_state=42)
X_train, X_test, y_train_lip, y_test_lip = train_test_split(X, y_lip, test_size=0.2, random_state=42)

model_dress = RandomForestClassifier(random_state=42)
model_dress.fit(X_train, y_train_dress)

model_lip = RandomForestClassifier(random_state=42)
model_lip.fit(X_train, y_train_lip)

joblib.dump(model_dress, 'model_dress.pkl')
joblib.dump(model_lip, 'model_lip.pkl')

model_dress_loaded = joblib.load('model_dress.pkl')
model_lip_loaded = joblib.load('model_lip.pkl')

joblib.dump(label_encoders, 'label_encoders.pkl')

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

suggested_dress_shade, suggested_lip_shade = suggest_shade('fair', 'blonde', 'pink', 'blue')
print(f'Suggested Dress Shade: {suggested_dress_shade}')
print(f'Suggested Lip Shade: {suggested_lip_shade}')