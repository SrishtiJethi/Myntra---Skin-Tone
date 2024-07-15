import pandas as pd
import random
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('fashion_dataset.csv')

label_encoders = joblib.load('label_encoders.pkl')
for column in df.columns:
    df[column] = label_encoders[column].transform(df[column].astype(str))

X = df[['skin_tone', 'hair_color', 'lip_color', 'eye_color', 'age', 'season']]
y_dress = df['dress_shade']

X_train, X_test, y_train_dress, y_test_dress = train_test_split(X, y_dress, test_size=0.2, random_state=42)

model_dress = GradientBoostingClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 5]
}

grid_search_dress = GridSearchCV(model_dress, param_grid, cv=5)
grid_search_dress.fit(X_train, y_train_dress)

best_model_dress = grid_search_dress.best_estimator_

joblib.dump(best_model_dress, 'model_dress.pkl')
y_pred_dress = best_model_dress.predict(X_test)
dress_accuracy = accuracy_score(y_test_dress, y_pred_dress)
print(f'Dress Shade Model Accuracy: {dress_accuracy}')

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



def suggest_shade(skin_tone, hair_color, lip_color, eye_color, age, season):
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
    
    return dress_shade, lip_shade


suggested_dress_shade, suggested_lip_shade = suggest_shade('fair', 'blonde', 'pink', 'blue', 25, 'spring')
print(f'Suggested Dress Shade: {suggested_dress_shade}')
print(f'Suggested Lip Shade: {suggested_lip_shade}')
