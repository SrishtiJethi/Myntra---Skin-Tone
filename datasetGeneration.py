import pandas as pd
import random
import joblib
from sklearn.preprocessing import LabelEncoder

skin_tones = ['fair', 'light', 'medium', 'olive', 'tan', 'dark', 'deep', 'porcelain', 'ebony', 'ivory']
hair_colors = ['blonde', 'light brown', 'brown', 'dark brown', 'black', 'red', 'auburn', 'chestnut', 'gray', 'platinum']
lip_colors = ['pink', 'peach', 'nude', 'red', 'purple', 'burgundy', 'mauve', 'coral', 'brown', 'plum']
eye_colors = ['blue', 'green', 'hazel', 'brown', 'gray', 'amber', 'violet', 'black', 'teal', 'turquoise']
dress_shades = ['pastel', 'earth tones', 'vibrant colors', 'neutral', 'metallic', 'jewel tones', 'monochrome', 'bright', 'muted', 'neon']
lip_shades = ['nude', 'coral', 'rose', 'berry', 'red', 'mauve', 'plum', 'peach', 'bronze', 'cherry']
seasons = ['spring', 'summer', 'fall', 'winter']
ages = range(15, 70)  

def get_compatible_dress_shade(skin_tone, hair_color, lip_color, eye_color):
    if skin_tone in ['fair', 'light', 'porcelain', 'ivory']:
        if hair_color in ['blonde', 'light brown', 'platinum']:
            return random.choice(['pastel', 'neutral', 'jewel tones'])
        elif hair_color in ['brown', 'dark brown', 'auburn']:
            return random.choice(['pastel', 'muted', 'jewel tones'])
        elif hair_color in ['red']:
            return random.choice(['jewel tones', 'earth tones', 'neutral'])
    elif skin_tone in ['medium', 'olive', 'tan']:
        if hair_color in ['blonde', 'light brown', 'chestnut']:
            return random.choice(['earth tones', 'vibrant colors', 'neutral'])
        elif hair_color in ['brown', 'dark brown', 'auburn']:
            return random.choice(['earth tones', 'vibrant colors', 'jewel tones'])
        elif hair_color in ['red']:
            return random.choice(['earth tones', 'neutral', 'vibrant colors'])
    elif skin_tone in ['dark', 'deep', 'ebony']:
        if hair_color in ['black', 'dark brown']:
            return random.choice(['vibrant colors', 'jewel tones', 'bright'])
        elif hair_color in ['brown', 'red', 'auburn']:
            return random.choice(['vibrant colors', 'earth tones', 'bright'])
        elif hair_color in ['gray']:
            return random.choice(['vibrant colors', 'jewel tones', 'metallic'])
    return random.choice(dress_shades)

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
    return random.choice(lip_shades)

data = []
for _ in range(10000):  
    skin_tone = random.choice(skin_tones)
    hair_color = random.choice(hair_colors)
    lip_color = random.choice(lip_colors)
    eye_color = random.choice(eye_colors)
    age = random.choice(ages)
    season = random.choice(seasons)

    dress_shade = get_compatible_dress_shade(skin_tone, hair_color, lip_color, eye_color)
    lip_shade = get_compatible_lip_shade(dress_shade, skin_tone, hair_color, lip_color)

    data.append({
        'skin_tone': skin_tone,
        'hair_color': hair_color,
        'lip_color': lip_color,
        'eye_color': eye_color,
        'age': age,
        'season': season,
        'dress_shade': dress_shade,
        'lip_shade': lip_shade
    })

df = pd.DataFrame(data)
df.to_csv('fashion_dataset.csv', index=False)
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

joblib.dump(label_encoders, 'label_encoders.pkl')


