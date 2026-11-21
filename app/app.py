from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '..', 'model', 'decision_tree_model.pkl')
features_path = os.path.join(BASE_DIR, '..', 'model', 'feature_names.pkl')

model = joblib.load(model_path)
feature_names = joblib.load(features_path)

FEATURE_LABELS = [
    ('mean_radius', 'Mean Radius (Rata-rata Jari-jari Sel)', '14.13', '0.01'),
    ('mean_texture', 'Mean Texture (Rata-rata Tekstur)', '19.29', '0.01'),
    ('mean_perimeter', 'Mean Perimeter (Rata-rata Keliling)', '91.97', '0.01'),
    ('mean_area', 'Mean Area (Rata-rata Luas Sel)', '654.89', '0.1'),
    ('mean_smoothness', 'Mean Smoothness (Kehalusan Kontur)', '0.096', '0.001'),
    ('mean_compactness', 'Mean Compactness (Kekompakan)', '0.104', '0.001'),
    ('mean_concavity', 'Mean Concavity (Tingkat Kecekungan)', '0.088', '0.001'),
    ('mean_concave_points', 'Mean Concave Points (Titik Cekung)', '0.048', '0.001'),
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    input_values = {}
    decision_path_explanation = []

    if request.method == 'POST':
        try:
            vals = []
            for key, label, default, step in FEATURE_LABELS:
                val = float(request.form.get(key, default))
                input_values[key] = val
                vals.append(val)

            features_array = np.array([vals])
            pred = model.predict(features_array)[0]
            proba = model.predict_proba(features_array)[0]

            if pred == 1:
                prediction = "GANAS (Malignant)"
                probability = round(proba[1] * 100, 2)
            else:
                prediction = "JINAK (Benign)"
                probability = round(proba[0] * 100, 2)

            # White-box rule explanation
            concave_pts = vals[7]
            texture = vals[1]
            area = vals[3]
            radius = vals[0]

            if concave_pts <= 0.051:
                decision_path_explanation.append(f"1. Mean Concave Points ({concave_pts}) <= 0.051 -> Mengarah ke cabang tumor Jinak.")
                if area <= 696.25:
                    decision_path_explanation.append(f"2. Mean Area ({area}) <= 696.25 -> Ukuran sel tergolong kecil/normal.")
                    if texture <= 21.56:
                        decision_path_explanation.append(f"3. Mean Texture ({texture}) <= 21.56 -> Kesimpulan akhir: JINAK.")
                    else:
                        decision_path_explanation.append(f"3. Mean Texture ({texture}) > 21.56 -> Kesimpulan akhir: JINAK.")
                else:
                    decision_path_explanation.append(f"2. Mean Area ({area}) > 696.25 -> Ukuran sel membesar.")
                    if texture <= 18.68:
                        decision_path_explanation.append(f"3. Mean Texture ({texture}) <= 18.68 -> Kesimpulan akhir: JINAK.")
                    else:
                        decision_path_explanation.append(f"3. Mean Texture ({texture}) > 18.68 -> Kesimpulan akhir: GANAS.")
            else:
                decision_path_explanation.append(f"1. Mean Concave Points ({concave_pts}) > 0.051 -> Mengarah ke cabang risiko tinggi.")
                if texture <= 15.425:
                    decision_path_explanation.append(f"2. Mean Texture ({texture}) <= 15.425 -> Tekstur relatif rendah.")
                    if concave_pts <= 0.079:
                        decision_path_explanation.append(f"3. Mean Concave Points ({concave_pts}) <= 0.079 -> Kesimpulan akhir: JINAK.")
                    else:
                        decision_path_explanation.append(f"3. Mean Concave Points ({concave_pts}) > 0.079 -> Kesimpulan akhir: GANAS.")
                else:
                    decision_path_explanation.append(f"2. Mean Texture ({texture}) > 15.425 -> Tekstur tinggi.")
                    if radius <= 14.43:
                        decision_path_explanation.append(f"3. Mean Radius ({radius}) <= 14.43 -> Kesimpulan akhir: GANAS.")
                    else:
                        decision_path_explanation.append(f"3. Mean Radius ({radius}) > 14.43 -> Kesimpulan akhir: GANAS.")

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        'index.html',
        feature_labels=FEATURE_LABELS,
        prediction=prediction,
        probability=probability,
        input_values=input_values,
        decision_path=decision_path_explanation
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
