from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np

app = Flask(__name__)
data = None

def classify_water_quality(df):
    ph_good = (7.0, 8.0)
    salinity_good = (15.0, 25.0)
    temperature_good = (27.0, 32.0)
    conditions = (
        (df['pH air'].between(ph_good[0], ph_good[1])) &
        (df['salinitas air'].between(salinity_good[0], salinity_good[1])) &
        (df['suhu air'].between(temperature_good[0], temperature_good[1]))
    )
    df['kualitas'] = np.where(conditions, 'Baik', 'Tidak Baik')
    return df

@app.route('/')
def index():
    return render_template('index.html', data=data)

@app.route('/analyze', methods=['POST'])
def analyze():
    global data
    file = request.files['file']
    df = pd.read_csv(file)
    
    # Mengubah format tanggal dan menangani tanggal yang tidak valid
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    df = df.dropna(subset=['Tanggal'])
    df['Tanggal'] = df['Tanggal'].dt.strftime('%Y-%m-%d')
    
    # Ubah tipe data kolom numerik
    df['pH air'] = pd.to_numeric(df['pH air'], errors='coerce')
    df['salinitas air'] = pd.to_numeric(df['salinitas air'], errors='coerce')
    df['suhu air'] = pd.to_numeric(df['suhu air'], errors='coerce')
    
    # Hapus baris dengan nilai yang tidak valid
    df = df.dropna(subset=['pH air', 'salinitas air', 'suhu air'])
    
    df = classify_water_quality(df)
    
    X = df[['pH air', 'salinitas air', 'suhu air']]
    y = df['kualitas']
    
    model = GaussianNB()
    model.fit(X, y)
    accuracy = model.score(X, y)
    
    # Pastikan urutan kolom yang benar
    df = df[['Tanggal', 'kualitas', 'salinitas air', 'suhu air', 'pH air']]
    
    data = df.to_dict(orient='records')
    return jsonify(data=data, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
