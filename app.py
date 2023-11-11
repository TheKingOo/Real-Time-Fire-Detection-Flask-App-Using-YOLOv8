from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__ ,static_url_path='/static')
model = YOLO('best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    result = model.predict(source=0, imgsz=640, conf=0.6, show=True)
    return result.imgs[0]

if __name__ == '__main__':
    app.run(debug=True)
