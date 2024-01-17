from ultralytics import YOLO
from flask import Flask, request
from flask_cors import cross_origin
import json

app = Flask(__name__)

@app.route('/api', methods=['POST'])
@cross_origin()
def api():
    # 在这里处理POST请求的逻辑

    # 获取POST请求的数据
    data = request.values.get("picPath")
    print(data)

    # Load a model
    model = YOLO('yolov8n-seg.pt')  # load an official model
    model = YOLO('/Users/lixizi/Documents/project/project_file/ultralytics-main-new/predict-lyz/train-small13-600-1089/weights/best.pt', task="detect")  # load a custom model

    # Predict with the model
    results = model(data)  # predict on an image
    # model.predict("/Users/lixizi/Documents/project/project_file/ultralytics-main-new/predict-lyz/image/insulator-defect.jpg", save=True)
    result = results[0]
    print(result)
    resultlist = []
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)


        resultlist.append({"label": class_id, "x1": cords[0], "y1": cords[1], "x2": cords[2], "y2": cords[3], "prob": conf})
        print("成功")
    print(resultlist)

    jsonstr = json.dumps(resultlist)
    print(jsonstr)

    return jsonstr

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)
