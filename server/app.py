from flask import Flask, jsonify, request
import cv2
import numpy as np
import base64

from .model import get_transcript


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img_str = request.json['img']
        img_str = base64.b64decode(img_str)
        nparr = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        transcripts = get_transcript(img)

        ret = {}
        if transcripts is None:
            ret['status'] = 'fail'
        else:
            ret['status'] = 'ok'
            ret['num_lines'] = len(transcripts)
            for i in range(len(transcripts)):
                ret[str(i + 1)] = transcripts[i]

        return jsonify(ret)


if __name__ == '__main__':
    app.run()
