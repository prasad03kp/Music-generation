
from predict import generate_midi
import os
from flask import send_file, request
import pretty_midi
import sys
if sys.version_info.major <= 2:
    from io import StringIO
else:
    from io import BytesIO
import time
import json

from flask import Flask
app = Flask(__name__, static_url_path='', static_folder=os.path.abspath('../static'))


@app.route('/predict', methods=['POST'])
def predict():
    now = time.time()
    values = json.loads(request.data)
    if sys.version_info.major <= 2:
        midi_data = pretty_midi.PrettyMIDI(StringIO(''.join(chr(v) for v in values)))
    else:
        midi_data = pretty_midi.PrettyMIDI(BytesIO(b''.join([v.to_bytes(1,'big') for v in values])))
    duration = float(request.args.get('duration'))
    ret_midi = generate_midi(midi_data, duration)
    return send_file(ret_midi, attachment_filename='return.mid', 
        mimetype='audio/midi', as_attachment=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    return send_file('../static/index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
