from flask import Flask, Response, render_template
from camera_tracking import main 
import asyncio
import config

app = Flask(__name__)

def generate_frames():
    token = config.get_token()
    chat_id = config.get_chat_id()

    output_path = 'child_safety_output.mp4'
    camera_idx = 0

    for frame in main(
        camera_index=camera_idx,
        output_path=output_path,
        bot_token=token,
        chat_id=chat_id
    ):
        yield frame


@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
