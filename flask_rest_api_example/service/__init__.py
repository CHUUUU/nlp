from flask import Flask, request, jsonify
from service.crawling_weather import get_weather_respond

app = Flask(__name__)

@app.route('/')
def index():
    return "hello"

@app.route('/keyboard')
def Keyboard():
    dataSend = {
    }
    return jsonify(dataSend)

@app.route('/weather', methods=['POST'])
def weather():
    content = request.get_json()
    content = content['userRequest']
    content = content['utterance']

    location = '마포구'
    respond_txt = get_weather_respond(location)

    if content == u"날씨":
        dataSend = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "carousel": {
                            "type" : "basicCard",
                            "items": [
                                {
                                    "title" : "",
                                    "description" : respond_txt
                                }
                            ]
                        }
                    }
                ]
            }
        }
    else :
        dataSend = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text" : "아직 공부하고있습니다."
                        }
                    }
                ]
            }
        }
    return jsonify(dataSend)

