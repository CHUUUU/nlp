from flask import Flask, request, jsonify

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
    print(content)
    content = content['userRequest']
    content = content['utterance']
    
    # print(content)    


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
                                    "description" : "날씨 연동 성공"
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
                        "simpleText":{
                            "text" : "아직 공부하고있습니다."
                        }
                    }
                ]
            }
        }
    return jsonify(dataSend)

