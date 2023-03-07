from flask import Flask, make_response, request # 注意需导入 make_response

app = Flask(__name__)

@app.route("/set_cookies")
def set_cookie():
    resp = make_response("success")
    resp.set_cookie("w3cshool", "w3cshool",max_age=3600<span style="background-color: initial; font-size: inherit;">)</span>