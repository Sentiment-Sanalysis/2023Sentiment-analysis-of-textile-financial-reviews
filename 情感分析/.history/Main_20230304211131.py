from flask import Flask,request # 下面要从request里面获取数据，所以要导入

app = Flask(__name__)


@app.route('/')
def hello_world():
    data = request.get_data() # 直接调用函数，并打印
    print(data)
    return 'Hello World!' # 这里就不改了，待会的效果直接去控制台看


if __name__ == '__main__':
    app.run(debug=True, port=80)
