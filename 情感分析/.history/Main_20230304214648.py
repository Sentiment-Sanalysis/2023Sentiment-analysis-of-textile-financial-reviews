from flask import Flask
app = Flask(__name__)

@app.route('/')

def hello_world():
       return 'hello worlddaaaadddd'
app.add_url_rule('/', 'hello', hello_world)

def hello_name(user_name):
       return 'Hello %s!' % user_name

if __name__ == '__main__':
   app.run(debug = True)
