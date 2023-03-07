from flask import Flask
app = Flask(__name__)

@app.route('/')

def hello_world():
       return 'Hello %s!' % name
app.add_url_rule('/', 'hello', hello_world)

if __name__ == '__main__':
   app.debug = True
   app.run()
   app.run(debug = True)
