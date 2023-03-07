from flask import Flask
app = Flask(__name__)

@app.route('/blog/<int:postID>')
def show_blog(postID):
   return 'Blog Number %d' % postID

@app.route('/rev/<float:revNo>')
def revision(revNo):
   return 'Revision Number %f' % revNo

@app.route('/hello/<int:score>')
def hello_name(score):
       return '你的分数是 %d' % score

if __name__ == '__main__':
   app.run()
