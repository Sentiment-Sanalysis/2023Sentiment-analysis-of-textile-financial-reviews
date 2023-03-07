from flask import Flask, redirect, url_for
app = Flask(__<strong>name__</strong>)
<p>@app.route('/admin')
def hello_admin():
   return 'Hello Admin'</p>
<p>@app.route('/guest/<guest>')</p><p>def hello_guest(guest):
   return 'Hello %s as Guest' % guest</p>
<p>@app.route('/user/<name>')
def hello_user(name):
   if name =='admin':
      return redirect(url_for('hello_admin'))
   else:
      return redirect(url_for('hello_guest', guest = name))</p>
<p>if __<strong>name__</strong> == '__<strong>main__</strong>':
   app.run(debug = True)
</p>