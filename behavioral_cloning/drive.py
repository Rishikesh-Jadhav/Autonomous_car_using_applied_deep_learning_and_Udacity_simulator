# download python socket.io to connect
# but first initialize the web application so install flask(used to build web apps)
from flask import Flask #(Flask is a class)

app = Flask(__name__)#'__main__'
#router decorator - we use this to tell flask what url we should use to trigger our function
@app.route('/home')
def greeting():
    return 'WELCOME!'

if __name__ == '__main__':
    app.run(port=3000) #run web app if the condition is satisfied and listen on port 3000(local host)
