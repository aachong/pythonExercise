from flask import Flask

app = Flask(__name__)

@app.route('/<string>',methods=['GET','POST'])
def hello_world(string):
    return 'BOS '+string+' . EOS'

if __name__=='__main__' :
  app.run()