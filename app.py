from flask import Flask,render_template,request,flash,url_for,redirect
from get_comments import *
from preprocess import *
import json

import pandas as pd
import re 

app = Flask(__name__)


def is_valid_youtube_link(link):
    pattern = r'^https?://(?:www\.)?youtube\.com/watch\?(?=.*v=\w+)(?:\S+)?$'
    match = re.match(pattern, link)
    return bool(match)


@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def sany():
    link = request.form.get('link')
    type = request.form['option']
    p_comm=12
    n_comm=24
    
    if(type=='yt'):
        checkValid=is_valid_youtube_link(link)
        
        if checkValid==0:
            flash('Entered url was not valid please enter valid link')
            message="Entered link was invalid"
            return render_template('index.html')
        
        video_comments(link)
        process_link()
        
        df = pd.read_csv('./comment.csv')
        
        sentiment_column = df['sentiment']
        
        
        for index,value in sentiment_column.iteritems():
            if value==1:
                p_comm+=1
            else:
                n_comm+=1
                
        
        data={'Comments': ' no of comments','Positive comments':p_comm,'Negative comments':n_comm}
        return render_template('output.html',data=data)
    else:
        pred=p_text(link)
        result =""
        if pred>0.5:
            result="Positive"
        else:
            result="Negative"
            
        return render_template('sen_out.html',sentence=link,result=result)
    


    





if __name__ == "__main__":
	app.run(port=3000)
