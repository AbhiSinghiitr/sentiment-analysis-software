from googleapiclient.discovery import build
import pandas as pd
import csv
from pytube import extract

api_key = 'AIzaSyAWvcYyhifqwreIpseWyyFGMljWOEbO0lI'



number = 0
comm=[]

def video_comments(url):

  video_id = extract.video_id(url)

  comment_count = 0

  youtube = build('youtube', 'v3', developerKey=api_key)

  video_response = youtube.commentThreads().list(
  part = 'snippet,replies',
  videoId = video_id
  ).execute()

  while video_response:
    for item in video_response['items']:
    
      comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
      arr=[comment]
      comm.append(arr)

      comment_count += 1

    if 'nextPageToken' in video_response:
      video_response = youtube.commentThreads().list(
          part = 'snippet,replies',
          videoId = video_id,
          pageToken = video_response['nextPageToken']
        ).execute()
    else:
        break
  with open('./comment.csv', 'w',encoding="utf-8", newline='') as filee:
    writer = csv.writer(filee)
    writer.writerow(["Comments"])
    writer.writerows(comm)
  filee.close();
