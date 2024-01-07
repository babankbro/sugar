from flask import Flask, request
from flask import render_template, Response
from flask_cors import CORS, cross_origin
import numpy as np
import json
import googlemaps

KEYS = "AIzaSyB4IFpFMh3jz9kqBIlIbUl4USXOXnIm-Mk"
gmaps = googlemaps.Client(key=KEYS)  

def get_distance_duration(lat1, lng1, lat2, lng2):
    origin_latitude = lat1
    origin_longitude = lng1
    destination_latitude = lat2
    destination_longitude = lng2
    distance = gmaps.distance_matrix([str(origin_latitude) + " " + str(origin_longitude)], 
                                    [str(destination_latitude) + " " + str(destination_longitude)], 
                                    mode='driving')['rows'][0]['elements'][0]

    return distance["distance"]["value"]/1000, distance["duration"]["value"]

def get_distance_duration_matrix(lats, lngs):
    DM = []
    DT = []
    for i in range(len(lats)):
        distances = []
        durations = []
        lat = lats[i]
        lng = lngs[i]
        for j in range(len(lngs)):
            latj = lats[j]
            lngj = lngs[j]
            print(i, j)
            if i == j:
                distances.append(0)
                durations.append(0)
            else:
                dis, du = get_distance_duration(lat, lng, latj, lngj)
                distances.append(dis)
                durations.append(du)
        
        DM.append(distances)
        DT.append(durations)
        
    return DM, DT


app = Flask(__name__, template_folder='./')
#socketio = SocketIO(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def home():
    #return "<br>My First app</br>"
    return render_template("index.html")



@app.route("/get_distance", methods = ["POST"])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def get_distance():
    try:
        print("DISTANCE 1")
        datas = request.get_json( )
        print("datas", datas)
        dataset = datas["dataset"]
        userid =  datas["userid"]
        print("dataset", dataset, userid)
        LAT = datas['LAT']
        LNG = datas['LNG']
    except: 
        print("DISTANCE 2")
        dataset = request.form['dataset']
        userid = request.form['userid']
        LAT = request.form['LAT']
        LNG = request.form['LNG']
    LAT = LAT[1:-1].split(",")
    LAT = [float(x) for x in LAT]
    LNG = LNG[1:-1].split(",")
    LNG = [float(x) for x in LNG]

    #LAT = [16.436590326601667, 16.436590326601667, 16.428069700492035]
    #LNG 

    

    #distance, duration = get_distance_duration(lat1, lng1, lat2, lng2)
    distance_matrix, duration_matrix = get_distance_duration_matrix(LAT, LNG)
    print(distance_matrix)
    print(duration_matrix)

    result = {"dataset":dataset, "userid":userid,
           "distance_matrix": json.dumps(distance_matrix), 
          "duration_matrix": json.dumps(duration_matrix)}

    return result

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port = 5012)