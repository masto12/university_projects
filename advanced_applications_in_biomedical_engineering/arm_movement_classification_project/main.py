#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Martti and Mariana

Script for making arm movement class predictions based on received accelerometer and gyroscope data

"""

import time
import json
import socket
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template
from flask_socketio import SocketIO
from threading import Thread, Event

app = Flask(__name__)
socketio = SocketIO(app)

thread = Thread()
thread_stop_event = Event()

def event_stream():
    
    def getMean(data):
        # Input: np.array
        # Returns mean value for given data
        return np.mean(data)
    
    def getPeaktoPeak(data):
        # Input: np.array
        # Return peak-to-peak value for given data
        return (max(data) - min(data))
    
    def diffStartEnd(data):
        # Input: np.array
        # Return difference between first maxima/minima and second maxima/minima
        # for given data
        
        # Find indices of max and min
        idx_max = list(data).index(max(data))
        idx_min = list(data).index(min(data))
        
        if idx_max < idx_min:
            return (max(data) -  min(data))
        elif idx_max > idx_min:
            return (min(data) -  max(data))
        else:
            # In case of min and max being equal
            return (max(data) -  min(data))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        host = "192.168.1.30" # WI-FI connection IP
        port = 4242
        
        # Emit server status data 
        socketio.emit('serverResponse', {'result': 'Connecting to server...', 'time': 'n/a'})
        s.connect((host, port))
        socketio.emit('serverResponse', {'result': 'Connected!', 'time': 'n/a'})

        # Create variable for accelerometer and gyroscope data
        # Number of samples wanted in variable samples
        samples = 100

        acc_res = []
        gyro_res = []
        
        # Initialize time
        init_t = time.time()
        
        # Record data for wanted amount of samples
        for n in range (0, samples):    
            data = s.recv(256)

            if data:
                decoded_data = data.decode("utf-8").split("\n")
                for msg in decoded_data:
                    try:
                    
                        package = json.loads(msg)

                        acc=package["accelerometer"]["value"]
                        gyro=package["gyroscope"]["value"]

                        # Add data to lists
                        acc_res.append(acc)
                        gyro_res.append(gyro)
                        
                        # Update time
                        t = time.time() - init_t

                        # Emit data for visualization
                        socketio.emit('serverResponse', {'result': 'Analysing movement...', 'time': int(t)})
                    
                    except:
                        continue

        features = {}
        # Transform data to numpy arrays
        acc_res = np.array(acc_res)
        gyro_res = np.array(gyro_res)

        # Loop through accelerometer and gyroscope data separately
        for j in range(0,2):
            if j == 0:
                results = acc_res
                mov = 'Acc'
            else:
                results = gyro_res
                mov = 'Gyro'
                
            # Mean
            features[mov + " mean X"] = getMean(results[:,0])
            features[mov + " mean Y"] = getMean(results[:,1])
            features[mov + " mean Z"] = getMean(results[:,2])
                
            # Peak-to-peak value
            features[mov + " peak-to-peak X"] = getPeaktoPeak( results[:,0] )
            features[mov + " peak-to-peak Y"] = getPeaktoPeak( results[:,1] )
            features[mov + " peak-to-peak Z"] = getPeaktoPeak( results[:,2] )
                
            # Difference between first maxima/minima and second maxima/minima
            features[mov + " diff first and last min-max X"] = diffStartEnd( results[:,0] )
            features[mov + " diff first and last min-max Y"] = diffStartEnd( results[:,1] )
            features[mov + " diff first and last min-max Z"] = diffStartEnd( results[:,2] )
            
        # Add feature valeus to a single variable
        values = list(features.values())

        # Load model
        filename = 'model.sav' # Add filename or path here to ML model
        cl = pickle.load(open(filename, 'rb'))
            
        # Predict and send result
        Y_pred = cl.predict([values])
        socketio.emit('serverResponse', {'result': Y_pred[0], 'time': int(t)})

@app.route('/')
def sessions():
    return render_template('index.html')

@socketio.on('sendData')
def handle_my_custom_event(json):
    global thread
    print('Client connected')
    
    # Start the thread only if the thread has not been started before.
    if not thread.is_alive():
        print("Starting Thread")
        thread = socketio.start_background_task(event_stream)

if __name__ == '__main__':
    print("Starting program")
    socketio.run(app, debug=True)
