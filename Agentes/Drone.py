import os
import pprint
from datetime import datetime

import airsim
import numpy as np
from airsim import ImageRequest


class RolDrone:
    Master = 1
    Slave = 0

class State:
    Stop = 0
    Armed = 1
    TakeOff = 2
    Fly = 3
    Landing = 4
    Disarmed = 5

class Drone():
    def iniciar_drone(self,n,L1,L2,GPS):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.setInfo(n=n,L1=L1,L2=L2,GPS=GPS)
        print(self.nombre," Conectado")

    def getGps(self):
        gps_data = self.client.getGpsData(gps_name=self.nombreGPS,vehicle_name=self.nombre)#self, gps_name=self.nombreGPS, vehicle_name=self.nombre)
        s = pprint.pformat(gps_data)
        return s
    def getPosition(self):
        return self.client.simGetGroundTruthKinematics(vehicle_name=self.nombre).position
    def orientacion(self):
        return self.client.getOrientation(vehicle_name=self.nombre)
    def allInfo(self):
        return self.client.getMultirotorState(vehicle_name=self.nombre)
    def takeoff(self):
        landed = self.client.getMultirotorState(vehicle_name=self.nombre).landed_state
        if landed == airsim.LandedState.Landed:
            print(self.nombre," taking off...")
            self.client.takeoffAsync(vehicle_name=self.nombre)
        else:
            print(self.nombre," already flying...")
            self.client.hoverAsync(vehicle_name=self.nombre).join()
    def moveZ(self,Z,velocidad):
        return self.client.moveToZAsync(z=-Z,velocity=velocidad,vehicle_name=self.nombre).join()
    def moveVel(self,vx,vy,vz,t):
        return self.client.moveByVelocityAsync(vx=-vx,vy=-vy,vz=-vz,duration=t,vehicle_name=self.nombre).join()
    def moveTo(self,horizontal,lateral,altura,v):
        return self.client.moveToPositionAsync(horizontal,lateral,-altura,v,vehicle_name=self.nombre).join()
    def moveToAcelerador(self,pitch,roll,throttle,yaw_rate,duration):
        return self.client.moveByAngleThrottleAsync(pitch=pitch, roll=roll, throttle=throttle,yaw_rate=yaw_rate, duration=duration,vehicle_name=self.nombre).join()
    def moveByAngleZ(self,pitch,roll,z,yaw,duration):
        return self.client.moveByAngleZAsync(pitch=pitch, roll=roll, z=z, yaw=yaw, duration=duration, vehicle_name = self.nombre).join()
    def saveImage(self):
        responses = self.client.simGetImages([ImageRequest("0", airsim.ImageType.Scene, False, False)],vehicle_name=self.nombre)
        response = responses[0]
        # get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        print(img1d)
        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        print(img_rgb)
        #img = Image.fromarray(img_rgb, 'RGB')
        #print(img)
        # original image is fliped vertically
        img_rgb = np.flipud(img_rgb)

        # write to png
        dt = datetime.now()
        airsim.write_png(os.path.normpath('./Logs/'+
                                          str(self.nombre)+
                                          str(dt.date())+
                                          str(dt.hour)+
                                          str(dt.minute)+
                                          str(dt.second)+
                                          str(dt.microsecond)+ '.png'), img_rgb)
    def getImage(self):

        png_image = self.client.simGetImage("0", airsim.ImageType.Scene,vehicle_name=self.nombre)

        print(png_image)
    def getLidar1(self):
        lindar = self.client.getLidarData(lidar_name=self.nombreLidar1, vehicle_name=self.nombre)

        dt = datetime.now()
        file = open('./Logs/'+
                    str(self.nombre) +
                    str(self.nombreLidar1) +
                    str(dt.date()) +
                    str(dt.hour) +
                    str(dt.minute) +
                    str(dt.second) +
                    str(dt.microsecond) +'.txt', 'w')
        file.write(str(lindar))
        file.close()
        return lindar
    def getLidar2(self):
        lindar = self.client.getLidarData(lidar_name=self.nombreLidar2, vehicle_name=self.nombre)

        dt = datetime.now()
        file = open('./Logs/'+
                    str(self.nombre) +
                    str(self.nombreLidar2) +
                    str(dt.date()) +
                    str(dt.hour) +
                    str(dt.minute) +
                    str(dt.second) +
                    str(dt.microsecond) + '.txt', 'w')
        file.write(str(lindar))
        file.close()
        return lindar
    def getCollision(self):
        return self.client.getCollisionInfo(vehicle_name=self.nombre)
    def giroDerecha(self,giro):
        return self.client.rotateByYawRateAsync( yaw_rate=giro, duration=(giro*3)/360, vehicle_name = self.nombre).join()
    def setInfo(self, n, L1, L2, GPS):
        self.nombre=n
        self.nombreLidar1=L1
        self.nombreLidar2=L2
        self.nombreGPS=GPS
        self.client.enableApiControl(True,vehicle_name=self.nombre)
        self.client.armDisarm(True,vehicle_name=self.nombre)

    def followMe(self,x,y,z,v):
        myPosition = self.getPosition()
        #distancia = np.linalg.norm(a-b)

        self.moveTo(x,y,z,v)

class Master(Drone):
    def __init__(self,jidSlave,n,L1,L2,GPS):
        self.iniciar_drone(n=n,L1=L1,L2=L2,GPS=GPS)
        print(self.nombre," Ha iniciado como Master")

class Slave(Drone):
    def __init__(self,n,L1,L2,GPS):
        self.iniciar_drone(n=n,L1=L1,L2=L2,GPS=GPS)
        print(self.nombre," Ha iniciado como Slave")
