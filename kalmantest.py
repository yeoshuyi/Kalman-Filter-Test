import time
import board
import adafruit_icm20x
import numpy as np
 
i2c = board.I2C()
imu =  adafruit_icm20x.ICM20948(i2c) #This is IMU object, remember to enable i2c on RPi

print(imu.magnetic, imu.gyro, imu.acceleration) #Chk sensor working

class CustomKalman:
    def __init__(self):
        self.gyroNoise = np.eye(4) * 0.01 #Uncalibrated Gyroscope Noise, this will be a pain to calibrate lol
        self.sensorNoise = np.eye(3) * 0.05 #Uncalibrated Measurement Noise
        self.estimateError = np.eye(4) * 0.001 #Uncertainty of estimate
        self.estimate = np.array([1.0, 0.0, 0.0, 0.0]) #Initial w,x,y,z in quarternion

    #Just some dead reckoning estimate of the new state
    def predict(self, gyroRead, dt):

        #Quarternions
        qw = self.estimate[0]
        qx = self.estimate[1]
        qy = self.estimate[2]
        qz = self.estimate[3]

        #Gyro Rates
        gx = gyroRead[0]
        gy = gyroRead[1]
        gz = gyroRead[2]

        transitionMatrix = np.array([
            [1.0, -0.5*gx*dt, -0.5*gy*dt, -0.5*gz*dt],
            [0.5*gx*dt, 1.0,   0.5*gz*dt, -0.5*gy*dt],
            [0.5*gy*dt, -0.5*gz*dt, 1.0,   0.5*gx*dt],
            [0.5*gz*dt,  0.5*gy*dt, -0.5*gx*dt, 1.0]
        ]) #Pulled this from some random website

        self.estimate = np.dot(transitionMatrix, self.estimate) #New estimate as dot product of gryo rate and new estimate
        abs = np.sqrt(np.sum(self.estimate * self.estimate)) #Abs scalar
        self.estimate = self.estimate / abs #Normalised (because EKF will break if not)
        
        self.estimateError = np.dot(np.dot(transitionMatrix, self.estimateError), transitionMatrix.transpose()) + self.gyroNoise #Wow this might get laggy!

        return self.estimate

    #Okay, now we incorporate the accelerometer and magnetometer, this function can be reused for both
    #So reference for accelerometer should be Down[0,0,1] and magnetometer should be North[1,0,0]
    def update(self, sensorMeasurement, reference):

        #Quarternions
        qw = self.estimate[0]
        qx = self.estimate[1]
        qy = self.estimate[2]
        qz = self.estimate[3]

        #Reference
        rx = reference[0]
        ry = reference[1]
        rz = reference[2]

        #Cool Jacobian thingy I found online for Euler > Quarternion

        jacob = np.array([
            [qw*rx + qy*rz - qz*ry,  qx*rx + qz*rz + qy*ry, -qy*rx + qw*rz + qx*ry, -qz*rx - qx*rz + qw*ry],
            [qz*rx + qw*ry - qx*rz,  qy*rx - qw*rz - qz*ry,  qx*rx + qw*rz + qy*ry,  qw*rx - qy*rz + qx*ry],
            [-qy*rx + qx*ry + qw*rz, qz*rx - qx*rz + qw*ry, -qw*rx - qz*rz + qx*ry,  qx*rx + qy*ry + qw*rz]
        ]) * 2.0
        
        #To rotate quarternions
        rotation = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx**2+qy**2)]
        ])

        #Convert reference to quarternion
        predicted = np.dot(rotation.transpose(), np.array([rx, ry, rz]))

        #Difference between actual accelero/magnetometer reading, and prediction based on gyro
        difference = sensorMeasurement - predicted

        #Total uncertainty based on gyro error and accelero/magnetometer error
        uncertainty = np.linalg.inv(np.dot(np.dot(jacob, self.estimateError), jacob.transpose()) + self.sensorNoise)

        #Kalman gain
        k = np.dot(np.dot(self.estimateError, jacob.transpose()), uncertainty)
                   
        #New estimate and error
        self.estimate = self.estimate + np.dot(k, difference)
        self.estimateError = np.dot((np.eye(4) - np.dot(k, jacob)), self.estimateError)
    
        return self.estimate

#Code starts here

imuObj = CustomKalman() #Initialise imu class
prevTime = time.time() #For dt calculation

def normalise(v):
    norm = np.linalg.norm(v)
    v = v if norm == 0 else v/norm
    return v 

while(True):

    #Calculate elapsed time
    curTime = time.time()
    dt = curTime - prevTime
    prevTime = time.time()

    normMagnetic = np.linalg.norm(imu.magnetic)

    imuObj.predict(imu.gyro, dt)
    imuObj.update(normalise(imu.magnetic), [1.0,0.0,0.0])
    imuObj.update(normalise(imu.acceleration), [0.0,0.0,1.0])

    print(imuObj.estimate)

    time.sleep(0.01) #Let the IMU relax...
