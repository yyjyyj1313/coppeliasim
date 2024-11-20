
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
from KinematicAnalysis import SPSAnalysis
import matplotlib.pyplot as plt

class Coppelia:
    def __init__(self) -> None:
        client = RemoteAPIClient()
        self.sim = client.require('sim')
        self.simIK = client.require('simIK')
        self.ka = SPSAnalysis()
        self.sps_zero_joint = np.zeros(6) 
        self.sps_work_init_joint = np.ones(6)*4 
        self.cr_zero_joint = np.zeros(6) 
        self.cr_work_init_joint = np.ones(6)*4 
        self.sps_config()
        # self.cr_config()

    def cr_config(self) -> None:
        self.cr_base = self.sim.getObject('/CR5')
        self.cr_joints = [self.sim.getObject('./cr_joint'+str(i+1)) for i in range(6)]
    
    def sps_config(self) -> None:
        self.sps_base = self.sim.getObject('/6SPS')
        self.sensor = self.sim.getObject('./ForceSensor')
        self.tips = []
        self.targets = []
        self.motors = [self.sim.getObject('./motor'+str(i+1)) for i in range(6)]
        # for i in range(5):
        #     self.tips.append(self.sim.getObject('./motor'+str(i+2)+'_tip'))
        #     self.targets.append(self.sim.getObject('./motor'+str(i+2)+'_target'))

        self.tip = self.sim.getObject('./tip')
        self.target = self.sim.getObject('./target')
        
        self.pose = self.sim.getObjectPose(self.tip,self.sps_base)
        print(self.quat2mat(self.pose))
        
        self.IK_env = self.simIK.createEnvironment()
        self.IK_group = self.simIK.createGroup(self.IK_env)
        self.simIK.setGroupCalculation(self.IK_env,self.IK_group,self.simIK.method_damped_least_squares,0.0001,3)
        for i in range(len(self.tips)):
            self.ik_element,self.simToIkMap,self.ikToSimMap = self.simIK.addElementFromScene(self.IK_env,self.IK_group,
                                                self.sps_base,self.tips[i],self.targets[i],
                                                self.simIK.constraint_position)
    
        self.sps_tipTask,_,_=self.simIK.addElementFromScene(self.IK_env,self.IK_group,
                                            self.sps_base,self.tip,self.target,
                                            self.simIK.constraint_pose)
        
    def quat2mat(self, arr):
        p_x,p_y,p_z,x,y,z,w = arr
        T = np.matrix([[0, 0, 0, p_x], [0, 0, 0, p_y], [0, 0, 0, p_z], [0, 0, 0, 1]])
        T[0, 0] = 1 - 2 * pow(y, 2) - 2 * pow(z, 2)
        T[0, 1] = 2 * (x * y - w * z)
        T[0, 2] = 2 * (x * z + w * y)

        T[1, 0] = 2 * (x * y + w * z)
        T[1, 1] = 1 - 2 * pow(x, 2) - 2 * pow(z, 2)
        T[1, 2] = 2 * (y * z - w * x)

        T[2, 0] = 2 * (x * z - w * y)
        T[2, 1] = 2 * (y * z + w * x)
        T[2, 2] = 1 - 2 * pow(x, 2) - 2 * pow(y, 2)
        return T

    def mat2quat(self, T):
        r11 = T[0, 0]
        r12 = T[0, 1]
        r13 = T[0, 2]
        r21 = T[1, 0]
        r22 = T[1, 1]
        r23 = T[1, 2]
        r31 = T[2, 0]
        r32 = T[2, 1]
        r33 = T[2, 2]
        w = (1 / 2) * np.sqrt(1 + r11 + r22 + r33)
        x = (r32 - r23) / (4 * w)
        y = (r13 - r31) / (4 * w)
        z = (r21 - r12) / (4 * w)
        return [T[0,3],T[1,3],T[2,3],x, y, z, w]

    
    def setFK(self) -> None: 
        self.simIK.setElementFlags(self.IK_env,self.IK_group,self.sps_tipTask,0)
        for i in range(len(self.motors)):
            self.simIK.setJointMode(self.IK_env,self.simToIkMap[self.motors[i]],self.simIK.jointmode_passive)
            
    def setIK(self) -> None: 
        self.simIK.setElementFlags(self.IK_env,self.IK_group,self.tipTask,1)
        for i in range(len(self.motors)):
            self.simIK.setJointMode(self.IK_env,self.simToIkMap[self.motors[i]],self.simIK.jointmode_ik)

    def set_cr_joints(self, q: np.asarray) -> None: 
        for i in range(len(q)):
            if q[i] != None:
                self.sim.setJointTargetPosition(self.cr_joints[i],q[i])
    
    def set_sps_joints(self, d: np.asarray) -> None: 
        
        for i in range(len(d)):
            if d[i] != None:
                if i == 0:
                    self.sim.setJointTargetPosition(self.motors[i],d[i])
                else:
                    self.sim.setJointTargetPosition(self.motors[i],-d[i])
        self.simIK.handleGroup(self.IK_env,self.IK_group,{'syncWorlds': True})
        
    def simulation(self) -> None:
        self.sim.setStepping(True)
        self.sim.startSimulation()
        pos = []
        # self.setFK()

        for t in np.linspace(0,2*np.pi,300):
            # d = self.ka.backward_kinematic(np.array([np.cos(t)*7,np.sin(t)*7,114,0,0,0]))/1000
            self.set_sps_joints(np.ones(6)*np.sin(t)*0.005)
            # self.sim.setJointTargetPosition(self.motors[0],np.sin(t)*0.005)
        #     # self.set_cr_joints(np.array([np.sin(t),np.sin(t),0,0,0,0])*np.pi)
            pos.append(self.sim.getObjectPosition(self.tip,self.sps_base))
        #     # track = self.sim.addDrawingObject(self.sim.drawing_points,5,0.0001,self.tip,1000)
        #     # self.sim.handleDynamics(0.01)
        #     res,F,T = self.sim.readForceSensor(self.sensor)
        #     if res == 1:
        #         print(F,T)
                
            self.sim.step()
            time.sleep(0.01)
        
        
        self.sim.stopSimulation()
        pos.append(self.sim.getObjectPosition(self.tip,self.sps_base))
        pos = np.array(pos)
        # plt.plot(pos[:,0],pos[:,1])
        # plt.show()
        
yyj = Coppelia()
yyj.simulation()