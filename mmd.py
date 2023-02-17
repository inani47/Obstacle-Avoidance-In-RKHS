import numpy as np
import torch
import random
from sklearn import mixture
from sys import getsizeof
import timeit
import math

import matplotlib.pyplot as plt
from IPython import display


class Agent:
    def __init__(self, position, goal, radius, num_samples):
        self.goal = goal
        self.dt = 0.2
        self.radius = radius
        self.position = position
        self.min_linear_velocity = 0.4
        self.max_linear_velocity = 1.5
        self.max_angular_velocity = 0.5
        self.head = math.atan2(goal[1]-position[1],goal[0]-position[0])
        self.linear_velocity = self.min_linear_velocity
        self.angular_velocity = 0
        vel_2d =self.linear_velocity*np.array([np.cos(self.head),np.sin(self.head)])
        self.linear_velocity_control_bounds = np.array([-0.2,0.2])
        self.angular_velocity_control_bounds = np.array([-0.2,0.2])
        self.lin_ctrl_list= []
        self.ang_ctrl_list= []
        self.num_samples = num_samples
        self.gamma= 0.1
        self.dirac_delta_distribution=np.random.randn(self.num_samples,1)*0.00000000001
        self.ones_mat=np.ones((1,self.dirac_delta_distribution.shape[0]))
        self.desired_mat = self.dirac_delta_distribution@self.ones_mat
        self.desired_coeffs=self.ones_mat/self.dirac_delta_distribution.shape[0]
        self.kernel_yy = np.exp(-self.gamma*np.power(self.desired_mat-self.desired_mat.T,2))
        self.mmd_term3 = self.desired_coeffs @ self.kernel_yy @ self.desired_coeffs.T
        

    def sampleControls(self, num_samples = 20):
        lb=[self.linear_velocity_control_bounds[0],self.angular_velocity_control_bounds[0]]
        ub=[self.linear_velocity_control_bounds[1],self.angular_velocity_control_bounds[1]]

        if(self.linear_velocity>(self.max_linear_velocity-ub[0])):
            ub[0]=self.max_linear_velocity-self.linear_velocity
        elif(self.linear_velocity<(self.min_linear_velocity-lb[0])):    
            lb[0]= -(self.linear_velocity-self.min_linear_velocity)
        if(self.angular_velocity>(self.max_angular_velocity-ub[1])):
            ub[1]=self.max_angular_velocity-self.angular_velocity
        elif(self.angular_velocity<(-self.max_angular_velocity-lb[1])):    
            lb[1]= -(self.angular_velocity+self.max_angular_velocity)   
        v_list=np.linspace(lb[0],ub[0],num_samples)
        w_list=[]
        for k in range(len(v_list)):
            w_list.append(np.linspace(lb[1],ub[1],num_samples))
        xv, yv = np.meshgrid(range(num_samples), range(num_samples))
        yv=yv.flatten()
        v_list=v_list[yv]
        v_list=np.append(v_list,0)
        w_list=np.append(w_list,0)
        self.lin_ctrl_list=v_list
        self.ang_ctrl_list=w_list

        print(self.lin_ctrl_list.shape)
        # print(self.ang_ctrl.shape)


    def update(self, controls):
        
        self.linear_velocity = controls[0] + self.linear_velocity
        self.linear_velocity = max(self.min_linear_velocity, self.linear_velocity )
        self.linear_velocity = min(self.max_linear_velocity,self.linear_velocity )
        
        self.angular_velocity = controls[1] + self.angular_velocity
        self.angular_velocity = max(-self.max_angular_velocity, self.angular_velocity)
        self.angular_velocity = min(self.max_angular_velocity, self.angular_velocity)
        
        self.head=self.head + self.angular_velocity*self.dt
        self.vel_2d = self.linear_velocity*np.array([np.cos(self.head),np.sin(self.head)])

        self.position = self.position + self.vel_2d * self.dt

    def goalReachingCost(self):
        desired_velocity=self.max_linear_velocity*(self.goal-self.position).T/np.linalg.norm(self.get_goal()-self.position)
        linear_velocity=self.linear_velocity+self.lin_ctrl_list
        angular_velocity=self.angular_velocity+self.ang_ctrl_list
        head=self.head+angular_velocity*self.dt
        cost=np.linalg.norm((linear_velocity*np.vstack((np.cos(head),np.sin(head)))).T-desired_velocity,axis=1)
        print(cost.shape)
        return cost

    def collsionConeCalc(self, obst  ):
        r_bot = self.position.reshape(1,1,1,2)
        r_obst = obst.get_position_samples().reshape(1,1,obst.get_num_samples(),2)
        r_rel = r_bot - r_obst
        lin_ctrls = self.lin_ctrl_list.reshape(self.lin_ctrl_list.shape[0],1,1,1)
        ang_ctrls = self.ang_ctrl_list.reshape(self.ang_ctrl_list.shape[0],1,1,1)
        # print(( lin_ctrls+ self.linear_velocity).shape)
        v_rob_2d = (lin_ctrls + self.linear_velocity)*np.concatenate((np.cos(self.head+(ang_ctrls + self.angular_velocity)*self.dt),
                                                                               np.sin(self.head+(ang_ctrls+self.angular_velocity)*self.dt)),axis=3)
        
        
        v_obs = obst.get_vel_2d().reshape(1,1,1,2)
        v_rel = v_rob_2d - v_obs
        # print( v_rel.shape)
        # print(r_rel.shape)
        cones=np.square(np.sum(r_rel*v_rel, axis=3))+ np.sum(np.square(v_rel), axis=3)*((self.radius +  obst.get_radius())**2 - np.sum(np.square(r_rel), axis=3))
        

        cones=cones.reshape(self.lin_ctrl_list.shape[0],obst.get_num_samples())
        # print(cones.shape)

        return cones
        
    def distributionMatchingCost(self, obst):

        if(np.linalg.norm(self.position - obst.get_position())) > 8:
            return 0
        collision_cones = self.collsionConeCalc(obst)
        collision_cones[collision_cones<0]=0
        collision_cones = collision_cones[..., np.newaxis]
        print(collision_cones.shape)
        print(self.ones_mat.shape)
        coeffs = self.desired_coeffs # equal weights for iid samples
        print(np.transpose(collision_cones@self.ones_mat, (0,2,1)).shape)
        kernel_xx = np.exp(-self.gamma*np.power(collision_cones@self.ones_mat-np.transpose(collision_cones@self.ones_mat, (0,2,1)),2)) 
        kernel_xy = np.exp(-self.gamma*np.power(collision_cones@self.ones_mat-self.desired_mat.T,2))
        mmd_term1 = coeffs @ kernel_xx @ coeffs.T
        mmd_term2 = coeffs @ kernel_xy @ (self.ones_mat/self.dirac_delta_distribution.shape[0]).T
        mmd = mmd_term1 - 2*mmd_term2 + self.mmd_term3 
        print(mmd.shape)
        return mmd

    def optimize(self, obst):
        # /print(self.distributionMatchingCost(obst).reshapeshape)
        indcs=np.argmin( np.squeeze(self.distributionMatchingCost(obst)) + self.goalReachingCost())
        # print(indcs)
        optimal_control=[self.lin_ctrl_list[indcs],self.ang_ctrl_list[indcs]]
        return optimal_control

    def get_position(self):
        return self.position

    def get_vel_2d(self):
        return self.vel_2d

    def get_goal(self):
        return self.goal

    def get_linear_velocity(self):
        return self.linear_velocity

    def get_angular_velocity(self):
        return self.angular_velocity

    def get_heading(self):
        return self.head


class Obstacle:
    def __init__(self, position, radius, noise_params, num_samples):
        self.dt = 0.2
        self.radius = radius
        self.position = position
        self.vel_2d = np.array([0,-1])
        self.num_samples = num_samples
        self.pos_noise_params = noise_params['position']
        self.vel_noise_params = noise_params['velocity']
        self.pos_noise = self.get_noise_samples(self.pos_noise_params)
        self.position_samples = self.position + self.pos_noise
        self.vel_noise = self.get_noise_samples(self.vel_noise_params)

    def get_noise_samples(self, params):
        samples=self.num_samples
        weights=params['weights']
        means=params['means']
        stds=params['stds']
        if(means.ndim==1):
            return np.vstack((np.random.randn(np.int16(round(samples*weights[0])),1)*stds[0]+means[0], np.random.randn(samples-np.int16(round(samples*weights[0])),1)*stds[1]+means[1]))
        else:
            cols=means.shape[0]
            gauss=[]
            for i in range(cols):
                if(i<(cols-1)):
                    sample=np.int16(round(samples*weights[i]))
                else:
                    sample=samples-np.sum(np.int16(np.round(samples*weights[:-1])))
                gauss.append(np.random.randn(sample,cols)*stds[:,i]+means[:,i])
            return np.vstack((gauss))

    def update(self):
        self.position = self.position + self.vel_2d * self.dt
        self.pos_noise = self.get_noise_samples(self.pos_noise_params)

        
        self.position_samples = self.position + self.pos_noise 



        
    def get_position(self):
        return self.position

    def get_position_samples(self):
        return self.position_samples

    def get_vel_2d(self):
        return self.vel_2d

    def get_num_samples(self):
        return self.num_samples

    def get_radius(self):
        return self.radius
            

        

obs_noise_params1 = {
    'position': {
        'weights': np.array([0.4, 0.6]),
        'means': np.array([[0.3, -0.2],[0.03,-0.04]]),
        'stds': np.array([[0.25, 0.05],[0.02,0.02]])
    },
    'velocity': {
        'weights': np.array([0.5, 0.5]),
        'means': np.array([[-0.0, 0.0],[-0.0,0.0]]),
        'stds': np.array([[0.01, 0.01],[0.01,0.01]])
    }
} 


bot =  Agent(position=np.array([7,0]), goal=np.array([7,17]), radius=0.6, num_samples = 20)

obst = Obstacle(position = np.array([7,15]), radius = 0.5, noise_params = obs_noise_params1, num_samples = 20)

while np.linalg.norm(bot.get_position() - bot.get_goal()) > 1.0:
    bot.sampleControls(num_samples=20)
    
    bot.update(bot.optimize(obst))
    obst.update()
    obs_pos = obst.get_position()
    obs_samples = obst.get_position_samples()

    

    circle1 = plt.Circle((bot.get_position()[0], bot.get_position()[1]), 0.5, color='b', zorder=2)

    obst_circ = plt.Circle(obs_pos, 0.5, color='r',  zorder=2)
    



    goal_plt = plt.Circle((bot.get_goal()[0], bot.get_goal()[1]), 0.1, color='g')



    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    # change default range so that new circles will work
    ax.set_xlim((0, 20))
    ax.set_ylim((0, 20))

    ax.add_patch(circle1)
    ax.add_patch(obst_circ)
    for i in range(20):
        ax.add_artist(plt.Circle(obs_samples[i,:], 0.5, color='#ffa804', zorder=1, alpha=0.2))

    ax.add_patch(goal_plt)
    ax.arrow(bot.get_position()[0],bot.get_position()[1], math.cos(bot.get_heading()),  math.sin(bot.get_heading()), head_width = 0.2, width = 0.05)

    plt.pause(0.3)
    display.clear_output(wait=True)
    display.display(plt.gcf())