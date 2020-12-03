#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:23:12 2020

@author: Casual Endvrs
Email: casual.endvrs@gmail.com
GitHub: https://github.com/Casual-Endvrs
Reddit: CasualEndvrs
Twitter: @CasualEndvrs
"""

import numpy as np
import os
import json

# from time import time as current_time

class raceCar():
    def __init__(self, track=None, AI_driver=None, car_dimensions=[0,0], 
        number_vision_rays=5, vision_ray_num_history=0, load_car=None, 
        params_dict=None):
        self.v_0 = 0. # initial velocity
        self.angle = 0. # angle of car
        self.pos = np.array([0., 0.]) # car position --> [x, y] --> [m]
        self.pos_pxl = np.array([0., 0.]) # car position --> [x, y] --> [pixels]
        self.vel = np.array([0., 0.]) # car velocity --> [v_x, v_y] --> [m/s]
        self.vel_mag = self.v_0 # magnitude of the car's velocity --> [m/s]
        self.acc = np.array([0., 0.]) # car acceleration --> [a_x, a_y] --> [m/s^2]
        self.acc_mag = 0. # magnitude of the car's acceleration --> [m/s^2]
        
        self.pos_old = np.array([0., 0.]) # previous position of the car --> [m]
        self.pos_old_pxl = np.array([0., 0.]) # previous position of the car --> [pixels]
        self.odometer = 0. # distance travelled by the car --> [m]
        self.raceTime = 1e-5 # duration of time spent on track --> [s]
        self.laps_completed = -1 # total number of laps completed
        self.check_points_crossed = 0 # total number of check points crossed
        
        self.number_vision_rays = number_vision_rays
        if self.number_vision_rays == 1 :
            angle_factor = 1
        else :
            angle_factor = float( self.number_vision_rays - 1 )
        num_rays_to_one_side = (self.number_vision_rays - 1) / 2
        self.measurement_angles = [ (i-num_rays_to_one_side)*np.pi/angle_factor for i in np.arange(self.number_vision_rays) ]
        
        self.turn_radius = 5.5 # turn radius [m]
        
        self.t_0_60 = 6.5 # 0-60 mph time [s]
        self.max_acc = 60*1.609/3.6 / self.t_0_60 # maximum acceleration [m/s^2]
        self.brake_acc_ratio = 2.6 # ratio of max brake deceleration to full power acceleration
        self.full_brakes = self.brake_acc_ratio*self.max_acc # maximum deceleration rate [m/s^2]
        self.max_corner_g = 0.98 # maximum cornering acceleration [g]
        self.max_corner_ms2 = 9.8*self.max_corner_g # maximum cornering acceleration [m/s^2]
        self.v_max_full_steering_loc = np.sqrt(self.turn_radius * self.max_corner_g)
        
        self.top_speed = 218 / 3.6 # gear limited top speed of car [m/s] # km/h / 3.6 = m/s
        self.max_speed_drag = 1.1*self.top_speed # power/drag limited top speed [m/s] # km/h / 3.6 = m/s
        
        self.throttle = 0 # 0 (coasting) --> 1 (full throttle) [%]
        self.brakes = 0 # 0 (coasting) --> 1 (full brakes) [%]
        self.steering = 0 # -1 (left) --> 0 (centered) --> 1 (right) [%]
        
        self.fps = 30 # fps the game is running at
        self.updates_per_frame = 3 # number of times to update the cars position per game update request
        self.dt = 1 / ( self.fps *self.updates_per_frame ) # dt for each update [s]
        
        self.raceTrack = track
        self.m_per_pxl = 1 # [meters/pixel]
        if track is not None :
            self.m_per_pxl = self.raceTrack.m_per_pxl
            self.angle = self.raceTrack.start_angle
        
        self.car_dimensions = np.array([4.23, 1.32], dtype=np.float32) / self.m_per_pxl # [length, width] in [pxl
        self.center_to_corner = self.car_dimensions / 2 # [length, width] distances from center of car to corners [pxl]
        self.car_corners = np.array([[0,0] for i in np.arange(4)])
        
        if track is not None :
            self.go_to_start_position()
        
        self.two_pi = 2*np.pi # [rad]
        
        self.location_is_pxl = True
        self.max_vision_distance = self.top_speed * 2
        if self.location_is_pxl :
            self.max_vision_distance /= self.m_per_pxl
        self.vision_ray_num_history = vision_ray_num_history
        self.num_input_distances = ( 1 + self.vision_ray_num_history ) * self.number_vision_rays
        self.normalize_distance_factor = 100
        self.normalize_v_mag_factor = 50
        
        self.AI_driver = AI_driver
        self.AI_num_inputs = len( self.measurement_angles ) + 2
        self.AI_num_outputs = 2
        self.score = 0
        self.input_distance_history = False
        if self.vision_ray_num_history != 0 :
            self.input_distance_history = True
            self.steps_between_history_points = int( self.fps / self.vision_ray_num_history )
            self.past_distances = np.ones([ self.steps_between_history_points*(1+self.vision_ray_num_history), self.number_vision_rays ])
            if self.raceTrack is not None :
                current_distances = self.get_distances()
                for i in np.arange( self.steps_between_history_points*(1+self.vision_ray_num_history) ) :
                    self.past_distances[i] = current_distances
        
        self.max_time_between_check_points = 15
        self.time_since_last_check_point = 0 # self.max_time_between_check_points-1.5 # stop cars from driving around behind the start/finish line
        self.crossed_check_point = False
        self.max_number_of_laps = 1 # set to None for unlimited laps
        
        self.ignore_car_corners = True
        
        self.use_check_point_min_time = False # require car to cross a checkpoint every self.max_time_between_check_points seconds
        
        self.load_saved_params(params=params_dict, load_car=load_car)
        
        if False : # used for diagnostics
            print( '\n\n\n' )
            print( 'Car Class - Parameters\n' )
            print( f't_0_60: {self.t_0_60}' )
            print( f'full_brakes: {self.full_brakes}' )
            print( f'max_corner_g: {self.max_corner_g}' )
            print( f'top_speed: {self.top_speed}' )
            print( f'max_speed_drag: {self.max_speed_drag}' )
            print( f'turn_radius: {self.turn_radius}' )
            print( f'car_dimensions: {self.car_dimensions}' )
            print( f'updates_per_frame: {self.updates_per_frame}' )
            print( '\n\n\n' )
    
    def calc_score(self, val=None) :
        avg_speed = self.odometer / self.raceTime
        if val is None :
            self.score = self.odometer # + np.sqrt( avg_speed )
            self.score /= 1e1
            if self.laps_completed > 0 :
                self.score += avg_speed
        else :
            self.score = val
        if self.AI_driver is not None :
            self.AI_driver.score = self.score
    
    def load_saved_params(self, params=None, load_car=None, supress_warnings=False) :
        # 't_0_60': 6.5,          # [s]
        # 'full_brakes': 4.125,   # [m/s^2]
        # 'max_corner_g': 0.98,   # [g]
        # 'top_speed': 218,       # [m/s]
        # 'max_speed_drag': 1.1,  # ratio of max_speed_drag/top_speed if <2, else [m/s]
        # 'turn_radius': 5.5,     # [m]
        # 'car_dimensions': [4.23, 1.32], # [m]
        # 'updates_per_frame': 5, # [-]
        
        if load_car is None :
            load_car = os.path.join( os.getcwd(), 'car_class.json' )
        
        file_params = None
        
        if os.path.isfile( load_car ) :
            with open( load_car, 'r' ) as f :
                file_params = json.load(f)
        
        if params is None :
            if file_params is not None :
                params = file_params
        else :
            if file_params is not None :
                file_params.update(params)
                params = file_params
            
        if params is not None :
            param_keys = params.keys()
            
            if 't_0_60' in param_keys :
                if params['t_0_60'] is not None :
                    self.t_0_60 = params['t_0_60']
                    self.max_acc = 60*1.609/3.6 / self.t_0_60
                    self.brake_acc_ratio = 2.6
                    self.full_brakes = self.brake_acc_ratio*self.max_acc
            
            if 'full_brakes' in param_keys :
                if params['full_brakes'] is not None :
                    self.full_brakes = params['full_brakes']
            
            if 'max_corner_g' in param_keys :
                if params['max_corner_g'] is not None :
                    self.max_corner_g = params['max_corner_g']
                    self.max_corner_ms2 = 9.8*self.max_corner_g
            
            if 'top_speed' in param_keys :
                if params['top_speed'] is not None :
                    self.top_speed = params['top_speed']
                    self.max_speed_drag = 1.1*self.top_speed
            
            if 'max_speed_drag' in param_keys :
                if params['max_speed_drag'] is not None :
                    self.max_speed_drag = params['max_speed_drag']
                    if self.max_speed_drag < 2 :
                        self.max_speed_drag = self.max_speed_drag * self.top_speed
                else :
                    self.max_speed_drag = 1.1 * self.top_speed
            if 'turn_radius' in param_keys :
                if params['turn_radius'] is not None :
                    self.turn_radius = params['turn_radius']
            
            if 'kinematic_updates_per_frame' in param_keys :
                if params['kinematic_updates_per_frame'] is not None :
                    self.updates_per_frame = int( params['kinematic_updates_per_frame'] )
            
            if 'car_dimensions' in param_keys :
                if params['car_dimensions'] is not None :
                    self.car_dimensions = np.array( params['car_dimensions'] )
            
            self.car_dimensions /= self.m_per_pxl
            self.center_to_corner = self.car_dimensions / 2
            if self.car_dimensions[0] < 1 :
                self.car_dimensions[0] = 1
            if self.car_dimensions[1] < 1 :
                self.car_dimensions[1] = 1
            
            self.v_max_full_steering_loc = np.sqrt(self.turn_radius * self.max_corner_g)
            self.calc_dt()
    
    def update_kinematics(self, throttle=None, steering=None, num_updates=None, dt=None):
        """
        This function is used to update the kinematics of the car.

        Parameters
        ----------
        throttle : float, optional
            Throttle position. -1 (full brakes) --> 0 (coasting) --> 1 (full throttle)
            The default is None.
        steering : float, optional
            Steering position. -1 (full left) --> 0 (centered) --> (full right)
            The default is None.
        num_updates : int, optional
            Number of iterations to update the cars kinematics. If left None the class default value will be used. The default is None.
        dt : float, optional
            Duration to up kinematics over each iteration. If left None the class default value will be used. The default is None.

        Returns
        -------
        bool
            True - if the car crashes.
            False - if the car does not crash.

        """
        
        if self.AI_driver is not None :
            AI_inputs = []
            distance_inputs = self.get_distances()
            
            if self.input_distance_history :
                self.past_distances[:-1] = self.past_distances[1:]
                self.past_distances[-1] = distance_inputs
                distance_inputs = np.reshape( self.past_distances[self.steps_between_history_points-1::self.steps_between_history_points], self.num_input_distances )
            AI_inputs.extend( distance_inputs )
            AI_inputs.append( self.vel_mag / self.normalize_v_mag_factor )
            AI_inputs = np.array(AI_inputs)
            
            AI_outputs = self.AI_driver.run_AI( AI_inputs )
            
            [throttle, steering] = AI_outputs
        
        if dt is None :
            dt = self.dt
        if num_updates is None :
            num_updates = self.updates_per_frame
        
        # update car throttle
        if throttle is not None :
            if np.abs(throttle) > 1 :
                throttle = np.sign(throttle)
            if throttle > 0 :
                self.throttle = throttle
                self.brakes = 0
            else :
                self.throttle = 0
                self.brakes = throttle
        
        # update car steering
        if steering is not None :
            self.steering = steering
        
        self.pos_old_pxl = np.copy( self.pos_pxl )
        
        for itr in np.arange(num_updates) :
            self.pos_pxl = self.pos / self.m_per_pxl
            if self.raceTrack is not None :
                if self.test_car_hit_wall() :
                    if not self.ignore_car_corners :
                        self.calc_score()
                        # print( 'first ending' )
                        return True
                [pxl, distance] = self.raceTrack.find_edge_distance(self.pos_pxl, self.angle, location_is_pxl=self.location_is_pxl, max_vision_distance=self.max_vision_distance)
                # [pxl, distance] = self.raceTrack.obsticle_detection(self.pos_pxl, self.angle, location_is_pxl=self.location_is_pxl, max_distance=self.max_vision_distance)
                if distance < 1e-1 : # test if there was a crash
                    self.calc_score()
                    # print( 'hit wall' )
                    return True # there was a crash
            
            # update steering
            if self.steering > 1 :
                self.steering = 1.
            elif self.steering < -1 :
                self.steering = -1.
            angle_before = self.angle
            
            #turn_radius = self.turn_radius
            #if self.vel_mag > self.v_max_full_steering_loc :
            #    turn_radius = self.vel_mag**2 / self.max_corner_ms2
            
            self.angle += self.steering*dt # *self.vel_mag/turn_radius
            
            angle_avg = (angle_before+self.angle) / 2.
            self.angle = self.angle % self.two_pi
            
            cos_dir = np.cos(angle_avg)
            sin_dir = np.sin(angle_avg)
            
            # update acceleration --> net acceleration due to brakes, throttle and wind resistance
            self.acc_mag = self.throttle*self.max_acc - self.brakes*self.full_brakes - self.wind_resistance()
            self.acc[0] = self.acc_mag*cos_dir
            self.acc[1] = self.acc_mag*sin_dir
            
            # update velocity
            vel_before = self.vel
            vel_mag_before = self.vel_mag
            self.vel_mag += self.acc_mag * dt
            if self.vel_mag < 0 : # cars don't have reverse, people don't need it in this universe as they drift into parking spots
                self.vel_mag = 0.
            if self.vel_mag > self.top_speed :
                self.vel_mag = self.top_speed
            self.vel[0] = self.vel_mag*cos_dir
            self.vel[1] = self.vel_mag*sin_dir
            
            # update position
            self.pos[0] += (vel_before[0]+self.vel[0])/2. * dt
            self.pos[1] += (vel_before[1]+self.vel[1])/2. * dt
            
            self.odometer += (vel_mag_before + self.vel_mag) / 2 * dt
        
        self.raceTime += dt * num_updates
        self.time_since_last_check_point += dt * num_updates
        
        # test if the AI has crossed a check_point recently
        if self.AI_driver is not None :
            if not self.crossed_check_point and self.time_since_last_check_point > self.max_time_between_check_points :
                self.crossed_check_point = True
                self.calc_score()
                # print( 'too long since last check point' )
                return True
        
        # test if the car has crossed the start_finish_line
        start_finish_line_crossed_direction = self.raceTrack.crossed_start_finish_line_test(self.pos_old_pxl, self.pos_pxl)
        if start_finish_line_crossed_direction == 1 :
            self.time_since_last_check_point = 0
            self.laps_completed += 1
            if self.max_number_of_laps is not None and self.laps_completed >= self.max_number_of_laps :
                self.calc_score()
                # print( 'completed laps' )
                return True
        elif start_finish_line_crossed_direction == -1 :
            self.calc_score(0)
            # print( 'start/finish line wrong direction' )
            return True
        
        
        # check if the car has crossed a check_point and direction
        check_point_cross_direction = self.raceTrack.crossed_check_point_test(self.pos_old_pxl, self.pos_pxl)
        if check_point_cross_direction == 1 :
            self.time_since_last_check_point = 0
            self.check_points_crossed += 1
            if not self.use_check_point_min_time :
                self.crossed_check_point = True
        elif check_point_cross_direction == -1 :
            self.calc_score()
            # print( 'check points wrong direction' )
            return True
        
        return False # there was no crash
    
    def wind_resistance(self) :
        """
        This function returns the aerodynamic drag of the car.

        Returns
        -------
        aero_drag : float
            Has units of Force, kg * m/s^2 .

        """
        # calculates the aerodynamic drag of the car
        # uses a theoretic max speed instead of proper eq'ns
        aero_drag = self.max_acc * ( self.vel_mag / self.max_speed_drag )**2
        
        return aero_drag
    
    def update_car_corners(self) :
        cos_theta = np.cos(self.angle)
        sin_theta = np.sin(self.angle)
        
        x_offset = self.center_to_corner[0]
        y_offset = self.center_to_corner[1]
        
        rotation_matrix = np.array( [[cos_theta, -sin_theta], [sin_theta, cos_theta]] )
        
        self.car_corners[0] = np.sum( rotation_matrix * np.array([x_offset, y_offset]), axis=1) + self.pos_pxl
        self.car_corners[1] = np.sum( rotation_matrix * np.array([-x_offset, y_offset]), axis=1) + self.pos_pxl
        self.car_corners[2] = np.sum( rotation_matrix * np.array([x_offset, -y_offset]), axis=1) + self.pos_pxl
        self.car_corners[3] = np.sum( rotation_matrix * np.array([-x_offset, -y_offset]), axis=1) + self.pos_pxl
    
    def test_car_hit_wall(self, update_corners=True) :
        if update_corners :
            self.update_car_corners()
        
        if self.raceTrack is not None :
            for corner in self.car_corners :
                if self.raceTrack.test_pxl_is_wall(corner, location_is_pxl=True) :
                    return True
        return False
    
    def calc_dt(self, fps=None, updates_per_frame=None) :
        """
        This function updates the parameters used to define the number of times per frame that the 
        car's kinematics are updated, updates_per_frame, and the duration for each update, dt.

        Parameters
        ----------
        fps : int, optional
            The frames per second that the game runs at. The default is None.
            If left as None the objects default will be used; class default is 30.
        updates_per_frame : int, optional
            Total number of iterations per frame to update the car. The default is None.
            If left as None the objects default will be used; class default is 30.

        Returns
        -------
        None.

        """
        if updates_per_frame is not None :
            self.updates_per_frame = updates_per_frame
        
        if fps is not None :
            self.fps = fps
        
        self.dt = 1 / ( self.fps * self.updates_per_frame )
    
    def go_to_start_position(self) :
        """
        This function resets all of the cars various parameters and moves it to the starting position.

        Returns
        -------
        None.

        """
        self.pos = np.array([0., 0.]) # car position --> [x, y] --> [m]
        self.pos_pxl = np.array([0., 0.]) # car position --> [x, y] --> [pixels]
        self.vel = np.array([0., 0.]) # car velocity --> [v_x, v_y] --> [m/s]
        self.vel_mag = self.v_0 # magnitude of the car's velocity --> [m/s]
        self.acc = np.array([0., 0.]) # car acceleration --> [a_x, a_y] --> [m/s^2]
        self.acc_mag = 0. # magnitude of the car's acceleration --> [m/s^2]
        
        self.pos_old = np.array([0., 0.]) # previous position of the car --> [m]
        self.pos_old_pxl = np.array([0., 0.]) # previous position of the car --> [pixels]
        self.odometer = 0. # distance travelled by the car --> [m]
        self.raceTime = 1e-5 # duration of time spent on track --> [s]
        self.laps_completed = -1 # total number of laps completed
        self.check_points_crossed = 0
        self.time_since_last_check_point = 0
        self.crossed_check_point = False
        
        self.score = 0
        
        if self.raceTrack is not None :
            self.angle = self.raceTrack.start_angle # angle of car
            self.pos = np.copy( self.raceTrack.start_point ) * self.m_per_pxl
            self.pos_pxl = np.copy( self.raceTrack.start_point )
            self.pos_old = np.copy( self.raceTrack.start_point ) * self.m_per_pxl
            self.pos_old_pxl = np.copy( self.raceTrack.start_point )
        
        self.update_car_corners()
    
    def get_distances(self, distances_only=True) :
        """
        This function returns a collection of distances from the car to the track boundary and the corresponding pixel
            along the track boundary. Each sets of pixels and distances is calculated using given angles based on the 
            orientation of the car.

        Returns
        -------
        results : list --> [ [ [pxl_x, pxl_y], distance ], ... ]
            List where each element contains the pixel of intersection and distance from the car to each intersection.

        """
        results = []
        
        for angle in self.measurement_angles :
            [pxl, distance] = self.raceTrack.find_edge_distance(self.pos_pxl, self.angle+angle, location_is_pxl=self.location_is_pxl, max_vision_distance=self.max_vision_distance)
            # [pxl, distance] = self.raceTrack.obsticle_detection(self.pos_pxl, 
            #                         self.angle+angle, 
            #                         location_is_pxl=self.location_is_pxl, 
            #                         max_distance=self.max_vision_distance
            #                         )
            distance *= self.m_per_pxl/self.normalize_distance_factor
            if distances_only :
                results.append( distance )
            else :
                results.append( [pxl, distance] )
        
        return results


def create_default_params():
    # create json preferences file for car_class.py
    preferences = {
        't_0_60': 6.5,          # [s]
        'full_brakes': 4.125,   # [m/s^2]
        'max_corner_g': 0.98,   # [g]
        'top_speed': 218,       # [m/s]
        'max_speed_drag': 1.1,  # ratio of max_speed_drag/top_speed if <2, else [m/s]
        'turn_radius': 5.5,     # [m]
        'car_dimensions': [4.23, 1.32], # [m]
        }
    
    with open( os.path.join(os.getcwd(), 'car_class.json'), 'w' ) as fil :
        json.dump( preferences, fil, indent=4 )




if __name__ == '__main__' :
    create_default_params()



