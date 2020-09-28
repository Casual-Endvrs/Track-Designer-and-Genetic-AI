#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 11:50:01 2020

@author: Casual Endvrs
Email: casual.endvrs@gmail.com
GitHub: https://github.com/Casual-Endvrs
Reddit: CasualEndvrs
Twitter: @CasualEndvrs
"""

import pygame
import sys
import os
from track_class import raceTrack
from car_class import raceCar
import numpy as np
from time import time as current_time
import json

def create_default_params() :
    # create json preferences file for AI_drivers.py
    preferences = {
        'use_AI_drivers_params': None,
        'track_file': 'track',
        'display_max_resolution': [1750, 1000],
        'car_file': None,
        'FPS': 30,
        'number_vision_rays': 5,
        }
    
    with open( os.path.join(os.getcwd(), 'single_player.json'), 'w' ) as fil :
        json.dump( preferences, fil, indent=4 )

def load_saved_params() :
    global track_file
    global display_max_resolution
    global car_file
    global FPS
    global number_vision_rays
    
    params_file = 'single_player.json'
    with open( os.path.join( cwd, params_file), 'r' ) as f :
        params = json.load(f)
        param_keys = params.keys()
    
    if 'use_AI_drivers_params' not in param_keys or params['use_AI_drivers_params'] is None :
        if 'track_file' in param_keys :
            if params['track_file'] is not None :
                track_file = params['track_file']
                
        if 'display_max_resolution' in param_keys :
            if params['display_max_resolution'] is not None :
                display_max_resolution = params['display_max_resolution']
                
        if 'car_file' in param_keys :
            if params['car_file'] is not None :
                car_file = params['car_file']
                
        if 'FPS' in param_keys :
            if params['FPS'] is not None :
                FPS = params['FPS']
                
        if 'number_vision_rays' in param_keys :
            if params['number_vision_rays'] is not None :
                number_vision_rays = params['number_vision_rays']
    else : # use 'use_AI_drivers_params' as a source file instead
        
        params_file = params['use_AI_drivers_params'] + '.json'
        with open( os.path.join( cwd, params_file), 'r' ) as f :
            params = json.load(f)
            param_keys = params.keys()
            
        if 'track_file' in param_keys :
            if params['track_file'] is not None :
                track_file = params['track_file']
                
        if 'display_max_resolution' in param_keys :
            if params['display_max_resolution'] is not None :
                display_max_resolution = params['display_max_resolution']
                
        if 'car_file' in param_keys :
            if params['car_file'] is not None :
                car_file = params['car_file']
                
        if 'FPS' in param_keys :
            if params['FPS'] is not None :
                FPS = params['FPS']
                
        if 'number_vision_rays' in param_keys :
            if params['number_vision_rays'] is not None :
                number_vision_rays = params['number_vision_rays']

if __name__ == '__main__' :
    cwd = os.getcwd()
    track_file = 'track'
    
    FPS = 30
    display_max_resolution = None
    
    load_saved_params()
    
    track = raceTrack( os.path.join(cwd, track_file) )
    if display_max_resolution is not None :
        wr = track.window_width / display_max_resolution[0] # width ratio
        hr = track.window_height / display_max_resolution[1] # height ratio
        scale_factor = max( wr, hr )
        if scale_factor < 1 :
            scale_factor = 1
    else :
        scale_factor = track.scale_factor
    
    car_file = 'cars/yellow_racecar.png'
    car_width_m = 2
    car_length_m = 5.7
    oversize_cars = (scale_factor-1)/2+1
    car = raceCar( track, number_vision_rays=number_vision_rays )
    car_dimensions = np.array( car.car_dimensions, dtype=np.int)
    car.ignore_car_corners = True
    car.max_number_of_laps = None
    car.go_to_start_position()
    car.calc_dt(fps=FPS, updates_per_frame=1)
    
    car_img = pygame.image.load( os.path.join(cwd, car_file) )
    car_img = pygame.transform.scale(car_img, car_dimensions[::-1])
    
    # Create width and height constants
    WINDOW_WIDTH = int( track.window_width / scale_factor )
    WINDOW_HEIGHT = int( track.window_height / scale_factor )
    
    # Initialise all the pygame modules
    pygame.init()
    
    # Create a game window
    game_window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    
    track_image = pygame.image.load(track.track_png)
    track_image = pygame.transform.scale(track_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Set title
    pygame.display.set_caption("Race Track!")
    
    game_running = True
    
    clock = pygame.time.Clock()
    
    font_size = 25
    txt_color = (255, 255, 255)
    font = pygame.font.SysFont('Tahoma', font_size, True, False)
    
    # frame rate information
    frame_rates = np.zeros(10)
    past_frame_time = current_time()
    
    # Game loop
    while game_running:
        clock.tick(FPS)
        current_frame_time = current_time()
        frame_rates[:-1] = frame_rates[1:]
        frame_rates[-1] = 1/(current_frame_time-past_frame_time)
        past_frame_time = current_frame_time
        avg_frame_rate = np.mean( frame_rates )
        
        # Loop through all active events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN :
                if event.key == pygame.K_w :
                    car.throttle = 1
                elif event.key == pygame.K_s :
                    car.brakes = 1
                elif event.key == pygame.K_a :
                    car.steering = -1
                elif event.key == pygame.K_d :
                    car.steering = 1
                elif event.key == pygame.K_x :
                    car.vel = np.array([0., 0.])
            elif event.type == pygame.KEYUP :
                if event.key == pygame.K_w :
                    car.throttle = 0
                elif event.key == pygame.K_s :
                    car.brakes = 0
                elif event.key == pygame.K_a :
                    car.steering = 0
                elif event.key == pygame.K_d :
                    car.steering = 0
            elif event.type == pygame.QUIT: # Close the program if the user presses the 'X'
                game_running = False
        
        crashed = car.update_kinematics()
        if crashed :
            car.go_to_start_position()
        
        # Content here
        game_window.blit(track_image, (0,0))
        
        if True : # displays numberic properties of the car
            x_pos = 3
            y_pos = 25
            text = font.render(f'fps: {avg_frame_rate:.0f}', True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render('Lap: %i' %car.laps_completed, True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render('CPs: %i/%i' %(car.check_points_crossed, track.num_check_points), True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render('Position: [%.1f, %.1f]' %(car.pos_pxl[0], car.pos_pxl[1]), True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render('Vel_mag: %.2f' %(3.6*car.vel_mag), True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render('Vel: [%.2f, %.2f]' %(3.6*car.vel[0], 3.6*car.vel[1]), True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render('Acc_mag: %.2f' %car.acc_mag, True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render('Acc: [%.2f, %.2f]' %(car.acc[0], car.acc[1]), True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render('Throttle: %s' %car.throttle, True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render('Brakes: %s' %car.brakes, True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render('Drag: %.3e' %car.wind_resistance(), True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render('Net Acc: %.3e' %car.acc_mag, True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render('Angle: %.1f*' %(car.angle/(2*np.pi)*360), True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render('Angle: %.1f*' %(car.angle), True, txt_color); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
        
        for arr in car.get_distances(distances_only=False) : # show "vision rays" for the car
            if np.isnan(arr[1]) :
                continue
            pygame.draw.line(game_window, (0, 255, 0), 
                             tuple([int(car.pos_pxl[0]/scale_factor), int(car.pos_pxl[1]/scale_factor)]), 
                             tuple([int(arr[0][0]/scale_factor), int(arr[0][1]/scale_factor)]), 2)
        
        car_rotated = pygame.transform.rotate( car_img, (-car.angle-np.pi/2)*360/(2*np.pi) )
        (car_offset_x, car_offset_y) = car_rotated.get_rect().center
        game_window.blit(car_rotated, tuple([int(car.pos_pxl[0]/scale_factor-car_offset_x), 
                                             int(car.pos_pxl[1]/scale_factor-car_offset_y)]))
        
        for corner in car.car_corners : # show the corners of the car
            pnt = [int(corner[0]/scale_factor), int(corner[1]/scale_factor)]
            pygame.draw.circle(game_window, (0, 255, 0), pnt, 1)
        
        
        # Update our display
        pygame.display.update()
    
    
    # Uninitialize all pygame modules and quit the program
    pygame.quit()
    sys.exit()


