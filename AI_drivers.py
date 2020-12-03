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

# add scaling to single and ai driver
# documentation
# function docs


import pygame
import sys
import os
from track_class import raceTrack
from car_class import raceCar
import numpy as np
from AI_class_genetic import genetic_population
from time import time as current_time
from multiProcessManagement import multiProcessManagement
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import pylab
import json

cwd = os.getcwd()

def create_default_params() :
    # create json preferences file for AI_drivers.py
    preferences = {
        'track_file': 'track',
        'display_max_resolution': [1750, 1000],
        'car_file': None,
        'genetic_AI_params_to_load': None,
        'track_params_to_load': None,
        'total_num_drivers': 300,
        'max_drivers_on_track': 15,
        'kinematic_updates_per_frame': 5,
        'number_vision_rays': 5,
        'vision_ray_num_history': 0,
        'training_mode': 1,
        'maximum_concurrent_training_threads': 4,
        'maximum_number_cars_per_training_thread': 15,
        'number_cars_display_vs_background': 35,
        }
    
    with open( os.path.join(os.getcwd(), 'AI_drivers.json'), 'w' ) as fil :
        json.dump( preferences, fil, indent=4 )

def train_drivers_in_background(elements, cars, result_q) :
    cars_complete = []
    for (itr, car) in enumerate(cars) :
        while True :
            crashed = car.update_kinematics()
            if crashed :
                cars_complete.append( car )
                break
    result_q.put( [elements, cars_complete] )

def prep_next_run() :
    global total_num_drivers
    global car_pool
    global driver_pool
    global all_drivers_out
    global num_drivers_on_track
    global drivers_to_run
    global next_driver_num
    global drivers_out
    global driver_pool
    global drivers_on_track
    global training_mode
    global training_in_background
    global drivers_train_background
    global threads_training_in_background
    global max_drivers_on_track
    global training_process
    global maximum_concurrent_training_threads
    global current_training_mode
    
    current_training_mode = training_mode
    
    for driver in np.arange( total_num_drivers ) :
        car_pool[driver].go_to_start_position()
        car_pool[driver].AI_driver = driver_pool.population[driver]
    all_drivers_out = False
    num_drivers_on_track = 0
    drivers_to_run = np.arange(total_num_drivers).tolist()
    next_driver_num = 0
    drivers_out = 0
    driver_pool.generation += 1
    drivers_on_track = []
    if training_mode == 0 : # full_display
        training_in_background = False
        del drivers_to_run[0] # otherwise there is a repeat as next_driver_num is already 0
    elif training_mode == 1 : # show a small subset of drivers
        keep_to_display = int( 0.75*total_num_drivers/100 * max_drivers_on_track )
        num_training_threads = int( (total_num_drivers - keep_to_display) / maximum_number_cars_per_training_thread )
        drivers_train_background = np.array_split(drivers_to_run[keep_to_display:], num_training_threads)
        drivers_to_run = drivers_to_run[:keep_to_display]
        training_in_background = True
        threads_training_in_background = num_training_threads
        del drivers_to_run[0] # otherwise there is a repeat as next_driver_num is already 0
        max_processes = maximum_concurrent_training_threads-1
    else : # full_speed
        next_driver_num = None
        all_drivers_out = True
        num_drivers_on_track = 0
        if multi_threading_version == 1 :
            drivers_train_background = np.array_split(drivers_to_run, maximum_concurrent_training_threads)
            threads_training_in_background = maximum_concurrent_training_threads
        else :
            num_training_threads = int( total_num_drivers / maximum_number_cars_per_training_thread )
            drivers_train_background = np.array_split(drivers_to_run, num_training_threads)
            threads_training_in_background = num_training_threads
        drivers_to_run = []
        training_in_background = True
        max_processes = maximum_concurrent_training_threads
    if training_mode == 1 or training_mode == 2 :
        result_q = multiprocessing.Queue()
        
        process_args = []
        
        if multi_threading_version == 1 :
            for elements in drivers_train_background :
                p_args = {'elements':elements, 'cars':car_pool[elements], 'result_q':result_q}
                training_process = multiprocessing.Process(target=train_drivers_in_background, kwargs=(p_args))
                training_process.start()
        elif multi_threading_version == 2 :
            for elements in drivers_train_background :
                process_args.append( [elements, car_pool[elements]] )
            p_args = {'fn':train_drivers_in_background, 'process_args':process_args, 'max_processes':max_processes, 'fn_result_q':result_q}
            training_process = multiprocessing.Process(target=multiProcessManagement, kwargs=(p_args))
            training_process.start()
    
    create_figure(score_history, scores)
    
    if training_mode == 1 or training_mode == 2 :
        return result_q

def create_figure(score_history, scores) :
    global score_history_fig
    global score_history_fig_size
    score_history_fig_size = (200, 200)
    avg_data = score_history[1][-1]
    min_data = np.min(scores)
    max_data = np.max(scores) + 1e-4
    score_rng = max_data-min_data
    step_size = score_rng / 20
    bins = np.arange( min_data, max_data+step_size, step_size )
    
    max_num_history_points = 15
    score_len = len(score_history[0][-max_num_history_points:])
    plot_x = np.arange( score_len ) / ( score_len - 1 ) * score_rng + min_data
    
    fig = pylab.figure(figsize=[2,2], dpi=100)
    
    ax = fig.gca()
    ax.hist( scores, bins=bins, color='g', rwidth=0.85 )
    ax.plot([avg_data, avg_data], ax.get_ylim(), 'm')
    
    ax2 = ax.twinx()
    ax2.plot( plot_x, score_history[0][-max_num_history_points:], 'r.-' )
    ax2.plot( plot_x, score_history[1][-max_num_history_points:], 'k.-' )
    ax2.plot( plot_x, score_history[2][-max_num_history_points:], 'b.-' )
    
    ax.set_yscale('log')
    ax2.set_yscale('log')
    
    ax.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    score_history_fig = pygame.image.fromstring(raw_data, score_history_fig_size, "RGB")
    plt.close('all')

def create_car_pool() :
    global total_num_drivers
    global driver_pool
    global track
    global number_vision_rays
    global vision_ray_num_history
    global FPS
    global kinematic_updates_per_frame
    global car_dimensions
    global scale_factor
    global oversize_cars
    
    car_pool = []
    for i in np.arange(total_num_drivers) :
        driver = driver_pool.add_new_individual_to_population()
        
        car = raceCar( track=track, AI_driver=driver, number_vision_rays=number_vision_rays, vision_ray_num_history=vision_ray_num_history, load_car=car_to_load )
        car.go_to_start_position()
        car.calc_dt(fps=FPS, updates_per_frame=kinematic_updates_per_frame)
        car_pool.append( car )
    
    car_dimensions = np.array( oversize_cars*car_pool[0].car_dimensions/scale_factor, dtype=np.int)
    
    return np.array( car_pool )

def load_saved_params(supress_warnings=False) :
    global track_file
    global display_max_resolution
    global car_file
    global genetic_AI_params_to_load
    global track_params_to_load
    global total_num_drivers
    global max_drivers_on_track
    global FPS
    global kinematic_updates_per_frame
    global number_vision_rays
    global vision_ray_num_history
    global training_mode
    global maximum_concurrent_training_threads
    global maximum_number_cars_per_training_thread
    # global AI_shape
    global IO_shapes
    
    # global AI_inputs
    global AI_outputs
    
    param_names = ['track_file',
                   'display_max_resolution',
                   'car_file',
                   'genetic_AI_params_to_load',
                   'track_params_to_load',
                   'total_num_drivers',
                   'max_drivers_on_track',
                   'FPS',
                   'kinematic_updates_per_frame',
                   'number_vision_rays',
                   'vision_ray_num_history',
                   'training_mode',
                   'maximum_concurrent_training_threads',
                   'maximum_number_cars_per_training_thread',
                   # 'AI_shape',
                   ]
    
    params_file = 'AI_drivers.json'
    with open( os.path.join( cwd, params_file), 'r' ) as f :
        params = json.load(f)
        param_keys = params.keys()
    
    if 'track_file' in param_keys :
        if params['track_file'] is not None :
            track_file = params['track_file']
            if '_image' in track_file or '.' in track_file :
                track_file = track_file.split('.')[0]
                track_file = track_file.split('_image')[0]
    
    if 'display_max_resolution' in param_keys :
        if params['display_max_resolution'] is not None :
            display_max_resolution = params['display_max_resolution']
    
    if 'car_file' in param_keys :
        if params['car_file'] is not None :
            car_file = params['car_file']
    
    if 'genetic_AI_params_to_load' in param_keys :
        if params['genetic_AI_params_to_load'] is not None :
            genetic_AI_params_to_load = params['genetic_AI_params_to_load']
    
    if 'track_params_to_load' in param_keys :
        if params['track_params_to_load'] is not None :
            track_params_to_load = params['track_params_to_load']
    
    if 'total_num_drivers' in param_keys :
        if params['total_num_drivers'] is not None :
            total_num_drivers = params['total_num_drivers']
    
    if 'max_drivers_on_track' in param_keys :
        if params['max_drivers_on_track'] is not None :
            max_drivers_on_track = params['max_drivers_on_track']
    
    if 'FPS' in param_keys :
        if params['FPS'] is not None :
            FPS = params['FPS']
    
    if 'kinematic_updates_per_frame' in param_keys :
        if params['kinematic_updates_per_frame'] is not None :
            kinematic_updates_per_frame = params['kinematic_updates_per_frame']
    
    if 'number_vision_rays' in param_keys :
        if params['number_vision_rays'] is not None :
            number_vision_rays = params['number_vision_rays']
    
    if 'vision_ray_num_history' in param_keys :
        if params['vision_ray_num_history'] is not None :
            vision_ray_num_history = params['vision_ray_num_history']
    
    if 'training_mode' in param_keys :
        if params['training_mode'] is not None :
            training_mode = params['training_mode']
    
    if 'maximum_concurrent_training_threads' in param_keys :
        if params['maximum_concurrent_training_threads'] is not None :
            maximum_concurrent_training_threads = params['maximum_concurrent_training_threads']
    
    if 'maximum_number_cars_per_training_thread' in param_keys :
        if params['maximum_number_cars_per_training_thread'] is not None :
            maximum_number_cars_per_training_thread = params['maximum_number_cars_per_training_thread']
    
    # if 'AI_shape' in param_keys :
    #     if params['AI_shape'] is not None :
    #         AI_shape = []
    #         layers = params['AI_shape']
    #         for layer in layers :
    #             if layer == 'AI_inputs' :
    #                 AI_inputs = number_vision_rays * (vision_ray_num_history+1) + 1
    #                 AI_shape.append( AI_inputs )
    #             elif layer == 'AI_outputs' :
    #                 AI_shape.append( AI_outputs )
    #             else :
    #                 AI_shape.append( layer )
    
    AI_inputs = number_vision_rays * (vision_ray_num_history+1) + 1
    IO_shapes = [AI_inputs, AI_outputs]

def human_time(duration) :
    tm_str = ''
    
    hr = int( duration//3600 )
    if hr >= 1 :
        tm_str += str(hr) + ':'
        duration -= hr*3600
   
    mn = int( duration//60 )
    if mn >= 1 :
        if mn < 10 :
            if hr > 0 :
                tm_str += '0'
            tm_str += str(mn) + ':'
        else :
            tm_str += str(mn) + ':'
        duration -= mn*60
    elif hr > 0 :
        tm_str += '00:'
    
    secs = int( duration//1 )
    if secs >= 1 :
        if secs < 10 :
            if hr > 0 or mn > 0 :
                tm_str += '0'
            tm_str += str(secs) + '.'
        else :
            tm_str += str(secs) + '.'
        duration -= secs
    elif hr > 0 or mn > 0 :
        tm_str += '00.'
    else :
        tm_str += '0.'
    
    millis = int( (duration*1000)//1 )
    if millis == 0 :
        tm_str += '000'
    elif millis < 10 :
        tm_str += '00' + str(millis)
    elif millis <100 :
        tm_str += '0' + str(millis)
    else :
        tm_str += str(millis)
    
    return tm_str

if __name__ == '__main__' :
    # track selection
    track_file = 'track'
    
    genetic_AI_params_to_load = None
    track_params_to_load = None
    
    FPS = 30
    display_max_resolution = None
    
    # driver accounting
    total_num_drivers = 500 # total number of drivers
    max_drivers_on_track = 20 # max drivers to have on the track at one time
    num_drivers_on_track = 0 # current number of drivers on track
    next_driver_num = 0 # element number to select
    
    # training procedures :
        # full display --> 0
        # show small subset --> 1
        # full multithreading --> 2
    training_mode = 1
    current_training_mode = training_mode
    multi_threading_version = 2 # leave this at 2 as 1 is broken
    maximum_concurrent_training_threads = 5
    maximum_number_cars_per_training_thread = 15
    
    # create driver pool
    kinematic_updates_per_frame = 3
    number_vision_rays = 9
    vision_ray_num_history = 0
    # AI_inputs = number_vision_rays * (vision_ray_num_history+1) + 1
    AI_outputs = 2
    # AI_shape = None
    IO_shapes = None
    
    load_saved_params()
    
    # driver accounting
    drivers_to_run = [] # list of all the drivers that have yet to run
    drivers_on_track = [] # element number of the drivers/car
    all_drivers_out = False # indicates when the last driver has left
    drivers_out = 0 # number of drivers that have crashed out
    last_gen_stats = [0., 0., 0.] # score stats for the previous generation
    
    score_history = [ [1e-2, 1e-2], [1e-2, 1e-2], [1e-2, 1e-2] ]
    scores = [ 1e-2 ]
    
    # create track
    track = raceTrack( os.path.join(cwd, track_file) )
    if display_max_resolution is not None :
        wr = track.window_width / display_max_resolution[0] # width ratio
        hr = track.window_height / display_max_resolution[1] # height ratio
        scale_factor = max( wr, hr )
        if scale_factor < 1 :
            scale_factor = 1
    else :
        scale_factor = track.scale_factor
    
    # create the driver pool
    car_file = 'racecar.png'
    car_file = 'blue_f1.png'
    oversize_cars = 2
    driver_pool = genetic_population(IO_shapes=IO_shapes, num_individuals=0, genetic_AI_params_to_load=genetic_AI_params_to_load)
    
    # create car pool
    car_dimensions = []
    car_pool = create_car_pool()
    
    # Create width and height constants
    WINDOW_WIDTH = int( track.window_width / scale_factor )
    WINDOW_HEIGHT = int( track.window_height / scale_factor )
    
    # Initialise all the pygame modules
    pygame.init()
    
    # Create a game window
    game_window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Set title
    pygame.display.set_caption("AI Race Track!")
    
    clock = pygame.time.Clock()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    font_size = 20
    font_clr = (250, 200, 150)
    font = pygame.font.SysFont('Tahoma', font_size, True, False)
    
    game_running = True
    pause_game = False
    lap_completed = False
    best_lap_time = None
    unlimited_FPS = False
    saved_current_generation = False
    
    training_in_background = False
    threads_training_in_background = 0
    
    # frame rate information
    frame_rates = np.zeros(10)
    past_frame_time = current_time()
    
    car_img_locs = ["cars/blue_f1.png", 
                    "cars/blue_racecar.png", 
                    "cars/green_racecar.png", 
                    "cars/purple_racecar.png", 
                    # "cars/racecar.png", 
                    "cars/red_f1.png", 
                    "cars/red_racecar.png", 
                    "cars/tan_f1.png", 
                    "cars/violet_racecar.png", 
                    "cars/yellow_f1.png", 
                    "cars/yellow_racecar.png", 
                    ]
    
    # track_image = pygame.image.load(track.racetrack_file)
    track_image = pygame.image.load(track.track_png)
    track_image = pygame.transform.scale(track_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    car_imgs = []
    for img_loc in car_img_locs :
        car_imgs.append( pygame.image.load( os.path.join(cwd, img_loc) ) )
        car_imgs[-1] = pygame.transform.scale(car_imgs[-1], car_dimensions[::-1])
    
    driver_cars = np.random.choice(len(car_imgs), size=total_num_drivers, replace=True)
    
    
    
    print( '---------------------------------------------------' )
    print( 'Training Controls' )
    print( '---------------------------------------------------' )
    print( 'To select setting press (key)' )
    print( '---------------------------------------------------' )
    print( 'Training mode:' )
    print( '0 (q): Show all cars' )
    print( '1 (w): Show a small representative group' )
    print( '\t\tRun the other cars in the background (multi-threaded)' )
    print( '2 (e): Run all cars in the background (multi-threaded)' )
    print( '\t\tNothing will be shown on screen except text updates' )
    print( '---------------------------------------------------' )
    
    
    
    
    
    result_q = prep_next_run()
    
    
    
    # Game loop
    while game_running:
        if not unlimited_FPS :
            clock.tick(FPS)
        current_frame_time = current_time()
        frame_rates[:-1] = frame_rates[1:]
        frame_rates[-1] = 1/(current_frame_time-past_frame_time)
        past_frame_time = current_frame_time
        avg_frame_rate = np.mean( frame_rates )
        # Loop through all active events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN :
                if event.key == pygame.K_SPACE :
                    pause_game = not pause_game
                elif event.key == pygame.K_q :
                    training_mode = 0
                elif event.key == pygame.K_w :
                    training_mode = 1
                elif event.key == pygame.K_e :
                    training_mode = 2
                elif event.key == pygame.K_u :
                    unlimited_FPS = not unlimited_FPS
                    
            elif event.type == pygame.QUIT: # Close the program if the user presses the 'X'
                game_running = False
        
        if all_drivers_out and num_drivers_on_track == 0 and not training_in_background and not saved_current_generation :
            saved_current_generation = True
            driver_pool.save_best_number()
        
        if not pause_game :
            for driver in drivers_on_track :
                crashed = car_pool[driver].update_kinematics()
                if crashed :
                    del drivers_on_track[ drivers_on_track.index(driver) ]
                    num_drivers_on_track -= 1
                    drivers_out += 1
            if not all_drivers_out and num_drivers_on_track < max_drivers_on_track :
                drivers_on_track.append( next_driver_num )
                num_drivers_on_track += 1
                if len(drivers_to_run) == 0 :
                    all_drivers_out = True
                    next_driver_num = None
                else :
                    next_driver_num = drivers_to_run[0]
                    del drivers_to_run[0]
            
            if saved_current_generation : # initial loop to setup drivers
                saved_current_generation = False
                
                scores = []
                for num in np.arange( total_num_drivers ) :
                    scores.append( driver_pool.population[num].score )
                last_gen_stats = [ np.max(scores), np.mean(scores), np.min(scores) ]
                score_history[0].append( last_gen_stats[0] )
                score_history[1].append( last_gen_stats[1] )
                score_history[2].append( last_gen_stats[2] )
                
                lap_completed = False
                best_lap_time = np.inf
                for car in car_pool :
                    laps_completed = car.laps_completed
                    if laps_completed > 0 :
                        lap_completed = True
                        lap_time = car.raceTime
                        if lap_time < best_lap_time :
                            best_lap_time = lap_time
                
                driver_pool.next_generation()
                result_q = prep_next_run()
        
        if training_in_background :
            try :
                [ elements, cars ] = result_q.get(block=False)
            except Exception as e :
                pass
            else :
                threads_training_in_background -= 1
                drivers_out += len(elements)
                for (key, val) in enumerate(elements) :
                    car_pool[val] = cars[key]
                    driver_pool.population[val] = cars[key].AI_driver
                if threads_training_in_background == 0 :
                    training_in_background = False
        
        # displace race track
        game_window.blit(track_image, (0,0))
        
        # display plot of score history
        game_window.blit(score_history_fig, (0, WINDOW_HEIGHT-score_history_fig_size[1]))
        
        for driver in drivers_on_track :
            car = car_pool[driver]
            car_img = car_imgs[ driver_cars[driver] ]
            car_rotated = pygame.transform.rotate( car_img, (-car.angle-np.pi/2)*360/(2*np.pi) )
            (car_offset_x, car_offset_y) = car_rotated.get_rect().center
            game_window.blit(car_rotated, tuple([int(car.pos_pxl[0]/scale_factor-car_offset_x), 
                                                 int(car.pos_pxl[1]/scale_factor-car_offset_y)]))
        
        if True : # displays important information
            x_pos = 3
            y_pos = 10
            text = font.render( f'fps: {avg_frame_rate:.0f}', True, font_clr); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render( f'Gen: {driver_pool.generation}', True, font_clr); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render( f'drivers left: {len(drivers_to_run)}', True, font_clr) ; game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render( f'on track: {num_drivers_on_track}/{max_drivers_on_track}', True, font_clr); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render( f'drivers done: {drivers_out}/{total_num_drivers}', True, font_clr); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            text = font.render( f'threads remaining: {threads_training_in_background}', True, font_clr); game_window.blit(text, (x_pos, y_pos) ); y_pos += 1.1*font_size
            text = font.render( f'training in background: {training_in_background}', True, font_clr) ; game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            if current_training_mode == training_mode :
                text = font.render( f'training mode: {training_mode}', True, font_clr); game_window.blit(text, (x_pos, y_pos) ); y_pos += 1.1*font_size
            else :
                text = font.render( f'training mode: {current_training_mode} --> {training_mode}', True, font_clr); game_window.blit(text, (x_pos, y_pos) ); y_pos += 1.1*font_size
            text = font.render( f'score rng: {last_gen_stats[0]:.3} / {last_gen_stats[1]:.3} / {last_gen_stats[2]:.3}', True, font_clr); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            if lap_completed :
                text = font.render( f'Fastest time: {human_time(best_lap_time)}', True, font_clr); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
                # text = font.render( f'Fastest time: {best_lap_time:.4}', True, font_clr); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            if pause_game :
                text = font.render( 'Paused', True, font_clr); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
            if unlimited_FPS :
                text = font.render( 'Unlimited FPS', True, font_clr); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
        
        # Update our display
        pygame.display.update()
    
    
    
    # Uninitialize all pygame modules and quit the program
    pygame.quit()
    sys.exit()
