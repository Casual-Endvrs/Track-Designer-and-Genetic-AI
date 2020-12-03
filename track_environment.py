#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:33:23 2020

@author: Casual Endvrs
Email: casual.endvrs@gmail.com
GitHub: https://github.com/Casual-Endvrs
Reddit: CasualEndvrs
Twitter: @CasualEndvrs
"""

import pygame
import numpy as np
import os
from multiProcessManagement import multiProcessManagement
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import pylab
import json

from time import time as current_time

from track_class import raceTrack
from car_class import raceCar
from trackDesigner_class import trackDesigner_class
from AI_class_genetic import genetic_population

class track_environment() :
    def __init__(self, game_control_params=None,
        params_dict=None,
        genetic_AI_params_to_load=None,
        track_params_to_load=None,
        car_file=None,
        track_file_loc=None,
        AI_shape=None,
        FPS=30,
        load_previous_gen=False,
        ) :
        
        # track selection
        self.params_dict = params_dict
        
        self.genetic_AI_params_to_load = genetic_AI_params_to_load
        self.track_params_to_load = track_params_to_load
        self.car_file = car_file
        
        self.FPS = FPS
        self.display_max_resolution = None
        
        # driver accountingcheck points
        self.total_num_drivers = 500 # total number of drivers
        self.max_drivers_on_track = 20 # max drivers to have on the track at one time
        self.num_drivers_on_track = 0 # current number of drivers on track
        self.next_driver_num = 0 # element number to select
        
        # training procedures :
            # full display --> 0
            # show small subset --> 1
            # full multithreading --> 2
        self.training_mode = 0
        self.current_training_mode = self.training_mode
        self.multi_threading_version = 2 # leave this at 2 as 1 is broken
        self.maximum_concurrent_training_threads = 5
        self.maximum_number_cars_per_training_thread = 15
        
        # create driver pool
        self.kinematic_updates_per_frame = 3
        self.number_vision_rays = 9
        self.vision_ray_num_history = 0
        # AI_inputs = number_vision_rays * (vision_ray_num_history+1) + 1
        self.AI_outputs = 2
        # AI_shape = None
        self.IO_shapes = AI_shape
        
        self.load_saved_params(game_control_params)
        
        # driver accounting
        self.drivers_to_run = [] # list of all the drivers that have yet to run
        self.drivers_on_track = [] # element number of the drivers/car
        self.all_drivers_out = False # indicates when the last driver has left
        self.drivers_out = 0 # number of drivers that have crashed out
        self.last_gen_stats = [0., 0., 0.] # score stats for the previous generation
        
        self.score_history = [ [1e-2, 1e-2], [1e-2, 1e-2], [1e-2, 1e-2] ]
        self.scores = [ 1e-2 ]
        
        # create track
        #self.track = raceTrack( os.path.join(self.track_folder, self.track_file) )
        self.track = raceTrack( self.track_file )
        if self.display_max_resolution is not None :
            wr = self.track.window_width / self.display_max_resolution[0] # width ratio
            hr = self.track.window_height / self.display_max_resolution[1] # height ratio
            self.scale_factor = max( wr, hr )
            if self.scale_factor < 1 :
                self.scale_factor = 1
        else :
            self.scale_factor = self.track.scale_factor
        
        self.track_file = 'track'
        if track_file_loc is not None :
            self.track_file = track_file_loc
        self.track_folder = self.track.working_folder # os.getcwd()
        
        if False : # used for diagnostics
            print( '\n\n\n' )
            print( 'Track Environment - Parameters\n' )
            print( f'track_file: {self.track_file}' )
            print( f'display_max_resolution: {self.display_max_resolution}' )
            print( f'car_file: {self.car_file}' )
            print( f'genetic_AI_params_to_load: {self.genetic_AI_params_to_load}' )
            print( f'track_params_to_load: {self.track_params_to_load}' )
            print( f'total_num_drivers: {self.total_num_drivers}' )
            print( f'max_drivers_on_track: {self.max_drivers_on_track}' )
            print( f'FPS: {self.FPS}' )
            print( f'kinematic_updates_per_frame: {self.kinematic_updates_per_frame}' )
            print( f'number_vision_rays: {self.number_vision_rays}' )
            print( f'vision_ray_num_history: {self.vision_ray_num_history}' )
            print( f'training_mode: {self.training_mode}' )
            print( f'maximum_concurrent_training_threads: {self.maximum_concurrent_training_threads}' )
            print( f'maximum_number_cars_per_training_thread: {self.maximum_number_cars_per_training_thread}' )
            print( '\n\n\n\n' )
        
        # create the driver pool
        self.oversize_cars = 2
        self.driver_pool = genetic_population(IO_shapes=self.IO_shapes, 
                    num_individuals=0, AI_params_dict=self.params_dict, 
                    genetic_AI_params_to_load=self.genetic_AI_params_to_load, 
                    load_previous_gen=load_previous_gen, )
        
        # create car pool
        self.car_dimensions = []
        self.car_pool = []
        self.create_car_pool()
        
        # Create width and height constants
        self.WINDOW_WIDTH = int( self.track.window_width / self.scale_factor )
        self.WINDOW_HEIGHT = int( self.track.window_height / self.scale_factor )
        
        # Initialise all the pygame modules
        pygame.init()
        
        # Create a game window
        #self.game_window = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.game_window = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        
        # Set title
        # pygame.display.set_caption("AI Race Track!")
        
        self.font_size = 20
        self.font_clr = (250, 200, 150)
        self.font = pygame.font.SysFont('Tahoma', self.font_size, True, False)
        
        self.game_running = True
        self.pause_game = False
        self.pause_after_gen = False
        self.lap_completed = False
        self.best_lap_time = None
        self.unlimited_FPS = False
        self.saved_current_generation = False
        
        self.training_in_background = False
        self.threads_training_in_background = 0
        
        self.result_q = self.prep_next_run()
        
        # frame rate information
        self.frame_rates = np.zeros(10)
        self.past_frame_time = current_time()
        
        self.car_img_locs = ["cars/blue_f1.png", 
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
        
        self.track_image = pygame.image.load(self.track.track_png)
        self.track_image = pygame.transform.scale(self.track_image, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        
        self.car_imgs = []
        for img_loc in self.car_img_locs :
            fil = os.path.join(self.track_folder, img_loc)
            self.car_imgs.append( pygame.image.load( fil ) )
            self.car_imgs[-1] = pygame.transform.scale(self.car_imgs[-1], self.car_dimensions[::-1])
        
        self.driver_cars = np.random.choice(len(self.car_imgs), size=self.total_num_drivers, replace=True)
        
        self.display_width = self.WINDOW_WIDTH
        self.display_height = self.WINDOW_HEIGHT
        self.score_history_fig_y_offset = self.display_height-self.score_history_fig_size[1]
    
    def train_drivers_in_background(self, elements, cars, result_q) :
        cars_complete = []
        for (itr, car) in enumerate(cars) :
            while True :
                crashed = car.update_kinematics()
                if crashed :
                    cars_complete.append( car )
                    break
        result_q.put( [elements, cars_complete] )
    
    def prep_next_run(self) :
        self.current_training_mode = self.training_mode
        
        for driver in np.arange( self.total_num_drivers ) :
            self.car_pool[driver].go_to_start_position()
            self.car_pool[driver].AI_driver = self.driver_pool.population[driver]
        self.all_drivers_out = False
        self.num_drivers_on_track = 0
        self.drivers_to_run = np.arange(self.total_num_drivers).tolist()
        self.next_driver_num = 0
        self.drivers_out = 0
        self.driver_pool.generation += 1
        self.drivers_on_track = []
        if self.training_mode == 0 : # full_display
            self.training_in_background = False
            del self.drivers_to_run[0] # otherwise there is a repeat as next_driver_num is already 0
        elif self.training_mode == 1 : # show_single_set
            keep_to_display = int( 0.75*self.total_num_drivers/100 * self.max_drivers_on_track )
            num_training_threads = int( (self.total_num_drivers - keep_to_display) / self.maximum_number_cars_per_training_thread )
            self.drivers_train_background = np.array_split(self.drivers_to_run[keep_to_display:], num_training_threads)
            self.drivers_to_run = self.drivers_to_run[:keep_to_display]
            self.training_in_background = True
            self.threads_training_in_background = num_training_threads
            del self.drivers_to_run[0] # otherwise there is a repeat as next_driver_num is already 0
        else : # full_speed
            self.next_driver_num = None
            self.all_drivers_out = True
            self.num_drivers_on_track = 0
            if self.multi_threading_version == 1 :
                self.drivers_train_background = np.array_split(self.drivers_to_run, self.maximum_concurrent_training_threads)
                self.threads_training_in_background = self.maximum_concurrent_training_threads
            else :
                num_training_threads = int( self.total_num_drivers / self.maximum_number_cars_per_training_thread )
                self.drivers_train_background = np.array_split(self.drivers_to_run, num_training_threads)
                self.threads_training_in_background = num_training_threads
            self.drivers_to_run = []
            self.training_in_background = True
        if self.training_mode == 1 or self.training_mode == 2 :
            result_q = multiprocessing.Queue()
            
            process_args = []
            
            if self.multi_threading_version == 1 :
                for elements in self.drivers_train_background :
                    p_args = {'elements':elements, 'cars':self.car_pool[elements], 'result_q':result_q}
                    self.training_process = multiprocessing.Process(target=self.train_drivers_in_background, kwargs=(p_args))
                    self.training_process.start()
            elif self.multi_threading_version == 2 :
                for elements in self.drivers_train_background :
                    process_args.append( [elements, self.car_pool[elements]] )
                p_args = {'fn':self.train_drivers_in_background, 'process_args':process_args, 'max_processes':self.maximum_concurrent_training_threads, 'fn_result_q':result_q}
                self.training_process = multiprocessing.Process(target=multiProcessManagement, kwargs=(p_args))
                self.training_process.start()
        
        self.create_figure()
        
        if self.training_mode == 1 or self.training_mode == 2 :
            return result_q

    def create_figure(self) :
        self.score_history_fig_size = (200, 200)
        avg_data = self.score_history[1][-1]
        min_data = np.min(self.scores)
        max_data = np.max(self.scores) + 1e-4
        score_rng = max_data-min_data
        step_size = score_rng / 20
        bins = np.arange( min_data, max_data+step_size, step_size )
        
        max_num_history_points = 15
        score_len = len(self.score_history[0][-max_num_history_points:])
        plot_x = np.arange( score_len ) / ( score_len - 1 ) * score_rng + min_data
        
        fig = pylab.figure(figsize=[2,2], dpi=100)
        
        ax = fig.gca()
        ax.hist( self.scores, bins=bins, color='g', rwidth=0.85 )
        ax.plot([avg_data, avg_data], ax.get_ylim(), 'm')
        
        ax2 = ax.twinx()
        ax2.plot( plot_x, self.score_history[0][-max_num_history_points:], 'r.-' )
        ax2.plot( plot_x, self.score_history[1][-max_num_history_points:], 'k.-' )
        ax2.plot( plot_x, self.score_history[2][-max_num_history_points:], 'b.-' )
        
        ax.set_yscale('log')
        ax2.set_yscale('log')
        
        ax.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        self.score_history_fig = pygame.image.fromstring(raw_data, self.score_history_fig_size, "RGB")
        plt.close('all')

    def create_car_pool(self) :
        self.car_pool = []
        for i in np.arange(self.total_num_drivers) :
            driver = self.driver_pool.add_new_individual_to_population()
            
            car = raceCar( track=self.track, AI_driver=driver, number_vision_rays=self.number_vision_rays, 
                vision_ray_num_history=self.vision_ray_num_history, load_car=self.car_file, 
                params_dict=self.params_dict )
            car.go_to_start_position()
            car.calc_dt(fps=self.FPS, updates_per_frame=self.kinematic_updates_per_frame)
            self.car_pool.append( car )
        
        self.car_dimensions = np.array( self.oversize_cars*self.car_pool[0].car_dimensions/self.scale_factor, dtype=np.int)
        
        self.car_pool = np.array( self.car_pool )
    
    def load_saved_params(self, params=None, params_file=None) :
        param_names = ['track_file',
                       'display_max_resolution',
                       'car_file',
                       'genetic_AI_params_to_load',
                       'track_params_to_load',
                       'car_to_load',
                       'total_num_drivers',
                       'max_drivers_on_track',
                       'FPS',
                       'kinematic_updates_per_frame',
                       'number_vision_rays',
                       'vision_ray_num_history',
                       'training_mode',
                       'maximum_concurrent_training_threads',
                       'maximum_number_cars_per_training_thread',
                       'AI_shape',
                       'display_width',
                       'display_height',
                       ]
        
        #params_file = 'AI_drivers.json'
        if params_file is not None :
            #params_file = 'AI_drivers.json'
            with open( os.path.join(os.getcwd(), params_file), 'r' ) as f :
                params_file = json.load(f)
                #param_keys = params.keys()
        
        if params is None :
            if params_file is not None :
                params = params_file
        else :
            if params_file is not None :
                params_file.update(params)
                params = params_file
        
        if params is not None :
            param_keys = params.keys()
        
        if params is not None :
            if 'track_file' in param_keys :
                if params['track_file'] is not None :
                    track_file = params['track_file']
                    if '_image' in track_file or '.' in track_file :
                        track_file = track_file.split('.')[0]
                        self.track_file = track_file.split('_image')[0]
            
            if 'display_max_resolution' in param_keys :
                if params['display_max_resolution'] is not None :
                    self.display_max_resolution = params['display_max_resolution']
            
            if 'car_file' in param_keys :
                if params['car_file'] is not None :
                    self.car_file = params['car_file']
            
            if 'genetic_AI_params_to_load' in param_keys :
                if params['genetic_AI_params_to_load'] is not None :
                    self.genetic_AI_params_to_load = params['genetic_AI_params_to_load']
            
            if 'track_params_to_load' in param_keys :
                if params['track_params_to_load'] is not None :
                    self.track_params_to_load = params['track_params_to_load']
            
            if 'car_to_load' in param_keys :
                if params['car_to_load'] is not None :
                    self.car_to_load = params['car_to_load']
            
            if 'total_num_drivers' in param_keys :
                if params['total_num_drivers'] is not None :
                    self.total_num_drivers = params['total_num_drivers']
            
            if 'max_drivers_on_track' in param_keys :
                if params['max_drivers_on_track'] is not None :
                    self.max_drivers_on_track = params['max_drivers_on_track']
            
            if 'FPS' in param_keys :
                if params['FPS'] is not None :
                    self.FPS = params['FPS']
            
            if 'kinematic_updates_per_frame' in param_keys :
                if params['kinematic_updates_per_frame'] is not None :
                    self.kinematic_updates_per_frame = params['kinematic_updates_per_frame']
            
            if 'number_vision_rays' in param_keys :
                if params['number_vision_rays'] is not None :
                    self.number_vision_rays = params['number_vision_rays']
            
            if 'vision_ray_num_history' in param_keys :
                if params['vision_ray_num_history'] is not None :
                    self.vision_ray_num_history = params['vision_ray_num_history']
            
            if 'training_mode' in param_keys :
                if params['training_mode'] is not None :
                    self.training_mode = params['training_mode']
            
            if 'maximum_concurrent_training_threads' in param_keys :
                if params['maximum_concurrent_training_threads'] is not None :
                    self.maximum_concurrent_training_threads = params['maximum_concurrent_training_threads']
            
            if 'maximum_number_cars_per_training_thread' in param_keys :
                if params['maximum_number_cars_per_training_thread'] is not None :
                    self.maximum_number_cars_per_training_thread = params['maximum_number_cars_per_training_thread']
            
            if 'AI_shape' in param_keys :
                if params['AI_shape'] is not None :
                    AI_shape = []
                    layers = params['AI_shape']
                    for layer in layers :
                        if layer == 'AI_inputs' :
                            AI_inputs = self.number_vision_rays * (self.vision_ray_num_history+1) + 1
                            AI_shape.append( AI_inputs )
                        elif layer == 'AI_outputs' :
                            AI_outputs = 2
                            AI_shape.append( AI_outputs )
                        else :
                            AI_shape.append( layer )
                    self.IO_shapes = AI_shape
            
            if 'display_width' in param_keys :
                if params['display_width'] is not None :
                    self.display_width = params['display_width']
            
            if 'display_height' in param_keys :
                if params['display_height'] is not None :
                    self.display_height = params['display_height']
            
            #AI_inputs = self.number_vision_rays * (self.vision_ray_num_history+1) + 1
            #self.IO_shapes = [AI_inputs, self.AI_outputs]
            
            #wr = self.display_width / self.display_max_resolution[0] # width ratio
            #hr = self.display_height / self.display_max_resolution[1] # height ratio
            #self.scale_factor = max(wr, hr)
            #if self.scale_factor < 1 :
            #    self.scale_factor = 1

    def human_time(self, duration) :
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
    
    def next_game_frame(self) :
        current_frame_time = current_time()
        self.frame_rates[:-1] = self.frame_rates[1:]
        self.frame_rates[-1] = 1/(current_frame_time-self.past_frame_time)
        self.past_frame_time = current_frame_time
        avg_frame_rate = np.mean(self. frame_rates )
        
        if self.all_drivers_out and self.num_drivers_on_track == 0 and not self.training_in_background and not self.saved_current_generation :
            self.saved_current_generation = True
            self.driver_pool.save_best_number()
        
        if not self.pause_game :
            for driver in self.drivers_on_track :
                crashed = self.car_pool[driver].update_kinematics()
                if crashed :
                    del self.drivers_on_track[ self.drivers_on_track.index(driver) ]
                    self.num_drivers_on_track -= 1
                    self.drivers_out += 1
            if not self.all_drivers_out and self.num_drivers_on_track < self.max_drivers_on_track :
                self.drivers_on_track.append( self.next_driver_num )
                self.num_drivers_on_track += 1
                if len(self.drivers_to_run) == 0 :
                    self.all_drivers_out = True
                    self.next_driver_num = None
                else :
                    self.next_driver_num = self.drivers_to_run[0]
                    del self.drivers_to_run[0]
            
            if self.saved_current_generation : # initial loop to setup drivers
                self.saved_current_generation = False
                
                self.scores = []
                for num in np.arange( self.total_num_drivers ) :
                    self.scores.append( self.driver_pool.population[num].score )
                self.last_gen_stats = [ np.max(self.scores), np.mean(self.scores), np.min(self.scores) ]
                self.score_history[0].append( self.last_gen_stats[0] )
                self.score_history[1].append( self.last_gen_stats[1] )
                self.score_history[2].append( self.last_gen_stats[2] )
                
                self.lap_completed = False
                self.best_lap_time = np.inf
                for car in self.car_pool :
                    laps_completed = car.laps_completed
                    if laps_completed > 0 :
                        self.lap_completed = True
                        lap_time = car.raceTime
                        if lap_time < self.best_lap_time :
                            self.best_lap_time = lap_time
                
                if self.pause_after_gen :
                    self.pause_game = True
                    self.pause_after_gen = False
                else :
                    self.driver_pool.next_generation()
                    self.result_q = self.prep_next_run()
        
        if self.training_in_background :
            try :
                [ elements, cars ] = self.result_q.get(block=False)
            except Exception as e :
                pass
            else :
                self.threads_training_in_background -= 1
                self.drivers_out += len(elements)
                for (key, val) in enumerate(elements) :
                    self.car_pool[val] = cars[key]
                    self.driver_pool.population[val] = cars[key].AI_driver
                if self.threads_training_in_background == 0 :
                    self.training_in_background = False
        
        # displace race track
        self.game_window.blit(self.track_image, (0,0))
        
        # display plot of score history
        self.game_window.blit(self.score_history_fig, (0, self.score_history_fig_y_offset))
        
        for driver in self.drivers_on_track :
            car = self.car_pool[driver]
            car_img = self.car_imgs[ self.driver_cars[driver] ]
            car_rotated = pygame.transform.rotate( car_img, (-car.angle-np.pi/2)*360/(2*np.pi) )
            (car_offset_x, car_offset_y) = car_rotated.get_rect().center
            self.game_window.blit(car_rotated, tuple([int(car.pos_pxl[0]/self.scale_factor-car_offset_x), 
                                                 int(car.pos_pxl[1]/self.scale_factor-car_offset_y)]))
        
        if True : # displays important information
            x_pos = 3
            y_pos = 10
            text = self.font.render( f'Gen: {self.driver_pool.generation}', True, self.font_clr); self.game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*self.font_size
            text = self.font.render( f'drivers left: {len(self.drivers_to_run)}', True, self.font_clr) ; self.game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*self.font_size
            text = self.font.render( f'on track: {self.num_drivers_on_track}/{self.max_drivers_on_track}', True, self.font_clr); self.game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*self.font_size
            text = self.font.render( f'drivers done: {self.drivers_out}/{self.total_num_drivers}', True, self.font_clr); self.game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*self.font_size
            text = self.font.render( f'threads remaining: {self.threads_training_in_background}', True, self.font_clr); self.game_window.blit(text, (x_pos, y_pos) ); y_pos += 1.1*self.font_size
            text = self.font.render( f'training in background: {self.training_in_background}', True, self.font_clr) ; self.game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*self.font_size
            if self.current_training_mode == self.training_mode :
                text = self.font.render( f'training mode: {self.training_mode}', True, self.font_clr); self.game_window.blit(text, (x_pos, y_pos) ); y_pos += 1.1*self.font_size
            else :
                text = self.font.render( f'training mode: {self.current_training_mode} --> {self.training_mode}', True, self.font_clr); self.game_window.blit(text, (x_pos, y_pos) ); y_pos += 1.1*self.font_size
            text = self.font.render( f'score rng: {self.last_gen_stats[0]:.3} / {self.last_gen_stats[1]:.3} / {self.last_gen_stats[2]:.3}', True, self.font_clr); self.game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*self.font_size
            if self.lap_completed :
                text = self.font.render( f'Fastest time: {self.human_time(self.best_lap_time)}', True, self.font_clr); self.game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*self.font_size
                # text = self.font.render( f'Fastest time: {best_lap_time:.4}', True, self.font_clr); self.game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*self.font_size
            if self.pause_game :
                text = self.font.render( 'Paused', True, self.font_clr); self.game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*self.font_size
            if self.pause_after_gen :
                text = self.font.render( 'Pause after this generation', True, self.font_clr); self.game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*self.font_size



def create_default_params() :
    # create json preferences file for AI_drivers.py
    preferences = {
        'track_file': 'track',
        'display_max_resolution': [1750, 950],
        'car_file': 'car_class.json',
        'genetic_AI_params_to_load': 'AI_class_genetic.json',
        'track_params_to_load': None,
        'car_to_load': 'car_class.json',
        'total_num_drivers': 500,
        'max_drivers_on_track': 15,
        'FPS': 30,
        'kinematic_updates_per_frame': 5,
        'number_vision_rays': 5,
        'vision_ray_num_history': 0,
        'training_mode': 1,
        'maximum_concurrent_training_threads': 4,
        'maximum_number_cars_per_training_thread': 15,
        }
    
    with open( os.path.join(os.getcwd(), 'game_control_params.json'), 'w' ) as fil :
        json.dump( preferences, fil, indent=4 )

if __name__ == '__main__' :
    create_default_params()

