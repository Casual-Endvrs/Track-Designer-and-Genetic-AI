#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:57:29 2020

@author: Casual Endvrs
Email: casual.endvrs@gmail.com
GitHub: https://github.com/Casual-Endvrs
Reddit: CasualEndvrs
Twitter: @CasualEndvrs
"""

import numpy as np
import pygame
import sys

from trackDesigner_class import trackDesigner_class

track = trackDesigner_class()

# Initialise all the pygame modules
pygame.init()

# Create a game window
[WINDOW_WIDTH, WINDOW_HEIGHT] = track.display_resolution
scale_factor = track.scale_factor
game_window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

# game control varaibles
FPS = 30
mouse_clicked = False

# Set title
pygame.display.set_caption("AI Race Track!")

clock = pygame.time.Clock()

from time import time as current_time
frame_rates = np.zeros(10)
past_frame_time = current_time()
font_size = 25
font = pygame.font.SysFont('Tahoma', font_size, True, False)

print( '--------------' )
print( 'Instructions: ' )
print( '--------------' )
print( 'To create a simple track simply click on three different' )
print( '\tpoints and close the loop by clicking on the first point' )
print( '\tagain. Existing points can be moved by clicking and dragging' )
print( '\tthem. New points can be added by clicking on the grey' )
print( '\tline in that is in the middle of the track.' )
print( '--------------' )
print( 'Available keyboard controls' )
print( '--------------' )
print( 'save track: s' )
print( 'load track: x' )
print( 'reset to blank canvas: r')
print( '--------------' )
print( 'move start/finish line: f' )
print( 'make/remove check point: c' )
print( 'delete selected point: d' )
print( '--------------' )

# Game loop
game_running = True
while game_running:
    clock.tick(FPS)
    
    # Loop through all active events
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN :
            pos = np.array( pygame.mouse.get_pos(), dtype=np.float32 )
            pos *= scale_factor
            point_set, point_idx = \
                track.test_selected_spline_or_track_point(pos)
            
            if not track.spline_created : # if the spline is not yet created
                if point_set == 'blank' : # if blank area clicked
                    track.add_point_to_list(pos) # append new point
                elif point_set == 'track_dot' : # elif track_dot is clicked
                    if point_idx == 0 : # if point is starting point
                        track.calculate_track_spline() # create spline
            else : # elif spline is created
                mouse_clicked = True
                if point_set == 'track_dot' : # if point clicked is track point
                    track.selected_point_idx = point_idx # select point
                elif point_set == 'spline' : # elif point clicked is spline point
                    idx = track.find_nearest_existing_point( pos )
                    if idx == 0 :
                        idx = -1
                    track.insert_new_point_at_idx( pos, idx ) # insert new point
                    track.selected_point_idx = idx # select new point
                elif point_set == 'blank' :
                    track.selected_point_idx = None
        elif event.type == pygame.MOUSEBUTTONUP : # on button un-click
            mouse_clicked = False
        elif event.type == pygame.KEYUP :
            if event.key == pygame.K_d :
                if track.selected_point_idx is not None :
                    track.delete_point( track.selected_point_idx ) # delete selected point
                    track.selected_point_idx = None
            elif event.key == pygame.K_f :
                if track.selected_point_idx is not None :
                    track.move_start_finish_line( track.selected_point_idx )
                    track.selected_point_idx = None
            elif event.key == pygame.K_c :
                if track.selected_point_idx is not None :
                    track.flip_check_point_status()
            elif event.key == pygame.K_s :
                track.save_track()
            elif event.key == pygame.K_x :
                track.load_track()
            elif event.key == pygame.K_r :
                track.reset_track()
            
            # elif event.key == pygame.K_u :
            #     track.spline_factor -= 0.1
            #     print()
            #     print( 'spline_factor' )
            #     print( track.spline_factor )
            #     track.track_is_updated = True
            # elif event.key == pygame.K_j :
            #     track.spline_factor += 0.1
            #     print()
            #     print( 'spline_factor' )
            #     print( track.spline_factor )
            #     track.track_is_updated = True
            
            # elif event.key == pygame.K_i :
            #     track.track_inside_val += 1e-3
            #     print()
            #     print( 'track_inside_val' )
            #     print( track.track_inside_val )
            #     track.track_is_updated = True
            # elif event.key == pygame.K_k :
            #     track.track_inside_val -= 1e-3
            #     print()
            #     print( 'track_inside_val' )
            #     print( track.track_inside_val )
            #     track.track_is_updated = True
            
            # elif event.key == pygame.K_o :
            #     track.track_outside_val += 1e-3
            #     print()
            #     print( 'track_outside_val' )
            #     print( track.track_outside_val )
            #     track.track_is_updated = True
            # elif event.key == pygame.K_l :
            #     track.track_outside_val -= 1e-3
            #     print()
            #     print( 'track_outside_val' )
            #     print( track.track_outside_val )
            #     track.track_is_updated = True
            
        elif event.type == pygame.QUIT: # Close the program if the user presses the 'X'
            game_running = False
    
    if mouse_clicked and track.selected_point_idx is not None :
        pos = np.array( pygame.mouse.get_pos(), dtype=np.float32 )
        pos *= scale_factor
        track.update_point_location(track.selected_point_idx, 
                                    pos)
        track.track_is_updated = True
    
    game_window.fill( (206,213,219) )
    
    current_frame_time = current_time()
    frame_rates[:-1] = frame_rates[1:]
    frame_rates[-1] = 1/(current_frame_time-past_frame_time)
    past_frame_time = current_frame_time
    avg_frame_rate = np.mean( frame_rates )
    x_pos = 3
    y_pos = 10
    font_color = (0, 0, 0)
    text = font.render( f'fps: {avg_frame_rate:.0f}', True, font_color ); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
    text = font.render( f'track length: {(track.track_length/1000):.3f} km', True, font_color ); game_window.blit(text, (x_pos, y_pos)); y_pos += 1.1*font_size
    
    track.update_scaled_variables()
    
    if track.spline_created :
        if track.track_is_updated :
            track.calculate_track_spline()
        radius = int(track.track_width_pxl / 2.01 / scale_factor )
        for [x_pxl, y_pxl] in track.track_spline_scaled :
            pygame.draw.circle(game_window, (0, 0, 0), [int(x_pxl), int(y_pxl)], radius)
        pygame.draw.lines(game_window, # surface
                          (0, 0, 0), # R, G, B, alpha
                          True, # add closing line
                          track.track_spline_scaled, # points to plot [[x0,y0], [x1,y1],...]
                          int(track.track_width_pxl/scale_factor), # line width
                          )
        pygame.draw.lines(game_window, # surface
                          (220, 220, 220), # R, G, B, alpha
                          True, # add closing line
                          track.track_spline_scaled, # points to plot [[x0,y0], [x1,y1],...]
                          3, # line width
                          )
    
    if track.track_limits_defined : # draw track boundaries
        pygame.draw.lines(game_window, # surface
                          (255, 0, 0), # R, G, B, alpha
                          True, # add closing line
                          track.track_inside_scaled, # points to plot [[x0,y0], [x1,y1],...]
                          2, # line width
                          )
        pygame.draw.lines(game_window, # surface
                          (0, 255, 0), # R, G, B, alpha
                          True, # add closing line
                          track.track_outside_scaled, # points to plot [[x0,y0], [x1,y1],...]
                          2, # line width
                          )
    
    for [x_pxl, y_pxl] in track.track_inside_scaled : # draw track_inside line markers
        pygame.draw.circle(game_window, (221, 51, 255), [int(x_pxl), int(y_pxl)], 2)
    
    for [x_pxl, y_pxl] in track.track_outside_scaled : # draw track_outside line markers
        pygame.draw.circle(game_window, (221, 51, 255), [int(x_pxl), int(y_pxl)], 2)
    
    if np.shape( track.track_points_scaled )[0] > 0 : # draw circle around start-finish point
        pygame.draw.circle(game_window, (255, 0, 0), track.track_points_scaled[0], 8)
    
    if track.start_finish_line_scaled is not None : # start-finish line
        [start, stp] = track.start_finish_line_scaled
        pygame.draw.line(game_window, (255, 0, 0), start, stp, 3)
    
    # draw circle around selected point
    if track.selected_point_idx is not None :
        pygame.draw.circle(game_window, (0, 255, 0), 
                           track.track_points_scaled[track.selected_point_idx], 10)
    
    # draw circles around check_point markers
    if len(track.check_points) > 0 :
        for idx in track.check_points :
            pygame.draw.circle(game_window, (255, 255, 0), track.track_points_scaled[idx], 7)
    
    # draw check point lines
    if np.shape(track.check_point_lines_scaled)[0] > 0 :
        for [pt_1, pt_2] in track.check_point_lines_scaled :
            pygame.draw.line(game_window, (255, 255, 0), pt_1, pt_2, 3)
    
    # draw track_points_scaled markers
    for [x_pxl, y_pxl] in track.track_points_scaled :
        pygame.draw.circle(game_window, (0, 100, 255), [x_pxl, y_pxl], 5)
    
    # Update display
    pygame.display.update()

# Uninitialize all pygame modules and quit the program
pygame.quit()
sys.exit()
