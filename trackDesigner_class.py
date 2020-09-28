#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:36:57 2020

@author: Casual Endvrs
Email: casual.endvrs@gmail.com
GitHub: https://github.com/Casual-Endvrs
Reddit: CasualEndvrs
Twitter: @CasualEndvrs
"""

import numpy as np
import os
import pickle
from scipy import interpolate
from PIL import Image, ImageDraw
from track_class import save_track_params
from bitarray import bitarray
import json

class trackDesigner_class() :
    def __init__(self, name_base=None, params_to_load=None) :
        self.display_max_resolution = [1750, 1000]
        self.display_resolution = self.display_max_resolution
        self.window_width = 1200 # 1750
        self.window_height = 1000 # 1000
        self.scale_factor = 1
        self.display_width = int( self.window_width / self.scale_factor )
        self.display_height = int( self.window_height / self.scale_factor )
        
        self.track_limits = [ bitarray(self.window_height) for i in np.arange(self.window_width) ]
        
        self.max_distance_sqrd_to_point = self.scale_factor * 12**2
        self.track_points = [] # list of all points the determine track route
        self.track_points_scaled = []
        self.start_finish_line = None
        self.start_finish_line_scaled = None
        self.check_points = []
        self.check_point_lines = []
        self.check_point_lines_scaled = []
        self.track_spline = np.array([])
        self.track_spline_scaled = np.array([])
        self.spline_created = False
        self.track_is_updated = False
        self.point_idx_selection = None
        self.selected_point_idx = None
        
        self.m_per_pxl = 0.2
        
        self.track_width_m = 8
        self.track_width_pxl = int( self.track_width_m / self.m_per_pxl )
        self.track_outside = []
        self.track_outside_scaled = []
        self.track_inside = []
        self.track_inside_scaled = []
        self.track_limits_defined = False
        self.track_length = 0
        self.track_length_pxl = 0
        
        self.spline_factor = 20
        self.track_inside_val = 0.1
        self.track_outside_val = 0.1
        
        self.track_name = None
        
        self.cwd = os.getcwd()
        self.load_saved_params(params_to_load)
        
        if name_base is not None :
            self.track_name = name_base
        if self.track_name is None :
            self.track_name = 'track'
    
    def reset_track(self) :
        self.track_limits = [ bitarray(self.window_height) for i in np.arange(self.window_width) ]
        
        self.track_points = [] # list of all points the determine track route
        self.track_points_scaled = []
        self.start_finish_line = None
        self.start_finish_line_scaled = None
        self.check_points = []
        self.check_point_lines = []
        self.check_point_lines_scaled = []
        self.track_spline = np.array([])
        self.track_spline_scaled = np.array([])
        self.spline_created = False
        self.track_is_updated = False
        self.point_idx_selection = None
        self.selected_point_idx = None
        
        self.track_width_pxl = int( self.track_width_m / self.m_per_pxl )
        self.track_outside = []
        self.track_outside_scaled = []
        self.track_inside = []
        self.track_inside_scaled = []
        self.track_limits_defined = False
        self.track_length = 0
        self.track_length_pxl = 0
    
    def save_track(self) :
        print( '\nsaving...' )
        self.create_track_limits_array()
        self.save_track_image()
        track_params = save_track_params(self)
        with open( os.path.join(self.cwd, self.track_name+'_params.pkl'), 'wb') as f :
            pickle.dump(track_params, f)
        print( 'save complete\n' )
    
    def load_track(self) :
        fil = os.path.join(self.cwd, self.track_name+'_params.pkl')
        if os.path.isfile( fil ) :
            with open( fil, 'rb') as f :
                track_params = pickle.load( f )
            
            self.window_width = track_params.window_width
            self.window_height = track_params.window_height
            self.track_limits = track_params.track_limits
            
            self.max_distance_sqrd_to_point = track_params.max_distance_sqrd_to_point
            self.track_points = track_params.track_points
            self.start_finish_line = track_params.start_finish_line
            self.check_points = track_params.check_points
            self.check_point_lines = track_params.check_point_lines
            self.track_spline = track_params.track_spline
            self.spline_created = track_params.spline_created
            self.track_is_updated = track_params.track_is_updated
            
            self.m_per_pxl = track_params.m_per_pxl
            
            self.track_width_pxl = track_params.track_width_pxl
            self.track_inside = track_params.track_inside
            self.track_outside = track_params.track_outside
            self.track_limits_defined = track_params.track_limits_defined
            self.track_length = track_params.track_length
            self.track_length_pxl = track_params.track_length_pxl
            
            self.spline_factor = track_params.spline_factor
            
            self.update_scaled_variables()
    
    def load_saved_params(self, params_to_load=None) :
        # 'track_name'
        # 'display_max_resolution'
        # 'track_resolution'
        # 'm_per_pxl'
        # 'track_width_m'
        
        if params_to_load is None :
            params_to_load = 'trackDesigner_class.json'
        
        fil = os.path.join(self.cwd, params_to_load)
        if os.path.isfile( fil ) :
            with open( fil, 'r') as f :
                params = json.load(f)
            
            param_keys = params.keys()
            
            if 'track_name' in param_keys :
                if params['track_name'] is not None :
                    self.track_name = params['track_name']
            
            if 'display_max_resolution' in param_keys :
                if params['display_max_resolution'] is not None :
                    self.display_max_resolution = params['display_max_resolution']
            
            if 'track_resolution' in param_keys :
                if params['track_resolution'] is not None :
                    self.window_width = int( params['track_resolution'][0] )
                    self.window_height = int( params['track_resolution'][1] )
            
            if 'm_per_pxl' in param_keys :
                if params['m_per_pxl'] is not None :
                    self.m_per_pxl = params['m_per_pxl']
            
            if 'track_width_m' in param_keys :
                if params['track_width_m'] is not None :
                    self.track_width_m = params['track_width_m']
            
            self.display_width = int( self.window_width / self.scale_factor )
            self.display_height = int( self.window_height / self.scale_factor )
            
            self.track_limits = [ bitarray(self.window_height) for i in np.arange(self.window_width) ]
            
            self.max_distance_sqrd_to_point = self.scale_factor * 15**2
            self.track_width_pxl = int( self.track_width_m / self.m_per_pxl )
            
            self.calculate_display_size_scale()
    
    def calculate_display_size_scale(self) :
        wr = self.window_width / self.display_max_resolution[0] # width ratio
        hr = self.window_height / self.display_max_resolution[1] # height ratio
        self.scale_factor = max( wr, hr )
        if self.scale_factor < 1 :
            self.scale_factor = 1
        self.display_resolution = [int(self.window_width/self.scale_factor), 
                                   int(self.window_height/self.scale_factor)]
    
    def update_scaled_variables(self) :
        self.check_point_lines_scaled = np.array( self.check_point_lines ) / self.scale_factor
        self.check_point_lines_scaled = self.check_point_lines_scaled.astype(np.int)
        
        self.track_points_scaled = np.array( self.track_points ) / self.scale_factor
        self.track_points_scaled = self.track_points_scaled.astype(np.int)
        
        self.track_spline_scaled = self.track_spline / self.scale_factor
        self.track_spline_scaled = self.track_spline_scaled.astype(np.int)
        
        self.track_inside_scaled = np.array( self.track_inside ) / self.scale_factor
        self.track_inside_scaled = self.track_inside_scaled.astype(np.int)
        
        self.track_outside_scaled = np.array( self.track_outside ) / self.scale_factor
        self.track_outside_scaled = self.track_outside_scaled.astype(np.int)
        
        if self.start_finish_line is not None :
            self.start_finish_line_scaled = self.start_finish_line / self.scale_factor
            self.start_finish_line_scaled = self.start_finish_line_scaled.astype(np.int)
    
    def flip_check_point_status(self) :
        if self.selected_point_idx is not None :
            if self.selected_point_idx in self.check_points :
                del self.check_points[ self.check_points.index(self.selected_point_idx) ]
            else :
                self.check_points.append( self.selected_point_idx )
                self.check_points.sort()
            self.selected_point_idx = None
            self.calculate_track_spline()
    
    def find_check_point_lines(self) :
        self.check_point_lines = []
        
        if len(self.check_points) > 0 :
            for indx in self.check_points :
                point = self.track_points[ indx ]
                idx_inside = self.test_pos_is_existing_point(point,
                                                             self.track_inside)
                idx_outside = self.test_pos_is_existing_point(point,
                                                             self.track_outside)
                line = [self.track_outside[idx_outside], self.track_inside[idx_inside]]
                self.check_point_lines.append( line )
    
    def find_nearest_existing_point(self, point, find_prev_point=False) :
        """
        This function will find the point amoung the track_points list that 
        directly follows the provided reference point. 

        Parameters
        ----------
        point : int or float
            This is the reference point.

        Returns
        -------
        idx : int
            This is the index of the point in track_points that directly follows 
            after the reference point.

        """
        if find_prev_point :
            advance_factor = -1
        else : advance_factor = 1
        
        track_spline_idx = self.test_pos_is_existing_point(point, self.track_spline, 
                                    max_distance_sqrd=self.max_distance_sqrd_to_point)
        track_spline_max_idx = np.shape( self.track_spline )[0]
        
        while True :
            idx = self.test_pos_is_existing_point(self.track_spline[track_spline_idx],
                                    max_distance_sqrd=self.max_distance_sqrd_to_point)
            if idx is not None :
                return idx
            
            track_spline_idx += advance_factor
            if track_spline_idx >= track_spline_max_idx :
                return 0

    def test_pos_is_existing_point(self, point, points_list=None, 
                                   max_distance_sqrd=None, 
                                   return_distance_sqrd=False) :
        """
        This function finds the index of the point in points_list that is closest 
        to the provided reference point. This index is returned if the distance 
        between the two points is less than max_distance. If the max_distance is 
        None the index will always be returned.

        Parameters
        ----------
        point : list [x, y]
            This is refered to as the reference point. Position to find nearest 
            point from.
        points_list : list of points, optional
            List of points to select nearest point from. The default is None
        max_distance_sqrd : int, float or None, optional
            Maximum distance the "nearest point" can be from the reference point. 
            If None the nearest point will always be returned. If this is an int 
            or float then the nearest point will only be returned if the distance
            squared is less than max_distance_sqrd. The default is None.
        return_distance_sqrd : bool, optional
            If True the distance squared will be returned with the index, else 
            only the index will be returned. The default is False.

        Returns
        -------
        closest_point_idx : int
            This is the index of the point in points_list that is closest to the 
            reference point.

        """
        if points_list is None :
            points_list = self.track_points
        
        if len( points_list ) > 0 :
            dist_sqrd = np.array( points_list ) - np.array( point )
            dist_sqrd = dist_sqrd**2
            dist_sqrd = np.sum(dist_sqrd, axis=1)
            min_dist = np.amin( dist_sqrd )
            closest_point_idx = np.where( dist_sqrd == min_dist )[0][0]
            if max_distance_sqrd is None or dist_sqrd[ closest_point_idx ] <= max_distance_sqrd :
                if return_distance_sqrd :
                    return closest_point_idx, np.sqrt( dist_sqrd[ closest_point_idx ] )
                return closest_point_idx
        if return_distance_sqrd :
            return None, None
        return None

    def add_point_to_list(self, point) :
        self.track_points.append( point )
        self.track_is_updated = True

    def calculate_track_spline(self) :
        x, y = zip(*self.track_points)
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
        
        self.calc_track_length()
        num_spline_points = int(self.track_length_pxl/self.spline_factor)
        
        f, u = interpolate.splprep([x,y], s=0, per=True)
        x, y = interpolate.splev(np.linspace(0, 1, num_spline_points), f)
        
        self.track_spline = np.transpose([x, y])
        self.spline_created = True
        self.track_is_updated = False
        
        self.calc_track_limits()

    def test_selected_spline_or_track_point(self, point) :
        track_points_idx = self.test_pos_is_existing_point(point,
                            max_distance_sqrd=self.max_distance_sqrd_to_point, 
                            return_distance_sqrd=False, 
                            )
        
        track_spline_idx = self.test_pos_is_existing_point(point, 
                            self.track_spline, 
                            max_distance_sqrd=self.max_distance_sqrd_to_point, 
                            return_distance_sqrd=False, 
                            )
        
        if track_spline_idx is None :
            if track_points_idx is None :
                return 'blank', None
            else :
                return 'track_dot', track_points_idx
        
        if track_points_idx is None :
            return 'spline', track_spline_idx
        
        if track_points_idx is not None :
            return 'track_dot', track_points_idx
        else :
            return 'spline', track_spline_idx

    def insert_new_point_at_idx(self, point, idx=-1) :
        if idx == -1 :
            self.track_points.append(point)
        else :
            self.track_points.insert(idx, point)
        
        if idx != -1 :
            for index in np.arange( len(self.check_points) ) :
                if self.check_points[index] >= idx :
                    self.check_points[index] += 1
            self.calculate_track_spline()
        
        self.track_is_updated = True
    
    def delete_point(self, idx) :
        if np.shape( self.track_points )[0] > 3 :
            check_points_updated = False
            del self.track_points[ idx ]
            if idx in self.check_points :
                del self.check_points[ self.check_points.index(idx) ]
            for index in np.arange( len(self.check_points) ) :
                if self.check_points[index] > idx :
                    self.check_points[index] -= 1
                    check_points_updated = True
            if check_points_updated :
                self.calculate_track_spline()
            self.track_is_updated = True
    
    def update_point_location(self, point_index, new_pos) :
        self.track_points[ point_index ] = new_pos
    
    def move_start_finish_line(self, idx):
        new_points_list = self.track_points[idx:]
        new_points_list.extend( self.track_points[:idx] )
        
        self.track_points = new_points_list
        
        max_idx = np.shape(self.track_points)[0]
        idx_shift = max_idx - idx
        for index in np.arange( len(self.check_points) ) :
            self.check_points[index] = (self.check_points[index]+idx_shift)%max_idx
        
        self.calculate_track_spline()
    
    def calc_track_limits(self) :
        half_track_width = self.track_width_pxl/2.
        spline_len = np.shape( self.track_spline )[0]
        
        self.track_inside = []
        self.track_outside = []
        
        for idx in np.arange(spline_len) :
            idx_1 = idx+1
            idx_2 = idx+2
            if idx_1 >= spline_len :
                idx_1 -= spline_len
            if idx_2 >= spline_len :
                idx_2 -= spline_len
            norm_vec = self.track_spline[idx_2,::-1] - self.track_spline[idx,::-1]
            norm_vec /= np.sqrt(np.sum(norm_vec**2))
            norm_vec[0] *= -1
            track_offset = half_track_width*norm_vec
            self.track_inside.append( self.track_spline[idx_1] + track_offset )
            self.track_outside.append( self.track_spline[idx_1] - track_offset )
        
        self.track_limits_defined = True
        
        self.start_finish_line = np.array( [self.track_outside[0], 
                                            self.track_inside[0]] )
        self.find_check_point_lines()
        
        # self.reduce_track_limit_lines()
        self.calc_track_length()
    
    def reset_track_limits_arr(self) :
        for bit_arr in self.track_limits :
            bit_arr.setall(False)
    
    def dist_point_to_line(self, point, line_start, line_end) :
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
        [x_0, y_0] = point
        [x_1, y_1] = line_start
        [x_2, y_2] = line_end
        
        num = np.abs( (y_2-y_1)*x_0 - (x_2-x_1)*y_0 + x_2*y_1 - y_2*x_1 )
        den = np.sqrt( (y_2-y_1)**2 + (x_2-x_1)**2 )
        
        return num/den
    
    def calc_track_length(self) :
        num_points = np.shape(self.track_points)[0]
        
        self.track_length_pxl = 0
        for i in np.arange(-1, num_points-1) :
            self.track_length_pxl += self.distance_between_points(self.track_points[i], 
                                                    self.track_points[i+1])
        
        self.track_length = self.m_per_pxl * self.track_length_pxl
    
    def distance_between_points(self, pt_1, pt_2) :
        pt_1 = np.array( pt_1 )
        pt_2 = np.array( pt_2 )
        return np.sqrt( np.sum( ( pt_1 - pt_2 )**2 ) )
    
    def create_track_limits_array(self) :
        self.reset_track_limits_arr()
        
        normal_spline_factor = self.spline_factor
        self.spline_factor /= 3
        self.calculate_track_spline()
        
        len_inside = len(self.track_inside)
        for i in np.arange(-1, len_inside-1) :
            pt1 = self.track_inside[i]
            pt2 = self.track_inside[i+1]
            lin_terms = self.calc_lin_terms(pt1, pt2)
            limit_pts = self.track_limits_locs(pt1, pt2, lin_terms[0], lin_terms[1], inside_limit=True)
            self.set_track_limits(limit_pts)
        
        len_outside = len(self.track_outside)
        for i in np.arange(-1, len_outside-1) :
            pt1 = self.track_outside[i]
            pt2 = self.track_outside[i+1]
            lin_terms = self.calc_lin_terms(pt1, pt2)
            limit_pts = self.track_limits_locs(pt1, pt2, lin_terms[0], lin_terms[1], inside_limit=False)
            self.set_track_limits(limit_pts)
        
        self.spline_factor = normal_spline_factor
        self.calculate_track_spline()
        
        self.check_stray_limit_pts()
        
        # self.print_track_limits_to_txt()
    
    def print_track_limits_to_txt(self) :
        with open( os.path.join(self.cwd, self.track_name + '_limits.txt'), 'w') as f :
            for bit_arr in self.track_limits :
                str_arr = bit_arr.to01()
                f.write( str_arr )
                f.write( '\n' )
    
    def check_stray_limit_pts(self) :
        """
        This function is used to find any "hanging" track limit points.
        A "hanging" point is a point that has been erroneously marked as a track limit.
        These points can cause cars to "crash" in the middle of the track.

        Returns
        -------
        None.

        """
        for x in np.arange(2, self.window_width-3) :
            for y in np.arange(2, self.window_height-5) :
                bits = self.track_limits[x-1][y-1:y+2]
                bits.append( self.track_limits[x][y-1:y+2] )
                bits.append( self.track_limits[x+1][y-1:y+2] )
                if not bits.any() : # if no limits points found
                    continue
                bits = self.track_limits[x-2][y-2:y+4]
                bits.extend( self.track_limits[x-1][y-2:y+4:5] )
                bits.extend( self.track_limits[x][y-2:y+4:5] )
                bits.extend( self.track_limits[x+1][y-2:y+4:5] )
                bits.extend( self.track_limits[x+2][y-2:y+4] )
                if np.sum( bits.tolist() ) > 1 :
                    continue
                
                for i in [x-1, x, x+1] :
                    for j in [y-1, y, y+1] :
                        self.track_limits[i][j] = False
        
        # for x in np.arange(1, self.window_width-1) :
        #     for y in np.arange(1, self.window_height-1) :
        #         if not self.track_limits[x][y] :
        #             continue
        #         bits = self.track_limits[x-1][y-1:y+2]
        #         bits.extend( self.track_limits[x][y-1:y+2:2] )
        #         bits.extend( self.track_limits[x+1][y-1:y+2] )
        #         if not bits.any() :
        #             self.track_limits[x][y] = 0
        
        # for x in np.arange(1, self.window_width-1) :
        #     for y in np.arange(1, self.window_height-2) :
        #         if not self.track_limits[x][y] and not self.track_limits[x][y+1] :
        #             continue
        #         bits = self.track_limits[x-1][y-1:y+2]
        #         bits.extend( self.track_limits[x][y-1:y+2:2] )
        #         bits.extend( self.track_limits[x+1][y-1:y+2:2] )
        #         bits.extend( self.track_limits[x+2][y-1:y+2] )
        #         if not bits.any() :
        #             self.track_limits[x][y] = 0
        #             self.track_limits[x][y+1] = 0
    
    def calc_lin_terms(self, pt1, pt2) :
        dx = pt2[1] - pt1[1]
        dy = pt2[0] - pt1[0]
        if dx == 0 :
            dx = 1e-5
        if dy == 0 :
            dy = 1e-5
        m_x = dx / dy
        b_x = pt1[1] - m_x * pt1[0]
        
        m_y = dy / dx
        b_y = pt1[0] - m_y * pt1[1]
        
        return [ [m_x, b_x], [m_y, b_y]]
    
    def track_limits_locs(self, pt1, pt2, x_terms, y_terms, inside_limit=False) :
        x_min = np.min([pt1[0], pt2[0]])
        x_max = np.max([pt1[0], pt2[0]])
        y_min = np.min([pt1[1], pt2[1]])
        y_max = np.max([pt1[1], pt2[1]])
        
        x_factor = 1
        if pt2[0]-pt1[0] < 0 :
            x_factor = -1
        y_factor = 1
        if pt2[1]-pt1[1] > 0 :
            y_factor = -1
        
        if inside_limit :
            x_factor *= -1
            y_factor *= -1
        
        limit_pts = []
        
        for x in np.arange(x_min, x_max+1) :
            y = x_terms[0] * x + x_terms[1]
            limit_pts.append([int(x), int(y)])
            limit_pts.append([int(x), int(y+y_factor)])
            limit_pts.append([int(x), int(y+2*y_factor)])
        
        for y in np.arange(y_min, y_max+1) :
            x = y_terms[0] * y + y_terms[1]
            limit_pts.append([int(x+x_factor), int(y)])
            limit_pts.append([int(x+2*x_factor), int(y)])
        
        return limit_pts
    
    def set_track_limits(self, limit_pts) :
        for [x,y] in limit_pts :
            try :
                self.track_limits[x][y] = 1
            except :
                pass
    
    def save_track_image(self) :
        self.calculate_track_spline()
        color = [ (255, 0, 0), (255, 255, 255) ]
        
        img = Image.new(mode="RGB", 
                            size=(self.window_width, self.window_height), 
                            color=(19,133,19) )
        
        img1 = ImageDraw.Draw(img)
        
        line_width = int( 2 * self.scale_factor )
        
        # draw track pavement
        track_spline_ints = np.array( self.track_spline, dtype=np.int32 )
        for i in np.arange(-1, np.shape(track_spline_ints)[0]-1 ) :
            # draw lines for pavement
            pt_1 = tuple( track_spline_ints[i] )
            pt_2 = tuple( track_spline_ints[i+1] )
            img1.line([pt_1, pt_2], fill=(0, 0, 0), width=self.track_width_pxl)
            
            # draw circles to fill in gaps for pavement
            x_0 = int( pt_1[0] - self.track_width_pxl/2. )
            y_0 = int( pt_1[1] - self.track_width_pxl/2. )
            x_1 = int( pt_1[0] + self.track_width_pxl/2. )
            y_1 = int( pt_1[1] + self.track_width_pxl/2. )
            img1.ellipse(xy=[x_0, y_0, x_1, y_1], outline=(0, 0, 0), fill=(0, 0, 0))
        
        # draw start-finish line
        img1.line([tuple(self.start_finish_line[0]), tuple(self.start_finish_line[1])], 
                  fill=(255, 0, 0), width=line_width+2)
        
        # draw check points
        for i in np.arange( np.shape(self.check_point_lines)[0] ) :
            pt_1 = tuple( self.check_point_lines[i][0] )
            pt_2 = tuple( self.check_point_lines[i][1] )
            img1.line([pt_1, pt_2], fill=(255, 255, 0), width=line_width+2)
            
        if True : # show curbs
            # draw track boundaries - inside
            track_inside_ints = np.array( self.track_inside, dtype=np.int32 )
            for i in np.arange(-1, np.shape(track_inside_ints)[0]-1 ) :
                pt_1 = tuple( track_inside_ints[i] )
                pt_2 = tuple( track_inside_ints[i+1] )
                img1.line([pt_1, pt_2], fill=color[ i%2 ], width=line_width)
                
            # draw track boundaries - outside
            track_outside_ints = np.array( self.track_outside, dtype=np.int32 )
            for i in np.arange(-1, np.shape(track_outside_ints)[0]-1 ) :
                pt_1 = tuple( track_outside_ints[i] )
                pt_2 = tuple( track_outside_ints[i+1] )
                img1.line([pt_1, pt_2], fill=color[ i%2 ], width=line_width)
        else : # show track limits
            for x in np.arange(self.window_width) :
                for y in np.arange(self.window_height) :
                    if self.track_limits[x][y] :
                        img.putpixel((x,y), (0, 0, 255))
        
        img.save( os.path.join(self.cwd, self.track_name+'_image.png') )

def create_default_params() :
    # create json preferences file for trackDesigner_class.py
    preferences = {
        'track_name': None, # name base to use during saving
        'display_max_resolution': [1750, 1000], # sets the maximum resolution that will be displayed to the user
        'track_resolution': [1200, 1000], # sets the resolution for the track
        'm_per_pxl': 0.2, # number of meters per pixel
        'track_width_m': 8, # width of the track in meters
        }
    
    with open( os.path.join(os.getcwd(), 'trackDesigner_class.json'), 'w' ) as fil :
        json.dump( preferences, fil, indent=4 )

if __name__ == "__main__" :
    create_default_params()
