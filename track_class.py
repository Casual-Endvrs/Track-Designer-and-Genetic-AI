#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 20:30:03 2020

@author: Casual Endvrs
Email: casual.endvrs@gmail.com
GitHub: https://github.com/Casual-Endvrs
Reddit: CasualEndvrs
Twitter: @CasualEndvrs
"""

import numpy as np
import pickle
import os
# from bitarray import bitarray

class save_track_params() :
    def __init__(self, track_params) :
        self.window_width = track_params.window_width
        self.window_height = track_params.window_height
        self.scale_factor = track_params.scale_factor
        
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

class raceTrack() :
    def __init__(self, track_file, track_params_to_load=None) :
        self.working_folder, self.name_base = os.path.split( track_file )
        self.track_png = self.name_base + '_image.png'
        self.pkl_file = self.name_base + '.pkl'
        
        self.window_width = None
        self.window_height = None
        self.scale_factor = None
        
        self.track_points = None
        self.start_finish_line = None
        self.check_points = None
        self.check_point_lines = None
        self.track_spline = None
        self.spline_created = None
        self.track_is_updated = None
        
        self.m_per_pxl = None
        
        self.track_width_pxl = None
        self.track_inside = None
        self.num_track_inside_pts = None
        self.track_outside = None
        self.num_track_outside_pts = None
        self.track_length = None
        self.track_length_pxl = None
        
        self.track_limits = None
        
        self.spline_factor = None
        self.track_inside_val = None
        self.track_outside_val = None
        
        self.start_point = None
        self.start_angle = 0
        
        self.num_check_points = None
        
        self.load_track_params()
        
        self.times = []
        
        self.check_point_lines_scaled = None
        self.track_points_scaled = None
        self.track_spline_scaled = None
        self.track_inside_scaled = None
        self.track_outside_scaled = None
        self.start_finish_line_scaled = None
        
        self.use_tl_np_arr = False # convert track_limits from bitarray to np.array
        if self.use_tl_np_arr :
            self.track_limits_to_np_arr()
        
        self.max_num_laps = 1
        self.max_time_between_check_points = 15 # seconds
    
    def load_track_params(self) :
        fil = os.path.join(self.working_folder, self.name_base+'_params.pkl')
        if os.path.isfile( fil ) :
            with open( fil, 'rb' ) as fil :
                track_params = pickle.load(fil)
            
            self.window_width = track_params.window_width
            self.window_height = track_params.window_height
            self.scale_factor = track_params.scale_factor
            
            self.track_points = track_params.track_points
            self.start_finish_line = track_params.start_finish_line
            self.check_points = track_params.check_points
            self.check_point_lines = track_params.check_point_lines
            self.track_spline = track_params.track_spline
            
            self.m_per_pxl = track_params.m_per_pxl
            
            self.track_width_pxl = track_params.track_width_pxl
            self.track_outside = track_params.track_outside
            self.track_inside = track_params.track_inside
            self.track_length = track_params.track_length
            self.track_length_pxl = track_params.track_length_pxl
            self.track_limits = track_params.track_limits
            self.spline_factor = track_params.spline_factor
            
            self.start_point = self.track_spline[0]
            self.start_angle = self.three_point_angle(self.track_spline[1], self.track_spline[0], self.track_spline[0]+[5,0])
            
            self.num_track_inside_pts = np.shape( self.track_inside )[0]
            self.num_track_outside_pts = np.shape( self.track_outside )[0]
            
            self.num_check_points = len( self.check_points )
            
            max_dists = []
            for limits in [self.track_inside, self.track_outside] :
                tmp = np.array( self.track_inside )
                tmp1 = np.copy( tmp )
                val = tmp1[-1]
                tmp1[1:] = tmp[:-1]
                tmp[0] = val
                diff = tmp-tmp1
                dists = np.sqrt( np.sum(diff**2, axis=1) )
                max_dists.append( max(dists) )
        
    def track_limits_to_np_arr(self) :
        bits_list  = self.track_limits
        new_limits = []
        
        for bit_arr in bits_list :
            lst = bit_arr.tolist()
            new_limits.append( lst )
        
        self.track_limits = np.array( new_limits, dtype=np.int )
    
    def find_edge_distance(self, current_location, angle, location_is_pxl=True, max_vision_distance=None):
        """
        The function calculates the distance to and location of the intersection of a line originating at at current_location and
            travelling in the diection of angle.

        Parameters
        ----------
        current_location : list
            Contains the x & y coordinates of the point to be used.
        angle : float
            The angle to travel from current_location.
        location_is_pxl : bool, optional
            Specifies if current_location is a pixel location or position in meters from the track origin. The default is True.

        Returns
        -------
        list
            Contains a list with the x & y coordinates of the point of intersection with the track boundary and a float which is the distance
                from the current_point to the boundary point of intersection.

        """
        # current location --> a location in pxls
        # angle --> angle in radians measured ccw with 0 being vertical down according to png image
        
        if self.use_tl_np_arr :
            [x,y] = current_location
            if not location_is_pxl :
                x /= self.m_per_pxl
                y /= self.m_per_pxl
            
            [x_max, y_max] = [self.window_width, self.window_height]
                
            if int(x)<0 or int(y)<0 or int(x)>=x_max or int(y)>=y_max:
                return [current_location, 0.]
            
            if self.track_limits[ int(x), int(y) ] == 1:
                return [current_location, 0.]
            
            x_disp = np.cos(angle) # x displacement
            y_disp = np.sin(angle) # y displacement
            
            if np.abs(x_disp) > np.abs(y_disp): # normalizing x_disp & y_disp WRT the larger term
                y_disp /= np.abs(x_disp)
                x_disp = np.sign(x_disp)
            else:
                x_disp /= np.abs(y_disp)
                y_disp = np.sign(y_disp)
            
            vision_limit = np.inf
            if max_vision_distance is not None :
                vision_limit = max_vision_distance
                if location_is_pxl :
                    vision_limit /= self.m_per_pxl
            
            if x_disp > 0 :
                x_steps = ( x_max - x ) / x_disp
            else :
                x_steps = np.abs( x / x_disp )
            
            if y_disp > 0 :
                y_steps = ( y_max - y ) / y_disp
            else :
                y_steps = np.abs( y / y_disp )
            
            rng = np.arange( np.min([x_steps, y_steps]) - 1, dtype=np.int )
            
            xs = np.array( x + x_disp * rng, dtype=np.int )
            ys = np.array( y + y_disp * rng, dtype=np.int )
            
            values = self.track_limits[ xs, ys ]
            
            idx = next((i for i, j in enumerate(values) if j), -1)
            x = xs[idx]
            y = ys[idx]
            
            distance = np.sqrt( (current_location[0]-x)**2 + (current_location[1]-y)**2 )
            
            if not location_is_pxl :
                distance *= self.m_per_pxl
            
            try :
                return [[int(x), int(y)], distance]
            except :
                print()
                # print( x )
                # print( y )
                print( idx )
                print( 'error in: find_edge_distance' )
                return [[0, 0], 0]
        
        else :
            [x,y] = current_location
            if not location_is_pxl :
                x /= self.m_per_pxl
                y /= self.m_per_pxl
            
            [x_max, y_max] = [self.window_width, self.window_height]
                
            if int(x)<0 or int(y)<0 or int(x)>=x_max or int(y)>=y_max:
                return [current_location, 0.]
            
            if self.track_limits[ int(x) ][ int(y) ] == 1:
                return [current_location, 0.]
            
            x_disp = np.cos(angle) # x displacement
            y_disp = np.sin(angle) # y displacement
            
            if np.abs(x_disp) > np.abs(y_disp): # normalizing x_disp & y_disp WRT the larger term
                y_disp /= np.abs(x_disp)
                x_disp = np.sign(x_disp)
            else:
                x_disp /= np.abs(y_disp)
                y_disp = np.sign(y_disp)
            
            while True:
                x += x_disp
                y += y_disp
                
                if x<0 or y<0 or x>=x_max or y>=y_max:
                    break
                
                if self.track_limits[ int(x) ][ int(y) ] != 0:
                    break
            
            distance = np.sqrt( (current_location[0]-x)**2 + (current_location[1]-y)**2 )
            
            if not location_is_pxl :
                distance *= self.m_per_pxl
            
            return [[int(x), int(y)], distance]
    
    def crossed_line_test(self, pos_old, pos_new, line, location_is_pxl=True) :
        """
        This function determines if a line has been crossed based on two pixels locations.

        Parameters
        ----------
        pos_old : list
            Used to define the start point of the line of travel.
        pos_new : list
            Used to define the end point of the line of travel.
        line : list
            list containining two lists. The 2 sub-lists contain the pixel locations that define the start and stop
            points of the line that is being tested if it was crossed. It this is left as None it will be assumed that the line
            to be tested is the start/finish line.
        location_is_pxl : bool, optional
            Specifies if current_location is a pixel location or position in meters from the track origin. The default is True.

        Returns
        -------
        int
            1 or -1 if the line was crossed. 1 indicates the line was crossed in the positive direction, -1 if the 
                line was crossed in the negative direction.
            0 if the line was not crossed.

        """
        
        [x1, y1] = pos_old
        [x2, y2] = pos_new
        [x3, y3] = line[0]
        [x4, y4] = line[1]
        
        x21 = x2 - x1
        y21 = y2 - y1
        x43 = x4 - x3
        y43 = y4 - y3
        
        den = (-x43 * y21 + x21 * y43)
        
        if den == 0 :
            return 0
        
        s = (-y21 * (x1 - x3) + x21 * (y1 - y3)) / den
        t = ( x43 * (y1 - y3) - y43 * (x1 - x3)) / den
        
        if s>=0 and s<=1 and t>=0 and t<=1 :
            return self.line_cross_direction(pos_new, line[0], line[1])
        else :
            return 0
    
    def line_cross_direction(self, p0, p1, p2) :
        # print( 'line_cross_direction' )
        """
        This function determines the direction that a line was crossed.
        This procedure is based on a cross product. No concern has been made for squared terms or magnitudes, only the sign of the cross product.

        Parameters
        ----------
        p0 : list
            This is the point to be tested to determine what side of the line it is on.
        p1 : list
            Start pixel of the line that was crossed.
        p2 : list
            End pixel of the line that was crossed.

        Returns
        -------
        direction : int
            1 or -1 if the line was crossed. 1 indicates the line was crossed in the positive direction, -1 if the 
                line was crossed in the negative direction.
            That is, 1 if the car is going in the correct direction, -1 if the car is travelling in the wrong direction for the track.

        """
        
        u = np.array(p0) - np.array(p1)
        v = np.array(p1) - np.array(p2)
        
        v = v[::-1]
        v[0] *= -1
        
        uv_mag_sqr = np.sum(u*v)
        
        return np.sign( uv_mag_sqr )
    
    def crossed_check_point_test(self, pos_old, pos_new) :
        """
        This function tests to see if the car has crossed any of the check points based on the car travelling from it's 
            old position, pos_old, to it's new position, pos_new.

        Parameters
        ----------
        pos_old : list
            x & y coordinates the car started at.
        pos_new : list
            x & y coordinates the car ended at.

        Returns
        -------
        int
            0 - if the car did not cross a check point.
            1 - if the car crossed a check point in the positive direction.
            -1 - if the car crossed a check point in the negative direction.

        """
        for line in self.check_point_lines :
            value = self.crossed_line_test(pos_old, pos_new, line)
            if value != 0 :
                return value
        return 0
    
    def crossed_start_finish_line_test(self, pos_old, pos_new) :
        return self.crossed_line_test(pos_old, pos_new, self.start_finish_line)
    
    def three_point_angle(self, pt_1, pt_2, pt_3) :
        # print( 'three_point_angle' )
        u = np.array(pt_2) - np.array(pt_1)
        v = np.array(pt_2) - np.array(pt_3)
        
        u_dot_v = np.sum( u*v )
        u_mag = np.sqrt( np.sum( u**2 ) )
        v_mag = np.sqrt( np.sum( v**2 ) )
        
        return np.arccos( u_dot_v / (u_mag * v_mag) )
    
    def test_pxl_is_wall(self, pxl, location_is_pxl=True) :
        [x,y] = pxl
        if not location_is_pxl :
            x /= self.m_per_pxl
            y /= self.m_per_pxl
        
        x = int( x )
        y = int( y )
        
        if x<=0 or x>=self.window_width-1 :
            return 1
        if y<=0 or y>=self.window_height-1 :
            return 1
        
        return self.track_limits[x][y]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def dist_point_to_line(self, point, line_start, line_end) :
    #     # print( 'dist_point_to_line' )
    #     # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
    #     [x_0, y_0] = point
    #     [x_1, y_1] = line_start
    #     [x_2, y_2] = line_end
        
    #     num = np.abs( (y_2-y_1)*x_0 - (x_2-x_1)*y_0 + x_2*y_1 - y_2*x_1 )
    #     den = np.sqrt( (y_2-y_1)**2 + (x_2-x_1)**2 )
        
    #     return num/den

    # def dist_point_to_point(self, pt_1, pt_2, sqrd=False) :
    #     # print( 'dist_point_to_point' )
    #     pt_1 = np.array( pt_1 )
    #     pt_2 = np.array( pt_2 )
    #     if sqrd :
    #         return np.sum( (pt_1-pt_2)**2 )
    #     else :
    #         return np.sqrt( np.sum( (pt_1-pt_2)**2 ) )

    # def line_line_intersection_point(self, l_1, l_2) :
    #     # print( 'line_line_intersection_point' )
    #     """
    #     This function determines if a line has been crossed based on two pixels locations.
    #     The math used here is based on :
    #         https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Intersection_of_two_lines
        
    #     Parameters
    #     ----------
    #     l_1 : list
    #         Used to define the start and end point of the first line.
    #     l_2 : list
    #         list containining two lists. The 2 sub-lists contain the pixel locations that define the start and stop
    #         points of the line that is being tested if it was crossed.
    #     # location_is_pxl : bool, optional
    #     #     Specifies if current_location is a pixel location or position in meters from the track origin. The default is True.
        
    #     Returns
    #     -------
    #     list or None
    #         None - there was no point of intersection found along the line segments.
    #         list - 1D list of two floats indicating the point of intersection.

    #     """
    #     [x_1, y_1] = l_1[0]
    #     [x_2, y_2] = l_1[1]
    #     [x_3, y_3] = l_2[0]
    #     [x_4, y_4] = l_2[1]
        
    #     x12 = x_1 - x_2
    #     x13 = x_1 - x_3
    #     x34 = x_3 - x_4
    #     y12 = y_1 - y_2
    #     y13 = y_1 - y_3
    #     y34 = y_3 - y_4
        
    #     den = x12 * y34 - y12 * x34
        
    #     t = (x13*y34 - y13*x34) / den
    #     u = (y12*x13 - x12*y13) / den
        
    #     if 0<=t<=1 and 0<=u<=1 :
    #         p_x = x_1 - t*x12
    #         p_y = y_1 - t*y12
    #         return [p_x, p_y]
    #     else :
    #         return None
    
    # def point_is_infront(self, point, car_pos, vec_angle) :
    #     vec_angle -= np.pi/2
    #     dx = np.cos( vec_angle )
    #     dy = np.sin( vec_angle )
        
    #     line_start = np.array( car_pos )
    #     line_end = line_start + np.array([dx, dy])
        
    #     return self.line_cross_direction(point, line_start, line_end)
    
    # def obsticle_detection(self, car_pos, vec_angle, location_is_pxl=False, max_distance=None, return_selected_points=False) :
    #     # measure each point's distance away from "vision line"
        
    #     ct = current_time()
        
    #     if max_distance is None :
    #         dist_factor = np.max([self.window_width, self.window_height])
    #         max_distance = np.inf
    #         max_offset = np.inf
    #     else :
    #         dist_factor = max_distance
    #         max_distance += self.barrier_gap
    #         max_offset = max_distance
    #         max_distance = max_distance**2
        
    #     # this is used to determine the "vision line" direction
    #     dx = np.cos( vec_angle )
    #     dy = np.sin( vec_angle )
    #     if np.abs(dx) > np.abs(dy) :
    #         dy /= np.abs(dx)
    #         dx = np.sign(dx)
    #         mag = np.sqrt( dx**2 + dy**2 )
    #         dx *= dist_factor/mag
    #         dy *= dist_factor/mag
    #     else :
    #         dx /= np.abs(dy)
    #         dy = np.sign(dy)
    #         mag = np.sqrt( dx**2 + dy**2 )
    #         dx *= dist_factor/mag
    #         dy *= dist_factor/mag
    #     car_pos = np.array( car_pos )
    #     vision_line_end = car_pos + np.array([dx, dy])
        
    #     # this is used to determine what side of the "vision line" the point is on
    #     vision_side_point = car_pos + np.array([-dy, dx])
        
    #     idxs_inner = np.arange( np.shape(self.track_inside)[0] )
    #     idxs_outer = np.arange( np.shape(self.track_outside)[0] )
        
    #     # print()
    #     POINTS_USED = []
        
    #     ct = current_time()
        
    #     # if the point is close then test the line from that point to its following neighbour
    #     dists = []
    #     pts = []
    #     for idx in idxs_inner :
    #         next_idx = idx + 1
    #         if next_idx >= self.num_track_inside_pts :
    #             next_idx = 0
    #         offsets = np.abs( self.track_inside[idx] - car_pos )
    #         if offsets[0] > max_offset or offsets[1] > max_offset :
    #             continue
    #         pt = self.line_line_intersection_point([car_pos, vision_line_end], [self.track_inside[idx], self.track_inside[next_idx]])
    #         if pt is not None :
    #             dists.append( self.dist_point_to_point(car_pos, pt, sqrd=True) )
    #             pts.append( pt )
    #         POINTS_USED.append( self.track_inside[idx] )
        
    #     for idx in idxs_outer :
    #         next_idx = idx + 1
    #         if next_idx >= self.num_track_outside_pts :
    #             next_idx = 0
    #         offsets = np.abs( self.track_inside[idx] - car_pos )
    #         if offsets[0] > max_offset or offsets[1] > max_offset :
    #             continue
    #         pt = self.line_line_intersection_point([car_pos, vision_line_end], [self.track_outside[idx], self.track_outside[next_idx]])
    #         if pt is not None :
    #             dists.append( self.dist_point_to_point(car_pos, pt, sqrd=True) )
    #             pts.append( pt )
    #         POINTS_USED.append( self.track_outside[idx] )
        
    #     self.times.append( current_time()-ct )
        
    #     if return_selected_points :
    #         return POINTS_USED
        
    #     if len( dists ) == 0 : # no obsticle detected
    #         return [ vision_line_end, self.dist_point_to_point(car_pos, vision_line_end, sqrd=False) ]
    #     else :
    #         # check those results to see which is the closest and infront of the car
    #         min_dist_sqrd = min(dists)
    #         # return the point where the shortest distance occurs and the shortest distance
    #         return [ pts[dists.index(min_dist_sqrd)], np.sqrt(min_dist_sqrd) ]
    
    
    
    
    















