#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:28:32 2020

@author: Casual Endvrs
Email: casual.endvrs@gmail.com
GitHub: https://github.com/Casual-Endvrs
Reddit: CasualEndvrs
Twitter: @CasualEndvrs
"""

from trackDesigner_class import create_default_params as tdc_defaults
from single_player import create_default_params as sp_defauts
from AI_drivers import create_default_params as AId_defaults
from car_class import create_default_params as new_car
from AI_class_genetic import create_default_params as AIc_defaults
from track_environment import create_default_params as TE_defaults

if __name__ == '__main__' :
    tdc_defaults()
    sp_defauts()
    #AId_defaults()
    new_car()
    AIc_defaults()
    TE_defaults()

