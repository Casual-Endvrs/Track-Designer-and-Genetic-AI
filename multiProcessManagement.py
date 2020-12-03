#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:55:27 2019

@author: Casual Endvrs
Email: casual.endvrs@gmail.com
GitHub: https://github.com/Casual-Endvrs
Reddit: CasualEndvrs
Twitter: @CasualEndvrs
"""

import multiprocessing
from types import GeneratorType

import numpy as np

def multiProcessManagement(fn, process_args, max_processes=2, daemonic=True, sort=False, fn_result_q=None, reporting=False):
    """
    This function allows any other function to be easily multi-threaded. If there is concern 
    about memory usage by the final results from each thread it is recommended to save the 
    results to disk using the Python module 'pickle.' Then the file name for each result can 
    be returned such that they can be easily found.

    Parameters
    ----------
    fn : function
        This is the function that is to be multi-threaded.
        Functions can be modified to be used here by:
            1) adding a keyword 'result_q' to the function definition.
            2) replacing the return statement with a result_q.put() statement.
                ex. return result --> result_q.put( result )
        Functions that do not have a return statement must still have a result_q.put( result ) statement. 
            result can be True to indicate that the function ran correctly or simply None. If this is the 
            case, the returned list from this function can simplely be ignored.
    
    process_args : list of lists, list of dictionaries or generator which returns a dictionary or 1D list
        If process_args contains a lists then all keywords for fn must be provided
            in the list with the only un-provided keyword being 'result_q' which must 
            be the last keyword in fn's definition.
        process_args can contain dictionaries as well. In this case every keyword
            argument does not beed to be specified. However, fn must have a keyword
            argument 'result_q' which is not provided in the dictionary or it will be
            over-written within this function.
        If a generator is used it should return a simple dictionary with keys for fn's
            keyword arguments or a dictionary with values for each keyword of fn (with the
            exception of the keyword 'result_q').
    
    max_processes : int, optional - Default is 2.
        This is the maximum number of threads that are to be used at any given time. 
        It is recommended that this value not exceed the total number of threads available 
        to the computer this will run on.
    
    daemonic : bool, optional - Default is True.
        Sets the value daemon for each thread upon its initialization, see Python's 
            multiprocessing module documentation for more. It is recommended to 
            leave this as True, otherwise thread termination will not occur properly
            in the case that the thread that this function is run in is closed pre-maturely.
    
    sort : bool, optional - Default is False.
        Currently unused. Will be used to determine if the results should be 
            sorted into the same order as the arguments of fn were provided.
    
    fn_result_q : multi-threading Queue - Default is None
        If a Queue is provided the results of the function will be put into the Queue.
        Else, the results will be given using return.
        If this is given then the function will force daemonic to False, else the function will fail.

    Returns
    -------
    results : list
        Returns a list containing all the results obtained from the function fn.

    """
    
    if fn_result_q is not None :
        daemonic = False
    
    if isinstance(process_args, GeneratorType) :
        use_generator = True
        processes_total = 2
        # This is set to 2 and advanced with each added process.
        # This ensures the end conditions are not meet.
        # processes_total is decrimented upon the final thread causing end conditions to be meet.
        p_args = next( process_args )
        if isinstance(p_args, list) :
            args_list_type = True
        else :
            args_list_type = False
    elif isinstance( process_args, (list, dict) ) :
        use_generator = False
        p_args = process_args[0]
        processes_total = len(process_args)
    else :
        print( 'process_args must be a list of (lists/dictionaries) or a generator function which provides a list or dictionary. Anything else will cause errors.' )
        print( 'process_args type: %s' %type(process_args) )
        return None
    
    if isinstance(p_args, list) :
        args_list_type = True
    elif isinstance(p_args, (dict, GeneratorType) ) :
        args_list_type = False
    else :
        print( 'p_args must be a list or dictionary of the required values for fn. Anything else will cause errors.' )
        print( 'p_args type: %s' %type(p_args) )
        return None
    
    if(max_processes<0):
        max_processes = 1
    process = []
    
    result_q = multiprocessing.Queue()
    process_number = 0
    processes_running = 0
    processes_complete = 0
    multi_running = True
    adding = True
    results = []

    while( multi_running ):
        while( adding ):
            if( processes_running == max_processes ):
                break
            elif( process_number == processes_total ):
                adding = False
                break
            
            if process_number != 0 :
                if use_generator :
                    p_args = next( process_args, None )
                    if p_args is None :
                        adding = False
                        processes_total -= 1
                        break
                    else :
                        processes_total += 1
                else :
                    p_args = process_args[process_number]
            
            if sort :
                if args_list_type :
                    p_args.append( process_number )
                else :
                    p_args['itr'] = process_number
            
            if args_list_type :
                p_args.append( result_q )
            else :
                p_args['result_q'] = result_q
            
            if args_list_type :
                process.append(multiprocessing.Process(target=fn, args=(p_args)))
            else :
                process.append(multiprocessing.Process(target=fn, kwargs=(p_args)))
            
            process[process_number].daemon = daemonic
            process[process_number].start()
            process_number += 1
            processes_running += 1
        if( processes_complete == processes_total ):
            multi_running = False
            break

        try:
            result = result_q.get(timeout=1)
        except:
            pass
        else:
            if fn_result_q is not None :
                fn_result_q.put( result )
            else :
                results.append(result)
            processes_complete += 1
            processes_running -= 1
    
    if sort :
        pass # sort results here
    
    if fn_result_q is None :
        return results
    # else :
    #     fn_result_q.put( results )










if __name__ == '__main__' :
    import numpy as np
    from random import random
    import time
    
    def test_fn(a, b, result_q) :
        time.sleep( 3*random() )
        result_q.put( a+b )
    
    
    
    if False : # run using a generator
        def arg_gen() :
            args = []
            a = np.arange( 10 )
            args = [ a, a+5 ]
            args = np.transpose( args )
            args = args.tolist()
            for arg_set in args :
                yield arg_set
                # yield {'a':arg_set[0], 'b':arg_set[1]}
        
        args = arg_gen()
    else : # run with list of lists
            args = []
            a = np.arange( 10 )
            args = [ a, a+5 ]
            args = np.transpose( args )
            args = args.tolist()
            
            # args_list = args
            # args = []
            # for st in args_list :
            #     args.append( {} )
            #     args[-1]['a'] = st[0]
            #     args[-1]['b'] = st[1]
    
    
    results = multiProcessManagement(test_fn, args, max_processes=2, daemonic=False, sort=False)
    print( results )

















