#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:51:16 2020

@author: Casual Endvrs
Email: casual.endvrs@gmail.com
GitHub: https://github.com/Casual-Endvrs
Reddit: CasualEndvrs
Twitter: @CasualEndvrs
"""

# add a parameter to randomly replace a gene with a completely random number that is not dependent upon and previous value.

import numpy as np
import os
import json
import pickle

class genetic_population() :
    def __init__(self, IO_shapes, num_individuals=0, AI_params_dict=None, genetic_AI_params_to_load=None, load_previous_gen=False) :
        self.IO_shapes = IO_shapes
        self.num_layers = 0
        self.num_individuals = num_individuals
        self.population = []
        self.old_population = []
        self.old_population_scores = None
        
        # individual selection parameters
        self.keep_best = 5
        self.propagate_best = 0.1
        self.minimum_score_percentage = 0.5
        
        # gene selection parameters
        self.propability_keep_f_gene = 0.95 # can be 0 --> 1 or None
        
        # population stats parameters
        self.breed_v_pop_stats_ratio = 0.5
        self.std_scaling_factor = 2
        self.use_weighted_stats = True
        
        # gene mix parameters
        self.mix_ratio_keep_f_gene = 0.9 # can be 0 --> 1
        self.mix_ratio_keep_m_gene = 1 - self.mix_ratio_keep_f_gene
        self.mix_genes_probability = 0.35
        self.mix_sum_requirement = 2*self.mix_genes_probability
        # if self.mix_genes_probability < 0.5 :
        #     self.mix_sum_requirement = 2 * ( 1 - np.sqrt( self.mix_genes_probability / 2 ) )
        # else :
        #     self.mix_sum_requirement = np.sqrt( 2 - 2 * self.mix_genes_probability )
        
        # gene mutation parameters
        self.mutation_probability_rate = 0.25
        self.mutate_uniform_gauss_ratio = 0.5
        self.mutation_max_factor_uniform_add_shift = 0.1
        self.mutation_max_factor_uniform_scaling = 0.1
        self.mutation_gauss_add_shift_sigma = 0.35
        self.mutation_gauss_scaling_sigma = 0.35
        
        self.generation = 0
        
        self.save_best_num = 5
        
        if num_individuals>0 :
            self.spawn_origin_population()
        
        self.cwd = os.getcwd()
        self.population_folder = None
        
        self.load_saved_params(params=AI_params_dict, genetic_AI_params_to_load=genetic_AI_params_to_load)
        
        if load_previous_gen :
            self.load_generation()
        
        ##### this was added
        self.set_AI_layer_sz( self.IO_shapes )
        
        self.min_score = 0
        
        if False : # used for diagnostics
            print( '\n\n\n' )
            print( 'Genetic AI - Parameters\n' )
            print( f'AI_shape: {self.IO_shapes}' )
            print( f'num_individuals: {self.num_individuals}' )
            print( f'keep_best: {self.keep_best}' )
            print( f'propagate_best: {self.propagate_best}' )
            print( f'breed_v_pop_stats_ratio: {self.breed_v_pop_stats_ratio}' )
            print( f'std_scaling_factor: {self.std_scaling_factor}' )
            print( f'propability_keep_f_gene: {self.propability_keep_f_gene}' )
            print( f'mix_ratio_keep_f_gene: {self.mix_ratio_keep_f_gene}' )
            print( f'mix_genes_probability: {self.mix_genes_probability}' )
            print( f'mutation_probability_rate: {self.mutation_probability_rate}' )
            print( f'mutate_uniform_gauss_ratio: {self.mutate_uniform_gauss_ratio}' )
            print( f'mutation_max_factor_uniform_add_shift: {self.mutation_max_factor_uniform_add_shift}' )
            print( f'mutation_max_factor_uniform_scaling: {self.mutation_max_factor_uniform_scaling}' )
            print( f'mutation_max_factor_gauss_add_shift: {self.mutation_gauss_add_shift_sigma}' )
            print( f'mutation_max_factor_gauss_scaling: {self.mutation_gauss_scaling_sigma}' )
            print( f'save_best_num: {self.save_best_num}' )
            print( f'minimum_score_percentage: {self.minimum_score_percentage}' )
            print( f'population_folder: {self.population_folder}' )
            print( f'load_previous_gen: {load_previous_gen}' )
            print( '\n\n\n' )
    
    def load_saved_params(self, params=None, genetic_AI_params_to_load=None, supress_warnings=False) :
        # 'AI_shape'
        # 'num_individuals'
        # 'keep_best'
        # 'propagate_best'
        # 'breed_v_pop_stats_ratio'
        # 'std_scaling_factor'
        # 'propability_keep_f_gene'
        # 'mix_ratio_keep_f_gene'
        # 'mix_genes_probability'
        # 'mutation_probability_rate'
        # 'mutate_uniform_gauss_ratio'
        # 'mutation_max_factor_uniform_add_shift'
        # 'mutation_max_factor_uniform_scaling'
        # 'mutation_max_factor_gauss_add_shift'
        # 'mutation_max_factor_gauss_scaling'
        # 'save_best_num'
        # 'minimum_score_percentage'
        # 'population_folder'
        
        if genetic_AI_params_to_load is None :
            genetic_AI_params_to_load = 'AI_class_genetic.json'
        
        file_params = None
        
        fil = os.path.join( self.cwd, genetic_AI_params_to_load)
        if os.path.isfile( fil ) :
            with open( fil, 'r' ) as f :
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
            if 'AI_shape' in param_keys :
                if params['AI_shape'] is not None :
                    self.set_AI_layer_sz( params['AI_shape'] )
            
            if 'num_individuals' in param_keys :
                if params['num_individuals'] is not None :
                    self.num_individuals = params['num_individuals']
            
            if 'keep_best' in param_keys :
                if params['keep_best'] is not None :
                    self.keep_best = params['keep_best']
            
            if 'propagate_best' in param_keys :
                if params['propagate_best'] is not None :
                    self.propagate_best = params['propagate_best']
            
            if 'breed_v_pop_stats_ratio' in param_keys :
                if params['breed_v_pop_stats_ratio'] is not None :
                    self.breed_v_pop_stats_ratio = params['breed_v_pop_stats_ratio']
            
            if 'std_scaling_factor' in param_keys :
                if params['std_scaling_factor'] is not None :
                    self.std_scaling_factor = params['std_scaling_factor']
            
            if 'propability_keep_f_gene' in param_keys :
                if params['propability_keep_f_gene'] is not None :
                    self.propability_keep_f_gene = params['propability_keep_f_gene']
            
            if 'mix_ratio_keep_f_gene' in param_keys :
                if params['mix_ratio_keep_f_gene'] is not None :
                    self.mix_ratio_keep_f_gene = params['mix_ratio_keep_f_gene']
                    self.mix_ratio_keep_m_gene = 1 - self.mix_ratio_keep_f_gene
            
            if 'mix_genes_probability' in param_keys :
                if params['mix_genes_probability'] is not None :
                    self.mix_genes_probability = params['mix_genes_probability']
                    self.mix_sum_requirement = 2*self.mix_genes_probability
                    # if self.mix_genes_probability < 0.5 :
                    #     self.mix_sum_requirement = 2 * ( 1 - np.sqrt( self.mix_genes_probability / 2. ) )
                    # else :
                    #     self.mix_sum_requirement = np.sqrt( 2 - 2. * self.mix_genes_probability )
            
            if 'mutation_probability_rate' in param_keys :
                if params['mutation_probability_rate'] is not None :
                    self.mutation_probability_rate = params['mutation_probability_rate']
            
            if 'mutate_uniform_gauss_ratio' in param_keys :
                if params['mutate_uniform_gauss_ratio'] is not None :
                    self.mutate_uniform_gauss_ratio = params['mutate_uniform_gauss_ratio']
            
            if 'mutation_max_factor_uniform_add_shift' in param_keys :
                if params['mutation_max_factor_uniform_add_shift'] is not None :
                    self.mutation_max_factor_uniform_add_shift = params['mutation_max_factor_uniform_add_shift']
            
            if 'mutation_max_factor_uniform_scaling' in param_keys :
                if params['mutation_max_factor_uniform_scaling'] is not None :
                    self.mutation_max_factor_uniform_scaling = params['mutation_max_factor_uniform_scaling']
            
            if 'mutation_max_factor_gauss_add_shift' in param_keys :
                if params['mutation_max_factor_gauss_add_shift'] is not None :
                    self.mutation_gauss_add_shift_sigma = params['mutation_max_factor_gauss_add_shift']
            
            if 'mutation_max_factor_gauss_scaling' in param_keys :
                if params['mutation_max_factor_gauss_scaling'] is not None :
                    self.mutation_gauss_scaling_sigma = params['mutation_max_factor_gauss_scaling']
            
            if 'save_best_num' in param_keys :
                if params['save_best_num'] is not None :
                    self.save_best_num = params['save_best_num']
            
            if 'minimum_score_percentage' in param_keys :
                if params['minimum_score_percentage'] is not None :
                    self.minimum_score_percentage = params['minimum_score_percentage']
            
            if 'population_folder' in param_keys :
                if params['population_folder'] is not None :
                    self.population_folder = os.path.join( self.cwd, 'Genetic AI Populations', params['population_folder'] )
    
    def set_AI_layer_sz(self, layer_sizes=None) :
        if layer_sizes is None :
            layer_sizes = self.IO_shapes
        
        self.shape = []
        for layer_sz in layer_sizes :
            if layer_sz == 'AI_input' :
                self.shape.append( self.IO_shapes[0] )
            elif layer_sz == 'AI_output' :
                self.shape.append( self.IO_shapes[1] )
            else :
                self.shape.append( int(layer_sz) )
        
        self.num_layers = len( self.shape ) - 1
    
    def next_generation(self, save_best=False) :
        if save_best :
            self.save_best_number()
        
        self.select_best_individuals()
        
        self.population = []
        
        if self.keep_best > 1 :
            num_to_keep = int( self.keep_best )
        else :
            num_to_keep = int( self.num_individuals * self.keep_best )
        
        if num_to_keep > len( self.old_population_scores ) :
            num_to_keep = len( self.old_population_scores )
        
        num_new_individuals = self.num_individuals-num_to_keep
        if self.breed_v_pop_stats_ratio > 1 :
            num_to_breed = int( self.breed_v_pop_stats_ratio )
            if self.breed_v_pop_stats_ratio > num_new_individuals :
                num_to_breed = num_new_individuals
            num_from_pop_stats = num_new_individuals - self.breed_v_pop_stats_ratio
        else :
            num_to_breed = int( self.breed_v_pop_stats_ratio * num_new_individuals )
            num_from_pop_stats = num_new_individuals - num_to_breed
        
        # create new individuals from pop stats
        self.create_individuals_from_pop_stats( num_from_pop_stats )
        
        # breed new individuals
        self.breed_new_generation( num_to_breed )
        
        
        
        # add previous best individuals
        self.population.extend( np.copy( self.old_population[-num_to_keep:] ).tolist() )
        
        np.random.shuffle( self.population )
        
        self.old_population = []
        self.old_population_scores = []
        
        for individual in self.population :
            
            individual.verify_layer_weights_in_limits()
    
    def breed_new_generation(self, num_to_breed) :
        potential_breeders = len(self.old_population_scores)
        
        for i in np.arange( num_to_breed ) :
            [f_num, m_num] = np.random.choice(potential_breeders, size=2, replace=False, p=self.old_population_scores)
            
            f = self.old_population[ f_num ]
            m = self.old_population[ m_num ]
            
            new_individual = self.breed_2_individuals(m, f)
            new_individual = self.mutate_individual( new_individual )
            
            self.population.append( new_individual )
    
    def create_individuals_from_pop_stats(self, num_new_from_pop_stats) :
        if num_new_from_pop_stats > 0 :
            num_layers = len( self.old_population[0].hidden_layers )
            num_old_population = len( self.old_population )
            
            new_population = [ self.spawn_single_individual() for i in np.arange(num_new_from_pop_stats) ]
            
            for layer_num in np.arange( num_layers ) :
                layer_weights = []
                for individual_num in np.arange( num_old_population ) :
                    layer_weights.append( self.old_population[individual_num].hidden_layers[layer_num].weights )
                if self.use_weighted_stats :
                    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
                    mns = np.average( layer_weights, weights=self.old_population_scores, axis=0 )
                    stds = np.array( [ layer_weights[i]-mns for i in np.arange(np.shape(layer_weights)[0])] )
                    stds = stds**2
                    stds = np.array( [ stds[i]*self.old_population_scores[i] for i in np.arange(np.shape(stds)[0]) ] )
                    factor = np.sum( self.old_population_scores ) / self.std_scaling_factor # This can be simplified since sum(o_p_scores)=1
                                # self.std_scaling_factor term added to artifically affect pop-stats method
                                # set self.std_scaling_factor = 1 to use normal weighted std
                    stds = np.sum( stds, axis=0 ) / factor
                    # stds = stds * self.num_individuals / (self.num_individuals-1) # ignored as this effect is <=1% for self.num_individuals >100
                    stds = np.sqrt( stds )
                else :
                    mns = np.mean( layer_weights, axis=0 )
                    stds = np.std( layer_weights, axis=0 ) * self.std_scaling_factor
                
                for individual in new_population :
                    individual.hidden_layers[layer_num].weights = np.random.normal( loc=mns, scale=stds )
                    individual = self.mutate_individual( individual )
            
            self.population.extend( new_population )
    
    def select_best_individuals(self) :
        if self.propagate_best >= 1 :
            keep_num_individuals = self.propagate_best
        else :
            keep_num_individuals = int( self.num_individuals * self.propagate_best )
        
        scores = []
        for individual in self.population :
            scores.append( individual.score )
        
        sorted_indices = sorted( np.arange(len(scores)), key=lambda x: scores[x] )
        
        population = np.array( self.population )[sorted_indices]
        scores = np.array( scores )[sorted_indices] # / np.max(scores)
        
        minimum_score = self.minimum_score_percentage * scores[-1]
        
        self.min_score = minimum_score
        
        viable_eles = scores >= minimum_score
        num_viable_eles = np.sum( viable_eles )
        if keep_num_individuals > num_viable_eles :
            keep_num_individuals = num_viable_eles
        
        self.old_population = population[-keep_num_individuals:]
        self.old_population_scores = scores[-keep_num_individuals:]
        
        self.old_population_scores /= np.sum(self.old_population_scores) # normalize scores such that their sum = 1
    
    def spawn_origin_population(self) :
        self.population = []
        for i in np.arange( self.num_individuals ) :
            self.population.append( self.spawn_single_individual() )
    
    def spawn_single_individual(self) :
        return genetic_individual( self.IO_shapes )
    
    def add_new_individual_to_population(self) :
        individual = self.spawn_single_individual()
        self.population.append( individual )
        self.num_individuals = len( self.population )
        return individual
    
    def breed_2_individuals(self, m, f) :
        for layer_num in np.arange( self.num_layers ) :
            child = self.spawn_single_individual()
            
            layer_shape = list( np.shape( child.hidden_layers[layer_num] ) )
            
            # get gene selection probabilities
            m_prob = np.random.uniform(0, 1, layer_shape )
            f_prob = np.random.uniform(0, 1, layer_shape )
            
            # select genes
            if self.propability_keep_f_gene is None :
                m_genes = np.array( m_prob >= f_prob )
                f_genes = np.array( f_prob > m_prob )
            else :
                m_genes = np.array( m_prob > self.propability_keep_f_gene )
                f_genes = np.array( np.logical_not( m_genes ) )
            
            # set selected genes
            child.hidden_layers[layer_num].weights[m_genes] = m.hidden_layers[layer_num].weights[m_genes]
            child.hidden_layers[layer_num].weights[f_genes] = f.hidden_layers[layer_num].weights[f_genes]
            
            # select mixing genes
            mix_prob = m_prob + f_prob
            mix_genes = mix_prob > self.mix_sum_requirement
            
            
            # get mixing genes
            m_genes[mix_genes] = False
            f_genes[mix_genes] = False
            
            # mix and set genes
            mixed_genes = self.mix_ratio_keep_m_gene * m.hidden_layers[layer_num].weights[mix_genes] + self.mix_ratio_keep_f_gene * f.hidden_layers[layer_num].weights[mix_genes]
            child.hidden_layers[layer_num].weights[mix_genes] = mixed_genes
        
        return child
    
    def mutate_individual(self, individual) :
        for layer_num in np.arange( self.num_layers ) :
            layer_shape = list( np.shape( individual.hidden_layers[layer_num].weights ) )
            
            mutation_probabilities = np.random.uniform(0, 1, layer_shape )
            
            mutate_genes = np.array( mutation_probabilities < self.mutation_probability_rate )
            
            num_mutating_genes = int( np.sum( mutate_genes ) )
            
            ratio_prop = np.random.uniform()
            mutation_terms_scaling = np.ones( num_mutating_genes )
            mutation_terms_addition = np.zeros( num_mutating_genes )
            if ratio_prop < self.mutate_uniform_gauss_ratio :
                if self.mutation_max_factor_uniform_scaling is not None :
                    mutation_terms_scaling *= np.random.uniform(1-self.mutation_max_factor_uniform_scaling, 1+self.mutation_max_factor_uniform_scaling, size=num_mutating_genes)
                if self.mutation_max_factor_uniform_add_shift is not None :
                    mutation_terms_addition += self.mutation_max_factor_uniform_add_shift * np.random.uniform( -1, 1, size=num_mutating_genes )
            else :
                if self.mutation_gauss_scaling_sigma is not None :
                    mutation_terms_scaling *= np.random.normal(loc=1, scale=self.mutation_gauss_scaling_sigma, size=num_mutating_genes)
                if self.mutation_gauss_add_shift_sigma is not None :
                    mutation_terms_addition += np.random.normal( loc=0, scale=self.mutation_gauss_add_shift_sigma, size=num_mutating_genes )
            
            individual.hidden_layers[layer_num].weights[mutate_genes] *= mutation_terms_scaling
            individual.hidden_layers[layer_num].weights[mutate_genes] += mutation_terms_addition
        
        return individual
    
    def load_generation(self, prep_next_gen=False) :
        if self.population_folder is not None :
            if os.path.isdir( self.population_folder ) :
                for root, dirs, fils in os.walk( self.population_folder ) :
                    break
                
                if len(fils) > 0 :
                    current_gen_file = sorted( fils, key=lambda x: int( x.split(' ')[-1].split('.')[0] ) )[-1]
                    self.generation = int( current_gen_file.split(' ')[-1].split('.')[0] )
                    current_gen_file = os.path.join( self.population_folder, current_gen_file )
                    
                    with open( current_gen_file, 'rb' ) as f :
                        saved_pop = pickle.load( f )
                    
                    if prep_next_gen :
                        self.generation += 1
                        saved_pop = np.array( saved_pop )
                        scores = []
                        for individual in saved_pop :
                            scores.append( individual.score )
                        scores = np.array( scores )
                        
                        sorted_indices = sorted( np.arange(len(scores)), key=lambda x: scores[x] )
                        
                        self.old_population = saved_pop[ sorted_indices ]
                        self.old_population_scores = scores[ sorted_indices ]
                        
                        self.old_population_scores /= np.sum(self.old_population_scores)
                        
                        self.next_generation( save_best=False )
                    else :
                        self.population = list( saved_pop )
    
    def save_best_number(self) :
        if self.population_folder is not None :
            if not os.path.isdir( self.population_folder ) :
                os.makedirs( self.population_folder )
            
            fil = 'Gen ' + str( self.generation ) + '.pkl'
            
            if self.save_best_num >= 1 :
                keep_num_individuals = self.save_best_num
            else :
                keep_num_individuals = int( self.num_individuals * self.save_best_num )
            
            scores = []
            for individual in self.population :
                scores.append( individual.score )
            
            population = np.array( self.population )
            
            sorted_indices = sorted( np.arange(len(scores)), key=lambda x: scores[x] )
            
            population_to_save = population[sorted_indices][-keep_num_individuals:]
            
            with open( os.path.join(self.population_folder, fil), 'wb' ) as f :
                pickle.dump( population_to_save, f )

class genetic_individual() :
    def __init__(self, shape) :
        self.shape = shape
        self.hidden_layers = []
        self.score = 0
        
        self.create_hiddel_layers()
    
    def create_hiddel_layers(self) :
        for i in np.arange( len(self.shape)-1 ) :
            num_inputs = self.shape[i]
            num_nodes = self.shape[i+1]
            new_layer = layer(num_nodes, num_inputs)
            self.hidden_layers.append( new_layer )
        self.hidden_layers[-1].activation_fn = self.hidden_layers[-1].tanh
    
    def run_AI(self, inputs) :
        inputs = np.array( inputs )
        
        for layer in self.hidden_layers :
            inputs = layer.evaluate_layer( inputs )
        return inputs
    
    def verify_layer_weights_in_limits(self) :
        for i in np.arange( len(self.shape)-1 ) :
            self.hidden_layers[i].verify_weights_in_limits()

class layer() :
    # https://cdn-images-1.medium.com/max/1600/1*p_hyqAtyI8pbt2kEl6siOQ.png
    def __init__(self, num_nodes, num_inputs, initial_weights=None, activation_fn=None) :
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.weights = initial_weights
        self.values = np.zeros( self.num_nodes )
        
        self.activation_fn = activation_fn
        if self.activation_fn is None :
            self.activation_fn = self.sigmoid
        
        self.elu_alpha = 1
        
        self.max_weight_val = 7.5
        
        if self.weights is None :
            self.randomize_weights()
    
    def verify_weights_in_limits(self) :
        idxs = np.where( np.abs(self.weights) > self.max_weight_val )
        self.weights[idxs] /= np.sign(self.weights[idxs])*self.max_weight_val/2
    
    def randomize_weights(self) :
        self.weights = np.random.uniform(-5, 5, [self.num_nodes, self.num_inputs+1])
    
    def evaluate_layer(self, inputs) :
        inputs = list( inputs )
        inputs.append( 1 ) # used for the bias
        inputs = np.array( inputs )
        self.values = self.weights.dot( inputs )
        self.activation_fn()
        return self.values
    
    def add_weight(self, w) :
        self.weights.append( w )
    
    # activation functions
    def linear(self) :
        pass
    
    def sigmoid(self) :
        self.values = 1. / (1. + np.exp(-1. * self.values))
    
    def tanh(self) :
        self.values = np.tanh(self.values)
    
    def relu(self) :
        truth_arr = self.values < 0
        self.values[truth_arr] = 0
    
    def leaky_relu(self) :
        truth_arr = self.values < 0
        self.values[truth_arr] *= 0.1
    
    def elu(self) :
        truth_arr = self.values < 0
        self.values[truth_arr] = self.elu_alpha * ( np.exp(self.values[truth_arr]) - 1 )
    
    # activation function derivatives
    def linear_deriv(self) :
        return 1
    
    def sigmoid_deriv(self, value) :
        f_x = 1. / (1. + np.exp(-1. * value))
        return f_x * (1 - f_x)
    
    def tanh_deriv(self, value) :
        return 1 - np.tanh(value)**2
    
    def relu_deriv(self, value) :
        if value < 0 :
            return 0
        else :
            return 1
    
    def leaky_relu_deriv(self, value) :
        if value < 0 :
            return 0.1
        else :
            return 1
    
    def elu_deriv(self, value) :
        if value < 0 :
            return self.elu_alpha * ( np.exp(value) - 1 ) + self.elu_alpha
        else :
            return 1


def create_default_params() :
    # create json preferences file for AI_class_genetic.py
    preferences = {
        'AI_shape': ['AI_input', 'AI_output'], # shape of the AI
        'keep_best': 0.1, # number/ratio to carry over between generations
        'propagate_best': 0.5, # number/ratio to propagate
        'minimum_score_percentage': 0.1, # minimum score, percent of maximum score, that can breed/gen stats
        
        'breed_v_pop_stats_ratio': 0.5, # number/ratio to breed, remained from pop stats
        'propability_keep_f_gene': 0.8, # probability to keep f gene instead of taking m gene
        'mix_ratio_keep_f_gene': 0.8, # ratio of f gene to keep during mixing
        'mix_genes_probability': 0.25, # probability to mix genes
        
        'mutation_probability_rate': 0.01, # probability of mutation
        'mutate_uniform_gauss_ratio': 0.5, # probability to use uniform distribution for mutations
        'mutation_max_factor_uniform_add_shift': 0.25, # maximum addition mutation factor for uniform
        'mutation_max_factor_uniform_scaling': 0.25, # maximum scaling mutation factor for uniform
        'mutation_max_factor_gauss_add_shift': 0.25, # maximum addition mutation factor for guassian
        'mutation_max_factor_gauss_scaling': 0.25, # maximum scaling mutation factor for guassian
        
        'std_scaling_factor': 1,
        
        'save_best_num': 0.35, # number/ratio of best individuals to save from generation
        }
    
    with open( os.path.join(os.getcwd(), 'AI_class_genetic.json'), 'w' ) as fil :
        json.dump( preferences, fil, indent=4 )








if __name__ == '__main__' :
    # AI = genetic_population([3,4])
    
    create_default_params()



