from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import os
import sys
import numpy as np
import json

from track_environment import track_environment

class MainWindow(QMainWindow) :
    def __init__(self) :
        super().__init__()
        self.main_folder = os.getcwd()
        self.AI_save_folder = None
        self.car_folder = None
        self.track_file = None
        self.track_image = None
        self.car_file = None
        self.AI_file = None
        self.game_control_file = None
        self.load_prev_gen = False
        
        self.FPS = 30
        
        self.display_width = 1750
        self.display_height = 950
        
        self.num_training_threads = 4
        self.num_cars_per_thread = 15
        self.num_individual_AIs = 500
        self.max_num_drivers_on_track = 15
        
        self.num_vision_rays = 5
        
        self.training_mode = 1
        self.kinematic_updates_per_frame = 5
        self.vision_ray_num_history = 0
        
        self.AI_shape = []
        self.AI_hidden_layers = []
        
        self.keep_best = 0.1
        self.propagate_best = 0.5
        self.minimum_score_percentage = 0.1
        
        self.breed_v_pop_stats_ratio = 0.5
        
        self.probability_keep_f_gene = 0.8
        self.mix_genes_probability = 0.25
        self.mix_ratio_keep_f_gene = 0.8
        
        self.mutation_probability_rate = 0.01
        self.mutate_uniform_gauss_ratio = 0.5
        
        self.mutation_max_factor_uniform_add_shift = 0.25
        self.mutation_max_factor_uniform_scaling = 0.25
        self.mutation_max_factor_gauss_add_shift = 0.25
        self.mutation_max_factor_gauss_scaling = 0.25
        
        self.std_scaling_factor = 1
        
        self.save_best_num = 0.35
        
        self.AI_inputs = None
        self.AI_outputs = 2
        
        title = 'Genetic AI - Race Track'
        self.setWindowTitle(title)
        
        self.main_layout = QVBoxLayout()
        
        left = 0
        top = 0
        width = 1275
        height = 700
        self.setGeometry(left, top, width, height)
        
        self.intro_widget = intro_page(self)
        self.game_params_widget = game_params_page(self)
        self.AI_params_widget = AI_params_page(self)
        self.training_tab = training_page(self)
        
        #self.intro_tab = QScrollArea()
        #self.intro_tab.setWidgetResizable(True)
        #self.intro_tab.setWidget(self.intro_widget)
        
        self.game_params_tab = QScrollArea()
        self.game_params_tab.setWidgetResizable(True)
        self.game_params_tab.setWidget(self.game_params_widget)
        
        self.AI_params_tab = QScrollArea()
        self.AI_params_tab.setWidgetResizable(True)
        self.AI_params_tab.setWidget(self.AI_params_widget)
        
        self.main_tabs = QTabWidget()
        
        self.main_tabs.addTab(self.intro_widget, "Introduction")
        self.main_tabs.addTab(self.game_params_tab, "Game Parameters")
        self.main_tabs.addTab(self.AI_params_tab, "AI Parameters")
        self.main_tabs.addTab(self.training_tab, "AI Training")
        self.main_tabs.setTabEnabled(3, False)
        
        self.main_layout.addWidget(self.main_tabs)
        self.setCentralWidget(self.main_tabs)
        
        self.show()
        
        self.update_AI_shape()
    
    def lock_settings_input(self) :
        self.game_params_widget.lock_inputs()
        self.AI_params_widget.lock_inputs()
    
    def begin_training(self) :
        self.lock_settings_input()
        self.training_tab.create_training_layout()
        self.main_tabs.setTabEnabled(3, True)
        self.main_tabs.setCurrentIndex(3)
    
    def update_AI_shape(self) :
        self.AI_inputs = self.num_vision_rays * (self.vision_ray_num_history+1) + 1
        
        self.AI_shape = [self.AI_inputs]
        
        for num in self.AI_hidden_layers :
            self.AI_shape.append(num)
        
        self.AI_shape.append(self.AI_outputs)
        self.AI_params_widget.update_AI_shape_statement()
    
    def get_AI_params_dict(self) :
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
        
        self.update_AI_shape()
        
        AI_params_dict = {}
        AI_params_dict['AI_shape'] = self.AI_shape
        AI_params_dict['num_individuals'] = self.num_individual_AIs
        AI_params_dict['keep_best'] = self.keep_best
        AI_params_dict['propagate_best'] = self.propagate_best
        AI_params_dict['breed_v_pop_stats_ratio'] = self.breed_v_pop_stats_ratio
        AI_params_dict['std_scaling_factor'] = self.std_scaling_factor
        AI_params_dict['probability_keep_f_gene'] = self.probability_keep_f_gene
        AI_params_dict['mix_ratio_keep_f_gene'] = self.mix_ratio_keep_f_gene
        AI_params_dict['mix_genes_probability'] = self.mix_genes_probability
        AI_params_dict['mutation_probability_rate'] = self.mutation_probability_rate
        AI_params_dict['mutate_uniform_gauss_ratio'] = self.mutate_uniform_gauss_ratio
        AI_params_dict['mutation_max_factor_uniform_add_shift'] = self.mutation_max_factor_uniform_add_shift
        AI_params_dict['mutation_max_factor_uniform_scaling'] = self.mutation_max_factor_uniform_scaling
        AI_params_dict['mutation_max_factor_gauss_add_shift'] = self.mutation_max_factor_gauss_add_shift
        AI_params_dict['mutation_max_factor_gauss_scaling'] = self.mutation_max_factor_gauss_scaling
        AI_params_dict['save_best_num'] = self.save_best_num
        AI_params_dict['minimum_score_percentage'] = self.minimum_score_percentage
        AI_params_dict['population_folder'] = self.AI_save_folder
        
        AI_params_dict['kinematic_updates_per_frame'] = self.kinematic_updates_per_frame
        
        return AI_params_dict
    
    def prep_AI_shape_to_parameter(self) :
        self.update_AI_shape()
        
        AI_shape_parameter = ['AI_input']
        
        num_hidden_layers = len(self.AI_shape)-2
        for i in np.arange(1, num_hidden_layers+1) :
            AI_shape_parameter.append(self.AI_shape[i])
        
        AI_shape_parameter.append('AI_output')
        self.AI_shape = AI_shape_parameter
    
    def get_game_control_params(self) :
        #'track_file',
        #'display_max_resolution',
        #'car_file',
        #'genetic_AI_params_to_load',
        #'track_params_to_load',
        #'car_to_load',
        #'total_num_drivers',
        #'max_drivers_on_track',
        #'FPS',
        #'kinematic_updates_per_frame',
        #'number_vision_rays',
        #'vision_ray_num_history',
        #'training_mode',
        #'maximum_concurrent_training_threads',
        #'maximum_number_cars_per_training_thread',
        #'AI_shape',
        #'display_width',
        #'display_height',
        
        gc_dict = {}
        gc_dict['track_file'] = self.track_file
        gc_dict['display_max_resolution'] = [self.display_width, self.display_height]
        gc_dict['car_file'] = self.car_file
        gc_dict['genetic_AI_params_to_load'] = None
        gc_dict['track_params_to_load'] = None
        gc_dict['car_to_load'] = None
        gc_dict['total_num_drivers'] = self.num_individual_AIs
        gc_dict['max_drivers_on_track'] = self.max_num_drivers_on_track
        gc_dict['FPS'] = self.FPS
        gc_dict['kinematic_updates_per_frame'] = self.kinematic_updates_per_frame
        gc_dict['number_vision_rays'] = self.num_vision_rays
        gc_dict['vision_ray_num_history'] = self.vision_ray_num_history
        gc_dict['training_mode'] = self.training_mode
        gc_dict['maximum_concurrent_training_threads'] = self.num_training_threads
        gc_dict['maximum_number_cars_per_training_thread'] = self.num_cars_per_thread
        gc_dict['AI_shape'] = self.AI_shape
        gc_dict['display_width'] = self.display_width
        gc_dict['display_height'] = self.display_height
        
        return gc_dict

class intro_page(QWidget) :
    def __init__(self, parent) :
        super(QWidget, self).__init__(parent)
        max_text_width = 150
        max_widget_width = 150
        max_input_width = 150
        min_input_width = 100
        
        self.parent = parent
        
        self.layout = QVBoxLayout(self)
        
        
        intro_instructions = '\n'.join([
            'Exiting Procedure during training',
            '--------------------------------------------------------',
            'It is important to not close the program while multi-threaded training '
            'is running in the background. If, during training, "training in background: True" '
            'is displayed on the screen, please select "Pause After This Generation". This will '
            'pause training after the current generation has finished training and has been saved. '
            'The next generation will not start making it save to exit the program.',
            '     - This is only a concern if you are using training mode 1 or 2. No multi-threaded '
            'training occurs during training mode 0.',
            '',
            'Selecting a Main Folder',
            '--------------------------------------------------------',
            'The main folder is where you should store the important files and folders that are '
            'required for training. If you use the pre-defined naming skeme the files and folders '
            'will be found and loaded for you. The pre-defined names are:',
            '     - "AI save folder" - The folder where AIs can be saved or loaded from.',
            '     - "cars" - The folder where the images for the cars are stored.',
            '     - "track_params.pkl" and "track_image.pkl" - Files that store the track information '
            'needed by the simulation and the track image that will be displayed.'
            '',
            'Setting Parameters',
            '--------------------------------------------------------',
            'There are several settings for both the game control and the training procedure. A basic '
            'description for each one is provided below the setting. Some of these settings require '
            'some extra information and they are discussed below.',
            'Most of the parameters do not need to be changed and training will run using default values. '
            'The parameters that must be set are: Main Folder, AI Save Folder, Car Images Folder and, Track File.',
            'The default values that are provided have been chosen because they are examples of values that work, '
            'not because they have been found to perform good results. This program is intented to allow users to '
            'toy around with a genetic algorithm and learn.',
            'There are also parameter files that can be edited and loaded allowing user to create '
            'settings that they prefer and quickly apply them.',
            'A new set of parameter files can be created by:',
            '     1) Open a terminal,',
            '     2) Navigator to the folder where you want to create the files,',
            '     3) Run the program "create_all_defaults.py".',
            '',
            'Training Modes',
            '--------------------------------------------------------',
            'There are 3 different training modes that affect the speed of training. These are:',
            '     0) All cars will be displayed on screen. This is the slowest method of training. '
            'This will show the user all cars so they can see how the entire generation is performing.',
            '     1) Some cars will be displayed, most will be trained in the background using multi-threading. '
            'The displayed cars will provide an indication to the performance of the generation, while '
            'minimizing the total training time.',
            '     2) All cars are trained in the background using multi-threading. The only display updates '
            'provided to the user are via the status text, in the upper left corner, and the generation '
            'training plots, in the lower left corner. This method is the fastest but there may not '
            'be a significant improvement over training mode 1.',
            '',
            'Important Training Terms',
            '--------------------------------------------------------',
            'Elite Selection (Keep Best):',
            '     - The best performing individuals will be carried over to the next generation. '
            'This helps to ensure that each subsequent generation will have a best performing '
            'individual whose score is atleast as good as the previous generation.',
            '',
            'Propagate Best:',
            '     - This sets the number of individuals that will be selected from each generation to create '
            'the next generation. For individuals to be considered they must have a score that is greater '
            'that the minimum score.',
            '',
            'Minimum Score Percent:',
            '     - This sets the minimum score necessary for an individual to contribute to the next generation, '
            'calculated as a given percentage of the maximum score achieved by an individual in that generation. '
            'This ensures that individuals who perform poorly are not allowed to negatively impact future generations.',
            '',
            'There are two possible paths for the creation of new individuals.',
            '--------------------------------------------------------',
            'Breeding:',
            '     - This method take two individuals and uses a collection of their genes to '
            'create a new individual. It is possible that the genes from one individual will '
            'be used and the other is selected. The user can specify the probability that '
            'the genes of one individual, female, is selected over the other\'s, male, genes. '
            'This can provide a preference to create individuals that are more alike one of their parents.'
            'This bias can be removed by setting "Probability to keep Female Gene" to 0.5.',
            '     - Genes can also be created by creating a mixture, or average, of the parental genes. '
            'This is controlled by the parameter "Probability to Mix Genes".'
            'The user can using "Female Gene Averaging Weight" to give a preference to one parent '
            'over the other. A simple average can be used by setting this to 0.5.',
            'Population statistics:',
            '     - This methood considers the generation as a whole as opposed to 2 randomly selected '
            'individuals. The average and standard deviation is calculated for a single gene over '
            'the entire population. New individuals are then created by selecting a random number '
            'using a normal distribution where the mean value is the distribution\'s central value '
            'and the standard deviation is the distribution\'s standard deviation. '
            'The user can set "Population Statistics Standard Deviation Scaling Factor" to '
            'increase/decresse the standard deviation by a multiplicative factor. This can help '
            'to increase/decrease the variation of the individuals created using this method.',
            '',
            'Gene Mutation',
            '--------------------------------------------------------',
            'Gene mutation can occur for all new individuals, regardless if they were created using '
            'breeding or populations statistics. The propbability that in individual gene will be '
            'mutated is set my the parameter "Probability for Gene Mutation". The probability of '
            'using one mutation method over the other is set my "Probability to use Uniform Distribution '
            'over normal Distribtuion for Mutations". The difference between the two methods is the '
            'distribution used to select the random mutationing factors. For both mutation methods '
            'the gene will first be shift, by addition, and then scaled, by multiplication.',
            'Uniform Distribtuion:',
            '     - Random numbers are selected using a uniform distribution. A maximum shift factor '
            'can be specified by the user ("Uniform Distribution Max Factor - Shift"), lets call this '
            'max_shift. The random number that be used will be between +/- max_shift. The maximum '
            'scaling factor ("Uniform Distribution Max Factor - Scaling") can be set my the user, lets '
            'call this max_scaling. The random number used to scale a gene will then be 1 +/- max_scaling.',
            'Normal Distribution:',
            '     - Random numbers are selected using a normal distribution. The random number used for '
            'scaling in this case will be selected using a normal distribution centered about 0 '
            'and using a standard deviation of set by "Normal (Gaussian) Distribution Max Factor - Shift". '
            'The scaling factor is found using a normal distribution centered about 1 with a standard '
            'deviation set by "Normal (Gaussian) Distribution Max Factor - Scaling".',
            '',
            'Status Text',
            '--------------------------------------------------------',
            'There are several different pieces of information provided to the user via the status text.',
            '     - Gen - current generation number.',
            '     - drivers left - number of drivers left that will be displayed on screen.',
            '     - on track - number of drivers that are on track.',
            '     - drivers done - the total number of drivers who have finished. This includes both those '
            'displayed on screen as well as those trained using multi-threading.',
            '     - threads remaining: this is the number of batches of drivers that need to be trained.',
            '     - training in background - indicated when training is running in the background (True), '
            'or there is no multi-threaded training running (False).',
            '     - training mode - shows the current training mode. An indicator is added when the training '
            'mode is switched. Changing the training mode will only occur at the start of training the '
            'next generation.',
            '     - score rng - displays the maximum individual score, mean generation score, minimum individual '
            'score for the previous generation.',
            '',
            'Score Plots',
            '--------------------------------------------------------',
            'A plot is provided at the bottom left of the screen to provide information generation scores. '
            'Note that a logarithm of the scores has been taken before the plots are created. This helps '
            'to ensure that the distribution of low scores are represented well. All of these plots are '
            'intended to be used to visualize training progress and not provide a numerical analysis.',
            '     - Red plot - maximum individual score for the last few generations.',
            '     - Black plot - mean generation score for the last few generations.',
            '     - Blue plot - minimum individual score for the last few generations.',
            '     - Green histogram - shows the distribution of score for the last generation. Low scores on '
            'the right, high scores to the left.',
            '     - Purple vertical line - indicates the mean score for the previous generation.'
            ])
        intro_text = QPlainTextEdit(readOnly=True, plainText=intro_instructions)
        intro_text.backgroundVisible = False
        intro_text.wordWrapMode = True
        #intro_text.setMinimumWidth(1250)
        intro_text.zoomIn(2)
        self.layout.addWidget(intro_text)
        
        """
        intro_instructions = '\n'.join([
            'Exiting Procedure during training',
            '--------------------------------------------------------',
            'It is important to not close the program while multi-threaded training '
            'is running in the background. If, during training, "training in background: True" '
            'is displayed on the screen, please select "Pause After This Generation". This will '
            'pause training after the current generation has finished training and has been saved. '
            'The next generation will not start making it save to exit the program.',
            '\tThis is only a concern if you are using training mode 1 or 2. No multi-threaded '
            'training occurs during training mode 0.',
            '',
            'Training Modes:',
            ''
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.wordWrapMode = True
        intro_text.setMaximumWidth(500)
        """
        self.layout.addWidget(intro_text)
        
        self.setLayout(self.layout)

class game_params_page(QWidget) :
    def __init__(self, parent) :
        super(QWidget, self).__init__(parent)
        max_text_width = 300
        max_input_width = 200
        min_input_width = 125
        
        self.parent = parent
        self.layout = QGridLayout(self)
        
        # create left control panel
        self.left_layout = QGridLayout()
        
        left_row = 0
        
        ##### Label indicating mandatory inputs
        label = QLabel('Mandatory Values - Folders and Track File Must Be Set')
        self.left_layout.addWidget(label, left_row, 0, 1, 3); left_row += 1
        
        # Insert horizontal seperation bar
        self.left_layout.addWidget(QHLine(), left_row, 0, 1, 3); left_row += 1
        
        ##### Select the folder that contains the parameters
        label = QLabel('Main Folder: ')
        label.setMaximumWidth(max_text_width)
        self.btn_main_data_folder = QPushButton("Select Folder")
        self.btn_main_data_folder.setMaximumWidth(max_input_width)
        self.btn_main_data_folder.clicked.connect(self.get_working_folder)
        self.left_layout.addWidget(label, left_row, 0)
        self.left_layout.addWidget(self.btn_main_data_folder, left_row, 1); left_row += 1
        
        label = QLabel('Current Folder: ')
        label.setMaximumWidth(max_text_width)
        self.lbl_main_folder = QLabel(self.parent.main_folder)
        self.left_layout.addWidget(label, left_row, 0)
        self.left_layout.addWidget(self.lbl_main_folder, left_row, 1, 1, 2); left_row += 1
        
        intro_instructions = '\n'.join([
            'This is the main folder where the other folders are likely to be found. '
            'If the folders "AI save folder" and "cars" and the file "track_params.pkl" '
            'are in this folder they will automatically be loaded, assuming these values '
            'not already been added.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.left_layout.addWidget(intro_text, left_row, 0, 1, 3); left_row += 1
        
        # Insert horizontal seperation bar
        self.left_layout.addWidget(QHLine(), left_row, 0, 1, 3); left_row += 1
        
        ##### Select folder to save AIs - AI_save_folder
        label = QLabel('AI Save Folder: ')
        label.setMaximumWidth(max_text_width)
        self.btn_AI_save_folder = QPushButton("Select AI Save Folder")
        self.btn_AI_save_folder.setMaximumWidth(max_input_width)
        self.btn_AI_save_folder.clicked.connect(self.get_AI_save_folder)
        self.left_layout.addWidget(label, left_row, 0)
        self.left_layout.addWidget(self.btn_AI_save_folder, left_row, 1); left_row += 1
        label = QLabel('Load Previous Generation Data: ')
        label.setMaximumWidth(max_text_width)
        self.input_load_prev_gen = QComboBox()
        self.input_load_prev_gen.addItem("False")
        self.input_load_prev_gen.addItem("True")
        self.update_load_prev_gen()
        self.input_load_prev_gen.currentIndexChanged.connect(self.update_load_prev_gen)
        self.left_layout.addWidget(label, left_row, 0)
        self.left_layout.addWidget(self.input_load_prev_gen, left_row, 1); left_row += 1
        label = QLabel('Current AI Save Folder: ')
        label.setMaximumWidth(max_text_width)
        self.lbl_AI_save_folder = QLabel('None')
        self.left_layout.addWidget(label, left_row, 0)
        self.left_layout.addWidget(self.lbl_AI_save_folder, left_row, 1, 1, 2); left_row += 1
        
        intro_instructions = '\n'.join([
            'Folder where AIs will be saved and can be loaded from. '
            'Set "Load Previous Generation Data" to True to load data from this folder.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.left_layout.addWidget(intro_text, left_row, 0, 1, 3); left_row += 1
        
        # Insert horizontal seperation bar
        self.left_layout.addWidget(QHLine(), left_row, 0, 1, 3); left_row += 1
        
        ##### Select the car images folfer to load
        label = QLabel('Car Images Folder: ')
        label.setMaximumWidth(max_text_width)
        self.btn_car_images_folder = QPushButton("Select Car Images Folder")
        self.btn_car_images_folder.setMaximumWidth(max_input_width)
        self.btn_car_images_folder.clicked.connect(self.get_car_folder)
        self.left_layout.addWidget(label, left_row, 0)
        self.left_layout.addWidget(self.btn_car_images_folder, left_row, 1); left_row += 1
        
        label = QLabel('Current Car Images Folder: ')
        label.setMaximumWidth(max_text_width)
        self.lbl_car_folder = QLabel( 'None' )
        self.left_layout.addWidget(label, left_row, 0)
        self.left_layout.addWidget(self.lbl_car_folder, left_row, 1, 1, 2); left_row += 1
        
        intro_instructions = '\n'.join([
            'Folder that contains the images for the cars that will be displayed on track.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.left_layout.addWidget(intro_text, left_row, 0, 1, 3); left_row += 1
        
        # Insert horizontal seperation bar
        self.left_layout.addWidget(QHLine(), left_row, 0, 1, 3); left_row += 1
        
        ##### Select the track file to load
        label = QLabel('Track File - *.pkl: ')
        label.setMaximumWidth(max_text_width)
        self.btn_track_file = QPushButton("Select Track File")
        self.btn_track_file.setMaximumWidth(max_input_width)
        self.btn_track_file.clicked.connect(self.get_track_file)
        self.left_layout.addWidget(label, left_row, 0)
        self.left_layout.addWidget(self.btn_track_file, left_row, 1); left_row += 1
        
        label = QLabel('Current Track File: ')
        label.setMaximumWidth(max_text_width)
        self.lbl_track_file = QLabel( 'None' )
        self.left_layout.addWidget(label, left_row, 0)
        self.left_layout.addWidget(self.lbl_track_file, left_row, 1, 1, 2); left_row += 1
        
        intro_instructions = '\n'.join([
            'This specifies the track parameters to be loaded. The file should be a *.pkl '
            'file and have a corresponding *.png file. For example, if the name base for the '
            'track file name is "track", then the parameter file should be "track_params.pkl" '
            'and the image file should be called "track_image.png".\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.left_layout.addWidget(intro_text, left_row, 0, 1, 3); left_row += 1
        
        # Insert horizontal seperation bar
        self.left_layout.addWidget(QHLine(), left_row, 0, 1, 3); left_row += 1
        label = QLabel('Non-Mandatory Values - Default Values Can Be Used')
        self.left_layout.addWidget(label, left_row, 0, 1, 3); left_row += 1

        # Insert horizontal seperation bar
        self.left_layout.addWidget(QHLine(), left_row, 0, 1, 3); left_row += 1
        
        ##### Select the game parameters file to load
        label = QLabel('Note: Loading parameters here will overwrite previous values')
        self.left_layout.addWidget(label, left_row, 0, 1, 3); left_row += 1
        label = QLabel('Game Parameters File - *.json: ')
        label.setMaximumWidth(max_text_width)
        self.btn_game_control_file = QPushButton("Select Game Parameters File")
        self.btn_game_control_file.setMaximumWidth(max_input_width)
        self.btn_game_control_file.clicked.connect(self.get_game_params_file)
        self.left_layout.addWidget(label, left_row, 0)
        self.btn_load_game_control_file = QPushButton("Load Game Parameters File")
        self.btn_load_game_control_file.setMaximumWidth(max_input_width+25)
        self.btn_load_game_control_file.clicked.connect(self.load_game_params_file)
        self.left_layout.addWidget(self.btn_game_control_file, left_row, 1)
        self.left_layout.addWidget(self.btn_load_game_control_file, left_row, 2); left_row += 1
        
        label = QLabel('Current Game Parameters File: ')
        label.setMaximumWidth(max_text_width)
        self.lbl_game_control_file = QLabel( 'None - Default will be used' )
        self.left_layout.addWidget(label, left_row, 0)
        self.left_layout.addWidget(self.lbl_game_control_file, left_row, 1, 1, 2); left_row += 1
        
        intro_instructions = '\n'.join([
            'Load a file which outlines the default parameters you wish to use. '
            'Files for the Track, Car and AI can all be specified in this file. '
            'Loading this file will replace Track, Car and AI files if they are specified.\n'
            'Note: The file to use for AI defaults will be selected but not loaded yet.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.left_layout.addWidget(intro_text, left_row, 0, 1, 3); left_row += 1
        
        # Insert horizontal seperation bar
        self.left_layout.addWidget(QHLine(), left_row, 0, 1, 3); left_row += 1
        
        ##### Select the car file to load
        label = QLabel('Car File - *.json: ')
        label.setMaximumWidth(max_text_width)
        self.btn_car_file = QPushButton("Select Car File")
        self.btn_car_file.setMaximumWidth(max_input_width)
        self.btn_car_file.clicked.connect(self.get_car_file)
        self.left_layout.addWidget(label, left_row, 0)
        self.left_layout.addWidget(self.btn_car_file, left_row, 1); left_row += 1
        
        label = QLabel('Current Car File: ')
        label.setMaximumWidth(max_text_width)
        self.lbl_car_file = QLabel( 'None - Default will be used' )
        self.left_layout.addWidget(label, left_row, 0)
        self.left_layout.addWidget(self.lbl_car_file, left_row, 1, 1, 2); left_row += 1
        
        intro_instructions = '\n'.join([
            'File that defines the parameters for the car to be used.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.left_layout.addWidget(intro_text, left_row, 0, 1, 3); left_row += 1
        
        # Insert horizontal seperation bar
        self.left_layout.addWidget(QHLine(), left_row, 0, 1, 3); left_row += 1
        
        ##### Select the AI file to load
        label = QLabel('AI File - *.json: ')
        label.setMaximumWidth(max_text_width)
        self.btn_AI_file = QPushButton("Select AI File")
        self.btn_AI_file.setMaximumWidth(max_input_width)
        self.btn_AI_file.clicked.connect(self.get_AI_file)
        self.left_layout.addWidget(label, left_row, 0)
        self.btn_load_AI_file = QPushButton("Load AI File - Overwrites AI Params")
        self.btn_load_AI_file.setMaximumWidth(max_input_width+25)
        self.btn_load_AI_file.clicked.connect(self.load_AI_params)
        self.left_layout.addWidget(self.btn_AI_file, left_row, 1)
        self.left_layout.addWidget(self.btn_load_AI_file, left_row, 2); left_row += 1
        
        label = QLabel('Current AI File: ')
        label.setMaximumWidth(max_text_width)
        self.lbl_AI_file = QLabel( 'None - Default will be used' )
        self.left_layout.addWidget(label, left_row, 0)
        self.left_layout.addWidget(self.lbl_AI_file, left_row, 1, 1, 2); left_row += 1
        
        intro_instructions = '\n'.join([
            'File that contains the default parameters to use.\n'
            'Note: you must select the file and select "Load AI File" to update values.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.left_layout.addWidget(intro_text, left_row, 0, 1, 3); left_row += 1
        
        self.left_layout.setRowStretch(left_row, 1)
        
        # add left control layer and insert seperation bar
        self.layout.addLayout(self.left_layout, 0, 1)
        self.layout.addWidget(QVLine(), 0, 2, 1, 1)
        
        # create right control panel
        self.right_layout = QGridLayout()
        right_row = 0
        
        ##### Label indicating mandatory inputs
        label = QLabel('Non-Mandatory Values - Default Values Can Be Used')
        self.right_layout.addWidget(label, right_row, 0, 1, 2); right_row += 1
        
        # Insert horizontal seperation bar
        self.right_layout.addWidget(QHLine(), right_row, 0, 1, 2); right_row += 1
        
        ##### Set the width and height of the display screen
        self.lbl_current_resolution = QLabel( f'Set the Game Display Resolution: {self.parent.display_width} x {self.parent.display_height}' )
        self.btn_resolution = QPushButton("Update Resolution")
        self.btn_resolution.setMaximumWidth(max_input_width)
        self.btn_resolution.clicked.connect(self.update_display_resolution)
        self.right_layout.addWidget(self.lbl_current_resolution, right_row, 0)
        self.right_layout.addWidget(self.btn_resolution, right_row, 1); right_row += 1
        
        resolution_layout = QHBoxLayout()
        label = QLabel('Width:')
        self.input_display_width = QLineEdit()
        self.input_display_width.setMaximumWidth(int(max_input_width/2))
        self.input_display_width.setMinimumWidth(min_input_width)
        resolution_layout.addWidget(label)
        resolution_layout.addWidget(self.input_display_width)
        label = QLabel('Height:')
        self.input_display_height = QLineEdit()
        self.input_display_height.setMaximumWidth(int(max_input_width/2))
        self.input_display_height.setMinimumWidth(min_input_width)
        resolution_layout.addWidget(label)
        resolution_layout.addWidget(self.input_display_height)
        self.right_layout.addLayout(resolution_layout, right_row, 0, 1, 2); right_row += 1
        
        intro_instructions = '\n'.join([
            'Specify the resolution to use when displaying the track during training. '
            'These are maximum values and the track display will be scaled to fit these '
            'values while maintaining the track\'s original aspect ratio.\n'
            'Note: The game window will not automatically resize to fit thise resolution, '
            'you will likely need to expand the window size during training.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.right_layout.addWidget(intro_text, right_row, 0, 1, 2); right_row += 1
        
        # Insert horizontal seperation bar
        self.right_layout.addWidget(QHLine(), right_row, 0, 1, 2); right_row += 1
        
        ##### Set current training mode
        label = QLabel('Current Training Mode: ')
        label.setMaximumWidth(max_text_width)
        self.input_training_mode = QComboBox()
        self.input_training_mode.addItem("0 - Display All Cars")
        self.input_training_mode.addItem("1 - Display Some")
        self.input_training_mode.addItem("2 - Train All in Background")
        self.input_training_mode.setCurrentIndex(self.parent.training_mode)
        self.input_training_mode.setMaximumWidth(max_input_width)
        self.input_training_mode.setMinimumWidth(min_input_width)
        self.input_training_mode.currentIndexChanged.connect(self.update_training_mode)
        self.right_layout.addWidget(label, right_row, 0)
        self.right_layout.addWidget(self.input_training_mode, right_row, 1); right_row += 1
        
        self.lbl_training_mode = QLabel( str(self.parent.training_mode) )
        self.update_training_mode()
        self.right_layout.addWidget(self.lbl_training_mode, right_row, 0); right_row += 1
        
        intro_instructions = '\n'.join([
            'Training mode determines how many cars will be shown during training versus '
            'how many will be trained in the background. The options are:\n'
            '\t1) Display All Cars - All of the cars will be displayed on screen. '
            'The number of cars on screen at one time will be limited.\n'
            '\t2) Display Some - A small group of cars will be displayed on screen. '
            'This will hopefully provide a decent representation of the generation. '
            'All of the other cars will be trained in the background\n'
            '\t3) Train All in Background - All cars will be trained in the background. '
            'No cars will be shown but you will see updates in the on-screen stats.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.right_layout.addWidget(intro_text, right_row, 0, 1, 2); right_row += 1
        
        # Insert horizontal seperation bar
        self.right_layout.addWidget(QHLine(), right_row, 0, 1, 2); right_row += 1
        
        ##### Set max number of concurrent training threads
        label = QLabel('Max Concurrent Training Threads: ')
        label.setMaximumWidth(max_text_width)
        self.input_num_threads = QLineEdit()
        self.input_num_threads.setMaximumWidth(max_input_width)
        self.input_num_threads.setMinimumWidth(min_input_width)
        self.right_layout.addWidget(label, right_row, 0)
        self.right_layout.addWidget(self.input_num_threads, right_row, 1); right_row += 1
        
        self.lbl_num_training_threads = QLabel( f'Current Maximum Number of Threads: {self.parent.num_training_threads}' )
        self.btn_training_threads = QPushButton( 'Update' )
        self.btn_training_threads.setMaximumWidth(max_input_width)
        self.btn_training_threads.clicked.connect(self.update_num_training_threads)
        self.right_layout.addWidget(self.lbl_num_training_threads, right_row, 0)
        self.right_layout.addWidget(self.btn_training_threads, right_row, 1); right_row += 1
        
        intro_instructions = '\n'.join([
            'The Max Concurrent Training Threads specifies the maximum number of '
            'threads that can be used at one time during training. For the fastest '
            'training it is recommended to set this to the number of physical cores you have. '
            'For example, if you have a 6 core, 12 thread machine, set this to 6.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.right_layout.addWidget(intro_text, right_row, 0, 1, 2); right_row += 1
        
        # Insert horizontal seperation bar
        self.right_layout.addWidget(QHLine(), right_row, 0, 1, 2); right_row += 1
        
        ##### Set number of cars per training thread
        label = QLabel('Number of Cars/Thread: ')
        label.setMaximumWidth(max_text_width)
        self.input_num_cars_per_thread = QLineEdit()
        self.input_num_cars_per_thread.setMaximumWidth(max_input_width)
        self.input_num_cars_per_thread.setMinimumWidth(min_input_width)
        self.right_layout.addWidget(label, right_row, 0)
        self.right_layout.addWidget(self.input_num_cars_per_thread, right_row, 1); right_row += 1
        
        self.lbl_input_num_cars_per_thread = QLabel( f'Current Number of Cars/Thread: {self.parent.num_cars_per_thread}' )
        self.btn_num_cars_per_thread = QPushButton( 'Update' )
        self.btn_num_cars_per_thread.setMaximumWidth(max_input_width)
        self.btn_num_cars_per_thread.clicked.connect(self.update_num_cars_per_thread)
        self.right_layout.addWidget(self.lbl_input_num_cars_per_thread, right_row, 0)
        self.right_layout.addWidget(self.btn_num_cars_per_thread, right_row, 1); right_row += 1
        
        intro_instructions = '\n'.join([
            'Number of Cars/Thread specifies the number of cars to be trained in one '
            'thread when trained in the background. It is recommended to leave this at 10-15.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.right_layout.addWidget(intro_text, right_row, 0, 1, 2); right_row += 1
        
        # Insert horizontal seperation bar
        self.right_layout.addWidget(QHLine(), right_row, 0, 1, 2); right_row += 1
        
        ##### Set total number of individual AIs
        label = QLabel('Total Number of AIs: ')
        label.setMaximumWidth(max_text_width)
        self.input_num_AIs = QLineEdit()
        self.input_num_AIs.setMaximumWidth(max_input_width)
        self.input_num_AIs.setMinimumWidth(min_input_width)
        self.right_layout.addWidget(label, right_row, 0)
        self.right_layout.addWidget(self.input_num_AIs, right_row, 1); right_row += 1
        
        self.lbl_num_AIs = QLabel( f'Current Number of Individuals: {self.parent.num_individual_AIs}' )
        self.btn_num_AIs = QPushButton( 'Update' )
        self.btn_num_AIs.setMaximumWidth(max_input_width)
        self.btn_num_AIs.clicked.connect(self.update_num_individual_AIs)
        self.right_layout.addWidget(self.lbl_num_AIs, right_row, 0)
        self.right_layout.addWidget(self.btn_num_AIs, right_row, 1); right_row += 1
        
        intro_instructions = '\n'.join([
            'Total number of AIs specifies the number of AIs that exist in each generation. '
            'Increasing this number can help to increase results at the cost of increasing '
            'training time.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.right_layout.addWidget(intro_text, right_row, 0, 1, 2); right_row += 1
        
        # Insert horizontal seperation bar
        self.right_layout.addWidget(QHLine(), right_row, 0, 1, 2); right_row += 1
        
        ##### Set number of AIs to be displayed track at once
        label = QLabel('Max Number of AIs on Track: ')
        label.setMaximumWidth(max_text_width)
        self.input_num_AIs_on_track = QLineEdit()
        self.input_num_AIs_on_track.setMaximumWidth(max_input_width)
        self.input_num_AIs_on_track.setMinimumWidth(min_input_width)
        self.right_layout.addWidget(label, right_row, 0)
        self.right_layout.addWidget(self.input_num_AIs_on_track, right_row, 1); right_row += 1
        
        self.lbl_num_AIs_on_track = QLabel( f'Current Max Number of Drivers on Track: {self.parent.max_num_drivers_on_track}' )
        self.btn_num_AIs_on_track = QPushButton( 'Update' )
        self.btn_num_AIs_on_track.setMaximumWidth(max_input_width)
        self.btn_num_AIs_on_track.clicked.connect(self.update_num_AIs_on_track)
        self.right_layout.addWidget(self.lbl_num_AIs_on_track, right_row, 0)
        self.right_layout.addWidget(self.btn_num_AIs_on_track, right_row, 1); right_row += 1
        
        intro_instructions = '\n'.join([
            'Max Number of AIs on Track specifies the number of AIs that will be displayed '
            'on screen during training. Lower this value if the display is choppy during training.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.right_layout.addWidget(intro_text, right_row, 0, 1, 2); right_row += 1
        
        # Insert horizontal seperation bar
        self.right_layout.addWidget(QHLine(), right_row, 0, 1, 2); right_row += 1
        
        ##### Set number of AIs to be displayed track at once
        label = QLabel('Number of Kinematic Updates per Frame: ')
        label.setMaximumWidth(max_text_width)
        self.input_num_updates_per_frame = QLineEdit()
        self.input_num_updates_per_frame.setMaximumWidth(max_input_width)
        self.input_num_updates_per_frame.setMinimumWidth(min_input_width)
        self.right_layout.addWidget(label, right_row, 0)
        self.right_layout.addWidget(self.input_num_updates_per_frame, right_row, 1); right_row += 1
        
        self.lbl_num_updates_per_frame = QLabel( f'Number Kinematic Updates/Frame: {self.parent.kinematic_updates_per_frame}' )
        self.btn_num_updates_per_frame = QPushButton( 'Update' )
        self.btn_num_updates_per_frame.setMaximumWidth(max_input_width)
        self.btn_num_updates_per_frame.clicked.connect(self.update_num_updates_per_frame)
        self.right_layout.addWidget(self.lbl_num_updates_per_frame, right_row, 0)
        self.right_layout.addWidget(self.btn_num_updates_per_frame, right_row, 1); right_row += 1
        
        intro_instructions = '\n'.join([
            'During each frame the position of the cars and the AI\'s response will be '
            'updated multiple times. This value specifies the number of times/frame '
            'these updates occur. It is recommended to leave this value at 5.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.right_layout.addWidget(intro_text, right_row, 0, 1, 2); right_row += 1
        
        # Insert horizontal seperation bar
        self.right_layout.addWidget(QHLine(), right_row, 0, 1, 2); right_row += 1
        
        ##### Set total number of individual AIs
        label = QLabel('Number of Vision Rays: ')
        label.setMaximumWidth(max_text_width)
        self.input_num_vision_rays = QLineEdit()
        self.input_num_vision_rays.setMaximumWidth(max_input_width)
        self.input_num_vision_rays.setMinimumWidth(min_input_width)
        self.right_layout.addWidget(label, right_row, 0)
        self.right_layout.addWidget(self.input_num_vision_rays, right_row, 1); right_row += 1
        
        self.lbl_num_vision_rays = QLabel( f'Current Number of Vision Rays: {self.parent.num_vision_rays}' )
        self.btn_num_vision_rays = QPushButton( 'Update' )
        self.btn_num_vision_rays.setMaximumWidth(max_input_width)
        self.btn_num_vision_rays.clicked.connect(self.update_num_vision_rays)
        self.right_layout.addWidget(self.lbl_num_vision_rays, right_row, 0)
        self.right_layout.addWidget(self.btn_num_vision_rays, right_row, 1); right_row += 1
        
        intro_instructions = '\n'.join([
            'The AI gets data based on vision rays. These vision rays point around the front '
            'of the car starting from directly out the left side, travelling around the front '
            'of the car to directly out the right side of the car. They are all evenly spaced.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.right_layout.addWidget(intro_text, right_row, 0, 1, 2); right_row += 1
        
        # Insert horizontal seperation bar
        self.right_layout.addWidget(QHLine(), right_row, 0, 1, 2); right_row += 1
        
        ##### Set total number of individual AIs
        label = QLabel('Number of Previous Vision Rays: ')
        label.setMaximumWidth(max_text_width)
        self.input_num_previous_vision_rays = QLineEdit()
        self.input_num_previous_vision_rays.setMaximumWidth(max_input_width)
        self.input_num_previous_vision_rays.setMinimumWidth(min_input_width)
        self.right_layout.addWidget(label, right_row, 0)
        self.right_layout.addWidget(self.input_num_previous_vision_rays, right_row, 1); right_row += 1
        
        self.lbl_num_previous_vision_rays = QLabel( f'Current Number of Previous Vision Rays: {self.parent.vision_ray_num_history}' )
        self.btn_num_previous_vision_rays = QPushButton( 'Update' )
        self.btn_num_previous_vision_rays.setMaximumWidth(max_input_width)
        self.btn_num_previous_vision_rays.clicked.connect(self.update_num_previous_vision_rays)
        self.right_layout.addWidget(self.lbl_num_previous_vision_rays, right_row, 0)
        self.right_layout.addWidget(self.btn_num_previous_vision_rays, right_row, 1); right_row += 1
        
        intro_instructions = '\n'.join([
            'A number of previous vision lines sets from previous time steps can be saved and passed to the AI. This '
            'is a method to give the AI historical information about where it was and how '
            'its position on track is changing. It is recommended to leave this at 0 unless '
            'you intend to use an AI with a couple hidden layers and a larger population size. '
            'This adds a lot of extra information for the AI to process and will increase '
            'training time.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        self.right_layout.addWidget(intro_text, right_row, 0, 1, 2); right_row += 1
        
        self.right_layout.setRowStretch(right_row, 1)
        
        # add right control layer
        self.layout.addLayout(self.right_layout, 0, 3)
        
        self.layout.addWidget(QHLine(), 1, 1, 1, 3)
        
        start_training_layout = QGridLayout()
        stl_row = 0
        
        # section to stop accepting options and pass training params
        label = QLabel('Do not use this section unless you have finished entering '
            'settings for the game and AI settings. This lock all the settings such '
            'that they cannot be changed again and begins the training process.')
        label.setWordWrap(True)
        label.setFont(QFont('Arial', 15))
        start_training_layout.addWidget(label, stl_row, 0, 1, 2); stl_row += 1
        
        #label = QLabel('Save Settings and begin Training')
        self.btn_save_settings_begin_training = QPushButton( 'Save Settings and begin Training' )
        self.btn_save_settings_begin_training.clicked.connect(self.save_settings_begin_training)
        #start_training_layout.addWidget(label, stl_row, 0)
        start_training_layout.addWidget(self.btn_save_settings_begin_training, stl_row, 1); stl_row += 1
        
        self.layout.addLayout(start_training_layout, 2, 1, 1, 3)
        
        self.layout.setRowStretch(3, 1)
        
        self.setLayout(self.layout)
        
        self.get_working_folder( self.parent.main_folder )
    
    def get_working_folder(self, folder=None) :
        if folder is None or isinstance(folder, bool) :
            folder = self.parent.main_folder
            folder = QFileDialog.getExistingDirectory(self, "Select Main Working Directory", folder)
        
        if folder != '' :
            self.parent.main_folder = folder
            self.lbl_main_folder.setText(folder)
            
            for root, dirs, fils in os.walk(folder) :
                break
            
            if self.parent.AI_save_folder is None and 'AI save folder' in dirs :
                self.get_AI_save_folder( os.path.join(folder, 'AI save folder') )
            if self.parent.car_folder is None and 'cars' in dirs :
                self.get_car_folder( os.path.join(folder, 'cars') )
            if self.parent.track_file is None and 'track_params.pkl' in fils :
                self.get_track_file( os.path.join(folder, 'track_params.pkl') )
    
    def get_AI_save_folder(self, folder=None) :
        if folder is None or isinstance(folder, bool) :
            if self.parent.AI_save_folder is None :
                folder = self.parent.main_folder
            else :
                folder = self.parent.AI_save_folder
            folder = QFileDialog.getExistingDirectory(self, "Select AI Save Directory", folder)
        
        if folder != '' :
            self.parent.AI_save_folder = folder
            self.lbl_AI_save_folder.setText(folder)
    
    def update_load_prev_gen(self) :
        idx = self.input_load_prev_gen.currentIndex()
        
        if idx == 0 :
            self.parent.load_prev_gen = False
        elif idx == 1 :
            self.parent.load_prev_gen = True
    
    def get_car_folder(self, folder=None) :
        if folder is None or isinstance(folder, bool) :
            folder = self.parent.main_folder
            folder = QFileDialog.getExistingDirectory(self, "Select Car Folder", folder)
        
        if folder != '' :
            self.parent.car_folder = folder
            self.lbl_car_folder.setText(folder)
    
    def get_track_file(self, file=None) :
        if file is None or isinstance(file, bool) :
            folder = self.parent.main_folder
            file = QFileDialog.getOpenFileName(self, "Select Track File", folder, "pkl(*.pkl)")[0]
        
        if file != '' :
            if file.split('.')[-1] != 'pkl' :
                title = "File Type Error"
                warning_msg = "Track Parameters file must be a *.pkl file type."
                warning_window = warningWindow(self)
                warning_window.build_window(title=title, msg=warning_msg)
                return None
            
            (dir, fil) = os.path.split(file)
            fil = fil.split('_')[0] + '_image.png'
            pic_file = os.path.join(dir, fil)
            
            if os.path.exists(pic_file) :
                self.parent.track_image = pic_file
                self.parent.track_file = file
                self.lbl_track_file.setText(os.path.split(file)[1])
            else :
                title = "Missing Track Image File"
                warning_msg = "The track image file for this track parameter file is missing.\n"
                warning_msg += 'If the parameter file name base is "track" (track_params.pkl),\n'
                warning_msg += '\tmake there the image is titled "track_image.png". \n'
                warning_msg += 'Track parameter files cannot be used if they do not have a matching image file.'
                warning_window = warningWindow(self)
                warning_window.build_window(title=title, msg=warning_msg)
    
    def get_car_file(self, file=None) :
        if not isinstance(file, str) :
            folder = self.parent.main_folder
            file = QFileDialog.getOpenFileName(self, "Select Car File", folder, "json(*.json)")[0]
        
        if file != '' :
            if file.split('.')[-1] != 'json' :
                title = "File Type Error"
                warning_msg = "Car Parameters file must be a *.json file type."
                warning_window = warningWindow(self)
                warning_window.build_window(title=title, msg=warning_msg)
                return None
            
            self.parent.car_file = file
            self.lbl_car_file.setText(os.path.split(file)[1])
    
    def get_AI_file(self, file=None) :
        if not isinstance(file, str) :
            folder = self.parent.main_folder
            file = QFileDialog.getOpenFileName(self, "Select AI File", folder, "json(*.json)")[0]
        
        if file != '' :
            if file.split('.')[-1] != 'json' :
                title = "File Type Error"
                warning_msg = "AI Parameters file must be a *.json file type."
                warning_window = warningWindow(self)
                warning_window.build_window(title=title, msg=warning_msg)
                return None
            
            self.parent.AI_file = file
            self.lbl_AI_file.setText(os.path.split(file)[1])
    
    def get_game_params_file(self) :
        folder = self.parent.main_folder
        file = QFileDialog.getOpenFileName(self, "Select Game Parameters File", folder, "json(*.json)")[0]
        
        if file != '' :
            if file.split('.')[-1] != 'json' :
                title = "File Type Error"
                warning_msg = "Game Parameters file must be a *.json file type."
                warning_window = warningWindow(self)
                warning_window.build_window(title=title, msg=warning_msg)
                return None
            
            self.parent.game_control_file = file
            self.lbl_game_control_file.setText(os.path.split(file)[1])
    
    def update_display_resolution(self) :
        width = self.input_display_width.text()
        height = self.input_display_height.text()
        
        try :
            width = int( width )
            height = int( height )
        except :
            title = "Value Entry Error"
            warning_msg = "Display resolution must be integer values."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.display_width = width
            self.parent.display_height = height
            self.lbl_current_resolution.setText( f'Set the Game Display Resolution: {self.parent.display_width} x {self.parent.display_height}' )
    
    def update_training_mode(self) :
        idx = self.input_training_mode.currentIndex()
        
        if idx in [0,1,2] :
            self.parent.training_mode = idx
            if idx == 0 :
                txt = '0 - Display all cars'
            elif idx == 1 :
                txt = '1 - Display some cars, train the rest in the background.'
            else :
                txt = '2 - Train all cars in the background.'
            self.lbl_training_mode.setText( txt )
    
    def update_num_training_threads(self) :
        num_threads = self.input_num_threads.text()
        
        try :
            num_threads = int( num_threads )
            if num_threads < 1 :
                raise ValueError('Invalid Entry')
        except :
            title = "Value Entry Error"
            warning_msg = "Number of threads must be an integer value greater than 0."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.num_training_threads = num_threads
            self.lbl_num_training_threads.setText( f'Current Maximum Number of Threads: {self.parent.num_training_threads}' )
    
    def update_num_cars_per_thread(self) :
        num_cars_per_thread = self.input_num_cars_per_thread.text()
        
        try :
            num_cars_per_thread = int( num_cars_per_thread )
            if num_cars_per_thread < 1 :
                raise ValueError('Invalid Entry')
        except :
            title = "Value Entry Error"
            warning_msg = "Number of cars per thread must be an integer value greater than 0."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.num_cars_per_thread = num_cars_per_thread
            self.lbl_input_num_cars_per_thread.setText( f'Current Number of Cars/Thread: {self.parent.num_cars_per_thread}' )
    
    def update_num_individual_AIs(self) :
        num_AIs = self.input_num_AIs.text()
        
        try :
            num_AIs = int( num_AIs )
            if num_AIs < 1 :
                raise ValueError('Invalid Entry')
        except :
            title = "Value Entry Error"
            warning_msg = "Number of individuals must be an integer value greater than 0."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.num_individual_AIs = num_AIs
            self.lbl_num_AIs.setText( f'Current Number of Individuals: {self.parent.num_individual_AIs}' )
    
    def update_num_AIs_on_track(self) :
        num_AIs_on_track = self.input_num_AIs_on_track.text()
        
        try :
            num_AIs_on_track = int( num_AIs_on_track )
            if num_AIs_on_track < 1 :
                raise ValueError('Invalid Entry')
        except :
            title = "Value Entry Error"
            warning_msg = "Number of drivers on track must be an integer value greater than 0."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.max_num_drivers_on_track = num_AIs_on_track
            self.lbl_num_AIs_on_track.setText( f'Current Max Number of Drivers on Track: {self.parent.max_num_drivers_on_track}' )
    
    def update_num_updates_per_frame(self) :
        num_updates_per_frame = self.input_num_updates_per_frame.text()
        
        try :
            num_updates_per_frame = int( num_updates_per_frame )
            if num_updates_per_frame < 1 :
                raise ValueError('Invalid Entry')
        except :
            title = "Value Entry Error"
            warning_msg = "Number of updates per frame must be an integer value greater than 0."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.kinematic_updates_per_frame = num_updates_per_frame
            self.lbl_num_updates_per_frame.setText( f'Number Kinematic Updates/Frame: {self.parent.kinematic_updates_per_frame}' )
    
    def update_num_vision_rays(self) :
        num_vision_rays = self.input_num_vision_rays.text()
        
        try :
            num_vision_rays = int( num_vision_rays )
            if num_vision_rays < 0 :
                raise ValueError('Invalid Entry')
        except :
            title = "Value Entry Error"
            warning_msg = "Number of Vision Rays must be an integer greater than 0."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.num_vision_rays = num_vision_rays
            self.lbl_num_vision_rays.setText( f'Current Number of Vision Rays: {self.parent.num_vision_rays}' )
            self.parent.AI_params_widget.update_AI_shape()
    
    def update_num_previous_vision_rays(self) :
        num_previous_vision_rays = self.input_num_previous_vision_rays.text()
        
        try :
            num_previous_vision_rays = int( num_previous_vision_rays )
            if num_previous_vision_rays < 0 :
                raise ValueError('Invalid Entry')
        except :
            title = "Value Entry Error"
            warning_msg = "Number of individuals must be an integer value equal to or greater than 0."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.vision_ray_num_history = num_previous_vision_rays
            self.lbl_num_previous_vision_rays.setText( f'Current Number of Previous Vision Rays: {self.parent.vision_ray_num_history}' )
            self.parent.AI_params_widget.update_AI_shape()
    
    def save_settings_begin_training(self) :
        if self.parent.AI_save_folder is not None and \
            self.parent.car_folder is not None and \
            self.parent.track_file is not None :
            self.parent.begin_training()
        else :
            title = "Missing Entry Error"
            warning_msg = "The Main Folder, AI Save Folder, Car Images Folder and Track file "
            warning_msg += "must all be set.\nPlease check and update these values and try again."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
    
    def lock_inputs(self) :
        self.btn_main_data_folder.setEnabled(False)
        self.btn_AI_save_folder.setEnabled(False)
        self.input_load_prev_gen.setEnabled(False)
        self.btn_load_AI_file.setEnabled(False)
        self.btn_car_images_folder.setEnabled(False)
        self.btn_track_file.setEnabled(False)
        self.btn_game_control_file.setEnabled(False)
        self.btn_load_game_control_file.setEnabled(False)
        self.btn_car_file.setEnabled(False)
        self.btn_AI_file.setEnabled(False)
        self.btn_resolution.setEnabled(False)
        self.input_display_width.setEnabled(False)
        self.input_display_height.setEnabled(False)
        self.input_training_mode.setEnabled(False)
        self.input_num_threads.setEnabled(False)
        self.btn_training_threads.setEnabled(False)
        self.input_num_cars_per_thread.setEnabled(False)
        self.btn_num_cars_per_thread.setEnabled(False)
        self.input_num_AIs.setEnabled(False)
        self.btn_num_AIs.setEnabled(False)
        self.input_num_AIs_on_track.setEnabled(False)
        self.btn_num_AIs_on_track.setEnabled(False)
        self.input_num_updates_per_frame.setEnabled(False)
        self.btn_num_updates_per_frame.setEnabled(False)
        self.input_num_previous_vision_rays.setEnabled(False)
        self.btn_num_previous_vision_rays.setEnabled(False)
        self.input_num_vision_rays.setEnabled(False)
        self.btn_num_vision_rays.setEnabled(False)
        self.btn_save_settings_begin_training.setEnabled(False)
    
    def load_AI_params(self) :
        if self.parent.AI_file is None :
            return None
        
        try :
            with open(self.parent.AI_file, 'r') as f :
                params = json.load(f)
        except :
            title = "Loading File Error"
            warning_msg = "An error occurred while attempting to load the file.\n"
            warning_msg += "Please check to ensure the correct file is located and the data is readable."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
            return None
        
        param_keys = params.keys()
        
        if 'AI_shape' in param_keys :
            if params['AI_shape'] is not None :
                shape = params['AI_shape']
                shape = ','.join([str(i) for i in shape[1:-1]])
                self.parent.AI_params_widget.input_AI_shape.setText( shape )
                self.parent.AI_params_widget.update_AI_shape()
        
        if 'num_individuals' in param_keys :
            if params['num_individuals'] is not None :
                num_individuals = params['num_individuals']
                self.parent.AI_params_widget.input_num_AIs.setText(str(num_individuals))
                self.parent.AI_params_widget.update_num_individual_AIs()
        
        if 'keep_best' in param_keys :
            if params['keep_best'] is not None :
                keep_best = params['keep_best']
                self.parent.AI_params_widget.input_keep_best.setText(str(keep_best))
                self.parent.AI_params_widget.update_keep_best()
        
        if 'propagate_best' in param_keys :
            if params['propagate_best'] is not None :
                propagate_best = params['propagate_best']
                self.parent.AI_params_widget.input_propagate_best.setText(str(propagate_best))
                self.parent.AI_params_widget.update_propagate_best()
        
        if 'breed_v_pop_stats_ratio' in param_keys :
            if params['breed_v_pop_stats_ratio'] is not None :
                breed_v_pop_stats_ratio = params['breed_v_pop_stats_ratio']
                self.parent.AI_params_widget.input_breed_v_pop_stats.setText(str(breed_v_pop_stats_ratio))
                self.parent.AI_params_widget.update_breed_v_pop_stats_ratio()
        
        if 'std_scaling_factor' in param_keys :
            if params['std_scaling_factor'] is not None :
                std_scaling_factor = params['std_scaling_factor']
                self.parent.AI_params_widget.input_std_scaling_factor.setText(str(std_scaling_factor))
                self.parent.AI_params_widget.update_std_scaling_factor()
        
        if 'propability_keep_f_gene' in param_keys :
            if params['propability_keep_f_gene'] is not None :
                propability_keep_f_gene = params['propability_keep_f_gene']
                self.parent.AI_params_widget.input_prob_keep_f_gene.setText(str(propability_keep_f_gene))
                self.parent.AI_params_widget.update_probability_keep_f_gene()
        
        if 'mix_ratio_keep_f_gene' in param_keys :
            if params['mix_ratio_keep_f_gene'] is not None :
                mix_ratio_keep_f_gene = params['mix_ratio_keep_f_gene']
                #mix_ratio_keep_m_gene = 1 - self.mix_ratio_keep_f_gene
                self.parent.AI_params_widget.input_mix_ratio_keep_f_gene.setText(str(mix_ratio_keep_f_gene))
                self.parent.AI_params_widget.update_mix_ratio_keep_f_gene()
        
        if 'mix_genes_probability' in param_keys :
            if params['mix_genes_probability'] is not None :
                mix_genes_probability = params['mix_genes_probability']
                #mix_sum_requirement = 2*mix_genes_probability
                self.parent.AI_params_widget.input_mix_genes_probability.setText(str(mix_genes_probability))
                self.parent.AI_params_widget.update_mix_genes_probability()
                # if self.mix_genes_probability < 0.5 :
                #     self.mix_sum_requirement = 2 * ( 1 - np.sqrt( self.mix_genes_probability / 2. ) )
                # else :
                #     self.mix_sum_requirement = np.sqrt( 2 - 2. * self.mix_genes_probability )
        
        if 'mutation_probability_rate' in param_keys :
            if params['mutation_probability_rate'] is not None :
                mutation_probability_rate = params['mutation_probability_rate']
                self.parent.AI_params_widget.input_mut_prob_rate.setText(str(mutation_probability_rate))
                self.parent.AI_params_widget.update_mutation_probability_rate()
        
        if 'mutate_uniform_gauss_ratio' in param_keys :
            if params['mutate_uniform_gauss_ratio'] is not None :
                mutate_uniform_gauss_ratio = params['mutate_uniform_gauss_ratio']
                self.parent.AI_params_widget.input_prob_uniform_v_normal.setText(str(mutate_uniform_gauss_ratio))
                self.parent.AI_params_widget.update_mutate_uniform_gauss_ratio()
        
        if 'mutation_max_factor_uniform_add_shift' in param_keys :
            if params['mutation_max_factor_uniform_add_shift'] is not None :
                mutation_max_factor_uniform_add_shift = params['mutation_max_factor_uniform_add_shift']
                self.parent.AI_params_widget.input_mut_max_uniform_add_shift.setText(str(mutation_max_factor_uniform_add_shift))
                self.parent.AI_params_widget.update_mutation_max_factor_uniform_add_shift()
        
        if 'mutation_max_factor_uniform_scaling' in param_keys :
            if params['mutation_max_factor_uniform_scaling'] is not None :
                mutation_max_factor_uniform_scaling = params['mutation_max_factor_uniform_scaling']
                self.parent.AI_params_widget.input_mut_max_uniform_scaling.setText(str(mutation_max_factor_uniform_scaling))
                self.parent.AI_params_widget.update_mutation_max_factor_uniform_scaling()
        
        if 'mutation_gauss_add_shift_sigma' in param_keys :
            if params['mutation_gauss_add_shift_sigma'] is not None :
                mutation_gauss_add_shift_sigma = params['mutation_gauss_add_shift_sigma']
                self.parent.AI_params_widget.input_mut_max_gauss_add_shift.setText(str(mutation_gauss_add_shift_sigma))
                self.parent.AI_params_widget.update_mutation_max_factor_gauss_add_shift()
        
        if 'mutation_max_factor_gauss_add_shift' in param_keys :
            if params['mutation_max_factor_gauss_add_shift'] is not None :
                mutation_max_factor_gauss_add_shift = params['mutation_max_factor_gauss_add_shift']
                self.parent.AI_params_widget.input_mut_max_gauss_add_shift.setText(str(mutation_max_factor_gauss_add_shift))
                self.parent.AI_params_widget.update_mutation_max_factor_gauss_add_shift()
        
        if 'mutation_max_factor_gauss_scaling' in param_keys :
            if params['mutation_max_factor_gauss_scaling'] is not None :
                mutation_max_factor_gauss_scaling = params['mutation_max_factor_gauss_scaling']
                self.parent.AI_params_widget.input_mut_max_gaussian_scaling.setText(str(mutation_max_factor_gauss_scaling))
                self.parent.AI_params_widget.update_mutation_max_factor_gauss_scaling()
        
        if 'save_best_num' in param_keys :
            if params['save_best_num'] is not None :
                save_best_num = params['save_best_num']
                self.parent.AI_params_widget.input_save_best_num.setText(str(save_best_num))
                self.parent.AI_params_widget.update_save_best_num()
        
        if 'minimum_score_percentage' in param_keys :
            if params['minimum_score_percentage'] is not None :
                minimum_score_percentage = params['minimum_score_percentage']
                self.parent.AI_params_widget.input_min_score_percentage.setText(str(minimum_score_percentage))
                self.parent.AI_params_widget.update_minimum_score_percentage()
    
    def load_game_params_file(self) :
        if self.parent.game_control_file is None :
            return None
        
        try :
            with open(self.parent.game_control_file, 'r') as f :
                params = json.load(f)
        except :
            title = "Loading File Error"
            warning_msg = "An error occurred while attempting to load the file.\n"
            warning_msg += "Please check to ensure the correct file is located and the data is readable."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
            return None
        
        folder = os.path.split(self.parent.game_control_file)[0]
        param_keys = params.keys()
        
        if 'display_max_resolution' in param_keys :
            if params['display_max_resolution'] is not None :
                res = params['display_max_resolution']
                self.input_display_width.setText(str(int(res[0])))
                self.input_display_height.setText(str(int(res[1])))
                self.update_display_resolution()
        
        if 'car_file' in param_keys :
            if params['car_file'] is not None :
                car_file = params['car_file']
                potential_car_file = os.path.join(folder, car_file)
                if os.path.exists(potential_car_file) :
                    self.get_car_file(potential_car_file)
        
        if 'genetic_AI_params_to_load' in param_keys :
            if params['genetic_AI_params_to_load'] is not None :
                genetic_AI_params_to_load = params['genetic_AI_params_to_load']
                potential_AI_file = os.path.join(folder, genetic_AI_params_to_load)
                if os.path.exists(potential_AI_file) :
                    self.get_AI_file(genetic_AI_params_to_load)
                    self.load_AI_params()
        
        if 'track_params_to_load' in param_keys :
            if params['track_params_to_load'] is not None :
                track_params_to_load = params['track_params_to_load']
                potential_track_file = os.path.join(folder, track_params_to_load)
                if os.path.exists(potential_track_file) :
                    self.get_track_file(potential_track_file)
        
        if 'total_num_drivers' in param_keys :
            if params['total_num_drivers'] is not None :
                total_num_drivers = params['total_num_drivers']
                self.input_num_AIs.setText(str(total_num_drivers))
                self.update_num_individual_AIs()
        
        if 'max_drivers_on_track' in param_keys :
            if params['max_drivers_on_track'] is not None :
                max_drivers_on_track = params['max_drivers_on_track']
                self.input_num_AIs_on_track.setText(str(max_drivers_on_track))
                self.update_num_AIs_on_track()
        
        if 'FPS' in param_keys :
            if params['FPS'] is not None :
                self.parent.FPS = params['FPS']
        
        if 'kinematic_updates_per_frame' in param_keys :
            if params['kinematic_updates_per_frame'] is not None :
                kinematic_updates_per_frame = params['kinematic_updates_per_frame']
                self.input_num_updates_per_frame.setText(str(kinematic_updates_per_frame))
                self.update_num_updates_per_frame()
        
        if 'number_vision_rays' in param_keys :
            if params['number_vision_rays'] is not None :
                number_vision_rays = params['number_vision_rays']
                self.input_num_vision_rays.setText(str(number_vision_rays))
                self.update_num_vision_rays()
        
        if 'vision_ray_num_history' in param_keys :
            if params['vision_ray_num_history'] is not None :
                vision_ray_num_history = params['vision_ray_num_history']
                self.input_num_previous_vision_rays.setText(str(vision_ray_num_history))
                self.update_num_previous_vision_rays()
        
        if 'training_mode' in param_keys :
            if params['training_mode'] is not None :
                training_mode = int( params['training_mode'] )
                if training_mode in [0,1,2] :
                    self.input_training_mode.setCurrentIndex(training_mode)
                    self.update_training_mode()
        
        if 'maximum_concurrent_training_threads' in param_keys :
            if params['maximum_concurrent_training_threads'] is not None :
                maximum_concurrent_training_threads = params['maximum_concurrent_training_threads']
                self.input_num_threads.setText(str(maximum_concurrent_training_threads))
                self.update_num_training_threads()
        
        if 'maximum_number_cars_per_training_thread' in param_keys :
            if params['maximum_number_cars_per_training_thread'] is not None :
                maximum_number_cars_per_training_thread = params['maximum_number_cars_per_training_thread']
                self.input_num_cars_per_thread.setText(str(maximum_number_cars_per_training_thread))
                self.update_num_cars_per_thread()

class AI_params_page(QWidget) :
    def __init__(self, parent) :
        super(QWidget, self).__init__(parent)
        max_text_width = 250
        max_widget_width = 150
        max_input_width = 150
        min_input_width = 100
        
        self.parent = parent
        
        self.layout = QVBoxLayout(self)
        
        intro_layout = QGridLayout()
        gli_row = 0
        
        ##### Set AI Shape
        label = QLabel('Size of Hidden Layers: ')
        label.setMaximumWidth(max_text_width)
        self.input_AI_shape = QLineEdit()
        self.input_AI_shape.setMaximumWidth(max_input_width)
        self.input_AI_shape.setMinimumWidth(min_input_width)
        intro_layout.addWidget(label, gli_row, 0)
        intro_layout.addWidget(self.input_AI_shape, gli_row, 1); gli_row += 1
        
        self.lbl_AI_shape = QLabel( f'Current AI shape: {self.parent.AI_shape}' )
        self.btn_AI_shape = QPushButton( 'Update' )
        self.btn_AI_shape.setMaximumWidth(max_input_width)
        self.btn_AI_shape.clicked.connect(self.update_AI_shape)
        intro_layout.addWidget(self.lbl_AI_shape, gli_row, 0)
        intro_layout.addWidget(self.btn_AI_shape, gli_row, 1); gli_row += 1
        
        intro_instructions = '\n'.join([
            'This parameter sets the size and number of hidden levels. '
            'For example, to add 2 hidden layers with 15 nodes in the first '
            'layer and 10 nodes in the second hidden layer, you should enter: '
            '\"15,10\", without the quotes. If you do not want any hidden layers, leave the entry '
            'box empty and click the Update button. The necessary number of '
            'input and output nodes will be automatically added.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        intro_layout.addWidget(intro_text, gli_row, 0, 1, 2); gli_row += 1
        
        # Insert horizontal seperation bar
        intro_layout.addWidget(QHLine(), gli_row, 0, 1, 2); gli_row += 1
        
        ##### Set number of best individuals to keep - keep best
        label = QLabel('Keep Best: ')
        label.setMaximumWidth(max_text_width)
        self.input_keep_best = QLineEdit()
        self.input_keep_best.setMaximumWidth(max_input_width)
        self.input_keep_best.setMinimumWidth(min_input_width)
        intro_layout.addWidget(label, gli_row, 0)
        intro_layout.addWidget(self.input_keep_best, gli_row, 1); gli_row += 1
        
        self.lbl_keep_best = QLabel( f'Current Number of best Individuals to keep: {self.parent.keep_best}' )
        self.btn_keep_best = QPushButton( 'Update' )
        self.btn_keep_best.setMaximumWidth(max_input_width)
        self.btn_keep_best.clicked.connect(self.update_keep_best)
        intro_layout.addWidget(self.lbl_keep_best, gli_row, 0)
        intro_layout.addWidget(self.btn_keep_best, gli_row, 1); gli_row += 1
        
        intro_instructions = '\n'.join([
            'This parameter is used to select the number of individuals that '
            'should be kept. These individuals are selected based on their fitness '
            'score. That is, the number set here will determine the number of '
            'individuals with the best scores to keep. This method of selection is '
            'called \"Elitism Selection.\"\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        intro_layout.addWidget(intro_text, gli_row, 0, 1, 2); gli_row += 1
        
        # Insert horizontal seperation bar
        intro_layout.addWidget(QHLine(), gli_row, 0, 1, 2); gli_row += 1
        
        ##### Set number of individuals to propagrate - propagate_best
        label = QLabel('Propagate Best: ')
        label.setMaximumWidth(max_text_width)
        self.input_propagate_best = QLineEdit()
        self.input_propagate_best.setMaximumWidth(max_input_width)
        self.input_propagate_best.setMinimumWidth(min_input_width)
        intro_layout.addWidget(label, gli_row, 0)
        intro_layout.addWidget(self.input_propagate_best, gli_row, 1); gli_row += 1
        
        self.lbl_propagate_best = QLabel( f'Current Max. Number of Individuals to Propagate: {self.parent.propagate_best}' )
        self.btn_propagate_best = QPushButton( 'Update' )
        self.btn_propagate_best.setMaximumWidth(max_input_width)
        self.btn_propagate_best.clicked.connect(self.update_propagate_best)
        intro_layout.addWidget(self.lbl_propagate_best, gli_row, 0)
        intro_layout.addWidget(self.btn_propagate_best, gli_row, 1); gli_row += 1
        
        intro_instructions = '\n'.join([
            'This value selects the maximum number of individuals that will be used for '
            'gene propagation. The maximum number of individuals will be used if they '
            'all have a score greater than minimum score.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        intro_layout.addWidget(intro_text, gli_row, 0, 1, 2); gli_row += 1
        
        # Insert horizontal seperation bar
        intro_layout.addWidget(QHLine(), gli_row, 0, 1, 2); gli_row += 1
        
        ##### Set minimum score to propagate - minimum_score_percentage
        label = QLabel('Minimum Score Percent: ')
        label.setMaximumWidth(max_text_width)
        self.input_min_score_percentage = QLineEdit()
        self.input_min_score_percentage.setMaximumWidth(max_input_width)
        self.input_min_score_percentage.setMinimumWidth(min_input_width)
        intro_layout.addWidget(label, gli_row, 0)
        intro_layout.addWidget(self.input_min_score_percentage, gli_row, 1); gli_row += 1
        
        self.lbl_min_score_percentage = QLabel( f'Current Minimum Score Percent: {self.parent.minimum_score_percentage}' )
        self.btn_min_score_percentage = QPushButton( 'Update' )
        self.btn_min_score_percentage.setMaximumWidth(max_input_width)
        self.btn_min_score_percentage.clicked.connect(self.update_minimum_score_percentage)
        intro_layout.addWidget(self.lbl_min_score_percentage, gli_row, 0)
        intro_layout.addWidget(self.btn_min_score_percentage, gli_row, 1); gli_row += 1
        
        intro_instructions = '\n'.join([
            'This value should be between 0 and 1. Individuals who have a score that '
            'is less than this percent of the highest fitness score will not be '
            'used for gene propagation. For example, if this value is set to 0.5 '
            'and the highest score is 50, then the lowest score an individual may '
            'have and still propagate its genes is 25.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        intro_layout.addWidget(intro_text, gli_row, 0, 1, 2); gli_row += 1
        
        # Insert horizontal seperation bar
        intro_layout.addWidget(QHLine(), gli_row, 0, 1, 2); gli_row += 1
        
        ##### Set ratio to breed vs population stats - breed_v_pop_stats_ratio
        label = QLabel('Ratio of Individuals to Create by Breeding vs Population Stats: ')
        label.setMaximumWidth(2*max_text_width)
        self.input_breed_v_pop_stats = QLineEdit()
        self.input_breed_v_pop_stats.setMaximumWidth(max_input_width)
        self.input_breed_v_pop_stats.setMinimumWidth(min_input_width)
        intro_layout.addWidget(label, gli_row, 0)
        intro_layout.addWidget(self.input_breed_v_pop_stats, gli_row, 1); gli_row += 1
        
        self.lbl_breed_v_pop_stats = QLabel( f'Current Ratio (Breed/Pop. Stats): {self.parent.breed_v_pop_stats_ratio}' )
        self.btn_breed_v_pop_stats = QPushButton( 'Update' )
        self.btn_breed_v_pop_stats.setMaximumWidth(max_input_width)
        self.btn_breed_v_pop_stats.clicked.connect(self.update_breed_v_pop_stats_ratio)
        intro_layout.addWidget(self.lbl_breed_v_pop_stats, gli_row, 0)
        intro_layout.addWidget(self.btn_breed_v_pop_stats, gli_row, 1); gli_row += 1
        
        intro_instructions = '\n'.join([
            'This value must be between 0 and 1. This sets the ratio of the number '
            'of individuals that will be created by gene propagation versus using '
            'population statistics. (Population statistics is discussed in more detail '
            'on the introduction page.) For example, if there are 100 new individuals that '
            'must be created and this value is set to 0.75, then 75 individuals will be '
            'created using gene propagation and 25 will be created using pop. stats. '
            'It is possible to set this value to either 0 or 1 if you want to use a '
            'specific method to create new individuals.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        intro_layout.addWidget(intro_text, gli_row, 0, 1, 2); gli_row += 1
        
        ##### Add intro_layout to layout
        self.layout.addLayout(intro_layout)
        
        self.layout.addWidget(QHLine())
        
        label = QLabel('Individuals are created using two different methods. The previous '
            'value sets the probability of selecting one or the other.\n')
        intro_text.setWordWrap(True)
        self.layout.addWidget(label)
        
        self.layout.addWidget(QHLine())
        
        middle_layout = QHBoxLayout()
        
        breed_params_layout = QGridLayout()
        bpl_row = 0
        
        ##### Set probability to  keep female gene - probability_keep_f_gene
        label = QLabel('Probability to keep Female Gene - Full Gene Selection: ')
        #label.setMaximumWidth(max_text_width)
        self.input_prob_keep_f_gene = QLineEdit()
        self.input_prob_keep_f_gene.setMaximumWidth(max_input_width)
        self.input_prob_keep_f_gene.setMinimumWidth(min_input_width)
        breed_params_layout.addWidget(label, bpl_row, 0)
        breed_params_layout.addWidget(self.input_prob_keep_f_gene, bpl_row, 1); bpl_row += 1
        
        self.lbl_prob_keep_f_gene = QLabel( f'Current Probability to keep female gene: {self.parent.probability_keep_f_gene}' )
        self.btn_prob_keep_f_gene = QPushButton( 'Update' )
        self.btn_prob_keep_f_gene.setMaximumWidth(max_input_width)
        self.btn_prob_keep_f_gene.clicked.connect(self.update_probability_keep_f_gene)
        breed_params_layout.addWidget(self.lbl_prob_keep_f_gene, bpl_row, 0)
        breed_params_layout.addWidget(self.btn_prob_keep_f_gene, bpl_row, 1); bpl_row += 1
        
        intro_instructions = '\n'.join([
            'During gene propagation, some genes from each parent are '
            'selected to be kept directly from the parent. This is a '
            'method to cause one parent to have a greater influence over '
            'the behaviour of future individuals. This value, '
            'which must be between 0 and 1, sets the probability that '
            'the female gene will be kept over the male gene. For example, '
            'if this value is set to 0.75 then there is a 75% chance that '
            'the female gene will be used in this case and a 25% chance that '
            'the male gene will be used.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        breed_params_layout.addWidget(intro_text, bpl_row, 0, 1, 2); bpl_row += 1
        
        # Insert horizontal seperation bar
        breed_params_layout.addWidget(QHLine(), bpl_row, 0, 1, 2); bpl_row += 1
        
        ##### Set probability to mix genes - mix_genes_probability
        label = QLabel('Probability to Mix Genes: ')
        #label.setMaximumWidth(max_text_width)
        self.input_mix_genes_probability = QLineEdit()
        self.input_mix_genes_probability.setMaximumWidth(max_input_width)
        self.input_mix_genes_probability.setMinimumWidth(min_input_width)
        breed_params_layout.addWidget(label, bpl_row, 0)
        breed_params_layout.addWidget(self.input_mix_genes_probability, bpl_row, 1); bpl_row += 1
        
        self.lbl_mix_genes_probability = QLabel( f'Current probability to mix genes: {self.parent.mix_genes_probability}' )
        self.btn_mix_genes_probability = QPushButton( 'Update' )
        self.btn_mix_genes_probability.setMaximumWidth(max_input_width)
        self.btn_mix_genes_probability.clicked.connect(self.update_mix_genes_probability)
        breed_params_layout.addWidget(self.lbl_mix_genes_probability, bpl_row, 0)
        breed_params_layout.addWidget(self.btn_mix_genes_probability, bpl_row, 1); bpl_row += 1
        
        intro_instructions = '\n'.join([
            'This value must be between 0 and 1. '
            'This sets the probability of genes being a combination, mix, '
            'of both parents as opposed to being purely from one parent. '
            'For example, if this value is set to 0.75 then there is a 75% '
            'chance that a gene will be a combination of the two parent genes '
            'and a 25% that the gene will be a direct carry over from a single parent.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        breed_params_layout.addWidget(intro_text, bpl_row, 0, 1, 2); bpl_row += 1
        
        # Insert horizontal seperation bar
        breed_params_layout.addWidget(QHLine(), bpl_row, 0, 1, 2); bpl_row += 1
        
        ##### Set ratio of female gene to keep during mixing - mix_ratio_keep_f_gene
        label = QLabel('Female Gene Averaging Weight - Gene Mixing Selection: ')
        #label.setMaximumWidth(max_text_width)
        self.input_mix_ratio_keep_f_gene = QLineEdit()
        self.input_mix_ratio_keep_f_gene.setMaximumWidth(max_input_width)
        self.input_mix_ratio_keep_f_gene.setMinimumWidth(min_input_width)
        breed_params_layout.addWidget(label, bpl_row, 0)
        breed_params_layout.addWidget(self.input_mix_ratio_keep_f_gene, bpl_row, 1); bpl_row += 1
        
        self.lbl_mix_ratio_keep_f_gene = QLabel( f'Current female gene averaging weight: {self.parent.mix_ratio_keep_f_gene}' )
        self.btn_mix_ratio_keep_f_gene = QPushButton( 'Update' )
        self.btn_mix_ratio_keep_f_gene.setMaximumWidth(max_input_width)
        self.btn_mix_ratio_keep_f_gene.clicked.connect(self.update_mix_ratio_keep_f_gene)
        breed_params_layout.addWidget(self.lbl_mix_ratio_keep_f_gene, bpl_row, 0)
        breed_params_layout.addWidget(self.btn_mix_ratio_keep_f_gene, bpl_row, 1); bpl_row += 1
        
        intro_instructions = '\n'.join([
            'This value must be between 0 and 1. '
            'This sets the weight that will be used for the female gene '
            'during the gene mixing process. For example, if this value is 0.75 '
            'then the female gene will have a weight of 0.75 and the male gene '
            'will have a weight of 0.25. This is a method to increase the influence '
            'of one parent over the other during gene mixing. To get a simple average '
            'set this value to 0.5.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        breed_params_layout.addWidget(intro_text, bpl_row, 0, 1, 2); bpl_row += 1
        
        middle_layout.addLayout(breed_params_layout)
        
        middle_layout.addWidget(QVLine())
        
        pop_stats_layout = QGridLayout()
        psl_row = 0
        
        ##### Set STD of the Mean reduction factor - std_scaling_factor
        label = QLabel('Population Statistics Standard Deviation Scaling Factor: ')
        #label.setMaximumWidth(max_text_width)
        self.input_std_scaling_factor = QLineEdit()
        self.input_std_scaling_factor.setMaximumWidth(max_input_width)
        self.input_std_scaling_factor.setMinimumWidth(min_input_width)
        pop_stats_layout.addWidget(label, psl_row, 0)
        pop_stats_layout.addWidget(self.input_std_scaling_factor, psl_row, 1); psl_row += 1
        
        self.lbl_std_scaling_factor = QLabel( f'Current scaling factor: {self.parent.std_scaling_factor}' )
        self.btn_std_scaling_factor = QPushButton( 'Update' )
        self.btn_std_scaling_factor.setMaximumWidth(max_input_width)
        self.btn_std_scaling_factor.clicked.connect(self.update_std_scaling_factor)
        pop_stats_layout.addWidget(self.lbl_std_scaling_factor, psl_row, 0)
        pop_stats_layout.addWidget(self.btn_std_scaling_factor, psl_row, 1); psl_row += 1
        
        intro_instructions = '\n'.join([
            'This value must be greater than 0. '
            'The mean and standard deviation for each individual gene '
            'is calculated using all individuals selected for propagation. '
            'This value can be set to increase, >1, or decrease, <1, the '
            'standard deviation for each gene. This can help to reduce/increase '
            'the variation of individuals that are created using this method.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        pop_stats_layout.addWidget(intro_text, psl_row, 0, 1, 2); psl_row += 1
        
        middle_layout.addLayout(pop_stats_layout)
        pop_stats_layout.setRowStretch(psl_row, 1)
        
        ##### Add middle_layout to layout
        self.layout.addLayout(middle_layout)
        
        self.layout.addWidget(QHLine())
        
        mutation_prob_layout = QGridLayout()
        mp_row = 0
        
        ##### Set mutation probability rate - mutation_probability_rate
        label = QLabel('Probability for Gene Mutation: ')
        label.setMaximumWidth(max_text_width)
        self.input_mut_prob_rate = QLineEdit()
        self.input_mut_prob_rate.setMaximumWidth(max_input_width)
        self.input_mut_prob_rate.setMinimumWidth(min_input_width)
        mutation_prob_layout.addWidget(label, mp_row, 0)
        mutation_prob_layout.addWidget(self.input_mut_prob_rate, mp_row, 1); mp_row += 1
        
        self.lbl_mut_prob_rate = QLabel( f'Current gene mutation rate: {self.parent.mutation_probability_rate}' )
        self.btn_mut_prob_rate = QPushButton( 'Update' )
        self.btn_mut_prob_rate.setMaximumWidth(max_input_width)
        self.btn_mut_prob_rate.clicked.connect(self.update_mutation_probability_rate)
        mutation_prob_layout.addWidget(self.lbl_mut_prob_rate, mp_row, 0)
        mutation_prob_layout.addWidget(self.btn_mut_prob_rate, mp_row, 1); mp_row += 1
        
        intro_instructions = '\n'.join([
            'This value must be between 0 and 1. '
            'This sets the probability that an individual gene will be mutated. '
            'For example, if this is set to 0.05 there is a 5% for each gene '
            'to be mutated.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        mutation_prob_layout.addWidget(intro_text, mp_row, 0, 1, 2); mp_row += 1
        
        # Insert horizontal seperation bar
        mutation_prob_layout.addWidget(QHLine(), mp_row, 0, 1, 2); mp_row += 1
        
        ##### Set mutation probability rate - mutate_uniform_gauss_ratio
        label = QLabel('Probability to use Uniform Distribution over Normal Distribution for Mutations: ')
        label.setMaximumWidth(2*max_text_width)
        self.input_prob_uniform_v_normal = QLineEdit()
        self.input_prob_uniform_v_normal.setMaximumWidth(max_input_width)
        self.input_prob_uniform_v_normal.setMinimumWidth(min_input_width)
        mutation_prob_layout.addWidget(label, mp_row, 0)
        mutation_prob_layout.addWidget(self.input_prob_uniform_v_normal, mp_row, 1); mp_row += 1
        
        self.lbl_prob_uniform_v_normal = QLabel( f'Current probability to use an uniform distribution: {self.parent.mutate_uniform_gauss_ratio}' )
        self.btn_prob_uniform_v_normal = QPushButton( 'Update' )
        self.btn_prob_uniform_v_normal.setMaximumWidth(max_input_width)
        self.btn_prob_uniform_v_normal.clicked.connect(self.update_mutate_uniform_gauss_ratio)
        mutation_prob_layout.addWidget(self.lbl_prob_uniform_v_normal, mp_row, 0)
        mutation_prob_layout.addWidget(self.btn_prob_uniform_v_normal, mp_row, 1); mp_row += 1
        
        intro_instructions = '\n'.join([
            'This value must be between 0 and 1. '
            'Individuals can be mutated using values generated from two different '
            'types of probability distributions: uniform and normal (Gaussian). '
            'Using a uniform distribution causes a hard limit on values that can be '
            'used for mutations and can increase the chances of larger mutations. '
            'Using a normal distribution allows for a small probability of very large '
            'mutations but has a higher probability of smaller mutations.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        mutation_prob_layout.addWidget(intro_text, mp_row, 0, 1, 2); mp_row += 1
        
        self.layout.addLayout(mutation_prob_layout)
        
        self.layout.addWidget(QHLine())
        
        label = QLabel('Individuals are mutated using two different methods. The previous '
            'value sets the probability of selecting one or the other.')
        intro_text.setWordWrap(True)
        self.layout.addWidget(label)
        
        self.layout.addWidget(QHLine())
        
        mutation_params_layout = QHBoxLayout()
        
        uniform_mutation_params_layout = QGridLayout()
        umpl_row = 0
        
        ##### Set mutation uniform addition shift - mutation_max_factor_uniform_add_shift
        label = QLabel('Uniform Distribution Max Factor - Shift: ')
        #label.setMaximumWidth(max_text_width)
        self.input_mut_max_uniform_add_shift = QLineEdit()
        self.input_mut_max_uniform_add_shift.setMaximumWidth(max_input_width)
        self.input_mut_max_uniform_add_shift.setMinimumWidth(min_input_width)
        uniform_mutation_params_layout.addWidget(label, umpl_row, 0)
        uniform_mutation_params_layout.addWidget(self.input_mut_max_uniform_add_shift, umpl_row, 1); umpl_row += 1
        
        self.lbl_mut_max_uniform_add_shift = QLabel( f'Current uniform max shift factor: {self.parent.mutation_max_factor_uniform_add_shift}' )
        self.btn_mut_max_uniform_add_shift = QPushButton( 'Update' )
        self.btn_mut_max_uniform_add_shift.setMaximumWidth(max_input_width)
        self.btn_mut_max_uniform_add_shift.clicked.connect(self.update_mutation_max_factor_uniform_add_shift)
        uniform_mutation_params_layout.addWidget(self.lbl_mut_max_uniform_add_shift, umpl_row, 0)
        uniform_mutation_params_layout.addWidget(self.btn_mut_max_uniform_add_shift, umpl_row, 1); umpl_row += 1
        
        intro_instructions = '\n'.join([
            'This value must be larger than 0. '
            'The value of a gene to be mutated will be shifted, by addition, by a random factor. '
            'The value of the factor will be chosen using a uniform distribution with a range of '
            '+/- the value set here.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        uniform_mutation_params_layout.addWidget(intro_text, umpl_row, 0, 1, 2); umpl_row += 1
        
        # Insert horizontal seperation bar
        uniform_mutation_params_layout.addWidget(QHLine(), umpl_row, 0, 1, 2); umpl_row += 1
        
        ##### Set mutation uniform scaling factor - mutation_max_factor_uniform_scaling
        label = QLabel('Uniform Distribution Max Factor - Scaling: ')
        #label.setMaximumWidth(max_text_width)
        self.input_mut_max_uniform_scaling = QLineEdit()
        self.input_mut_max_uniform_scaling.setMaximumWidth(max_input_width)
        self.input_mut_max_uniform_scaling.setMinimumWidth(min_input_width)
        uniform_mutation_params_layout.addWidget(label, umpl_row, 0)
        uniform_mutation_params_layout.addWidget(self.input_mut_max_uniform_scaling, umpl_row, 1); umpl_row += 1
        
        self.lbl_mut_max_uniform_scaling = QLabel( f'Current uniform max scaling factor: {self.parent.mutation_max_factor_uniform_scaling}' )
        self.btn_mut_max_uniform_scaling = QPushButton( 'Update' )
        self.btn_mut_max_uniform_scaling.setMaximumWidth(max_input_width)
        self.btn_mut_max_uniform_scaling.clicked.connect(self.update_mutation_max_factor_uniform_scaling)
        uniform_mutation_params_layout.addWidget(self.lbl_mut_max_uniform_scaling, umpl_row, 0)
        uniform_mutation_params_layout.addWidget(self.btn_mut_max_uniform_scaling, umpl_row, 1); umpl_row += 1
        
        intro_instructions = '\n'.join([
            'This value must be larger than 0. '
            'The value of a gene to be mutated will be scaled, by multiplication, by a random factor. '
            'The value of the factor will be chosen using a uniform distribution with a range of '
            '+/- the value set here.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        uniform_mutation_params_layout.addWidget(intro_text, umpl_row, 0, 1, 2); umpl_row += 1
        
        uniform_mutation_params_layout.addWidget(QVLine())
        
        mutation_params_layout.addLayout(uniform_mutation_params_layout)
        mutation_params_layout.addWidget(QVLine())
        
        normal_mutation_params_layout = QGridLayout()
        nmpl_row = 0
        
        ##### Set mutation gaussian addition shift - mutation_max_factor_gauss_add_shift
        label = QLabel('Normal (Gaussian) Distribution Max Factor - Shift: ')
        #label.setMaximumWidth(max_text_width)
        self.input_mut_max_gauss_add_shift = QLineEdit()
        self.input_mut_max_gauss_add_shift.setMaximumWidth(max_input_width)
        self.input_mut_max_gauss_add_shift.setMinimumWidth(min_input_width)
        normal_mutation_params_layout.addWidget(label, nmpl_row, 0)
        normal_mutation_params_layout.addWidget(self.input_mut_max_gauss_add_shift, nmpl_row, 1); nmpl_row += 1
        
        self.lbl_mut_max_gauss_add_shift = QLabel( f'Current normal max shift factor: {self.parent.mutation_max_factor_gauss_add_shift}' )
        self.btn_mut_max_gauss_add_shift = QPushButton( 'Update' )
        self.btn_mut_max_gauss_add_shift.setMaximumWidth(max_input_width)
        self.btn_mut_max_gauss_add_shift.clicked.connect(self.update_mutation_max_factor_gauss_add_shift)
        normal_mutation_params_layout.addWidget(self.lbl_mut_max_gauss_add_shift, nmpl_row, 0)
        normal_mutation_params_layout.addWidget(self.btn_mut_max_gauss_add_shift, nmpl_row, 1); nmpl_row += 1
        
        intro_instructions = '\n'.join([
            'This value must be larger than 0. '
            'The value of a gene to be mutated will be shifted, by addition, by a random factor. '
            'The value of the factor will be chosen using a normal (Gaussian) distribution with a '
            'standard deviation equal to the value set here and a mean of 0.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        normal_mutation_params_layout.addWidget(intro_text, nmpl_row, 0, 1, 2); nmpl_row += 1
        
        # Insert horizontal seperation bar
        normal_mutation_params_layout.addWidget(QHLine(), nmpl_row, 0, 1, 2); nmpl_row += 1
        
        ##### Set mutation gaussian scaling factor - mutation_max_factor_gauss_scaling
        label = QLabel('Normal (Gaussian) Distribution Max Factor - Scaling: ')
        #label.setMaximumWidth(max_text_width)
        self.input_mut_max_gaussian_scaling = QLineEdit()
        self.input_mut_max_gaussian_scaling.setMaximumWidth(max_input_width)
        self.input_mut_max_gaussian_scaling.setMinimumWidth(min_input_width)
        normal_mutation_params_layout.addWidget(label, nmpl_row, 0)
        normal_mutation_params_layout.addWidget(self.input_mut_max_gaussian_scaling, nmpl_row, 1); nmpl_row += 1
        
        self.lbl_mut_max_gaussian_scaling = QLabel( f'Current normal max scaling factor: {self.parent.mutation_max_factor_gauss_scaling}' )
        self.btn_mut_max_gaussian_scaling = QPushButton( 'Update' )
        self.btn_mut_max_gaussian_scaling.setMaximumWidth(max_input_width)
        self.btn_mut_max_gaussian_scaling.clicked.connect(self.update_mutation_max_factor_gauss_scaling)
        normal_mutation_params_layout.addWidget(self.lbl_mut_max_gaussian_scaling, nmpl_row, 0)
        normal_mutation_params_layout.addWidget(self.btn_mut_max_gaussian_scaling, nmpl_row, 1); nmpl_row += 1
        
        intro_instructions = '\n'.join([
            'This value must be larger than 0. '
            'The value of a gene to be mutated will be scaled, by multiplication, by a random factor. '
            'The value of the factor will be chosen using a normal (Gaussian) distribution with a '
            'standard deviation equal to the value set here and a mean of 1.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        normal_mutation_params_layout.addWidget(intro_text, nmpl_row, 0, 1, 2); nmpl_row += 1
        
        mutation_params_layout.addLayout(normal_mutation_params_layout)
        
        ##### Add middle_layout to layout
        self.layout.addLayout(mutation_params_layout)
        
        self.layout.addWidget(QHLine())
        
        final_layout = QGridLayout()
        fl_row = 0
        
        ##### Set number of best individuals to save - save_best_num
        label = QLabel('Save Best Individuals: ')
        label.setMaximumWidth(max_text_width)
        self.input_save_best_num = QLineEdit()
        self.input_save_best_num.setMaximumWidth(max_input_width)
        self.input_save_best_num.setMinimumWidth(min_input_width)
        final_layout.addWidget(label, fl_row, 0)
        final_layout.addWidget(self.input_save_best_num, fl_row, 1); fl_row += 1
        
        self.lbl_save_best_num = QLabel( f'Current number of individuals to save: {self.parent.save_best_num}' )
        self.btn_save_best_num = QPushButton( 'Update' )
        self.btn_save_best_num.setMaximumWidth(max_input_width)
        self.btn_save_best_num.clicked.connect(self.update_save_best_num)
        final_layout.addWidget(self.lbl_save_best_num, fl_row, 0)
        final_layout.addWidget(self.btn_save_best_num, fl_row, 1); fl_row += 1
        
        intro_instructions = '\n'.join([
            'This value must be either be between 0 and 1 or and integer value greater than 0. '
            'If the value is between 0 and 1 then it will be treated as a percent and this '
            'percent of the population will be saved. If the value is an integer equal to 1 or '
            'greater, then this number of individuals will be saved. Individuals selected to be saved '
            'are selected based on their scores and the best individuals are chosen.\n'
            ])
        intro_text = QLabel(intro_instructions)
        intro_text.setWordWrap(True)
        final_layout.addWidget(intro_text, fl_row, 0, 1, 2); fl_row += 1
        
        self.layout.addLayout(final_layout)
        
        self.layout.addWidget(QHLine())
        
        start_training_layout = QGridLayout()
        stl_row = 0
        
        # section to stop accepting options and pass training params
        label = QLabel('Do not use this section unless you have finished entering '
            'settings for the game and AI settings. This lock all the settings such '
            'that they cannot be changed again and begins the training process.')
        label.setWordWrap(True)
        label.setFont(QFont('Arial', 15))
        start_training_layout.addWidget(label, stl_row, 0, 1, 2); stl_row += 1
        
        #label = QLabel('Save Settings and begin Training')
        self.btn_save_settings_begin_training = QPushButton( 'Save Settings and begin Training' )
        self.btn_save_settings_begin_training.clicked.connect(self.parent.game_params_widget.save_settings_begin_training)
        #start_training_layout.addWidget(label, stl_row, 0)
        start_training_layout.addWidget(self.btn_save_settings_begin_training, stl_row, 1); stl_row += 1
        
        self.layout.addLayout(start_training_layout)
        
        self.setLayout(self.layout)
    
    def update_AI_shape(self) :
        hidden_layers_str = self.input_AI_shape.text().split(',')
        
        try:
            num_nodes = []
            if hidden_layers_str == [''] :
                num_nodes = []
            else :
                for i in np.arange(len(hidden_layers_str)) :
                    if hidden_layers_str[i] == '' :
                        pass
                    else :
                        num_nodes.append(int(hidden_layers_str[i]))
                        if num_nodes[-1] < 1 :
                            raise ValueError('Invalid Entry')
        except:
            title = "Value Entry Error"
            warning_msg = "AI layers must be an integer values greater than 0."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            hidden_layers = []
            for i in num_nodes :
                hidden_layers.append( i )
            
            self.parent.AI_hidden_layers = hidden_layers
            
            self.parent.update_AI_shape()
    
    def update_AI_shape_statement(self) :
        self.lbl_AI_shape.setText( f'Current AI shape: {self.parent.AI_shape}' )
    
    def update_keep_best(self) :
        keep_best = self.input_keep_best.text()
        
        try:
            keep_best = float(keep_best)
            if keep_best < 0 :
                raise ValueError('Invalid Entry')
            
            if keep_best < 1 :
                keep_best = int( keep_best*self.parent.num_individual_AIs )
            else :
                keep_best = int( keep_best )
        except:
            title = "Value Entry Error - Keep Best"
            warning_msg = "Entry value must be a float between 0 and 1 or an integer equal to or greater than 1."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.keep_best = keep_best
            
            self.lbl_keep_best.setText( f'Current Number of best Individuals to keep: {self.parent.keep_best}' )
    
    def update_propagate_best(self) :
        propagate_best = self.input_propagate_best.text()
        
        try:
            propagate_best = float(propagate_best)
            if propagate_best < 0 :
                raise ValueError('Invalid Entry')
            
            if propagate_best < 1 :
                propagate_best = int( propagate_best*self.parent.num_individual_AIs )
            else :
                propagate_best = int( propagate_best )
        except:
            title = "Value Entry Error - Propagate Best"
            warning_msg = "Entry value must be a float between 0 and 1 or an integer equal to or greater than 1."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.propagate_best = propagate_best
            
            self.lbl_propagate_best.setText( f'Current Max. Number of Individuals to Propagate: {self.parent.propagate_best}' )
    
    def update_minimum_score_percentage(self) :
        minimum_score_percentage = self.input_min_score_percentage.text()
        
        try:
            minimum_score_percentage = float(minimum_score_percentage)
            if minimum_score_percentage < 0 or minimum_score_percentage > 1 :
                raise ValueError('Invalid Entry')
        except:
            title = "Value Entry Error - Minimum Score Percentage"
            warning_msg = "Entry value must be a float between 0 and 1."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.minimum_score_percentage = minimum_score_percentage
            self.lbl_min_score_percentage.setText( f'Current Minimum Score Percent: {self.parent.minimum_score_percentage}' )
    
    def update_breed_v_pop_stats_ratio(self) :
        breed_v_pop_stats = self.input_breed_v_pop_stats.text()
        
        try:
            breed_v_pop_stats = float(breed_v_pop_stats)
            if breed_v_pop_stats < 0 :
                raise ValueError('Invalid Entry')
            
            if breed_v_pop_stats < 1 :
                breed_v_pop_stats = int( breed_v_pop_stats*self.parent.num_individual_AIs )
            else :
                breed_v_pop_stats = int( breed_v_pop_stats )
        except:
            title = "Value Entry Error - Breed vs Population Stats Ratio"
            warning_msg = "Entry value must be a float between 0 and 1 or an integer equal to or greater than 1."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.breed_v_pop_stats_ratio = breed_v_pop_stats
            self.lbl_breed_v_pop_stats.setText( f'Current Ratio (Breed/Pop. Stats): {self.parent.breed_v_pop_stats_ratio}' )
    
    def update_probability_keep_f_gene(self) :
        probability_keep_f_gene = self.input_prob_keep_f_gene.text()
        
        try:
            probability_keep_f_gene = float(probability_keep_f_gene)
            if probability_keep_f_gene < 0 or probability_keep_f_gene > 1 :
                raise ValueError('Invalid Entry')
        except:
            title = "Value Entry Error - Probability to Keep Female Gene"
            warning_msg = "Entry value must be a float between 0 and 1."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.probability_keep_f_gene = probability_keep_f_gene
            self.lbl_prob_keep_f_gene.setText( f'Current Probability to keep female gene: {self.parent.probability_keep_f_gene}' )
    
    def update_mix_genes_probability(self) :
        mix_genes_probability = self.input_mix_genes_probability.text()
        
        try:
            mix_genes_probability = float(mix_genes_probability)
            if mix_genes_probability < 0 or mix_genes_probability > 1 :
                raise ValueError('Invalid Entry')
        except:
            title = "Value Entry Error - Mix Genes Probability"
            warning_msg = "Entry value must be a float between 0 and 1."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.mix_genes_probability = mix_genes_probability
            self.lbl_mix_genes_probability.setText( f'Current probability to mix genes: {self.parent.mix_genes_probability}' )
    
    def update_mix_ratio_keep_f_gene(self) :
        mix_ratio_keep_f_gene = self.input_mix_ratio_keep_f_gene.text()
        
        try:
            mix_ratio_keep_f_gene = float(mix_ratio_keep_f_gene)
            if mix_ratio_keep_f_gene < 0 or mix_ratio_keep_f_gene > 1 :
                raise ValueError('Invalid Entry')
        except:
            title = "Value Entry Error - Ratio Keep Female Gene"
            warning_msg = "Entry value must be a float between 0 and 1."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.mix_ratio_keep_f_gene = mix_ratio_keep_f_gene
            self.lbl_mix_ratio_keep_f_gene.setText( f'Current female gene averaging weight: {self.parent.mix_ratio_keep_f_gene}' )
    
    def update_mutation_probability_rate(self) :
        mutation_probability_rate = self.input_mut_prob_rate.text()
        
        try:
            mutation_probability_rate = float(mutation_probability_rate)
            if mutation_probability_rate < 0 or mutation_probability_rate > 1 :
                raise ValueError('Invalid Entry')
        except:
            title = "Value Entry Error - Mutation Probability Rate"
            warning_msg = "Entry value must be a float between 0 and 1."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.mutation_probability_rate = mutation_probability_rate
            self.lbl_mut_prob_rate.setText( f'Current gene mutation rate: {self.parent.mutation_probability_rate}' )
    
    def update_mutate_uniform_gauss_ratio(self) :
        prob_uniform_v_normal = self.input_prob_uniform_v_normal.text()
        
        try:
            prob_uniform_v_normal = float(prob_uniform_v_normal)
            if prob_uniform_v_normal < 0 or prob_uniform_v_normal > 1 :
                raise ValueError('Invalid Entry')
        except:
            title = "Value Entry Error - Mutate Uniform vs Normal Distribution Ratio"
            warning_msg = "Entry value must be a float between 0 and 1."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.mutate_uniform_gauss_ratio = prob_uniform_v_normal
            self.lbl_prob_uniform_v_normal.setText( f'Current probability to use an uniform distribution: {self.parent.mutate_uniform_gauss_ratio}' )
    
    def update_mutation_max_factor_uniform_add_shift(self) :
        mutation_max_factor_uniform_add_shift = self.input_mut_max_uniform_add_shift.text()
        
        try:
            mutation_max_factor_uniform_add_shift = float(mutation_max_factor_uniform_add_shift)
            if mutation_max_factor_uniform_add_shift < 0 :
                raise ValueError('Invalid Entry')
        except:
            title = "Value Entry Error - Mutation, Max Uniform Shift"
            warning_msg = "Entry value must be greater than 0."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.mutation_max_factor_uniform_add_shift = mutation_max_factor_uniform_add_shift
            self.lbl_mut_max_uniform_add_shift.setText( f'Current uniform max shift factor: {self.parent.mutation_max_factor_uniform_add_shift}' )
    
    def update_mutation_max_factor_uniform_scaling(self) :
        mutation_max_factor_uniform_scaling = self.input_mut_max_uniform_scaling.text()
        
        try:
            mutation_max_factor_uniform_scaling = float(mutation_max_factor_uniform_scaling)
            if mutation_max_factor_uniform_scaling < 0 :
                raise ValueError('Invalid Entry')
        except:
            title = "Value Entry Error - Mutation, Max Uniform Scaling"
            warning_msg = "Entry value must be greater than 0."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.mutation_max_factor_uniform_scaling = mutation_max_factor_uniform_scaling
            self.lbl_mut_max_uniform_scaling.setText( f'Current uniform max scaling factor: {self.parent.mutation_max_factor_uniform_scaling}' )
    
    def update_mutation_max_factor_gauss_add_shift(self) :
        mutation_max_factor_gauss_add_shift = self.input_mut_max_gauss_add_shift.text()
        
        try:
            mutation_max_factor_gauss_add_shift = float(mutation_max_factor_gauss_add_shift)
            if mutation_max_factor_gauss_add_shift < 0 :
                raise ValueError('Invalid Entry')
        except:
            title = "Value Entry Error - Mutation, Normal Shift Standard Deviation"
            warning_msg = "Entry value must be greater than 0."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.mutation_max_factor_gauss_add_shift = mutation_max_factor_gauss_add_shift
            self.lbl_mut_max_gauss_add_shift.setText( f'Current normal max shift factor: {self.parent.mutation_max_factor_gauss_add_shift}' )
    
    def update_mutation_max_factor_gauss_scaling(self) :
        mutation_max_factor_gauss_scaling = self.input_mut_max_gaussian_scaling.text()
        
        try:
            mutation_max_factor_gauss_scaling = float(mutation_max_factor_gauss_scaling)
            if mutation_max_factor_gauss_scaling < 0 :
                raise ValueError('Invalid Entry')
        except:
            title = "Value Entry Error - Mutation, Normal Scaling Standard Deviation"
            warning_msg = "Entry value must be greater than 0."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.mutation_max_factor_gauss_scaling = mutation_max_factor_gauss_scaling
            self.lbl_mut_max_gaussian_scaling.setText( f'Current normal max scaling factor: {self.parent.mutation_max_factor_gauss_scaling}' )
    
    def update_std_scaling_factor(self) :
        std_scaling_factor = self.input_std_scaling_factor.text()
        
        try:
            std_scaling_factor = float(std_scaling_factor)
            if std_scaling_factor < 0 :
                raise ValueError('Invalid Entry')
        except:
            title = "Value Entry Error - Population Stats, Scale Standard Deviation Factor"
            warning_msg = "Entry value must be greater than 0."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.std_scaling_factor = std_scaling_factor
            self.lbl_std_scaling_factor.setText( f'Current scaling factor: {self.parent.std_scaling_factor}' )
    
    def update_save_best_num(self) :
        save_best_num = self.input_save_best_num.text()
        
        try:
            save_best_num = float(save_best_num)
            if save_best_num < 0 :
                raise ValueError('Invalid Entry')
            
            if save_best_num < 1 :
                save_best_num = int( save_best_num*self.parent.num_individual_AIs )
            else :
                save_best_num = int( save_best_num )
        except:
            title = "Value Entry Error - Number of Best to Save"
            warning_msg = "Entry value must be a float between 0 and 1 or an integer equal to or greater than 1."
            warning_window = warningWindow(self)
            warning_window.build_window(title=title, msg=warning_msg)
        else :
            self.parent.save_best_num = save_best_num
            self.lbl_save_best_num.setText( f'Current AI shape: {self.parent.save_best_num}' )
    
    def lock_inputs(self) :
        self.input_AI_shape.setEnabled(False)
        self.btn_AI_shape.setEnabled(False)
        self.input_keep_best.setEnabled(False)
        self.btn_keep_best.setEnabled(False)
        self.input_propagate_best.setEnabled(False)
        self.btn_propagate_best.setEnabled(False)
        self.input_min_score_percentage.setEnabled(False)
        self.btn_min_score_percentage.setEnabled(False)
        self.input_breed_v_pop_stats.setEnabled(False)
        self.btn_breed_v_pop_stats.setEnabled(False)
        self.input_prob_keep_f_gene.setEnabled(False)
        self.btn_prob_keep_f_gene.setEnabled(False)
        self.input_mix_genes_probability.setEnabled(False)
        self.btn_mix_genes_probability.setEnabled(False)
        self.input_mix_ratio_keep_f_gene.setEnabled(False)
        self.btn_mix_ratio_keep_f_gene.setEnabled(False)
        self.input_std_scaling_factor.setEnabled(False)
        self.btn_std_scaling_factor.setEnabled(False)
        self.input_mut_prob_rate.setEnabled(False)
        self.btn_mut_prob_rate.setEnabled(False)
        self.input_prob_uniform_v_normal.setEnabled(False)
        self.btn_prob_uniform_v_normal.setEnabled(False)
        self.input_mut_max_uniform_add_shift.setEnabled(False)
        self.btn_mut_max_uniform_add_shift.setEnabled(False)
        self.input_mut_max_uniform_scaling.setEnabled(False)
        self.btn_mut_max_uniform_scaling.setEnabled(False)
        self.input_mut_max_gauss_add_shift.setEnabled(False)
        self.btn_mut_max_gauss_add_shift.setEnabled(False)
        self.input_mut_max_gaussian_scaling.setEnabled(False)
        self.btn_mut_max_gaussian_scaling.setEnabled(False)
        self.input_save_best_num.setEnabled(False)
        self.btn_save_best_num.setEnabled(False)
        self.btn_save_settings_begin_training.setEnabled(False)

class training_page(QWidget) :
    def __init__(self, parent) :
        super(QWidget, self).__init__(parent)
        
        #self.track_env = self.parent.track_env
        self.frame_time = 1./30
        
        self.parent = parent
        
        self.layout = QGridLayout(self)
        
        self.setLayout(self.layout)
    
    def create_training_layout(self) :
        max_widget_width = 200
        
        self.parent.prep_AI_shape_to_parameter()
        AI_params_dict = self.parent.get_AI_params_dict()
        game_control_params = self.parent.get_game_control_params()
        AI_shape = self.parent
        
        self.track_env = track_environment(
                             game_control_params=game_control_params,
                             params_dict = AI_params_dict,
                             genetic_AI_params_to_load = None,
                             track_params_to_load = None,
                             car_file = self.parent.car_file,
                             track_file_loc = self.parent.track_file,
                             AI_shape = self.parent.AI_shape,
                             FPS = self.parent.FPS,
                             load_previous_gen = self.parent.load_prev_gen,
                             )
        
        self.game_widget = pygame_widget(self.track_env, self)
        self.layout.addWidget(self.game_widget, 0, 0, 1, 1)
        
        self.layout.addWidget(QVLine(), 0, 1)
        
        control_layout = QGridLayout() # QVBoxLayout()
        cntrl_row = 0
        
        btn = QPushButton("Pause Game")
        btn.setMaximumWidth(max_widget_width)
        btn.clicked.connect(self.pause_game)
        control_layout.addWidget(btn, cntrl_row, 0); cntrl_row += 1
        
        btn = QPushButton("Pause After This Generation")
        btn.setMaximumWidth(max_widget_width)
        btn.clicked.connect(self.flip_pause_after_gen)
        control_layout.addWidget(btn, cntrl_row, 0); cntrl_row += 1
        
        self.input_training_mode = QComboBox()
        self.input_training_mode.addItem("0 - Display All Cars")
        self.input_training_mode.addItem("1 - Display Some")
        self.input_training_mode.addItem("2 - Train All in Background")
        self.input_training_mode.setCurrentIndex(self.track_env.training_mode)
        self.input_training_mode.setMaximumWidth(max_widget_width)
        self.input_training_mode.currentIndexChanged.connect(self.update_training_mode)
        control_layout.addWidget(self.input_training_mode, cntrl_row, 0); cntrl_row += 1
        
        hline = QHLine()
        hline.setMaximumWidth( max_widget_width )
        control_layout.addWidget(hline, cntrl_row, 0); cntrl_row += 1
        
        exit_warning = '\n'.join([
            'Warning: Do not exit the program while "training in background: True" '
            'is displayed in the game information. This indicates that multithreading '
            'is being used for training. Exiting while this is True will leave '
            'running threads hanging causing the program to exit incorrectly.\n'
            ])
        label = QLabel(exit_warning)
        label.setMaximumWidth( max_widget_width )
        label.setWordWrap(True)
        control_layout.addWidget(label, cntrl_row, 0); cntrl_row += 1
        
        hline = QHLine()
        hline.setMaximumWidth( max_widget_width )
        control_layout.addWidget(hline, cntrl_row, 0); cntrl_row += 1
        
        exit_warning = '\n'.join([
            'To exit correctly, please click "Pause After This Generation" and '
            'then wait for "training in background" to be False.\n'
            ])
        label = QLabel(exit_warning)
        label.setMaximumWidth( max_widget_width )
        label.setWordWrap(True)
        control_layout.addWidget(label, cntrl_row, 0); cntrl_row += 1
        
        hline = QHLine()
        hline.setMaximumWidth( max_widget_width )
        control_layout.addWidget(hline, cntrl_row, 0); cntrl_row += 1
        
        control_layout.setRowStretch(cntrl_row, 1)
        
        #control_layout.addStretch()
        self.layout.addLayout(control_layout, 0, 2)
        
        self.timer = QTimer()
        self.timer.timeout.connect( self.get_next_frame )
        self.timer.start( self.frame_time )
    
    def get_next_frame(self) :
        self.game_widget.update_game()
    
    def pause_game(self) :
        self.track_env.pause_game = not self.track_env.pause_game
    
    def flip_pause_after_gen(self) :
        self.track_env.pause_after_gen = not self.track_env.pause_after_gen
    
    def update_training_mode(self) :
        idx = self.input_training_mode.currentIndex()
        self.track_env.training_mode = idx

class pygame_widget(QWidget) :
    def __init__(self, track_env, parent=None) :
        super(pygame_widget, self).__init__(parent)
        self.parent = parent
        self.track_env = track_env
        self.surface = self.track_env.game_window
        self.display_width = self.track_env.display_width
        self.display_height = self.track_env.display_height
        self.FPS = self.track_env.FPS
        self.data = self.surface.get_buffer().raw
        self.image = QImage(self.data, self.display_width, self.display_height, QImage.Format_RGB32)
    
    def paintEvent(self, event=None) :
        qp = QPainter()
        qp.begin(self)
        qp.drawImage(0, 0, self.image)
        qp.end()
        self.parent.update()
    
    def update_game(self) :
        self.track_env.next_game_frame()
        self.data = self.surface.get_buffer().raw
        self.image = QImage(self.data, self.display_width, self.display_height, QImage.Format_RGB32)

class QHLine(QFrame):
    """
    """
    def __init__(self):
        """
        

        Returns
        -------
        None.

        """
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

class QVLine(QFrame):
    """
    """
    def __init__(self):
        """
        

        Returns
        -------
        None.

        """
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)

class warningWindow(QDialog):
    """
    """
    def __init__(self, *args, **kwargs):
        """
        

        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(warningWindow, self).__init__(*args, **kwargs)
        self.title = ''
        self.msg = ''
    
    def set_title(self, title) :
        """
        

        Parameters
        ----------
        title : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.title = title
    
    def set_msg(self, msg) :
        """
        

        Parameters
        ----------
        msg : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.msg = msg
    
    def set_text_msgs(self, title, msg) :
        """
        

        Parameters
        ----------
        title : TYPE
            DESCRIPTION.
        msg : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.title = title
        self.msg = msg
    
    def build_window(self, title=None, msg=None) :
        """
        

        Parameters
        ----------
        title : TYPE, optional
            DESCRIPTION. The default is None.
        msg : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if title is not None :
            self.title = title
        if msg is not None :
            self.msg = msg
        
        self.setWindowTitle(self.title)
        
        QBtn = QDialogButtonBox.Ok # | QDialogButtonBox.Cancel
        
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)

        self.layout = QVBoxLayout()
        
        label = QLabel(self.msg)
        self.layout.addWidget(label)
        
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        
        self.exec_()

if __name__ == '__main__' :
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec_()
    sys.exit()
