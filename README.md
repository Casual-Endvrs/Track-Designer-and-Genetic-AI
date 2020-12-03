Track Designer and Genetic AI
----------------------------
This is a program designed to allow users to quickly and easily create a simple race track.

There should be no need to edit the python files as all parameters that are intended to be changed are stored in *.json files. Documentation provides a description for what each parameter does in each json file. (Note: this is true for all track designing json files. The parameters for the AI trainer can be editted within the program and descriptions are provided there.)

To create a track please begin with the following steps.
1) Create all *.json files by running "create_all_defaults.py". This is a good time to edit them for your purposes.
2) Create a track using "trackDesigner.py".
3) To test the track to ensure it is drivable for the AI use "single_player.py".
4) Train the AI using "AI_drivers.py".





Genetic AI
----------------------------
This program is used to train a basic genetic algorithm AI to drive around a track.

To start the AI Training program run "AI Trainer - GUI.py".

The Introduction page contains information about how to proceed, what some parameters do and more specific information about types of selection and propagation.

It is recommended that you do not use hidden layers in your AIs as this will significantly increase the complexity and training time required.

Some recommended setting changes are:
	- increase "Number of Vision Rays" to 15
	- increase "Total Number of AIs" to 1,000 or more (especially if you are using hidden layers)
	- decrease "Max Number of AIs on Track" to 10-12 (This will cause the training shown on screen to be smoother)


