# FO24Z-gr34

# Basic usage
1. install all required python dependencies with command
pip install -r requirements.txt

2. run application with right configuration file
python heat_equation.py --config <configuration_file>

in case of our base configuraiton, run
python heat_equation.py --config config.json

# Configuration
Config.json file allows to configure:
• wait: time interval between updates  
• x_size, y_size: domain dimensions  
• T: total simulation duration  
• dt: time step used in updates  
• plots: number of plot outputs through out simulation
• points: coordinates with initial temperature values 
• default_val: default temperature of points
