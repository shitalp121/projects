#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U scikit-fuzzy')


# In[2]:


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# input variables
temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')

# output variable
automation_level = ctrl.Consequent(np.arange(0, 101, 1), 'automation_level')

# fuzzy sets for input variables
temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 50])
temperature['medium'] = fuzz.trimf(temperature.universe, [0, 50, 100])
temperature['high'] = fuzz.trimf(temperature.universe, [50, 100, 100])

humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['medium'] = fuzz.trimf(humidity.universe, [0, 50, 100])
humidity['high'] = fuzz.trimf(humidity.universe, [50, 100, 100])

# fuzzy sets for output variable
automation_level['low'] = fuzz.trimf(automation_level.universe, [0, 0, 50])
automation_level['medium'] = fuzz.trimf(automation_level.universe, [0, 50, 100])
automation_level['high'] = fuzz.trimf(automation_level.universe, [50, 100, 100])

#  fuzzy rules
rule1 = ctrl.Rule(temperature['low'] & humidity['low'], automation_level['low'])
rule2 = ctrl.Rule(temperature['medium'] & humidity['medium'], automation_level['medium'])
rule3 = ctrl.Rule(temperature['high'] & humidity['high'], automation_level['high'])

# Create control system
automation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

# Create simulation
automation_sim = ctrl.ControlSystemSimulation(automation_ctrl)

# Set input values
automation_sim.input['temperature'] = 25  # We can replace with actual temperature value
automation_sim.input['humidity'] = 70  # We can replace with actual humidity value

# Compute the automation level
automation_sim.compute()

# Access the defuzzified results
# Printing the automation level
print("Automation Level:", automation_sim.output['automation_level'])

temperature.view(sim=automation_sim)
humidity.view(sim=automation_sim)
automation_level.view(sim=automation_sim)



# In[ ]:





# In[ ]:




