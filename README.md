# Lunar Lander
## Description
This projects goal was to implement a smart agent to learn OpenAI Gym's
lunar lander environment, such that it can consistently land within the
designated zone. The agent uses on policy Q-Learning on top of a
neural network that trains on experience replay. 

## Project structure
* LunarLander
    * lunar_lander
        * nn_learning_agent.py
    * main.py
    * README.md

## How To Run
1. Install Python 3.5

2. Please install scikit-learn and numpy
   http://scikit-learn.org/stable/install.html

3. Using Python 3.5, run `main.py`. U may with to modify the variable `record`, 
which is currently set to False, as well as use a different api_key if you plan to upload to OpenAI

4. Some results are printed to standard output after 100 episodes.

5. If `record` is set to True, the results will be uploaded to OpenAI, 
and the url will be printed to standard out. Currently the agent is set 
to train for 25,000 episodes which can take several hours.

6. Results that I recently uploaded can be found here. https://gym.openai.com/evaluations/eval_SpmmaCg7QEqeU47M8kkkw
