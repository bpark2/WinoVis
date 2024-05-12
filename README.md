# WinoVis

WinoVis is a novel dataset specifically designed to probe text-to-image models on pronoun disambiguation within multimodal contexts. Utilizing GPT-4 for prompt generation and Diffusion Attentive Attribution Maps (DAAM) for heatmap analysis, we propose a novel evaluation framework that isolates the models' ability in pronoun disambiguation from other visual processing challenges.

## Getting Started
First, download a local copy of this git repo. After you've installed the repo, you can install all dependencies by navigating to project and using the command `pip install -r requirements.txt`.

## Running WinoVis
WinoVis consists of two key functionalities. 
  ⋅ Generating images and overlaying their heatmaps onto them
  ⋅ Evaluating the performance of your chosen diffusion model in the task of WinoVis using default or user-specified parameters.
  
The code for WinoVis has been designed to allow the user to specify their desired diffusion model, as well as what functions to execute and with what parameters.
