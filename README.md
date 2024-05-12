# WinoVis

WinoVis is a novel dataset specifically designed to probe text-to-image models on pronoun disambiguation within multimodal contexts. Utilizing GPT-4 for prompt generation and Diffusion Attentive Attribution Maps (DAAM) for heatmap analysis, we propose a novel evaluation framework that isolates the models' ability in pronoun disambiguation from other visual processing challenges.

## Getting Started
First, download a local copy of this git repo. After you've installed the repo, you can install all dependencies by navigating to the project and using the command `pip install -r requirements.txt`.

## Running WinoVis
WinoVis consists of two key functionalities. 
  ⋅ Generating images and overlaying their heatmaps onto them
  ⋅ Evaluating the performance of your chosen diffusion model in the task of WinoVis using default or user-specified parameters.
  
The code for WinoVis has been designed to allow the user to specify their desired diffusion model, as well as what functions to execute and with what parameters.
Examples:
  * `python __init__.py stabilityai/stable-diffusion-2-base 0`- Here the user specifies SD 2.0 as their model for image generation and uses 0 to indicate that they would like to run both image generation and model evaluation.
  * `python __init__.py stabilityai/stable-diffusion-2-base 1`- Here, the user uses 1 to indicate they only want to generate and overlay images.
  * `python __init__.py stabilityai/stable-diffusion-2-base 2`- By using `2` the user has specified they only want to evaluate a model which has presumably already generated images and their heatmaps.
  * `python __init__.py stabilityai/stable-diffusion-2-base 0 0.4 0.6` - This input matches the first example but provides two additional parameters, the decision boundary (0.4) and overlap threshold (0.6).
