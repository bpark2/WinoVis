# WinoVis

WinoVis is a novel dataset specifically designed to probe text-to-image models on pronoun disambiguation within multimodal contexts ([paper link](https://arxiv.org/abs/2405.16277)). Utilizing [GPT-4](https://openai.com/index/gpt-4-research/) for prompt generation and [Diffusion Attentive Attribution Maps (DAAM)](https://github.com/castorini/daam) for heatmap analysis, we propose a novel evaluation framework that isolates the models' ability in pronoun disambiguation from other visual processing challenges.

<div align="center">
  <img src="https://github.com/bpark2/WinoVis/blob/master/bee_example.png" width="255" height="312">
</div>

## Getting Started
First, download a local copy of this git repo. After you've installed the repo, you can install all dependencies by navigating to the project and using the command `pip install -r requirements.txt`.

## Running WinoVis
WinoVis consists of two key functionalities. 
  ⋅ Generating images and overlaying their heatmaps onto them
  ⋅ Evaluating the performance of your chosen diffusion model in the task of WinoVis using default or user-specified parameters.
  
The code for WinoVis has been designed to allow the user to specify their desired diffusion model, as well as what functions to execute and with what parameters. 
Models which have been tested/used with WinoVis include the following:
  * [CompVis/stable-diffusion-v1-1](https://huggingface.co/CompVis/stable-diffusion-v1-1)
  * [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
  * [stabilityai/stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base)
  * [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

Examples:
  * `python __init__.py stabilityai/stable-diffusion-2-base 0`- Here the user specifies SD 2.0 as their model for image generation and uses `0` to indicate that they would like to run both image generation and model evaluation.
  * `python __init__.py stabilityai/stable-diffusion-2-base 1`- Here, the user uses `1` to indicate they only want to generate and overlay images.
  * `python __init__.py stabilityai/stable-diffusion-2-base 2`- By using `2` the user has specified they only want to evaluate a model which has presumably already generated images and their heatmaps.
  * `python __init__.py stabilityai/stable-diffusion-2-base 0 0.4 0.6` - This input matches the first example but provides two additional parameters, the decision boundary (0.4) and overlap threshold (0.6).
If the user does not provide the values for the decision boundary and overlap threshold, the default values of 0.4 discussed in the paper are used.

When running WinoVis you will be prompted to specify the name of your dataset from a list of any datasets included in the `data` folder. If you generated a custom dataset matching the WinoVis format and wish to use it, please ensure it is located in the `data` folder so that it can be used.

If you are also evaluating your model, after images have been generated you will be prompted to provide the name of the heatmap pickle file (saved in the `heatmaps` folder) and the dataset matching that heatmap folder (for example `wsv.pkl`). Afterwards you will be presented with a performance breakdown of your specified model.

## Interpreting Output
Generated images are added to a folder titled `model-name_wsv_images` which can be found in `images`. The versions of these images with heatmaps overlayed are stored in a similarly titled folder which specifies the model name and heatmap threshold. This folder can be found in `hm_images`.

## Citation
```
@inproceedings{tbd,
    title = "Picturing Ambiguity: A Visual Twist on the Winograd Schema Challenge",
    author = 
    year = "2024",
    url = "",
}
```
