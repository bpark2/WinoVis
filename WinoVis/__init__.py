import os
import daam
import torch
import pickle
import random
import argparse
import numpy as np
from PIL import Image
from tabulate import tabulate
from matplotlib import pyplot as plt
from diffusers import StableDiffusionPipeline, DiffusionPipeline

def make_im_subplot(*args):
    fig, ax = plt.subplots(*args)
    for ax_ in ax.flatten():
        ax_.set_xticks([])
        ax_.set_yticks([])
    return fig,ax

def prompt_fname(root,prompt,extension):
    '''
    This function prompts the user to provide a file name and checks whether the file
    exists and if it has the specified extension.
    :param root: root directory containing files user is being prompted to specify
    :param prompt: prompt provided to user which specifies which file to select
    :param extension: expected file extension
    :return: filename provided by user
    '''
    if not root == "":
        os.chdir(root)
    pkl_files = [f for f in os.listdir() if os.path.isfile(f) and f.__contains__(extension)]
    while True:
        print(pkl_files)
        fname = input(prompt)
        if os.path.isfile(fname) and fname.__contains__(extension):
            if not root == "":
                os.chdir("..")
                return root+"/"+fname
            else:
                return fname
        else:
            print("File does not exist or does not contain extension \'"+extension+"\'")

steps = [50]
model_ids = ['stabilityai/stable-diffusion-2-base','CompVis/stable-diffusion-v1-1','runwayml/stable-diffusion-v1-5']#'CompVis/stable-diffusion-v1-1','runwayml/stable-diffusion-v1-5'
heatmap_threshold = 90
decision_boundary = 0.4
overlap_threshold = 0.4
random.seed(42)

def generate_images():
    '''
    This method prompts stable diffusion to generate images using prompts from WSV. Once all
    images for a model are generated, versions of each images with their heatmaps overlayed
    are generated and saved.
    '''
    fname = prompt_fname("data", "Please provide filename for dataset: ", ".pkl")
    heatmaps = {}  # sentence:[entity1_hm, entity2_hm, pron_hm]
    with open(fname, 'rb') as pkl_f:
        wsv = pickle.load(pkl_f)

    count = 0
    for id in model_ids:#in the case where we want to generate images using multiple SD models
        print("Generating images using "+id)
        try:
            if id.__contains__("xl-base"):
                model = DiffusionPipeline.from_pretrained(id, use_auth_token=True, torch_dtype=torch.float16, use_safetensors=True, variant='fp16')
            else:
                model = StableDiffusionPipeline.from_pretrained(id)#loading pretrained model weights
        except Exception as error:
            print("The following error has occured:\n"+str(error))
            return
        model = model.to('cuda')#moving model to gpu
        save_id = id.split("/")[1]

        if os.path.isfile("heatmaps/" + str(save_id) + "_heatmaps.pkl"):#we've already generated some images/heatmaps of this dataset
            with open("heatmaps/" + str(save_id) + "_heatmaps.pkl", 'rb') as pkl_f:
                heatmaps = pickle.load(pkl_f)
        for step in steps:#in the case where we want to generate an image for multiple ranges of steps
            #preparing save folder
            img_save_folder = "images/"+save_id+"_"+fname.rsplit("/",1)[1].replace(".pkl","")+"_images"
            if not os.path.isdir(img_save_folder):
                os.makedirs(img_save_folder, exist_ok=True)
            for prompt in wsv.keys():#for each WSV sample
                if not os.path.isfile(img_save_folder+"/" + prompt.replace(".", "") + ".png"): #if not already generated
                    #prompting Stable Diffusion while tracing heatmaps
                    with daam.trace(model) as trc:
                        output_image = model(prompt, num_inference_steps=step).images[0]
                    #calculating heatmaps
                    global_heat_map = trc.compute_global_heat_map()
                    #saving heatmaps to heatmap dictionary
                    # Heatmap dictionaries --> {prompt:[ent1_hm, ent2_hm, pron_hm]}
                    ent1_hm = global_heat_map.compute_word_heat_map(
                        wsv[prompt]['options'][0].replace("the ", "").strip())
                    ent2_hm = global_heat_map.compute_word_heat_map(
                        wsv[prompt]['options'][1].replace("the ", "").strip())
                    pron_hm = global_heat_map.compute_word_heat_map(wsv[prompt]['pronoun'])
                    # saving output image
                    heatmaps[prompt.replace(".","")] = [ent1_hm, ent2_hm, pron_hm]
                    output_image.save(img_save_folder+"/"+prompt.replace(".","")+".png")
                else:
                    print(img_save_folder+"/" + prompt.replace(".", "").replace("\"","") + ".png \n Already Generated ^")
                count += 1
                print(str(count)+"/"+str(len(wsv.keys()))+" images generated")
            #saving dictionary of heatmaps to .pkl file
            with open("heatmaps/"+str(save_id)+"_heatmaps.pkl", 'wb') as fp:
                pickle.dump(heatmaps, fp)
                print("Heatmaps saved to pkl file")

            #Overlaying heatmaps on to generated images
            print("Generating images with heatmap overlays")
            # folder = "images/"+save_id+"_"+fname.rsplit("/",1)[1].replace(".pkl","")+"_images"
            heatmaps_fname = "heatmaps/"+str(save_id)+"_heatmaps.pkl"  # file containing heatmaps of generated images
            with open(heatmaps_fname, 'rb') as pkl_f:
                img_heatmaps = pickle.load(pkl_f)
            # updating the save name
            save_folder = img_save_folder.rsplit("/", 1)[1].replace(".pkl", "")

            if not os.path.isdir("hm_images/"+save_folder + "_" + str(heatmap_threshold) + "p"):
                os.makedirs("hm_images/"+save_folder + "_" + str(heatmap_threshold) + "p",exist_ok=True)

            # open dictionary of WSV sentences and their answers: filename is folder after model_id and before "_images" plus .pkl
            for image in os.listdir(img_save_folder):
                prompt = image.replace(".png", "")
                output_image = Image.open(img_save_folder + "/" + image)
                ent1_hm = img_heatmaps[prompt][0]
                ent2_hm = img_heatmaps[prompt][1]
                pron_hm = img_heatmaps[prompt][2]

                # thresholds for percentiles (specified by heatmap_threshold)
                thresholds = {ent1_hm: np.percentile(torch.flatten(ent1_hm.heatmap.cpu()).numpy(), heatmap_threshold),
                              ent2_hm: np.percentile(torch.flatten(ent2_hm.heatmap.cpu()).numpy(), heatmap_threshold),
                              pron_hm: np.percentile(torch.flatten(pron_hm.heatmap.cpu()).numpy(), heatmap_threshold)}

                # preparing diagram
                plt.rcParams['figure.figsize'] = (8, 8)

                # updating heatmaps to be thresholded
                ent1_hm.heatmap = torch.where(ent1_hm.heatmap < thresholds[ent1_hm], torch.tensor(0),
                                              ent1_hm.heatmap)  # torch.Tensor(np.where(ent1_hm.heatmap.cpu() < thresholds[ent1_hm], torch.tensor(1), ent1_hm.heatmap.cpu(), 0))
                ent2_hm.heatmap = torch.where(ent2_hm.heatmap < thresholds[ent2_hm], torch.tensor(0),
                                              ent2_hm.heatmap)  # torch.Tensor(np.where(ent2_hm.heatmap.cpu() < thresholds[ent2_hm], ent2_hm.heatmap.cpu(), 0))
                pron_hm.heatmap = torch.where(pron_hm.heatmap < thresholds[pron_hm], torch.tensor(0),
                                              pron_hm.heatmap)  # torch.Tensor(np.where(pron_hm.heatmap.cpu() < thresholds[pron_hm], pron_hm.heatmap.cpu(), 0))

                fig, ax = make_im_subplot(2, 2)

                # Original image
                ax[0, 0].imshow(output_image)
                # Entity 1 heat map
                ent1_hm.plot_overlay(output_image, ax=ax[0, 1])
                # Entity 2 heat map
                ent2_hm.plot_overlay(output_image, ax=ax[1, 0])
                # Pronoun heat map
                pron_hm.plot_overlay(output_image, ax=ax[1, 1])

                fig.suptitle(image.rsplit(".", 1)[0])

                # saving image to folder

                plt.savefig("hm_images/"+save_folder + "_" + str(heatmap_threshold) + "p/" + prompt + "_heatmap.png")
                plt.close(fig)

def get_binary_mask(heatmap):#note: heatmaps are already thresholded
    return torch.where(heatmap != 0, torch.tensor(1), heatmap)

def get_iou(heatmap1, heatmap2):
    '''
    This function calculates the Intersection over Union (IoU) of two heatmaps.
    :param heatmap1: heatmap of first image
    :param heatmap2: heatmap of second image
    :return: IoU of two heatmaps
    '''
    intersect = torch.logical_and(heatmap1, heatmap2)
    union = torch.logical_or(heatmap1, heatmap2)
    i_area = torch.nansum(intersect)
    u_area = torch.nansum(union)
    iou =  i_area / u_area
    return iou #dividing area of intersection by area of union

def measure_acc(iou_db,iou_ol):
    '''
    This function calculates the accuracy of a given model based on its generated heatmaps using
    the specified iou_db and iou_ol
    :param iou_db: IoU decision boundary - used to determine if a decision is made by the model. If both the heatmaps of
    each entity have an iou value below the iou_db, the model's decision is considered a 'neither' decision
    :param iou_ol: IoU Overlap - used to determine when the heatmap of two entities is significantly overlapped and cannot be assessed
    '''
    hm_fname = prompt_fname("heatmaps/","Provide heatmap file of your desired model: ",".pkl")
    with open(hm_fname, 'rb') as pkl_f:
        heatmaps = pickle.load(pkl_f)

    wsv_fname = prompt_fname("data","Provide file name of wsv dictionary matching the heatmaps: ", ".pkl")
    with open(wsv_fname, 'rb') as pkl_f:
        wsv = pickle.load(pkl_f)

    model_name = hm_fname.rsplit("/", 1)[1].split("_")[0]
    preds = [[0,0],#guess\actual| entity 1 | entity 2
                [0,0],#entity 1 |
                [0,0],#entity 2 |
                [0,0]]#neither  |
                      #invalid  |

    #getting list of images generated (which were not removed due to captioning)
    image_path = 'images/'+model_name+'_wsv_images'
    if not os.path.isdir(image_path):
        print("Missing folder containing images generated by "+model_name+".\nPlease make sure images are generated before measuring accuracy")
        return
    folds = list(os.listdir(image_path))
    y_true = []
    y_pred = []

    for key in wsv:
        if key+".png" in folds:#only evaluating heatmaps which correspond to an image contained within the 'folds' folder
            answer = wsv[key]['answer']
            ent1_hm = heatmaps[key][0]
            ent2_hm = heatmaps[key][1]
            pron_hm = heatmaps[key][2]

            #calculating threshold values
            thresholds = {ent1_hm: np.percentile(torch.flatten(ent1_hm.heatmap.cpu()).numpy(), heatmap_threshold),
                          ent2_hm: np.percentile(torch.flatten(ent2_hm.heatmap.cpu()).numpy(), heatmap_threshold),
                          pron_hm: np.percentile(torch.flatten(pron_hm.heatmap.cpu()).numpy(), heatmap_threshold)}

            #applying heatmap thresholds to each respective heatmap
            ent1_hm.heatmap = torch.where(ent1_hm.heatmap < thresholds[ent1_hm], torch.tensor(0), ent1_hm.heatmap)
            ent2_hm.heatmap = torch.where(ent2_hm.heatmap < thresholds[ent2_hm], torch.tensor(0), ent2_hm.heatmap)
            pron_hm.heatmap = torch.where(pron_hm.heatmap < thresholds[pron_hm], torch.tensor(0), pron_hm.heatmap)

            #converting to binary mask
            ent1_binary = get_binary_mask(ent1_hm.heatmap)
            ent2_binary = get_binary_mask(ent2_hm.heatmap)
            pron_binary = get_binary_mask(pron_hm.heatmap)

            #calculating IoU overlaps
            ent_overlap = get_iou(ent1_binary, ent2_binary)
            ent1_pron_overlap = get_iou(ent1_binary, pron_binary)
            ent2_pron_overlap = get_iou(ent2_binary, pron_binary)

            if ent_overlap > iou_ol:#unevaluable - iou_ent1_pron_overlap and iou_ent_overlap > iou_ent2_pron_overlap:
                if answer == 0:
                    preds[3][0] += 1
                else:
                    preds[3][1] += 1
            elif ent1_pron_overlap < iou_db and ent2_pron_overlap < iou_db:#neither
                if answer == 0:
                    preds[2][0] += 1
                else:
                    preds[2][1] += 1
            elif ent1_pron_overlap > ent2_pron_overlap:#guessed entity 1
                y_pred.append(0)
                if answer == 0:#answer: entity 1
                    preds[0][0] += 1
                    y_true.append(0)
                else:#answer: entity 2
                    preds[0][1] += 1
                    y_true.append(1)
            elif ent2_pron_overlap > ent1_pron_overlap:#guessed entity 2
                y_pred.append(1)
                if answer == 1:#answer: entity 2
                    preds[1][1] += 1
                    y_true.append(1)
                else:#answer: entity 1
                    preds[1][0] += 1
                    y_true.append(0)
    data = {
        'Predicted\\Actual': ['Entity 1', 'Entity 2', 'Neither', 'Overlapped', 'Totals'],
        'Entity 1': [str(preds[0][0]), str(preds[1][0]), str(preds[2][0]), str(preds[3][0]), str(preds[0][0] + preds[1][0] + preds[2][0] + preds[3][0])],
        'Entity 2': [str(preds[0][1]), str(preds[1][1]), str(preds[2][1]), str(preds[3][1]), str(
        preds[0][1] + preds[1][1] + preds[2][1] + preds[3][1])]
    }
    #printing out table and metrics
    print(tabulate(data, headers="keys", tablefmt="fancy_grid"))
    precision = preds[0][0]/(preds[0][0]+preds[0][1])
    recall = preds[0][0]/(preds[0][0]+preds[1][0])
    f1 = (2*precision*recall)/(precision+recall)
    accuracy = (preds[0][0] + preds[1][1])/(preds[0][0] + preds[0][1] + preds[1][0] + preds[1][1])
    certainty = (preds[0][0] + preds[0][1] + preds[1][0] + preds[1][1])/(preds[0][0] + preds[0][1] + preds[1][0] + preds[1][1] + preds[2][0] + preds[2][1])
    print("Precision: "+str(round(precision,4)))
    print("Recall: "+str(round(recall,4)))
    print("F1: "+str(round(f1,4)))
    print("Accuracy: "+str(round(accuracy,4)))
    print("Certainty: "+str(round(certainty,4)))


parser = argparse.ArgumentParser(description='WinoVis')
parser.add_argument('diff_mod', metavar='diffusion model', type=str, nargs=1)
parser.add_argument('functions', metavar='functions', type=int, nargs=1, help='specifies which functions to run.\n 0: runs image generation\n 1: runs model evaluation\n 2: runs image generation and model evaluation')
parser.add_argument('db', metavar='decision boundary', type=float, nargs='?', help='float value used to specify decision boundary (0.4 by default)')
parser.add_argument('ot', metavar='overlap threshold', type=float, nargs='?', help='float value used to specify the overlap threshold (0.4 by default)')
args = parser.parse_args()

if not str(args.diff_mod).__contains__('stable-diffusion-xl'):#likely will remove this statement later, just here for now to remind me to test sdxl
    if args.diff_mod == None:#no diffusion model specified
        print("Missing specification of diffusion model. Common choices include " + ' '.join(
            model_ids) + ". SD 2.0 will be used")
        model_ids = ['stabilityai/stable-diffusion-2-base']
    else:
        model_ids = [str(args.diff_mod[0])]
    if args.functions[0] == 0: #generates images and runs measure_acc since no functions are specified
        generate_images()
        if not args.db == None and args.db <= 1.0:
            decision_boundary = args.db
        if not args.ot == None and args.ot <= 1.0:
            overlap_threshold = args.ot
        measure_acc(decision_boundary,overlap_threshold)
    elif args.functions[0] == 1:#generate_images specified
        generate_images()
    elif args.functions[0] == 2:#measure_acc specified
        if not args.db == None and args.db <= 1.0:
            decision_boundary = args.db
        if not args.ot == None and args.ot <= 1.0:
            overlap_threshold = args.ot
        measure_acc(decision_boundary,overlap_threshold)
else: #user specified sdxl - still need to modify code to work for both
    print('SDXL support not yet tested. May lead to unexpected errors.')
    model_ids = ["stabilityai/stable-diffusion-xl-base-1.0"]
    generate_images()
    if not args.db == None and args.db <= 1.0:
        decision_boundary = args.db
    if not args.ot == None and args.ot <= 1.0:
        overlap_threshold = args.ot
    measure_acc(decision_boundary,overlap_threshold)