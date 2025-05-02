
import os, sys
import json
from argparse import ArgumentParser, Namespace

def get_folder_list(dataset):
    if dataset == "dycheck":
        folder_list = ["apple", "block", "spin", "paper-windmill", "space-out", "teddy", "wheel"]       
    elif dataset == "dynerf":
        folder_list = ["coffee_martini", "cook_spinach", "cut_roasted_beef", "flame_salmon_1", "flame_steak", "sear_steak"]
    elif dataset == "nvidia":
        folder_list = ["Balloon1", "Balloon2", "Jumping", "dynamicFace","Playground", "Skating", "Truck", "Umbrella"]
    elif dataset == "hypernerf":
        folder_list = ["aleks-teapot", "chickchicken", "cut-lemon1", "hand1", "slice-banana", "torchocolate", 
                       "americano", "cross-hands1", "espresso", "keyboard", "oven-mitts", "split-cookie", "tamping", 
                       "3dprinter", "broom", "chicken", "peel-banana"]
    elif dataset == "dnerf":
        folder_list = ["bouncingballs", "hellwarrior", "hook", "jumpingjacks", "lego", "mutant", "standup", "trex"]
    elif dataset == "panoptic_sports":
        folder_list = ["basketball", "boxes", "football", "juggle", "softball", "tennis"]
    return folder_list

   
def collect_psnr_ssim_lpips_memory(folder_list, output_path):
    
    psnr_results = {}
    ssim_results = {}
    lpips_results= {}
    total_memory = {}
    
    output_folder_name = output_path.split("/")[-1]

    with open(os.path.join(output_path, output_folder_name+ "_psnr_ssim_lpips_memory.txt"), 'w') as f:
        f.write("")
    
    for folder in folder_list:
        json_path = os.path.join(output_path, folder, "results.json")
        model_path = os.path.join(output_path, folder, "point_cloud")

        if not os.path.exists(json_path):
            with open(os.path.join(output_path, output_folder_name+ "_psnr_ssim_lpips_memory.txt"), 'a') as f:
                f.write(f"{folder} : \n")
            continue
            
        with open(json_path) as f:
            results = json.load(f)

        result_key = list(results.keys())[0]      
        
        psnr_results[folder] = results[result_key]['PSNR']
        ssim_results[folder] = results[result_key]['SSIM']
        lpips_results[folder] = results[result_key]['LPIPS-vgg']

        model_folder_list = os.listdir(model_path)
        model_folder_list.sort()
        model_folder = model_folder_list[-1]
        
        total_path = os.path.join(model_path, model_folder)
        
        total_size = sum(os.path.getsize(os.path.join(total_path, f)) for f in os.listdir(total_path)) / (1000*1000)
        
        total_memory[folder] = total_size
        
        print(f"{folder} : {results[result_key]['PSNR']} {results[result_key]['SSIM']} {results[result_key]['LPIPS-vgg']} {total_size} MB")

        with open(os.path.join(output_path, output_folder_name+ "_psnr_ssim_lpips_memory.txt"), 'a') as f:
            f.write(f"{folder} : {results[result_key]['PSNR']} {results[result_key]['SSIM']} {results[result_key]['LPIPS-vgg']} {total_size} MB\n")
                    
        
if __name__ == "__main__":
        
    parser = ArgumentParser(description="collection parameters")
    
    parser.add_argument('--output_path', type=str, default="./output/dycheck")
    parser.add_argument('--dataset', type=str, default="dycheck")

    args = parser.parse_args(sys.argv[1:])

    folder_list = get_folder_list(args.dataset)

    collect_psnr_ssim_lpips_memory(folder_list, args.output_path)
  
		
    
    




