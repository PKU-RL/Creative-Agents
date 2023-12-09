import sys
sys.path.append('./img2vox')
sys.path.append('./txt2img')
sys.path.append('./vox2seq')
sys.path.append('./seq2building')
from txt2img.test_text_to_image_lora import text2img
from img2vox.runner import run_img2vox
from vox2seq.vox2seq import vox2seq
from seq2building.Building import Build



import os
import shutil

def main(save_dir = "./results", prompt_path = "./results/validation_prompt.txt", remove_mid = False):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print("----------[INFO] Generating Image from Text Descriptions--------------")
    text2img(save_dir = os.path.join(save_dir,"image"), image_sample_num = 1, prompt_path=prompt_path)
    print(f'----------[INFO] Text Descriptions Are Saved in {os.path.join(save_dir,"image")}--------------')

    print("----------[INFO] Generating Voxel from Images--------------")
    run_img2vox(pipline= True)
    print(f'----------[INFO] Voxels Are Saved in {os.path.join(save_dir,"voxel")}--------------')
    

    print("----------[INFO] Generating Sequences from Voxel--------------")
    vox2seq(save_path = os.path.join(save_dir,"seq"), max_step=800)
    print(f'----------[INFO] Sequences Are Saved in {os.path.join(save_dir,"seq")}--------------')


    print("----------[INFO] Building Structures from Sequences--------------")
    Build(save_pth= os.path.join(save_dir,"building"), full_record= False)
    print(f'----------[INFO] Buildings Are Saved in {os.path.join(save_dir,"building")}--------------')

    if remove_mid:
        remove_middle_results([os.path.join(save_dir,"image"),os.path.join(save_dir,"voxel"),os.path.join(save_dir,"seq")])

def remove_middle_results(paths):
    for p in paths:
        shutil.rmtree(p)
        os.mkdir(p)
    
if __name__ == "__main__":
    main()
