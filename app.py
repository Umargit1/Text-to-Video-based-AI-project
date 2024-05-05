from flask import Flask, request, jsonify,render_template

import shutil
import torch
import sys
sys.path.append('/content/drive/MyDrive/MyProject/SadTalker/')
import torch
from TTS.api import TTS
from time import strftime
import os
from argparse import ArgumentParser
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path





driven_audio='/content/drive/MyDrive/MyProject/SadTalker/output.wav'
source_image='/content/drive/MyDrive/MyProject/SadTalker/examples/source_image/happy.png'
ref_eyeblink=None
ref_pose=None
checkpoint_dir='/content/drive/MyDrive/MyProject/SadTalker/checkpoints'
result_dir='/content/drive/MyDrive/MyProject/SadTalker/static/results'
pose_stylee=0
size=256
expression_scale=1
input_yaw=None
input_pitch=None
input_roll=None
enhancer='gfpgan'
background_enhancer=None
cpu=False
face3dvis=False
still=False
preprocess='crop'
verbose=False
old_version=False
net_recon='resnet50'
init_paths=None
use_last_fc=False
bfm_folder='/content/drive/MyDrive/MyProject/SadTalker/checkpoints/checkBFM_Fitting/'
bfm_model='BFM_model_front.mat'
text='This is a test.'

    # default renderer parameters
focal=1015
center=112.
camera_d=10.
z_near=5.
z_far=15.






app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

    



@app.route('/process', methods=['POST'])
def process():

    print('i was here')
    # Get image file from request
    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    driven_audio='/content/drive/MyDrive/MyProject/SadTalker/static/uploads/audio.wav'

    # Get text from request
    text = request.form['text']
    temp_folder='static/uploads'
    source_image = os.path.join(temp_folder, file.filename)
    file.save(source_image)


    # print("Image saved as:", file_path)
    print("Text received:", text)
    main(driven_audio,text,source_image,file.filename)



    # Return a response
    return render_template('display.html', filename=file.filename)














print('Imports done!')



def tts(driven_audio,text):
  
  # Get device
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # Init TTS
  # Init TTS with the target model name
  tts = TTS("tts_models/en/ljspeech/tacotron2-DCA").to(device)
  tts.tts_to_file(text=text, file_path=driven_audio)








def main(driven_audio,text,source_image,filename):
    #torch.backends.cudnn.enabled = False
    audio=driven_audio
    inputt=text
    tts(audio,inputt)

    print('Done with TTS')

    pic_path = source_image
    audio_path = driven_audio
    save_dir = result_dir
    os.makedirs(save_dir, exist_ok=True)
    pose_style = pose_stylee
    device = 'cuda'
    batch_size = 2
    input_yaw_list = None
    input_pitch_list = None
    input_roll_list = None
    ref_eyeblink = None
    ref_pose = None

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'),size, old_version, preprocess)

    #init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
    
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, preprocess,
                                                                             source_image_flag=True, pic_size=size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3dface render
    if face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=expression_scale, still_mode=still, preprocess=preprocess, size=size)
    
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=enhancer, background_enhancer=background_enhancer, preprocess=preprocess, img_size=size)
    

    
    # Construct the full path where you want to save the video
    output_path = os.path.join(save_dir, filename + '.mp4')

    # Rename the generated video to the desired filename
    os.rename(result, output_path)

    # Print the path of the saved video
    print('The generated video is saved as:', output_path)
    # if not verbose:
    #     shutil.rmtree(save_dir)
    

if __name__ == '__main__':
    app.run(debug=True)

    
    