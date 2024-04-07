import argparse
import time
from datetime import date

from subprocess import call
import GPUtil
import os




parser = argparse.ArgumentParser()
parser.add_argument('--gpus_to_use', type=str)
#parser.add_argument('--dataset', type=str, default='cifar10')
#parser.add_argument('--test_orig_models', action='store_true')
#parser.add_argument('--on_slurm', action='store_true')
#parser.add_argument('--output_on', type=str)
#parser.add_argument('--only_clean', action='store_true')
#parser.add_argument('--use_ema', action='store_true')

args = parser.parse_args()

if args.gpus_to_use is None:
    gpus_to_use = list(range(8))
else:
    gpus_to_use = [float(c) for c in args.gpus_to_use.split(' ')]


ckpts = [
    None, 
    '../robust-clip/ViT-L-14_openai_imagenet_txtSup_True_imagenet_vit-l-sup-10k-3adv-lr1e-5_wd_1e-4_fFTvv_final.pt',
    '../robust-clip/ViT-L-14_openai_imagenet_txtSup_False_vit-l-unsup-clean-0p1-lr1e-5_ZUSEW_final.pt',
    '../robust-clip/ViT-L-14_openai_imagenet_txtSup_False_imagenet_vit-l-unsup-clean-0p0-lr1e-5-wd-1e-4_mCGle_final.pt',
    # new models
    '../robust-clip/ViT-L-14_openai_imagenet_txtSup_True_imagenet_vit-l-tecoa-eps4-2epoch_BaLvU_final.pt',
    #'../robust-clip/ViT-L-14_openai_imagenet_txtSup_True_imagenet_vit-l-tecoa-eps2-2epoch_OM5H5_final.pt',
    '../robust-clip/ViT-L-14_openai_imagenet_txtSup_False_imagenet_vit-l-fareP-2epoch_qZvwS_final.pt',
    '../robust-clip/ViT-L-14__mnt_cschlarmann37_project_multimodal_clip-finetune_ViT-L-14_openai_imagenet_txtSup_False_imagenet_vit-l-fareP-eps2-2epoch_17yHI_temp_checkpoints_fallback_16000.pt_imagenet_txtSup_False_imagenet_vit-l-fareP-eps2-2epoch_9JNNU_final.pt',
    # perceptual metrics models
    # '../robust-clip/CLIP-ViT-B-32-DataComp.XL-s13B-b90K_none_imagenet_ce_imagenet_TECOA4_oGFNb.pt',
    # '../robust-clip/CLIP-ViT-B-32-DataComp.XL-s13B-b90K_none_imagenet_l2_imagenet_FARE4_t1gfj.pt',
    # 'hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K',
    # '../robust-clip/CLIP-ViT-B-32-laion2B-s34B-b79K-original.pt',
    # '../robust-clip/CLIP-ViT-B-32-laion2B-s34B-b79K_none_imagenet_ce_imagenet_TECOA4_9j55I.pt',
    # '../robust-clip/CLIP-ViT-B-32-laion2B-s34B-b79K_none_imagenet_l2_imagenet_FARE4_dCUnU.pt',
    # '../lipsim/model.ckpt-1.pth',
    # '../lipsim/model.ckpt-435_margin=0.5.pth',
    # '../lipsim/model.ckpt-435_margin=0.2.pth',
    '../robust-clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg_none_imagenet_ce_imagenet_TECOA4_cREDo.pt',
    '../robust-clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg_none_imagenet_l2_imagenet_FARE4_VFul3.pt',
    #'hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg',
    # # '../robust-clip/CLIP-ViT-B-16-DataComp.XL-s13B-b90K.pt',
    # # '../robust-clip/CLIP-ViT-B-16-DataComp.XL-s13B-b90K_none_imagenet_ce_imagenet_TECOA4_LFNiy.pt',
    # # '../robust-clip/CLIP-ViT-B-16-DataComp.XL-s13B-b90K_none_imagenet_l2_imagenet_FARE4_W1Hzc.pt',
    # '../robust-clip/CLIP-ViT-B-16-laion2B-s34B-b88K.pt',
    '../robust-clip/CLIP-ViT-B-16-laion2B-s34B-b88K_none_imagenet_ce_imagenet_TECOA4_mMRaV.pt',
    # '../robust-clip/CLIP-ViT-B-16-laion2B-s34B-b88K_none_imagenet_l2_imagenet_FARE4_SKirS.pt',
    # '../R-LPIPS/checkpoints/latest_net_linf_ref.pth',
    # 'dreamsim:open_clip_vitb32',
    # 'dreamsim:clip_vitb32',
    # 'dreamsim:dino_vitb16',
    # 'dreamsim:ensemble',
    # mlp
    #('../robust-clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg_none_imagenet_ce_imagenet_TECOA4_cREDo.pt', 'tecoa-4-cREDo'),
    #('../robust-clip/CLIP-ViT-B-16-laion2B-s34B-b88K_none_imagenet_ce_imagenet_TECOA4_mMRaV.pt', 'tecoa-4-mMRaV'),
    #('../robust-clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg_none_imagenet_l2_imagenet_FARE4_VFul3.pt', 'fare-4-VFul3'),
    # ('../robust-clip/CLIP-ViT-B-16-laion2B-s34B-b88K_none_imagenet_l2_imagenet_FARE4_SKirS.pt', 'fare-4-SKirS'),
    # ('../robust-clip/CLIP-ViT-B-32-laion2B-s34B-b79K_none_imagenet_ce_imagenet_TECOA4_9j55I.pt', 'tecoa-4-9j55I'),
    # ('../robust-clip/CLIP-ViT-B-32-laion2B-s34B-b79K_none_imagenet_l2_imagenet_FARE4_dCUnU.pt', 'fare-4-dCUnU'),
    # ('../robust-clip/CLIP-ViT-B-16-laion2B-s34B-b88K.pt', 'openclip-laion-vitb16'),
    # ('../robust-clip/CLIP-ViT-B-32-laion2B-s34B-b79K-original.pt', 'openclip-laion-vitb32'),
    #('hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg', 'openclip-convnext-base-w',),
    # lora
    #('../robust-clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg_none_imagenet_ce_imagenet_TECOA4_cREDo.pt', 'tecoa-4-cREDo',  'tecoa-4-cREDo'),
    # ('../robust-clip/CLIP-ViT-B-16-laion2B-s34B-b88K_none_imagenet_ce_imagenet_TECOA4_mMRaV.pt', 'tecoa-4-mMRaV', 'tecoa-4-mMRaV'),
    #('../robust-clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg_none_imagenet_l2_imagenet_FARE4_VFul3.pt', 'fare-4-VFul3', 'fare-4-VFul3'),
    # ('../robust-clip/CLIP-ViT-B-16-laion2B-s34B-b88K_none_imagenet_l2_imagenet_FARE4_SKirS.pt', 'fare-4-SKirS', 'fare-4-SKirS'),
    # ('../robust-clip/CLIP-ViT-B-32-laion2B-s34B-b79K_none_imagenet_ce_imagenet_TECOA4_9j55I.pt', 'tecoa-4-9j55I', 'tecoa-4-9j55I'),
    # ('../robust-clip/CLIP-ViT-B-32-laion2B-s34B-b79K_none_imagenet_l2_imagenet_FARE4_dCUnU.pt', 'fare-4-dCUnU', 'fare-4-dCUnU'),
    # ('../robust-clip/CLIP-ViT-B-16-laion2B-s34B-b88K.pt', 'openclip-laion-vitb16', 'openclip-laion-vitb16'),
    # ('../robust-clip/CLIP-ViT-B-32-laion2B-s34B-b79K-original.pt', 'openclip-laion-vitb32', 'openclip-laion-vitb32'),
    #('hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg', 'openclip-convnext-base-w', 'openclip-convnext-base-w')
][-1:]  # -16:-15

norm = ['Linf', 'L2'][0]
epss = [0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 6, 9, 12, 15][6:7]
splits = [
   'test_imagenet',
   'test_no_imagenet',
   'all',
   'cnn',
   'color', 
   'deblur', 
   'frameinterp',
   'superres',
   'traditional',
   'none'
   ][-1:]

    
str_to_run = (
    'other_tasks_clip.py --n_ex 5000 --batch_size 100 --device cuda:0'
    #' --norm Linf'
    ' --n_iter 10000'
    ' --alpha_init 2.'
    #' --attack_name apgd'
    #' --dataset bapps-bin'
    #' --img_res 224'
    #' --modelname ViT-L-14'
    #' --loss binary-margin'
    #' --attack_name square --loss margin'
    #' --save_to_dict --dictname acc_dets_new.json'
    f' --norm {norm}'
    #' --n_restarts 1 --use_rs'
    ' --int_fts_subset m7+out --int_fts_pool cls_tkn+norm'
    )

models_to_run = []
for ckpt in ckpts:
    for eps in epss:
        for split in splits:
            if split != 'none':
                dataset = 'nights' if 'imagenet' in split else 'bapps-bin'
            else:
                dataset = 'things'
            if ckpt is not None:
                if isinstance(ckpt, str):
                    str_curr = str_to_run + f' --ckptpath {ckpt}'
                elif isinstance(ckpt, (list, tuple)):
                    if len(ckpt) == 2:
                        str_curr = str_to_run + f' --ckptpath {ckpt[0]} --mlp_head {ckpt[1]}'
                    elif len(ckpt) == 3:
                        str_curr = str_to_run + f' --ckptpath {ckpt[0]}  --lora_weights {ckpt[2]}'
            else:
                str_curr = str_to_run + ''
            str_curr += f' --eps {eps}'
            str_curr += f' --split {split}'
            str_curr += f' --dataset {dataset}'
            models_to_run.append(str_curr)

    
    




cart_prod = [a \
for a in models_to_run \
#
]

for job in cart_prod:
    print(job)

time.sleep(5)



if True:
    count = 0
    while True: # infinite loop
      gpu_ids  = GPUtil.getAvailable(order = 'first', limit = 8, \
        maxLoad = .98, maxMemory = .5) # get free gpus listd
      if len(gpu_ids) >0:
        print(gpu_ids)
        for id in (gpu_ids):
          if id in gpus_to_use:
            temp_list = cart_prod[count]
      
            #id_env = '8' if id == 0 else str(id)
            #
            
            # following is the command as a string
            # it varies according to purpose.
            command_to_exec = '' +\
            ' CUDA_VISIBLE_DEVICES='+str(id)+\
            ' python3' +\
            ' ' + temp_list\
            + ' &' # for going to next iteration without job in background.
      
            print("Command executing is " + command_to_exec)
            call(command_to_exec, shell=True)
            print('done executing in '+str(id))
            count += 1
            if count == len(cart_prod) and False:
                print('all processes started')
                time.sleep(10)
                empty_gpus = []
                while len(empty_gpus) < len(gpus_to_use):
                    time.sleep(120)
                    empty_gpus = GPUtil.getAvailable(order = 'first', limit = 8, \
                        maxLoad = .05, maxMemory = .05)
                    print('some processes are still running')
                print('all processes should be done')
        
        time.sleep(120) # wait for processes to start
      else:
        print('No gpus free waiting for 30 seconds')
        time.sleep(30)
    
else:
    '''for i, cmd in enumerate(cart_prod):
        modelname = cmd.split(' --topcl_name ')[1].split(' --')[0]
        argsmain = cmd.replace(f' --topcl_name {modelname}', '')
        command_to_exec = f'args_main="{argsmain}" modelname={modelname} sbatch preamble_slurm.sh '
        print(f'starting cmd {i}: {command_to_exec}')
        #time.sleep(10)
        call(command_to_exec, shell=True)
        time.sleep(60)
    '''
    with open(f'./cmds_s/cmds_test_{date.today()}.txt', 'a') as f:
        for i, cmd in enumerate(cart_prod):
            command_to_exec = '' +\
                'CUDA_VISIBLE_DEVICES=' + str(i * 1) +\
                ' python3' +\
                ' ' + cmd\
                + ' &'
            f.write(command_to_exec + '\n')
        f.write('\n')
        f.flush()

