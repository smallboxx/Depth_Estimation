import os
import torch
import random
import numpy as np

def get_dataset_txt(datasetname, path, split_ratio):
    if os.path.exists('train_method') == False:
        os.mkdir("train_method")
    dirname = 'train_method'
    for name in datasetname:
        if name == 'DiverseDepth':
            r = os.path.join(path,name) + '/rgbs'
            d = os.path.join(path,name) + '/depths'
            subfolders = os.listdir(r)
            with open(dirname+'/DiverseDepth_train.txt','w') as ft:
                with open(dirname+'/DiverseDepth_val.txt','w') as fv:
                    for subname in subfolders:
                        rgb_jpgs = os.listdir(os.path.join(r,subname))
                        depth_pngs = os.listdir(os.path.join(d,subname))
                        N = len(rgb_jpgs)
                        train_num = int(split_ratio*N)
                        for i in range(N):
                            rgb_rlpath = os.path.join(os.path.join(r,subname),rgb_jpgs[i])
                            depth_rlpath = os.path.join(os.path.join(d,subname),depth_pngs[i])
                            if i <= train_num:
                                ft.write(f'{rgb_rlpath}${depth_rlpath} \n')
                            else:
                                fv.write(f'{rgb_rlpath}${depth_rlpath} \n')
            ft.close()
            fv.close()

        if name == 'Holopix50k':
            p = os.path.join(path,name) + '/train'
            fpaths = [file for file in os.listdir(p) if file.endswith('.jpg')]
            N = len(fpaths)
            train_num = int(split_ratio*N)
            with open(dirname+'/Holopix50k_train.txt','w') as ft:
                with open(dirname+'/Holopix50k_val.txt','w') as fv:
                    for i in range(N):
                        rgb_rlpath = os.path.join(p,fpaths[i]).replace('\\','/')
                        depth_rlpath = os.path.join(p,fpaths[i].split('.')[0]+'_dsp_refined.png').replace('\\','/')
                        if i <= train_num:
                            ft.write(f'{rgb_rlpath}${depth_rlpath} \n')
                        else:
                            fv.write(f'{rgb_rlpath}${depth_rlpath} \n')
            ft.close()
            fv.close()

        if name == 'HR-WSI':
            ftrain = os.path.join(path,name) + '/train'
            fval   = os.path.join(path,name) + '/val'
            with open(dirname+'/HR-WSI_train.txt','w') as ft:
                with open(dirname+'/HR-WSI_val.txt','w') as fv:
                        rgb_train = os.listdir(os.path.join(ftrain,'imgs'))
                        rgb_val   = os.listdir(os.path.join(fval,'imgs'))
                        for rgb_t in rgb_train:
                            rgb1 = ftrain+'/imgs/'+rgb_t
                            dep1 = ftrain+'/gts/'+rgb_t.replace('.jpg','.png')
                            ft.write(f'{rgb1}${dep1} \n')

                        for rgb_v in rgb_val:
                            rgb2 = ftrain+'/imgs/'+rgb_v
                            dep2 = fval+'/gts/'+rgb_v.replace('.jpg','.png')
                            fv.write(f'{rgb2}${dep2} \n')
            ft.close()
            fv.close()

        if name == 'ReDWeb_V1':
            r = os.path.join(path,name) + '/Imgs'
            d = os.path.join(path,name) + '/RDs'
            fpath = os.listdir(r)
            N = len(fpath)
            train_num = int(split_ratio*N)
            with open(dirname+'/ReDWeb_V1_train.txt','w') as ft:
                with open(dirname+'/ReDWeb_V1_val.txt','w') as fv:
                    for i in range(N):
                        rgb_rlpath = os.path.join(r,fpath[i]).replace('\\','/')
                        depth_rlpath = os.path.join(d,fpath[i].replace('.jpg','.png')).replace('\\','/')
                        if i <= train_num:
                            ft.write(f'{rgb_rlpath}${depth_rlpath} \n')
                        else:
                            fv.write(f'{rgb_rlpath}${depth_rlpath} \n')
            ft.close()
            fv.close()
        
        if name == 'replica_fullplus':
            r = os.path.join(path,name) + '/rgb/replica'
            d = os.path.join(path,name) + '/depth_zbuffer/replica'
            subfolders = os.listdir(r)
            with open(dirname+'/replica_fullplus_train.txt','w') as ft:
                with open(dirname+'/replica_fullplus_val.txt','w') as fv:
                    for subname in subfolders:
                        rgb_jpgs = os.listdir(os.path.join(r,subname))
                        depth_pngs = os.listdir(os.path.join(d,subname))
                        N = len(rgb_jpgs)
                        train_num = int(split_ratio*N)
                        for i in range(N):
                            rgb_rlpath = os.path.join(os.path.join(r,subname),rgb_jpgs[i]).replace('\\','/')
                            depth_rlpath = os.path.join(os.path.join(d,subname),depth_pngs[i]).replace('\\','/')
                            if i <= train_num:
                                ft.write(f'{rgb_rlpath}${depth_rlpath} \n')
                            else:
                                fv.write(f'{rgb_rlpath}${depth_rlpath} \n')
            ft.close()
            fv.close()

        if name == 'taskonomy':
            r = os.path.join(path,name) + '/rgbs'
            d = os.path.join(path,name) + '/depths'
            subfolders = os.listdir(r)
            with open(dirname+'/taskonomy_train.txt','w') as ft:
                with open(dirname+'/taskonomy_val.txt','w') as fv:
                    for subname in subfolders:
                        rgb_jpgs = os.listdir(os.path.join(r,subname))
                        depth_pngs = os.listdir(os.path.join(d,subname))
                        N = len(rgb_jpgs)
                        train_num = int(split_ratio*N)
                        for i in range(N):
                            rgb_rlpath = os.path.join(os.path.join(r,subname),rgb_jpgs[i]).replace('\\','/')
                            depth_rlpath = os.path.join(os.path.join(d,subname),depth_pngs[i]).replace('\\','/')
                            if i <= train_num:
                                ft.write(f'{rgb_rlpath}${depth_rlpath} \n')
                            else:
                                fv.write(f'{rgb_rlpath}${depth_rlpath} \n')
            ft.close()
            fv.close()
    print('perset_txt generate complete!')


def get_train_method(datasetname, txtpath, seed=1, train_sample_nums=500, val_sample_nums=100):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if os.path.exists('train_method') == False:
        os.mkdir("train_method")
    dirname = 'train_method'

    N_SET = len(datasetname)
    N_TRAIN_PER = train_sample_nums//N_SET
    N_VAL_PER   = val_sample_nums//N_SET

    with open(dirname+'/train_inside.txt','w') as train_insidef:
        with open(dirname+'/val_inside.txt', 'w') as val_insidef:
            with open(dirname+'/train_outside.txt','w') as train_outsidef:
                with open(dirname+'/val_outside.txt', 'w') as val_outsidef:
                    tset_idxrange, vset_idxrange= {}, {}
                    tempidx_t, tempidx_v = 0, 0
                    trainlist, vallist = [], []

                    for name in datasetname:
                        train_path = os.path.join(txtpath,name+'_train.txt')
                        val_path   = os.path.join(txtpath,name+'_val.txt')
                        with open(train_path,'r') as ft:
                            with open(val_path,'r') as fv:
                                tpaths = ft.readlines()
                                vpaths = fv.readlines()
                                tset_idxrange[name] = [tempidx_t,tempidx_t+len(tpaths)]
                                vset_idxrange[name] = [tempidx_v,tempidx_v+len(vpaths)]
                                tempidx_t += len(tpaths)
                                tempidx_v += len(vpaths)
                                trainlist += [t.replace('\n','') for t in tpaths]
                                vallist   += [v.replace('\n','') for v in vpaths]
                        ft.close()
                        fv.close()

                    print(tset_idxrange, vset_idxrange)
                    
                    train_range = [tset_idxrange['taskonomy'][0], tset_idxrange['taskonomy'][1]]
                    val_range   = [vset_idxrange['taskonomy'][0], vset_idxrange['taskonomy'][1]]
        
                    tsample = train_range[1]
                    select_list = torch.randint(train_range[0], train_range[1], (1, tsample-1)).squeeze(0)
                    import cv2
                    num = 0
                    n_sample = 0
                    for check in select_list:
                        num+=1
                        print(f'{num}/{tsample-1}')
                        gt_path =trainlist[check].split('$')[1].strip(' ')
                        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
                        assert depth is not None, f"Failed to read depth! path:{gt_path}"
                        depth[depth > 23000] = 0
                        depth = depth / 512.0
                        max_dp=depth.max()
                        if max_dp > 15.0:
                            n_sample+=1
                            train_outsidef.write(f'{trainlist[check]}\n')

                    num = 0
                    for check in select_list:
                        if n_sample==0:
                            break
                        num+=1
                        print(f'{num}/{tsample-1}')
                        gt_path =trainlist[check].split('$')[1].strip(' ')
                        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
                        assert depth is not None, f"Failed to read depth! path:{gt_path}"
                        depth[depth > 23000] = 0
                        depth = depth / 512.0
                        max_dp=depth.max()
                        if max_dp < 15.0 and n_sample > 0:
                            n_sample-=1
                            train_insidef.write(f'{trainlist[check]}\n')

                    n_sample = 0
                    vsample = val_range[1]
                    select_list = torch.randint(val_range[0], val_range[1], (1, vsample-1)).squeeze(0)
                    num = 0
                    for check in select_list:
                        num+=1
                        print(f'{num}/{vsample-1}')
                        gt_path =vallist[check].split('$')[1].strip(' ')
                        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
                        assert depth is not None, f"Failed to read depth! path:{gt_path}"
                        depth[depth > 23000] = 0
                        depth = depth / 512.0
                        max_dp=depth.max()
                        if max_dp > 15.0:
                            n_sample+=1
                            val_outsidef.write(f'{vallist[check]}\n')

                    vsample = val_range[1]
                    num = 0
                    for check in select_list:
                        if n_sample==0:
                            break
                        num+=1
                        print(f'{num}/{vsample-1}')
                        gt_path =vallist[check].split('$')[1].strip(' ')
                        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
                        assert depth is not None, f"Failed to read depth! path:{gt_path}"
                        depth[depth > 23000] = 0
                        depth = depth / 512.0
                        max_dp=depth.max()
                        if max_dp < 15.0 and n_sample>0:
                            n_sample-=1
                            val_insidef.write(f'{vallist[check]}\n')


    train_outsidef.close()
    train_insidef.close()
    val_outsidef.close()
    val_insidef.close()

    print('txt generate complete!')

if __name__ == '__main__':
    datasetname = ['taskonomy']
    path = 'Data'
    # split_ratio = 0.8
    # get_dataset_txt(datasetname, path, split_ratio) 
    # ---txtfiles put into {train_method} folder--- #
    txtpath = 'train_method'
    get_train_method(datasetname,txtpath)