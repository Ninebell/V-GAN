import os
import sys

from PIL import Image, ImageEnhance
from keras.preprocessing.image import Iterator
from scipy.ndimage import rotate
from skimage import filters
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import measure

import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib
import random

def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]
    
    if sort:
        filenames = sorted(filenames)
    
    return filenames

def image_shape(filename):
    img = Image.open(filename)
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape

def imagefiles2arrs(filenames):
    img_shape = image_shape(filenames[0])
    if len(img_shape)==3:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    elif len(img_shape)==2:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)
    
    for file_index in range(len(filenames)):
        img = Image.open(filenames[file_index])
        images_arr[file_index] = np.asarray(img).astype(np.float32)
    
    return images_arr

def STARE_files(data_path):
    img_dir=os.path.join(data_path, "images")
    vessel_dir=os.path.join(data_path,"1st_manual")
    mask_dir=os.path.join(data_path,"mask")
    
    img_files=all_files_under(img_dir, extension=".ppm")
    vessel_files=all_files_under(vessel_dir, extension=".ppm")
    mask_files=all_files_under(mask_dir, extension=".ppm")
    
    return img_files, vessel_files, mask_files
def DRIVE_files(data_path):
    img_dir=os.path.join(data_path, "images")
    vessel_dir=os.path.join(data_path,"1st_manual")
    mask_dir=os.path.join(data_path,"mask")
    
    img_files=all_files_under(img_dir, extension=".tif")
    vessel_files=all_files_under(vessel_dir, extension=".gif")
    mask_files=all_files_under(mask_dir, extension=".gif")
    
    return img_files, vessel_files, mask_files

def crop_imgs(imgs,pad):
    """
    crop images (4D tensor) by [:,pad:-pad,pad:-pad,:] 
    """
    return imgs[:,pad:-pad,pad:-pad,:]

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
        
def discriminator_shape(n, d_out_shape):
    if len(d_out_shape)==1: # image gan
        return (n, d_out_shape[0])
    elif len(d_out_shape)==3:   # pixel, patch gan
        return (n, d_out_shape[0], d_out_shape[1], d_out_shape[2])
    return None

def input2discriminator(real_img_patches, real_vessel_patches, fake_vessel_patches, d_out_shape):
    real=np.concatenate((real_img_patches,real_vessel_patches), axis=3)
    fake=np.concatenate((real_img_patches,fake_vessel_patches), axis=3)

    d_x_batch=np.concatenate((real,fake), axis=0)
    
    # real : 1, fake : 0
    d_y_batch=np.ones(discriminator_shape(d_x_batch.shape[0], d_out_shape))
    d_y_batch[real.shape[0]:,...] = 0
    
    return d_x_batch, d_y_batch
 
def input2gan(real_img_patches, real_vessel_patches, d_out_shape):    
    g_x_batch=[real_img_patches,real_vessel_patches]
    # set 1 to all labels (real : 1, fake : 0)
    g_y_batch=np.ones(discriminator_shape(real_vessel_patches.shape[0], d_out_shape))
    return g_x_batch, g_y_batch
    
def print_metrics(itr, **kargs):
    print("*** Round {}  ====> ".format(itr))
    for name, value in kargs.items():
        print("{} : {}, ".format(name, value))
    print("")
    sys.stdout.flush()

class TrainBatchFetcher(Iterator):
    """
    fetch batch of original images and vessel images
    """
    def __init__(self, train_imgs, train_vessels, batch_size, train_indices, shape):
        self.train_imgs=train_imgs
        self.train_vessels=train_vessels
        self.shape = shape
        self.train_indices = train_indices
        self.n_train_imgs=len(self.train_indices)
        self.batch_size=batch_size

    def next(self):
        indices=list(np.random.choice(self.n_train_imgs, self.batch_size))
        train_imgs = []
        train_vessels = []

        for idx in indices:
            if self.batch_size != 2:
                print(idx)
            train_imgs.append(np.load(self.train_imgs[self.train_indices[idx]]))
            train_vessels.append(np.load(self.train_vessels[self.train_indices[idx]]))
        train_imgs=np.asarray(train_imgs)
        train_vessels=np.asarray(train_vessels)
        train_vessels=train_vessels.reshape(self.shape)
        return train_imgs, train_vessels

    @staticmethod
    def load_numpy(pathes):
        datas = [np.load(path) for path in pathes]
        return datas

def AUC_ROC(true_vessel_arr, pred_vessel_arr, save_fname):
    """
    Area under the ROC curve with x axis flipped
    """
    fpr, tpr, _ = roc_curve(true_vessel_arr, pred_vessel_arr)
    save_obj({"fpr":fpr, "tpr":tpr}, save_fname)
    AUC_ROC=roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    return AUC_ROC

def plot_AUC_ROC(fprs,tprs,method_names,fig_dir):
    # set font style
    font={'family':'serif'}
    matplotlib.rc('font', **font)

    # sort the order of plots manually for eye-pleasing plots
    colors=['r','b','y','g','#7e7e7e','m','k','c'] if len(fprs)==8 else ['r','y','m','k','g']
    indices=[6,2,5,3,4,7,1,0] if len(fprs)==8 else [3,1,2,4,0] 
    
    # print auc  
    print( "****** ROC AUC ******")
    print("CAVEAT : AUC of V-GAN with 8bit images might be lower than the floating point array (check <home>/pretrained/auc_roc*.npy)")
    for index in indices:
        if method_names[index]!='CRFs' and method_names[index]!='2nd_manual':
            print("{} : {:04}".format(method_names[index],auc(fprs[index],tprs[index])))
    
    # plot results
    for index in indices:
        if method_names[index]=='CRFs':
            plt.plot(fprs[index],tprs[index],colors[index]+'*',label=method_names[index].replace("_"," "))
        elif method_names[index]=='2nd_manual':
            plt.plot(fprs[index],tprs[index],colors[index]+'*',label='Human')
        else:
            plt.plot(fprs[index],tprs[index],colors[index],label=method_names[index].replace("_"," "))
    
    plt.title('ROC Curve')
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.xlim(0,0.3)
    plt.ylim(0.7,1.0)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(fig_dir,"ROC.png"))
    plt.close()

def plot_AUC_PR(precisions, recalls, method_names, fig_dir):
    # set font style
    font={'family':'serif'}
    matplotlib.rc('font', **font)
    
    # sort the order of plots manually for eye-pleasing plots
    colors=['r','b','y','g','#7e7e7e','m','k','c'] if len(precisions)==8 else ['r','y','m','k','g']
    indices=[6,2,5,3,4,7,1,0] if len(precisions)==8 else [3,1,2,4,0] 

    # print auc  
    print("****** Precision Recall AUC ******")
    print("CAVEAT : AUC of V-GAN with 8bit images might be lower than the floating point array (check <home>/pretrained/auc_pr*.npy)")
    for index in indices:
        if method_names[index]!='CRFs' and method_names[index]!='2nd_manual':
            print("{} : {:04}".format(method_names[index],auc(recalls[index],precisions[index])))
    
    # plot results
    for index in indices:
        if method_names[index]=='2nd_manual':
            plt.plot(recalls[index],precisions[index],colors[index]+'*',label='Human')
        else:
            plt.plot(recalls[index],precisions[index],colors[index],label=method_names[index].replace("_"," "))
    
    plt.title('Precision Recall Curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.5,1.0)
    plt.ylim(0.5,1.0)
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(fig_dir,"Precision_recall.png"))
    plt.close()
    
def AUC_PR(true_vessel_img, pred_vessel_img, save_fname):
    """
    Precision-recall curve
    """
    precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(),  pos_label=1)
    save_obj({"precision":precision, "recall":recall}, save_fname)
    AUC_prec_rec = auc(recall, precision)
    return AUC_prec_rec

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def best_f1_threshold(precision, recall, thresholds):
    best_f1=-1
    for index in range(len(precision)):
        curr_f1=2.*precision[index]*recall[index]/(precision[index]+recall[index])
        if best_f1<curr_f1:
            best_f1=curr_f1
            best_threshold=thresholds[index]

    return best_f1, best_threshold

def threshold_by_f1(true_vessels, generated, masks, flatten=True, f1_score=False):
    vessels_in_mask, generated_in_mask = pixel_values_in_mask(true_vessels, generated, masks)
    precision, recall, thresholds = precision_recall_curve(vessels_in_mask.flatten(), generated_in_mask.flatten(),  pos_label=1)
    best_f1,best_threshold=best_f1_threshold(precision, recall, thresholds)
        
    pred_vessels_bin=np.zeros(generated.shape)
    pred_vessels_bin[generated>=best_threshold]=1
    
    if flatten:
        if f1_score:
            return pred_vessels_bin[masks==1].flatten(), best_f1
        else:
            return pred_vessels_bin[masks==1].flatten()
    else:
        if f1_score:
            return pred_vessels_bin, best_f1
        else:
            return pred_vessels_bin
    
def misc_measures(true_vessels, pred_vessels, masks):
    thresholded_vessel_arr, f1_score = threshold_by_f1(true_vessels, pred_vessels, masks, f1_score=True)
    true_vessel_arr=true_vessels[masks == 1].flatten()
    
    cm=confusion_matrix(true_vessel_arr, thresholded_vessel_arr)
    acc=1.*(cm[0,0]+cm[1,1])/np.sum(cm)
    sensitivity=1.*cm[1,1]/(cm[1,0]+cm[1,1])
    specificity=1.*cm[0,0]/(cm[0,1]+cm[0,0])
    return f1_score, acc, sensitivity, specificity

def dice_coefficient(true_vessels, pred_vessels, masks):
    thresholded_vessels=threshold_by_f1(true_vessels, pred_vessels, masks, flatten=False)
    
    true_vessels = true_vessels.astype(np.bool)
    thresholded_vessels = thresholded_vessels.astype(np.bool)
    
    intersection = np.count_nonzero(true_vessels & thresholded_vessels)
    
    size1 = np.count_nonzero(true_vessels)
    size2 = np.count_nonzero(thresholded_vessels)
    
    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc

def pad_imgs(imgs, img_size):
    img_h,img_w=imgs.shape[1], imgs.shape[2]
    target_h,target_w=img_size[0],img_size[1]
    if len(imgs.shape)==4:
        d=imgs.shape[3]
        padded=np.zeros((imgs.shape[0],target_h, target_w,d))
    elif len(imgs.shape)==3:
        padded=np.zeros((imgs.shape[0],img_size[0],img_size[1]))
    padded[:,(target_h-img_h)//2:(target_h-img_h)//2+img_h,(target_w-img_w)//2:(target_w-img_w)//2+img_w,...]=imgs
    
    return padded
    
def random_perturbation(imgs):
    for i in range(imgs.shape[0]):
        im=Image.fromarray(imgs[i,...].astype(np.uint8))
        en=ImageEnhance.Brightness(im)
        im=en.enhance(random.uniform(0.8,1.2))
        en=ImageEnhance.Color(im)
        im=en.enhance(random.uniform(0.8,1.2))
        en=ImageEnhance.Contrast(im)
        im=en.enhance(random.uniform(0.8,1.2))
        imgs[i,...]= np.asarray(im).astype(np.float32)
    return imgs 


def save_numpy(root_path, data, idx):
    for i, fundus in enumerate(data):
        path = ("{0}_{1}".format(root_path,(idx+i)))
        print(path)
        np.save(path, fundus)

def normalize(imgs):
    for index in range(imgs.shape[0]):
        mean=np.mean(imgs[index,...][imgs[index,...,0] > 40.0],axis=0)
        std=np.std(imgs[index,...][imgs[index,...,0] > 40.0],axis=0)
        assert len(mean)==3 and len(std)==3
        imgs[index,...]=(imgs[index,...]-mean)/std
    return imgs

def _multi_save_imgs(target_dir, augmentation, img_size, dataset, mask=False, tag=None):
    if dataset=='DRIVE':
        img_files, vessel_files, mask_files = DRIVE_files(target_dir)
    elif dataset=='STARE':
        img_files, vessel_files, mask_files = STARE_files(target_dir)

    # load images
    fundus_idx = 0
    vessel_idx = 0
    fundus_imgs=imagefiles2arrs(img_files)


    vessel_imgs=imagefiles2arrs(vessel_files)/255
    fundus_imgs=pad_imgs(fundus_imgs, img_size)

    vessel_imgs=pad_imgs(vessel_imgs, img_size)
    assert(np.min(vessel_imgs)==0 and np.max(vessel_imgs)==1)
    if mask:
        mask_imgs=imagefiles2arrs(mask_files)/255
        mask_imgs=pad_imgs(mask_imgs, img_size)
        assert(np.min(mask_imgs)==0 and np.max(mask_imgs)==1)
    # augmentation
    if augmentation:
        # augment the original image (flip, rotate)
        all_fundus_imgs=[fundus_imgs]
        all_vessel_imgs=[vessel_imgs]
        flipped_imgs=fundus_imgs[:,:,::-1,:]    # flipped imgs
        flipped_vessels=vessel_imgs[:,:,::-1]

        all_fundus_imgs.append(flipped_imgs)
        all_vessel_imgs.append(flipped_vessels)

        print(target_dir+"/fundus_{0}/".format(tag))
        all_fundus_imgs = np.asarray(all_fundus_imgs)
        all_vessel_imgs = np.asarray(all_vessel_imgs)
        all_fundus_imgs = np.concatenate(all_fundus_imgs, axis=0)
        all_vessel_imgs = np.concatenate(all_vessel_imgs, axis=0)

        all_fundus_imgs_resize = [np.resize(img, (img.shape[0]//2,img.shape[1]//2,img.shape[2])) for img in all_fundus_imgs]
        all_vessel_imgs_resize = [np.resize(img, (img.shape[0]//2,img.shape[1]//2)) for img in all_vessel_imgs]

        all_fundus_imgs_resize = np.asarray(all_fundus_imgs_resize)
        all_vessel_imgs_resize = np.asarray(all_vessel_imgs_resize)

        normalize_fundus = normalize(all_fundus_imgs)
        normalize_fundus_resize = normalize(all_fundus_imgs_resize)
        # for index in range(all_fundus_imgs.shape[0]):
        #     mean=np.mean(all_fundus_imgs[index,...][all_fundus_imgs[index,...,0] > 40.0],axis=0)
        #     std=np.std(all_fundus_imgs[index,...][all_fundus_imgs[index,...,0] > 40.0],axis=0)
        #     assert len(mean)==3 and len(std)==3
        #     all_fundus_imgs[index,...]=(all_fundus_imgs[index,...]-mean)/std

        save_numpy(target_dir+"/fundus_{0}/fundus_img".format(tag), normalize_fundus, fundus_idx)
        save_numpy(target_dir+"/vessel_{0}/vessel_img".format(tag), all_vessel_imgs, vessel_idx)

        fundus_idx = fundus_idx + len(all_fundus_imgs)
        vessel_idx = vessel_idx + len(all_vessel_imgs)

        save_numpy(target_dir+"/fundus_{0}/fundus_img".format(tag), normalize_fundus_resize, fundus_idx)
        save_numpy(target_dir+"/vessel_{0}/vessel_img".format(tag), all_vessel_imgs_resize, vessel_idx)

        for angle in range(6,360,6):  # rotated imgs 3~360
            print(angle)
            all_fundus_imgs = []
            all_vessel_imgs = []
            all_fundus_imgs.append(random_perturbation(rotate(fundus_imgs, angle, axes=(1, 2), reshape=False)))
            all_fundus_imgs.append(random_perturbation(rotate(flipped_imgs, angle, axes=(1, 2), reshape=False)))
            all_vessel_imgs.append(rotate(vessel_imgs, angle, axes=(1, 2), reshape=False))
            all_vessel_imgs.append(rotate(flipped_vessels, angle, axes=(1, 2), reshape=False))

            all_fundus_imgs = np.asarray(all_fundus_imgs)
            all_vessel_imgs = np.asarray(all_vessel_imgs)

            all_fundus_imgs = np.concatenate(all_fundus_imgs, axis=0)
            all_vessel_imgs = np.concatenate(all_vessel_imgs, axis=0)

            all_fundus_imgs_resize = [np.resize(img, (img.shape[0]//2,img.shape[1]//2,img.shape[2])) for img in all_fundus_imgs]
            all_vessel_imgs_resize = [np.resize(img, (img.shape[0]//2,img.shape[1]//2)) for img in all_vessel_imgs]

            all_fundus_imgs_resize = np.asarray(all_fundus_imgs_resize)
            all_vessel_imgs_resize = np.asarray(all_vessel_imgs_resize)

            normalize_fundus = normalize(all_fundus_imgs)
            normalize_fundus_resize = normalize(all_fundus_imgs_resize)

            save_numpy(target_dir+"/fundus_{0}/fundus_img".format(tag), normalize_fundus, fundus_idx)
            save_numpy(target_dir+"/vessel_{0}/vessel_img".format(tag), all_vessel_imgs, vessel_idx)

            fundus_idx = fundus_idx + len(all_fundus_imgs)
            vessel_idx = vessel_idx + len(all_vessel_imgs)

            save_numpy(target_dir+"/fundus_{0}/fundus_img".format(tag), normalize_fundus_resize, fundus_idx)
            save_numpy(target_dir+"/vessel_{0}/vessel_img".format(tag), all_vessel_imgs_resize, vessel_idx)

            fundus_idx = fundus_idx + len(all_fundus_imgs)
            vessel_idx = vessel_idx + len(all_vessel_imgs)

        fundus_imgs=np.concatenate(all_fundus_imgs,axis=0)
        vessel_imgs=np.round((np.concatenate(all_vessel_imgs,axis=0)))

    # z score with mean, std of each image
    n_all_imgs=fundus_imgs.shape[0]
    for index in range(n_all_imgs):
        mean=np.mean(fundus_imgs[index,...][fundus_imgs[index,...,0] > 40.0],axis=0)
        std=np.std(fundus_imgs[index,...][fundus_imgs[index,...,0] > 40.0],axis=0)
        assert len(mean)==3 and len(std)==3
        fundus_imgs[index,...]=(fundus_imgs[index,...]-mean)/std

    if mask:
        return fundus_imgs, vessel_imgs, mask_imgs
    else:
        return fundus_imgs, vessel_imgs


def _origin_save_imgs(target_dir, augmentation, img_size, dataset, mask=False, tag=None):
    if dataset=='DRIVE':
        img_files, vessel_files, mask_files = DRIVE_files(target_dir)
    elif dataset=='STARE':
        img_files, vessel_files, mask_files = STARE_files(target_dir)

    # load images
    fundus_idx = 0
    vessel_idx = 0
    fundus_imgs=imagefiles2arrs(img_files)


    vessel_imgs=imagefiles2arrs(vessel_files)/255
    fundus_imgs=pad_imgs(fundus_imgs, img_size)

    vessel_imgs=pad_imgs(vessel_imgs, img_size)
    assert(np.min(vessel_imgs)==0 and np.max(vessel_imgs)==1)
    if mask:
        mask_imgs=imagefiles2arrs(mask_files)/255
        mask_imgs=pad_imgs(mask_imgs, img_size)
        assert(np.min(mask_imgs)==0 and np.max(mask_imgs)==1)
    # augmentation
    if augmentation:
        # augment the original image (flip, rotate)
        all_fundus_imgs=[fundus_imgs]
        all_vessel_imgs=[vessel_imgs]
        flipped_imgs=fundus_imgs[:,:,::-1,:]    # flipped imgs
        flipped_vessels=vessel_imgs[:,:,::-1]

        all_fundus_imgs.append(flipped_imgs)
        all_vessel_imgs.append(flipped_vessels)

        print(target_dir+"/fundus_{0}/".format(tag))
        all_fundus_imgs = np.asarray(all_fundus_imgs)
        all_vessel_imgs = np.asarray(all_vessel_imgs)
        all_fundus_imgs = np.concatenate(all_fundus_imgs, axis=0)
        all_vessel_imgs = np.concatenate(all_vessel_imgs, axis=0)
        for index in range(all_fundus_imgs.shape[0]):
            mean=np.mean(all_fundus_imgs[index,...][all_fundus_imgs[index,...,0] > 40.0],axis=0)
            std=np.std(all_fundus_imgs[index,...][all_fundus_imgs[index,...,0] > 40.0],axis=0)
            assert len(mean)==3 and len(std)==3
            all_fundus_imgs[index,...]=(all_fundus_imgs[index,...]-mean)/std

        save_numpy(target_dir+"/fundus_{0}/fundus_img".format(tag), all_fundus_imgs, fundus_idx)
        save_numpy(target_dir+"/vessel_{0}/vessel_img".format(tag), all_vessel_imgs, vessel_idx)
        fundus_idx = fundus_idx + len(all_fundus_imgs)
        print(fundus_idx)
        vessel_idx = vessel_idx + len(all_vessel_imgs)

        for angle in range(3,360,3):  # rotated imgs 3~360
            print(angle)
            all_fundus_imgs = []
            all_vessel_imgs = []
            all_fundus_imgs.append(random_perturbation(rotate(fundus_imgs, angle, axes=(1, 2), reshape=False)))
            all_fundus_imgs.append(random_perturbation(rotate(flipped_imgs, angle, axes=(1, 2), reshape=False)))
            all_vessel_imgs.append(rotate(vessel_imgs, angle, axes=(1, 2), reshape=False))
            all_vessel_imgs.append(rotate(flipped_vessels, angle, axes=(1, 2), reshape=False))

            all_fundus_imgs = np.asarray(all_fundus_imgs)
            all_vessel_imgs = np.asarray(all_vessel_imgs)

            all_fundus_imgs = np.concatenate(all_fundus_imgs, axis=0)
            all_vessel_imgs = np.concatenate(all_vessel_imgs, axis=0)

            for index in range(all_fundus_imgs.shape[0]):
                mean=np.mean(all_fundus_imgs[index,...][all_fundus_imgs[index,...,0] > 40.0],axis=0)
                std=np.std(all_fundus_imgs[index,...][all_fundus_imgs[index,...,0] > 40.0],axis=0)
                assert len(mean)==3 and len(std)==3
                all_fundus_imgs[index,...]=(all_fundus_imgs[index,...]-mean)/std

            save_numpy(target_dir+"/fundus_{0}/fundus_img".format(tag), all_fundus_imgs, fundus_idx)
            save_numpy(target_dir+"/vessel_{0}/vessel_img".format(tag), all_vessel_imgs, vessel_idx)
            fundus_idx = fundus_idx + len(all_fundus_imgs)
            vessel_idx = vessel_idx + len(all_vessel_imgs)

        fundus_imgs=np.concatenate(all_fundus_imgs,axis=0)
        vessel_imgs=np.round((np.concatenate(all_vessel_imgs,axis=0)))

    # z score with mean, std of each image
    n_all_imgs=fundus_imgs.shape[0]
    for index in range(n_all_imgs):
        mean=np.mean(fundus_imgs[index, ...][fundus_imgs[index, ..., 0] > 40.0], axis=0)
        std=np.std(fundus_imgs[index, ...][fundus_imgs[index, ..., 0] > 40.0], axis=0)
        assert len(mean)==3 and len(std)==3
        fundus_imgs[index,...]=(fundus_imgs[index,...]-mean)/std

    if mask:
        return fundus_imgs, vessel_imgs, mask_imgs
    else:
        return fundus_imgs, vessel_imgs


def save_imgs(target_dir, augmentation, img_size, dataset, mask=False, tag=None):
    os.makedirs(target_dir+"/fundus_{0}/".format(tag),exist_ok=True)
    os.makedirs(target_dir+"/vessel_{0}/".format(tag),exist_ok=True)
    if tag == "multi":
        return _multi_save_imgs(target_dir,augmentation,img_size,dataset,mask,tag)
    else:
        return _origin_save_imgs(target_dir,augmentation,img_size,dataset,mask,tag)


def get_imgs(target_dir, augmentation, img_size, dataset, mask=False):
    
    if dataset=='DRIVE':
        img_files, vessel_files, mask_files = DRIVE_files(target_dir)
    elif dataset=='STARE':
        img_files, vessel_files, mask_files = STARE_files(target_dir)
        
    # load images    
    fundus_imgs=imagefiles2arrs(img_files)
    vessel_imgs=imagefiles2arrs(vessel_files)/255
    fundus_imgs=pad_imgs(fundus_imgs, img_size)
    vessel_imgs=pad_imgs(vessel_imgs, img_size)
    assert(np.min(vessel_imgs)==0 and np.max(vessel_imgs)==1)
    if mask:
        mask_imgs=imagefiles2arrs(mask_files)/255
        mask_imgs=pad_imgs(mask_imgs, img_size)
        assert(np.min(mask_imgs)==0 and np.max(mask_imgs)==1)

    # augmentation
    if augmentation:
        # augment the original image (flip, rotate)
        all_fundus_imgs=[fundus_imgs]
        all_vessel_imgs=[vessel_imgs]
        flipped_imgs=fundus_imgs[:,:,::-1,:]    # flipped imgs
        flipped_vessels=vessel_imgs[:,:,::-1]
        all_fundus_imgs.append(flipped_imgs)
        all_vessel_imgs.append(flipped_vessels)
        for angle in range(3,360,3):  # rotated imgs 3~360
            all_fundus_imgs.append(random_perturbation(rotate(fundus_imgs, angle, axes=(1, 2), reshape=False)))
            all_fundus_imgs.append(random_perturbation(rotate(flipped_imgs, angle, axes=(1, 2), reshape=False)))
            all_vessel_imgs.append(rotate(vessel_imgs, angle, axes=(1, 2), reshape=False))
            all_vessel_imgs.append(rotate(flipped_vessels, angle, axes=(1, 2), reshape=False))
        fundus_imgs=np.concatenate(all_fundus_imgs,axis=0)
        vessel_imgs=np.round((np.concatenate(all_vessel_imgs,axis=0)))
    
    # z score with mean, std of each image
    n_all_imgs=fundus_imgs.shape[0]
    for index in range(n_all_imgs):
        mean=np.mean(fundus_imgs[index,...][fundus_imgs[index,...,0] > 40.0],axis=0)
        std=np.std(fundus_imgs[index,...][fundus_imgs[index,...,0] > 40.0],axis=0)
        assert len(mean)==3 and len(std)==3
        fundus_imgs[index,...]=(fundus_imgs[index,...]-mean)/std
    
    if mask:
        return fundus_imgs, vessel_imgs, mask_imgs
    else:
        return fundus_imgs, vessel_imgs

def pixel_values_in_mask(true_vessels, pred_vessels,masks):
    assert np.max(pred_vessels)<=1.0 and np.min(pred_vessels)>=0.0
    assert np.max(true_vessels)==1.0 and np.min(true_vessels)==0.0
    assert np.max(masks)==1.0 and np.min(masks)==0.0
    assert pred_vessels.shape[0]==true_vessels.shape[0] and masks.shape[0]==true_vessels.shape[0]
    assert pred_vessels.shape[1]==true_vessels.shape[1] and masks.shape[1]==true_vessels.shape[1]
    assert pred_vessels.shape[2]==true_vessels.shape[2] and masks.shape[2]==true_vessels.shape[2]
     
    return true_vessels[masks==1].flatten(), pred_vessels[masks==1].flatten() 

def remain_in_mask(imgs,masks):
    imgs[masks==0]=0
    return imgs

def load_images_under_dir(path_dir):
    files=all_files_under(path_dir)
    return imagefiles2arrs(files)

def crop_to_original(imgs, ori_shape):
    pred_shape=imgs.shape
    assert len(pred_shape)<4

    if ori_shape == pred_shape:
        return imgs
    else: 
        if len(imgs.shape)>2:
            ori_h,ori_w =ori_shape[1],ori_shape[2]
            pred_h,pred_w=pred_shape[1],pred_shape[2]
            return imgs[:,(pred_h-ori_h)//2:(pred_h-ori_h)//2+ori_h,(pred_w-ori_w)//2:(pred_w-ori_w)//2+ori_w]
        else:
            ori_h,ori_w =ori_shape[0],ori_shape[1]
            pred_h,pred_w=pred_shape[0],pred_shape[1]
            return imgs[(pred_h-ori_h)//2:(pred_h-ori_h)//2+ori_h,(pred_w-ori_w)//2:(pred_w-ori_w)//2+ori_w]

def difference_map(ori_vessel, pred_vessel, mask):
    thresholded_vessel=threshold_by_f1(np.expand_dims(ori_vessel, axis=0),np.expand_dims(pred_vessel, axis=0),
                                np.expand_dims(mask, axis=0), flatten=False)
    
    thresholded_vessel=np.squeeze(thresholded_vessel, axis=0)
    diff_map=np.zeros((ori_vessel.shape[0],ori_vessel.shape[1],3))
    diff_map[(ori_vessel==1) & (thresholded_vessel==1)]=(0,255,0)   #Green (overlapping)
    diff_map[(ori_vessel==1) & (thresholded_vessel!=1)]=(255,0,0)    #Red (false negative, missing in pred)
    diff_map[(ori_vessel!=1) & (thresholded_vessel==1)]=(0,0,255)    #Blue (false positive)

    return diff_map

class Scheduler:
    def __init__(self, n_itrs_per_epoch_d, n_itrs_per_epoch_g, schedules, init_lr):
        self.schedules=schedules
        self.init_dsteps=n_itrs_per_epoch_d
        self.init_gsteps=n_itrs_per_epoch_g
        self.init_lr=init_lr
        self.dsteps=self.init_dsteps
        self.gsteps=self.init_gsteps
        self.lr=self.init_lr

    def get_dsteps(self):
        return self.dsteps
    
    def get_gsteps(self):
        return self.gsteps
    
    def get_lr(self):
        return self.lr
        
    def update_steps(self, n_round):
        key=str(n_round)
        if key in self.schedules['lr_decay']:
            self.lr=self.init_lr*self.schedules['lr_decay'][key]
        if key in self.schedules['step_decay']:
            self.dsteps=max(int(self.init_dsteps*self.schedules['step_decay'][key]),1)
            self.gsteps=max(int(self.init_gsteps*self.schedules['step_decay'][key]),1)
