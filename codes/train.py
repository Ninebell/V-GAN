import numpy as np
from codes.model import GAN, discriminator_pixel, discriminator_image, discriminator_patch1, discriminator_patch2, generator, discriminator_dummy
import codes.utils as utils
import os
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm
from keras import backend as K


# arrange arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    '--ratio_gan2seg',
    type=int,
    help="ratio of gan loss to seg loss",
    required=True
    )
parser.add_argument(
    '--gpu_index',
    type=str,
    help="gpu index",
    required=True
    )
parser.add_argument(
    '--discriminator',
    type=str,
    help="type of discriminator",
    required=True
    )
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help="batch size",
    required=False
    )
parser.add_argument(
    '--dataset',
    type=str,
    help="dataset name",
    required=True
    )
# FLAGS,_= parser.parse_known_args()

# training settings 
# os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu_index
n_rounds=50
# batch_size=FLAGS.batch_size
batch_size=2
n_filters_d=32
n_filters_g=32
val_ratio=0.05
init_lr=2e-4
schedules={'lr_decay':{},  # learning rate and step have the same decay schedule (not necessarily the values)
           'step_decay':{}}
# alpha_recip=1./FLAGS.ratio_gan2seg if FLAGS.ratio_gan2seg>0 else 0
ratio_gan2seg = 10
alpha_recip=1./ratio_gan2seg if ratio_gan2seg>0 else 0
rounds_for_evaluation=range(n_rounds)

# set dataset
# dataset=FLAGS.dataset
dataset='DRIVE'
discriminator='image'
img_size= (640,640) if dataset=='DRIVE' else (720,720) # (h,w)  [original img size => DRIVE : (584, 565), STARE : (605,700) ]
img_out_dir="{}/segmentation_results_{}_{}".format(dataset,discriminator,ratio_gan2seg)
model_out_dir="{}/model_{}_{}".format(dataset,discriminator,ratio_gan2seg)
auc_out_dir="{}/auc_{}_{}".format(dataset,discriminator,ratio_gan2seg)
train_dir="../data/{}/training/".format(dataset)
test_dir="../data/{}/test/".format(dataset)
if not os.path.isdir(img_out_dir):
    os.makedirs(img_out_dir)
if not os.path.isdir(model_out_dir):
    os.makedirs(model_out_dir)
if not os.path.isdir(auc_out_dir):
    os.makedirs(auc_out_dir)


# train_imgs, train_vessels =utils.save_imgs(train_dir, augmentation=True, img_size=img_size, dataset=dataset)
# exit(0)
# set training and validation dataset
# train_imgs, train_vessels =utils.get_imgs(train_dir, augmentation=True, img_size=img_size, dataset=dataset)
print("load train data")
train_imgs = os.listdir(train_dir+"/fundus/")
train_imgs = [os.path.join(train_dir+"/fundus/", file_name) for file_name in train_imgs]
# train_imgs = train_imgs[:100]

train_vessels = os.listdir(train_dir+"/vessel/")
train_vessels = [os.path.join(train_dir+"/vessel/", file_name) for file_name in train_vessels]
# train_vessels = train_vessels[:100]
# train_vessels=np.expand_dims(train_vessels, axis=3)
# n_all_imgs=train_imgs.shape[0]
n_all_imgs=len(train_imgs)
n_train_imgs=int((1-val_ratio)*n_all_imgs)
train_indices=np.random.choice(n_all_imgs,n_train_imgs,replace=False)
train_indices = sorted(train_indices)
train_indices = np.asarray(train_indices)
train_batch_fetcher=utils.TrainBatchFetcher(train_imgs, train_vessels, batch_size, train_indices, (-1,img_size[0],img_size[1],1))
# val_imgs, val_vessels=train_imgs[np.delete(range(n_all_imgs),train_indices),...], train_vessels[np.delete(range(n_all_imgs),train_indices),...]
validate_indices = []
idx = 0
for i in range(n_all_imgs):
    if idx < len(train_indices) and train_indices[idx] == i:
        idx = idx + 1
        continue
    else:
        validate_indices.append(i)
validate_indices = np.asarray(validate_indices, np.int)
validate_batch_fetcher=utils.TrainBatchFetcher(train_imgs, train_vessels, batch_size, validate_indices, (-1, img_size[0], img_size[1], 1))
# set test dataset
print("load test data")
test_imgs, test_vessels, test_masks=utils.get_imgs(test_dir, augmentation=False, img_size=img_size, dataset=dataset, mask=True)

print("train set: {0}, validate set: {1}, test set: {2}".format(len(train_indices), len(validate_indices), test_imgs.shape[0]))
# create networks
g = generator(img_size, n_filters_g)
if discriminator=='pixel':
    d, d_out_shape = discriminator_pixel(img_size, n_filters_d,init_lr)
elif discriminator=='patch1':
    d, d_out_shape = discriminator_patch1(img_size, n_filters_d,init_lr)
elif discriminator=='patch2':
    d, d_out_shape = discriminator_patch2(img_size, n_filters_d,init_lr)
elif discriminator=='image':
    d, d_out_shape = discriminator_image(img_size, n_filters_d,init_lr)
else:
    d, d_out_shape = discriminator_dummy(img_size, n_filters_d,init_lr)

gan=GAN(g,d,img_size, n_filters_g, n_filters_d,alpha_recip, init_lr)
g.summary()
d.summary()
gan.summary()
validate_losses = []
validate_accuracies = []
# start training
scheduler=utils.Scheduler(n_train_imgs//batch_size, n_train_imgs//batch_size, schedules, init_lr) if alpha_recip>0 else utils.Scheduler(0, n_train_imgs//batch_size, schedules, init_lr)
figure = plt.figure()
ax1 = figure.add_subplot(311)
ax2 = figure.add_subplot(312)
ax3 = figure.add_subplot(313)

train_loss = 0
train_accuracy = 0

validate_loss = 0
validate_accuracy = 0

print("training {} images :".format(n_train_imgs))
for n_round in range(n_rounds):

    train_loss = 0
    train_accuracy = 0
    # train D
    utils.make_trainable(d, True)
    for i in tqdm(range(scheduler.get_dsteps()), desc='D'):
        real_imgs, real_vessels = next(train_batch_fetcher)
        d_x_batch, d_y_batch = utils.input2discriminator(real_imgs, real_vessels, g.predict(real_imgs,batch_size=batch_size), d_out_shape)
        result = d.train_on_batch(d_x_batch, d_y_batch)
        # utils.print_metrics((n_round+1)*(i+1),acc=result[1],loss=result[0],type='D')

    # train G (freeze discriminator)
    utils.make_trainable(d, False)
    for i in tqdm(range(scheduler.get_gsteps()), desc='G'):
        real_imgs, real_vessels = next(train_batch_fetcher)
        g_x_batch, g_y_batch=utils.input2gan(real_imgs, real_vessels, d_out_shape)
        result = gan.train_on_batch(g_x_batch, g_y_batch)
        train_loss += result[0]
        train_accuracy += result[1]
        # utils.print_metrics((n_round+1)*(i+1),acc=result[1],loss=result[0],type='GAN')

    train_loss /= scheduler.get_gsteps()
    train_accuracy /= scheduler.get_gsteps()

    # evaluate on validation set
    if n_round in rounds_for_evaluation:
        validate_loss = 0
        validate_accuracy = 0
        for i in range(len(validate_indices)//2):
            # D
            val_real_imgs, val_real_vessels = next(validate_batch_fetcher)
            d_x_test, d_y_test=utils.input2discriminator(val_real_imgs, val_real_vessels, g.predict(val_real_imgs,batch_size=batch_size), d_out_shape)
            loss, acc=d.evaluate(d_x_test,d_y_test, batch_size=batch_size, verbose=0)
            # utils.print_metrics(n_round+1, loss=loss, acc=acc, type='D')
            # G
            gan_x_test, gan_y_test=utils.input2gan(val_real_imgs, val_real_vessels, d_out_shape)
            loss,acc=gan.evaluate(gan_x_test,gan_y_test, batch_size=batch_size, verbose=0)
            validate_loss += loss
            validate_accuracy += acc
            # utils.print_metrics(n_round+1, acc=acc, loss=loss, type='GAN')
        # save the model and weights with the best validation loss

        with open(os.path.join(model_out_dir,"g_{}_{}_{}.json".format(n_round,discriminator,ratio_gan2seg)),'w') as f:
            f.write(g.to_json())
        g.save_weights(os.path.join(model_out_dir,"g_{}_{}_{}.h5".format(n_round,discriminator,ratio_gan2seg)))

        validate_loss /= len(validate_indices)//2
        validate_accuracy /= len(validate_indices)//2

        ax1.plot(n_round, train_loss, 'rs', label='train_loss')
        ax1.plot(n_round, validate_loss, 'bs', label='validate_loss')

        ax2.plot(n_round, train_accuracy, 'ro', label='train_accuracy')
        ax2.plot(n_round, validate_accuracy, 'bo', label='validate_accuracy')

    plt.pause(0.1)

    # update step sizes, learning rates
    scheduler.update_steps(n_round)
    K.set_value(d.optimizer.lr, scheduler.get_lr())
    K.set_value(gan.optimizer.lr, scheduler.get_lr())

    # evaluate on test images
    if n_round in rounds_for_evaluation:
        generated=g.predict(test_imgs,batch_size=batch_size)
        generated=np.squeeze(generated, axis=3)
        vessels_in_mask, generated_in_mask = utils.pixel_values_in_mask(test_vessels, generated , test_masks)
        auc_roc=utils.AUC_ROC(vessels_in_mask,generated_in_mask,os.path.join(auc_out_dir,"auc_roc_{}.npy".format(n_round)))
        auc_pr=utils.AUC_PR(vessels_in_mask, generated_in_mask,os.path.join(auc_out_dir,"auc_pr_{}.npy".format(n_round)))
        utils.print_metrics(n_round+1, auc_pr=auc_pr, auc_roc=auc_roc, type='TESTING')

        # print test images
        segmented_vessel=utils.remain_in_mask(generated, test_masks)
        for index in range(segmented_vessel.shape[0]):
            Image.fromarray((segmented_vessel[index,:,:]*255).astype(np.uint8)).save(os.path.join(img_out_dir,str(n_round)+"_{:02}_segmented.png".format(index+1)))
        ax3.plot(n_round, auc_pr, 'rs', label='auc_pr')
        ax3.plot(n_round, auc_roc, 'bo', label='auc_roc')

plt.show()
