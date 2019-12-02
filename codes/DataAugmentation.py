import codes.utils as utils
import os

if __name__ == "__main__":
    # batch_size=FLAGS.batch_size
    batch_size = 2
    n_filters_d = 32
    n_filters_g = 32
    ratio_gan2seg = 10
    alpha_recip = 1. / ratio_gan2seg if ratio_gan2seg > 0 else 0

    # set dataset
    # dataset=FLAGS.dataset
    dataset = 'DRIVE'
    discriminator = 'image'
    img_size = (640, 640) if dataset == 'DRIVE' else (
    720, 720)  # (h,w)  [original img size => DRIVE : (584, 565), STARE : (605,700) ]
    img_out_dir = "{}/segmentation_results_{}_{}".format(dataset, discriminator, ratio_gan2seg)
    model_out_dir = "{}/model_{}_{}".format(dataset, discriminator, ratio_gan2seg)
    auc_out_dir = "{}/auc_{}_{}".format(dataset, discriminator, ratio_gan2seg)
    train_dir = "../data/{}/training/".format(dataset)
    test_dir = "../data/{}/test/".format(dataset)
    if not os.path.isdir(img_out_dir):
        os.makedirs(img_out_dir)
    if not os.path.isdir(model_out_dir):
        os.makedirs(model_out_dir)
    if not os.path.isdir(auc_out_dir):
        os.makedirs(auc_out_dir)

    train_imgs, train_vessels =utils.save_imgs(train_dir, augmentation=True, img_size=img_size, dataset=dataset)
