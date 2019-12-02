import matplotlib.pyplot as plt
import os
import numpy as np
if __name__ =="__main__":
    dataset = "DRIVE"
    img_dir = "../data/"+dataset+"/training/fundus"
    img_list = os.listdir(img_dir)
    print(len(img_list))
    fig = plt.figure()

    for i, img in enumerate(img_list[:100]):
        print(i//10+1, i%10+1, i%10 + 1)
        ax = fig.add_subplot(10, 10, i+1)

        img_path = os.path.join(img_dir,img)
        img_np = np.load(img_path)
        # print(img_np.shape)
        min_val = np.min(img_np)
        img_np = img_np - min_val
        max_val = np.max(img_np)
        img_np /= max_val
        plt.imshow(img_np)
    plt.show()
