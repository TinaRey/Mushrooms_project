from skimage import io
import os


data_dir = "./Mushrooms"

def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True

img_sizes = set()
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    for filename in os.listdir(folder_path):
        f = os.path.join(folder_path, filename)
        if verify_image(f)==False:
            print(f+str(verify_image(f)))


#Display images and their label:
# plt.figure(figsize=(4, 4))
# for data, labels in train_ds.take(1):
#     print(data.shape)
#     print(labels.shape)
#     for i in range(4):
#         print(data[i].shape)
#         ax = plt.subplot(2, 2, i + 1)
#         plt.imshow(data[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.show()
