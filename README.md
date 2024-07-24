For the purpose of road lane segmenation, the segformer model based on transofrmers has been trained with good reustls as shown in the result image.
This model training needs pairs of images and thier binary masks.
Based on this article https://medium.com/@diazoangga/implementing-lane-detection-in-carla-using-lanenet-330d8fd8720c, the mask code has been applied on the Tusimple dataset for the training dataset creation.
Trainiing process has been done for about 600 epochs on 200K pairs of images and masks.
