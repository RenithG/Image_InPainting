# Image_InPainting
Implemented Image Inpainting algorithm using GAN in PyTorch framework.

For step by step code walkthrough and to know more about Image Inpainting algorithm and its applications, you can have a look at my article in this link [How to Repair your Damaged Images with Deep Learning](https://medium.com/@renithprem/how-to-repair-your-damaged-images-with-deep-learning-cc404aec144?source=friends_link&sk=389ba8a4fcc2bc6aeb4916f956a33739)

For Training the Image Inpainting with regular / square masking based on GAN, run the following python command.

  python Train_Image_Inpainting.py

Place the training images under DATASET_NAME folder. The model will be trained from the scratch. You can see the sampled reconstructed output images for every sampling period based on SAMPLE_INTERVAL value. For higher iterations, you can see the output results are improvising.
The Trained model will be saved as "inpaint_model.pth"

For Testing the trained model, then run the follwing command,

  python Test_Image_Inpainting.py
  
You can give your own test image and verify the output for your trained model. Give the path of your test image to TEST_IMAGE and path of your trained model to TRAINED_MODEL_NAME.
  
TO-DO :
  1. To implement with irregular masking inputs.
  2. User providing custom mask inputs.
