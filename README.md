<h1 align="center">Google Summer of Code 2021 - TensorFlow</h1>
<p align="center">
  <a href="https://summerofcode.withgoogle.com/projects/#4563139045097472">
    <img src="Images/readme.jpg" alt="Logo" width="369.9" height="150">
  </a>
</p>
<h2 align="center">Improving TensorFlow GAN library </h2>
<h4 align="center">Mentor : <a href = "https://github.com/margaretmz">Margaret Maynard Reid</a> </h4>
<p align="center"><strong>
  <a href="https://summerofcode.withgoogle.com/projects/#4563139045097472">Project Link</a> |
  <a href="https://github.com/tensorflow/gan">TF-GAN Library</a>
  </strong>
</p>

## Project Abstract
[TensorFlow GAN](https://github.com/tensorflow/gan) is a lightweight library that provides a convenient way to train and evaluate GAN models.The main aim of this project is to update the TF-GAN library by adding more recent variants of GANs that has more applications. For this we selected ESRGAN<sup>1</sup> for the task of Image-Super-Resolution and ControlGAN<sup>2</sup> for Text-to-Image translation. Along with these examples we also add more functionalities to TF-GAN library that can help in the process of training, evaluation or inference of GAN models. 
This project also aims to add notebook tutorials for these examples which can also help users to gain insights into the implementation, training and evaluation process for these models and can help in exploring different useful features of the TF-GAN library.  This also allows users to train the models directly on Google Colaboratory.

## Project Scope
### ESRGAN<sup>1</sup> - Enhanced Super-Resolution Adversarial Network
Image Super-Resolution is the process of reconstructing high resolution (HR) image from a given low resolution (LR) image. Such a task has numerous application in today's world. The [Super-Resolution GAN](https://arxiv.org/abs/1609.04802) model was a major breathrough in this field and was capable of generating photorealistic images, however the model also generated artifacts that reduced the overall visual quality. To overcome this, the ESRGAN<sup>1</sup> model was proposed with three major changes made to the SRGAN model :

1. Using Residual-in-Residual Dense Block (RRDB) without batch normalization as basic network building unit
2. Using an improved method to calculate adversarial loss used in [RelativisticGAN](https://arxiv.org/abs/1807.00734v3) 
3. Improving perceptual loss function by using features before activation. 

Through this project, the ESRGAN model was added as an [example](esrgan) to TF-GAN library ([#47](https://github.com/tensorflow/gan/pull/47)). Additionally two [notebook files](esrgan/colab_notebooks) for end-to-end training of the model on GPU as well as TPU are also implemented which can be directly run on Google Colaboratory ([#48](https://github.com/tensorflow/gan/pull/48)). The model was trained on the DIV2K dataset and was able to achieve great results. Some of the results obtained are displayed in the tutorial notebook.Evaluation metrics such as FID and Inception Scores, for evaluating the model was also calculated using TF-GAN. The Relativistic Average GAN loss used in the model was also added as a loss function to TF-GAN ([#46](https://github.com/tensorflow/gan/pull/46)). The ESRGAN example has not been merged to TF-GAN yet, and is in review. 

### ControlGAN<sup>2</sup> - Controllable Text-to-Image Generation
The Controllable text-to-image Generative Adversarial Network is used for the task of generating high-quality images based on textual descriptions and can make changes to certain visual attributes of the image based on the same. This can potentially have numerous applications in areas such as art generation, UI designing and image editing. The generator of ControlGAN makes use of two attention modules - Spatial Attention Module and Channel-Wise Attention module.  The discriminator used is also different from other GAN networks, and checks the correlation between subregions of the generated image and the descriptions. Perceptual loss function is also used for improving the quality of the generated images. 

This is a work in progress and although the basic implementation is completed, the model is currently being trained on the Caltech-UCSD Birds dataset. Once the training process is done it will also be added to the TF-GAN library. 

## Tasks
|Tasks|Code|Status|TF-GAN PR|
|:-:|:-:|:-:|:-:|
|<br />Train and Evaluate ESRGAN<sup>1</sup><br /><img width=1/>|[Link](esrgan)|In Review|[#47](https://github.com/tensorflow/gan/pull/47)|
|<br />Write Colab notebook for training and evaluating <br>ESRGAN using TF-GAN|[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nivedwho/GSoC-TF-GAN/blob/main/esrgan/colab_notebooks/ESRGAN_GPU.ipynb)|In Review|[#48](https://github.com/tensorflow/gan/pull/48)|
|<br />Add TPU support for ESRGAN<br /><img width=1/>|[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nivedwho/GSoC-TF-GAN/blob/main/esrgan/colab_notebooks/ESRGAN_TPU.ipynb)|In Review|[#48](https://github.com/tensorflow/gan/pull/48)|
|<br />Add RaGAN loss function to TF-GAN<br /><img width=1/>|[Link](https://github.com/tensorflow/gan/pull/46/commits/8d9bab792d9573f45bc53b3fe84bee26cce95b98)|In Review|[#46](https://github.com/tensorflow/gan/pull/46)|
|<br />Train and Evaluate ControlGAN<sup>2</sup> model<br /><img width=1/>|[Link](ControlGAN/)| In progress |
|<br />Write Colab notebook for training and evaluating <br>ControlGAN using TF-GAN|Link| In progress |

## What's Next? 
Currently almost all the text-to-image generation models are being trained on datasets such as CUB and COCO for benchmarking their performance and as far as we know only results for such models are publicly available. Once the implementation of ControlGAN is completed, we plan to extend it to serve some real-world applications in areas such as art generation or image editing and for doing so we are looking for other relevant datasets to train the model. At the same time we are also looking for ways to improve its performance. 

At present there are not a lot of publicly available resources exploring the area of text-image generation and as a result we are also planning to publish a tutorial / blog post discussing the implementation, training process of ControlGAN and the results obtained.

### Acknowledgement
I would like to thank [Margaret Maynard Reid](https://github.com/margaretmz) and [Joel Shor](https://github.com/joel-shor) for their valuable guidance and mentorship. I would like to also thank [Google Cloud Platform](https://cloud.google.com/) and [TPU Research Cloud](https://sites.research.google/trc/) for extending their support which has helped in accelerating the development of this project.

### Reference
[1] [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)

[2] [Controllable Text-to-Image Generation](https://arxiv.org/abs/1909.07083v2)
