<br />
<h1 align="center">Improving TensorFlow GAN library</h1>
<p align="center">
  <a href="https://summerofcode.withgoogle.com/projects/#4563139045097472">Project Link</a> |
  <a href="https://github.com/tensorflow/gan">TF-GAN Library</a>
</p>
<p align="center">
  <a href="https://summerofcode.withgoogle.com/projects/#4563139045097472">
    <img src="Images/readme.jpg" alt="Logo" width="300" height="200">
  </a>
</p>
<br>

## Project Abstract
[TensorFlow GAN](https://github.com/tensorflow/gan) is a lightweight library that provides a convenient way to train and evaluate GAN models. GANs have come a long way in the past few years and in through this project more recent GAN models with better performance and more applications such as Image Super Resolution and Text-to-Image translation will be added as examples to the library.  Additionally multiple Colab notebooks will also be added to demonstrate the performances of each of these example models and also to explore various functionalities of the library. 

**Mentor**: [@margaretmz](https://github.com/margaretmz)

## Tasks
|Tasks|Code|Status|PR|
|:-:|:-:|:-:|:-:|
|Train and Evaluate ESRGAN<sup>1</sup> model|[Link](esrgan)| :heavy_check_mark: |[#45](https://github.com/tensorflow/gan/pull/45)|
|Write Colab notebook for training <br>and evaluating ESRGAN using TF-GAN |[Link](esrgan/colab_notebook)|  :heavy_check_mark:|[#45](https://github.com/tensorflow/gan/pull/45)|
|Add TPU support for ESRGAN|[Link](esrgan/colab_notebook)| :heavy_check_mark: |[#45](https://github.com/tensorflow/gan/pull/45)|
|Add RaGAN loss function to TF-GAN|[Link](esrgan/colab_notebook)| :heavy_check_mark: |[#45](https://github.com/tensorflow/gan/pull/45)|
|Train and Evaluate ControlGAN<sup>2</sup> model|[Link](ControlGAN/)| In progress |
|Write Colab notebook for training <br>and evaluating ControlGAN using TF-GAN |Link| In progress |
|Work on new datasets|Link| Pending |
|Write blog posts/tutorials for ControlGAN|Link| Pending |

## Work Done
### ESRGAN<sup>1</sup>- Enhanced Super-Resolution Adversarial Network
Image Super-Resolution is the process of enhancing the resolution of low resolution(LR) image.  Such a task has numerous application in today's world. The [Super-Resolution GAN](https://arxiv.org/abs/1609.04802) model was a major breathrough in this field and was capable of generating photorealistic images, however the model also generated artifacts that reduced the overall visual quality. To overcome this, the ESRGAN<sup>1</sup> model was proposed with three major changes made to the SRGAN model : Using Residual-in-Residual Dense Block (RRDB) without batch normalization as basic network building unit; Using an improved method to calculate adversarial loss used in [RelativisticGAN](https://arxiv.org/abs/1807.00734v3) ; Improving perceptual loss function by using features before activation. 
Through this project, the ESRGAN model was added as an example to TF-GAN library. Additionally notebook files for end-to-end training of the model on GPU as well as TPU are also implemented which can be directly run on Google Colaboratory.  The model was trained on the DIV2K dataset and was able to achieve great results. Evaluation metrics such as FID and Inception Scores, for evaluating the model was also calculated using TF-GAN. The Relativistic Average GAN loss used in the model was also added as a loss function to TF-GAN. 

### ControlGAN<sup>2</sup> - Controllable Text-to-Image Generation

### Acknowledgement
I would like to thank [Google Cloud Platform](https://cloud.google.com/) and [TPU Research Cloud](https://sites.research.google/trc/) whose support has really accelerated the development of this project.  

### Reference
[1] [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)
[2] [Controllable Text-to-Image Generation](https://arxiv.org/abs/1909.07083v2)
