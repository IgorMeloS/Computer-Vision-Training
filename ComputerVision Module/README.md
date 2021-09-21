# Computer Vision Module (compvis)

# Computer Vision Module (compvis) from PyImageSearch

The compvis module is inspired by the PyImageSearch module. I've grabbed some repositories ([1](https://github.com/Sid2697/pyimage-Learning), [2](https://github.com/meizhoubao/pyimagesearch) and [3](https://github.com/dloperab/PyImageSearch-CV-DL-CrashCourse)) as example and several instruction on the [website](https://www.pyimagesearch.com/). I gathered all the information obtained to train. During this process, I've decided to edit the module name, sub module and some codes. However, I emphasize once again the author's credits.

I present the Computer Vision Module in the folder compvis. The module is a pipeline to simplify the code process for any computer vision task. The folder Practical Examples contains examples using the the compvis module to build and evaluate the deployed models.

## The Computer Vision Module (compvis) is organized as:

- [compvis](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis) module

  - [datsets](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis/datasets) sub-module
    - [SimpleDatasetLoader](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/datasets/simpledatasetoader.py) class

  - [preprocessing] sub-module
    - [ImageToArrayPreprocessor](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/preprocessing/imagetoarraypreprocessor.py) class
    - [SimplePreprocessor](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/preprocessing/simplepreprocessor.py) class

  - [ann](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis/ann) sub-module
    - [ANN](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/ann/neuralnetwork.py) class
    - [cnns](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis/ann/cnns) sub-sub-module
      - [ShallowNet](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/ann/cnns/shallownet.py) class


## The Practical Examples

Inside each folder, we find the files for the examples

1 - [Simple Image Classifier](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/Pratical%20Examples/1%20-%20Simple%20Image%20Classifier)
  - [simple_image_classifier.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/1%20-%20Simple%20Image%20Classifier/simple_image_classifier.ipynb)
2 - [ANN Image Classifier](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/Pratical%20Examples/2%20-%20ANN_Image_Classifier)
  - [ann_animals.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/2%20-%20ANN_Image_Classifier/ann_animals.ipynb)
  - [ann_mnist.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/2%20-%20ANN_Image_Classifier/ann_mnist.ipynb)
  - [cifar10_ann.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/2%20-%20ANN_Image_Classifier/cifar10_ann.ipynb)
3 - [Simple CNN - ShallowNet](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/Pratical%20Examples/3%20-%20Simple%20CNN)
  - [Animals_shallownet.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/3%20-%20Simple%20CNN/Aninals_shallownet.ipynb)
  - [CIFAR10_shallownet.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/3%20-%20Simple%20CNN/CIFAR10_shallownet.ipynb)
