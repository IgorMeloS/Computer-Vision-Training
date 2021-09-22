# Computer Vision Module (compvis) from PyImageSearch

The compvis module is inspired by the PyImageSearch module. I've grabbed some repositories ([1](https://github.com/Sid2697/pyimage-Learning), [2](https://github.com/meizhoubao/pyimagesearch) and [3](https://github.com/dloperab/PyImageSearch-CV-DL-CrashCourse)) as example and several instruction on the [website](https://www.pyimagesearch.com/). I gathered all the information obtained to train. During this process, I've decided to edit the module name, sub module and some codes. However, I emphasize once again the author's credits.

I present the Computer Vision Module in the folder compvis. The module is a pipeline to simplify the code process for any computer vision task. The folder Practical Examples contains examples using the the compvis module to build and evaluate the deployed models.

## The Computer Vision Module (compvis) is organized as:

- [compvis](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis) module

  - [datsets](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis/datasets) sub-module
    - [SimpleDatasetLoader](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/datasets/simpledatasetoader.py) class

  - [preprocessing](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis/preprocessing) sub-module
    - [ImageToArrayPreprocessor](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/preprocessing/imagetoarraypreprocessor.py) class
    - [SimplePreprocessor](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/preprocessing/simplepreprocessor.py) class

  - [ann](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis/ann) sub-module
    - [ANN](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/ann/neuralnetwork.py) class
    - [cnns](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis/ann/cnns) sub-sub-module
      - [ShallowNet](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/ann/cnns/shallownet.py) class
      - [LeNet](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/ann/cnns/lenet.py) class


## The Practical Examples

The practical examples follow a logical order. Using the compvis module, We start from a simple image classifier into the most advanced. Each folder presented in the Practical Examples corresponds to a specific model or techniques (image preprocessing, learn rate, regularization etc). Several dataset are considered along the examples, all of them are cited, due to a space limitation, we do not up load them.

1 - [Simple Image Classifier](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/Pratical%20Examples/1%20-%20Simple%20Image%20Classifier)

Image classification wirh Logistic Regression and K-Nearest Neighbors on the Animals dataset. Logistic Regression accuracy 59% and K-Nearest Neighbors accuracy 60%
  - [simple_image_classifier.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/1%20-%20Simple%20Image%20Classifier/simple_image_classifier.ipynb)

2 - [ANN Image Classifier](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/Pratical%20Examples/2%20-%20ANN_Image_Classifier)

Image classification using Artificial Neural Network. Example made with the class ANN from the computer vision module, but also with TensorFlow library. The dataset considered are Animals, MNIST (8x8) and CIFAR10. The accuracy for the Animals dataset is 59%, for MNIST 97% and for CIFAR10 57%.
  - [ann_animals.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/2%20-%20ANN_Image_Classifier/ann_animals.ipynb)
  - [ann_mnist.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/2%20-%20ANN_Image_Classifier/ann_mnist.ipynb)
  - [cifar10_ann.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/2%20-%20ANN_Image_Classifier/cifar10_ann.ipynb)

3 - [Simple CNN - ShallowNet](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/Pratical%20Examples/3%20-%20Simple%20CNN)

Image classification using Convolution Neural Network, specifically ShallowNet architecture from the compvis module. The ShalloNet model is composed by one convolutional layer and the full conected layers. The dataset used are Animals (accuracy of 70%) and CIFAR10 (accuracy of 62%).
  - [Animals_shallownet.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/3%20-%20Simple%20CNN/Aninals_shallownet.ipynb)
  - [CIFAR10_shallownet.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/3%20-%20Simple%20CNN/CIFAR10_shallownet.ipynb)

4 - [LeNet on MNIST](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/Pratical%20Examples/4%20-%20LeNet%20on%20MNIST)

Image classification using the LeNet architecture on the MNIST (28x28) dataset. The obtained accuracy was 98% without over-fit.
  -[lenet_mnist.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/4%20-%20LeNet%20on%20MNIST/lenet_mnist.ipynb)
