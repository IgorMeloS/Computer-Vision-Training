# Computer Vision Module (compvis) based on PyImageSearch

The compvis module is inspired by the PyImageSearch module. I've grabbed some repositories ([1](https://github.com/Sid2697/pyimage-Learning), [2](https://github.com/meizhoubao/pyimagesearch) and [3](https://github.com/dloperab/PyImageSearch-CV-DL-CrashCourse)) as example and several instruction on the [website](https://www.pyimagesearch.com/). I gathered all the information obtained to train. During this process, I've decided to edit the module name, sub module and some codes. However, I emphasize once again the author's credits.

I present the Computer Vision Module in the folder compvis. The module is a pipeline to simplify the code process for any computer vision task. The folder Practical Examples contains examples using the the compvis module to build and evaluate the deployed models.

## The Computer Vision Module (compvis) is organized as:

- [compvis](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis) module

  - [ann](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis/ann) sub-module
    - [ANN](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/ann/neuralnetwork.py) class
    - [cnns](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis/ann/cnns) sub-sub-module
      - [FCHeadNet](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/nn/cnns/fcheadnet.py) class
      - [LeNet](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/ann/cnns/lenet.py) class
      - [MiniVGG](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/ann/cnns/minivgg.py) class
      - [ShallowNet](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/ann/cnns/shallownet.py) class

  - [datsets](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis/datasets) sub-module
    - [SimpleDatasetLoader](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/datasets/simpledatasetoader.py) class

  - [io](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis/io) sub-module
    - [HDF5DatasetWriter](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/io/hdf5datasetwriter.py) class

  - [preprocessing](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/compvis/preprocessing) sub-module
    - [ImageToArrayPreprocessor](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/preprocessing/imagetoarraypreprocessor.py) class
    - [ResizeAR](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/preprocessing/resizear.py) class
    - [SimplePreprocessor](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/compvis/preprocessing/simplepreprocessor.py) class


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
  - [lenet_mnist.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/4%20-%20LeNet%20on%20MNIST/lenet_mnist.ipynb)

5 - [MiniVGG on CIFAR10](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/Pratical%20Examples/5%20-%20MiniVGG%20on%20CIFAR10)

Image classifications using the Mini VGG architecture on the CIFAR10 dataset. In this example some regularization techniques was implemented. The accuracy for this model was 80% on the test set.
  - [minivgg_cifar10.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/5%20-%20MiniVGG%20on%20CIFAR10/minivgg_cifar10.ipynb)

6 - [Learning Rate Scheduler](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/Pratical%20Examples/6%20-%20Learning%20%20Rate%20Schedulers)

Image classification using MiniVGG architecture on the CIFAR10. In this example, we consider the Learning Rate Scheduler from TensorFlow, passing a piecewise function to change the learn rate every 5 epochs. The best result have 79% of accuracy. This result is a little bit smaller than the previous example, on other hand, the overfit was reduced.
  - [leaning_rate.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/6%20-%20Learning%20%20Rate%20Schedulers/learning_rate.ipynb)

7 - [DataAugumentation  and aspect ratio](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/Pratical%20Examples/7%20-%20DataAugumentation%20%20and%20aspect%20ratio)

Image classification using Mini VGG network on Animals dataset. In this example, was considered Data Augmentation regularization, to the image preprocessing, we've resized all the images maintaining the aspect ratio. The accuracy was 74% without overfit.
  - [animals_data_augmentation.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/7%20-%20DataAugumentation%20%20and%20aspect%20ratio/animals_data_augmentation.ipynb)

8 - [Feature extraction and HDF5 file](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/Pratical%20Examples/8%20-%20Feature%20extraction%20and%20HDF5%20file)

Image classification using transfer learning. We consider the Feature Extraction using the convolutional (the body of the network) layers from the VGG16 pre-trained model. To store all obtained features, we utilize a dataset writer outputting a HDF5 file. To realize the classification, we consider the Logistic Regression Model from Scikit-Learn.  The accuracy obtained for the Animals dataset on the test set was 99%, our best result until now. For the 17 Flowers the accuracy was 92%.
  - [feature_extr_hdf5_VGG16_17flowers.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/8%20-%20Feature%20extraction%20and%20HDF5%20file/feature_extr_hdf5_VGG16_17flowers.ipynb)
  - [feature_extr_hdf5_VGG16_animals.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/8%20-%20Feature%20extraction%20and%20HDF5%20file/feature_extr_hdf5_VGG16_animals.ipynb)

9 - [Transfer Learning with Fine Tuning](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/ComputerVision%20Module/Pratical%20Examples/9%20-%20Transfer%20Learning%20with%20Fine%20Tuning)

Image classification using transfer learning. We consider the fine-tuning method using a VGG16 pre-trained model. The training process is made in two step. The first step is the feature extraction and classification made by a defined head function, in this moment the weights from VGG16 are not tweaked. The second step the head function communicates with some layer from the body, realizing the fine-tuning over the weights from the pre-trained model. We've considered the 17 Flowers dataset, the accuracy was 95%, a considerable increase. On the other hand, the model seems in over-fit.
  - [finetune.py](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/9%20-%20Transfer%20Learning%20with%20Fine%20Tuning/finetune.py)
  - [test_model.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/ComputerVision%20Module/Pratical%20Examples/9%20-%20Transfer%20Learning%20with%20Fine%20Tuning/test_model.ipynb)
