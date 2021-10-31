# Computer Vision Training

This project is a part of my personal training in Computer Vision. Here, we can find two folders, Computer Vision Module (compvis) and Practical Examples. The compvis module is a pipeline that builds any model practically. It covers several CNN architectures, and also, some skills for image preprocessing.

The compvis module is inspired by the [PyImageSearch](https://www.pyimagesearch.com/) module, so the credits for this module is to the author of the website Adrian Rosebrock. How did I have access to the module? I've grabbed this repository on GitHub [1](https://github.com/Sid2697/pyimage-Learning) and I found this website [2](https://blog.csdn.net/zimiao552147572/article/details/106718957#t1), all of them contain the module. Why did I change the name of module? During my training, typing line by line to understand the module, I've decided to change the name of the module, and also, some modifications in the module (name changing and including class), for a questions of simplicity and coding preference.

The folder Practical Example contains examples how to use the module. As I'm not enrolled at PyImageSearch, I should follow the examples from [1](https://github.com/Sid2697/pyimage-Learning). The problem was, in this repository there's no explanation how to use the module, neither discussion about the results. I learned how to use it by myself and reading the PyImageSearch’s blog. To understand each model, I did a deep reading about the papers of each model. Each example presented here is followed by a model explanation, discussion about the results and evidently, a detailed explanation of how to use the module.

## The Computer Vision Module (compvis) is organized as:

- [compvis](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/compvis) module

  - [callbacks](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/compvis/callbacks) sub-module
    - [EpochCheckPoint](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/callbacks/epochcheckpoint.py) class
    - [TrainingMonitor](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/callbacks/trainingmonitor.py) class

  - [datasets](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/compvis/datasets) sub-module
    - [SimpleDatasetLoader](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/datasets/simpledatasetoader.py) class

  - [io](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/compvis/io) sub-module
    - [HDF5DatasetGenerator](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/io/hdf5datasetgenerator.py) class
    - [HDF5DatasetWriter](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/io/hdf5datasetwriter.py) class

  - [nn](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/compvis/nn) sub-module
    - [ANN](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/nn/neuralnetwork.py) class
    - [cnns](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/compvis/nn/cnns) sub-sub-module
      - [AlexNet](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/nn/cnns/alexNet.py) class
      - [DeeperGoogLeNet](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/nn/cnns/deepergooglenet.py) class
      - [FCHeadNet](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/nn/cnns/fcheadnet.py) class
      - [LeNet](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/nn/cnns/lenet.py) class
      - [MiniGoogLeNet](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/nn/cnns/minigooglenet.py) class
      - [MiniVGG](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/nn/cnns/minivgg.py) class
      - [ShallowNet](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/nn/cnns/shallownet.py) class
    - [lr](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/compvis/nn/lr) sub-sub-module
      - [LRFunc](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/nn/lr/learning_rate_functions.py) class

  - [preprocessing](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/compvis/preprocessing) sub-module
    - [CropPreprocessor](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/preprocessing/croppreprocessor.py) class
    - [ImageToArrayPreprocessor](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/preprocessing/imagetoarraypreprocessor.py) class
    - [MeanPreprocessor](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/preprocessing/meanpreprocessor.py) class
    - [PatchPreprocessor](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/preprocessing/patchpreprocessor.py) class
    - [ResizeAR](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/preprocessing/resizear.py) class
    - [SimplePreprocessor](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/preprocessing/simplepreprocessor.py) class

  - [utils](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/compvis/utils) sub-module

    - [rank5_accuracy](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/compvis/utils/ranked.py) attribute


## The Practical Examples

The practical examples follow a logical order. Using the compvis module, We start from a simple image classifier into the most advanced. Each folder presented in the Practical Examples corresponds to a specific model or techniques (image preprocessing, learning rate scheduler, regularization etc). Several datasets are considered along the examples, all of them are cited, due to a space limitation, I did not up load the datasets, but you find the link to download all of them in the proper moment.

1 - [Simple Image Classifier](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/Practical%20Examples/1%20-%20Simple%20Image%20Classifier)

Image classification with Logistic Regression and K-Nearest Neighbors on the Animals dataset. Logistic Regression accuracy 59% and K-Nearest Neighbors accuracy 60%
  - [simple_image_classifier.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/1%20-%20Simple%20Image%20Classifier/simple_image_classifier.ipynb)

2 - [ANN Image Classifier](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/Practical%20Examples/2%20-%20ANN_Image_Classifier)

Image classification using Artificial Neural Network. Example made with the class ANN from the computer vision module, but also with TensorFlow library. The dataset considered are Animals, MNIST (8x8) and CIFAR10. The accuracy for the Animals dataset is 59%, for MNIST 97% and for CIFAR10 57%.
  - [ann_animals.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/2%20-%20ANN_Image_Classifier/ann_animals.ipynb)
  - [ann_mnist.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/2%20-%20ANN_Image_Classifier/ann_mnist.ipynb)
  - [cifar10_ann.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/2%20-%20ANN_Image_Classifier/cifar10_ann.ipynb)

3 - [Simple CNN - ShallowNet](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/Practical%20Examples/3%20-%20Simple%20CNN)

Image classification using Convolution Neural Network, specifically ShallowNet architecture from the compvis module. The ShalloNet model is composed by one convolutional layer and the full conected layers. The dataset used are Animals (accuracy of 70%) and CIFAR10 (accuracy of 62%).
  - [Animals_shallownet.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/3%20-%20Simple%20CNN/Animals_shallownet.ipynb)
  - [CIFAR10_shallownet.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/3%20-%20Simple%20CNN/CIFAR10_shallownet.ipynb)

4 - [LeNet on MNIST](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/Practical%20Examples/4%20-%20LeNet%20on%20MNIST)
Image classification using the LeNet architecture on the MNIST (28x28) dataset. The obtained accuracy was 98% without over-fit.
  - [lenet_mnist.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/4%20-%20LeNet%20on%20MNIST/lenet_mnist.ipynb)

5 - [MiniVGG on CIFAR10](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/Practical%20Examples/5%20-%20MiniVGG%20on%20CIFAR10)

Image classifications using the Mini VGG architecture on the CIFAR10 dataset. In this example some regularization techniques was implemented. The accuracy for this model was 80% on the test set.
  - [minivgg_cifar10.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/5%20-%20MiniVGG%20on%20CIFAR10/minivgg_cifar10.ipynb)

6 - [Learning Rate Scheduler](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/Practical%20Examples/6%20-%20Learning%20%20Rate%20Schedulers)

Image classification using MiniVGG architecture on the CIFAR10. In this example, we consider the Learning Rate Scheduler from TensorFlow, passing a piecewise function to change the learn rate every 5 epochs. The best result have 79% of accuracy. This result is a little bit smaller than the previous example, on other hand, the overfit was reduced.
  - [leaning_rate.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/6%20-%20Learning%20%20Rate%20Schedulers/learning_rate.ipynb)

7 - [DataAugumentation  and aspect ratio](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/Practical%20Examples/7%20-%20DataAugumentation%20%20and%20aspect%20ratio)

Image classification using Mini VGG network on Animals dataset. In this example, was considered Data Augmentation regularization, to the image preprocessing, we've resized all the images maintaining the aspect ratio. The accuracy was 74% without overfit.
  - [animals_data_augmentation.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/7%20-%20DataAugumentation%20%20and%20aspect%20ratio/animals_data_augmentation.ipynb)

8 - [Feature extraction and HDF5 file](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/Practical%20Examples/8%20-%20Feature%20extraction%20and%20HDF5%20file)

Image classification using transfer learning. We consider the Feature Extraction using the convolutional (the body of the network) layers from the VGG16 pre-trained model. To store all obtained features, we utilize a dataset writer outputting a HDF5 file. To realize the classification, we consider the Logistic Regression Model from Scikit-Learn.  The accuracy obtained for the Animals dataset on the test set was 99%, our best result until now. For the 17 Flowers the accuracy was 92%.
  - [feature_extr_hdf5_VGG16_17flowers.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/8%20-%20Feature%20extraction%20and%20HDF5%20file/feature_extr_hdf5_VGG16_17flowers.ipynb)
  - [feature_extr_hdf5_VGG16_animals.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/8%20-%20Feature%20extraction%20and%20HDF5%20file/feature_extr_hdf5_VGG16_animals.ipynb)

9 - [Transfer Learning with Fine Tuning](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/Practical%20Examples/9%20-%20Transfer%20Learning%20with%20Fine%20Tuning)

Image classification using transfer learning. We consider the fine-tuning method using a VGG16 pre-trained model. The training process is made in two step. The first step is the feature extraction and classification made by a defined head function, in this moment the weights from VGG16 are not tweaked. The second step the head function communicates with some layer from the body, realizing the fine-tuning over the weights from the pre-trained model. We've considered the 17 Flowers dataset, the accuracy was 95%, a considerable increase. On the other hand, the model seems in over-fit.
  - [finetune.py](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/9%20-%20Transfer%20Learning%20with%20Fine%20Tuning/finetune.py)
  - [test_model.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/9%20-%20Transfer%20Learning%20with%20Fine%20Tuning/test_model.ipynb)

10 - [Dos vs Cats dataset with AlexNet and transfer learning](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/Practical%20Examples/10%20-%20Dogs%20vs%20Cats)

Using HDF5 datawriter and generator to manege the dataset.Implementation of AlexNet architecture, training the classification model on Dog vs Cats dataset from Kaggle. Rank-1: 94.08% with crop method on the test set. Using feature extraction with ResNet50, we obtain 0.9856 as accuracy score. The classification model was trained with 2500 images, just 10% of the original training set.

  - [data_building.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/10%20-%20Dogs%20vs%20Cats/data_building.ipynb)
  - [training_alexnet.py](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/10%20-%20Dogs%20vs%20Cats/training_alexnet.py)
  - [testing_alexnet.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/10%20-%20Dogs%20vs%20Cats/testing_alexnet.ipynb)
  - [features_extraction.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/10%20-%20Dogs%20vs%20Cats/features_extraction.ipynb)
  - [training_on_features.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/10%20-%20Dogs%20vs%20Cats/training_on_features.ipynb)

11 - [GoogLeNet](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/Practical%20Examples/11%20-%20GoogLeNet)

Training the Mini GoogLeNet on the CIFAR10 and, Deeper GoogLeNet on the Tiny ImageNet dataset. The results with the Mini GoogLeNet on CIFAR10 show an accuracy of 91% on training set, the best result until now, using this module. The results on Tiny ImageNet was also good, we've obtained an error rate of 0.55, a good result for the Tiny ImageNet challenge, not the best, but convicently. We also used the HDF5DatasetGenerator to read the images on the batch instead of allocate all images on the memory.

  - [buildind_tiny_dataset.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/11%20-%20GoogLeNet/buildind_tiny_dataset.ipynb)
  - [minigoogle_cifar10.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/11%20-%20GoogLeNet/minigoogle_cifar10.ipynb)
  - [googlenet.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/11%20-%20GoogLeNet/googlenet.ipynb)
  - [evaluating.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/11%20-%20GoogLeNet/evaluating.ipynb)

12 - [ResNet](https://github.com/IgorMeloS/Computer-Vision-Training/tree/main/Practical%20Examples/12%20-%20ResNet)

Training ResNet architecture on CIFAR10 and Tiny ImageNet dataset. The accuracy on the validation set of CIFAR10 was 92.8%, the best result in this training folder. On the Tiny ImageNet the error rate using ResNet was 51%, showing that the model reached a good generalization, but the model is not well efficient to real world problems.

  - [ResNet_cifar10.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/12%20-%20ResNet/ResNet_cifar10.ipynb)
  - [ResNet_Tiny.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/12%20-%20ResNet/ResNet_Tiny.ipynb)
  - [evaluating.ipynb](https://github.com/IgorMeloS/Computer-Vision-Training/blob/main/Practical%20Examples/12%20-%20ResNet/evaluating.ipynb)
## Datasets

The examples considered here are trained with classical datasets as MNIST, CIFAR-10, CALTECH, Tiny ImageNet, COCO, Kaggle challenges datasets and others.

## Usage

If you want use the compvis module, Add the following line to your `~/.profile file.`

`export PYTHONPATH=$PYTHONPATH:/path/you/want/to/add`

## References

Along the project, I cite the papers that refer models or techniques, in the proper moment. I also would like to highlight the importance of the Massive Open Online Courses (MOOCs), that offers to us many possibilities to learn practically (sometimes free or not, but always in the web). I’ve used many of them in this project as

- [PyImageSearch](https://www.pyimagesearch.com/) offers to us free examples for real applications in computer vision, the best website about Computer Vision.

- [Learn OpenCV](https://learnopencv.com/) website specialized about OpenCV.

- [Machine Learning Mastery](https://machinelearningmastery.com/blog/) is  an excellent website that proposes several practical examples for Machine Learning projects in your blog.

- [freeCodeCamp.org](https://www.youtube.com/c/Freecodecamp) YouTube channel with code examples about technical courses.

- [Deep Learning and Computer Vision A-Z™: OpenCV, SSD & GANs](https://www.udemy.com/share/101rbO3@p_-13AH-2kf-7X-QYKG5iB-Ze6U-hHXsq7ou2gG5-Jqa4J7QiBBbb-HGpTF6oN7b/) on the Udemy platform.

**Keep in mind, the best sources are always the original papers of each Convolutional Neural Network model. From them, we can understand better the idea behind each model**.
