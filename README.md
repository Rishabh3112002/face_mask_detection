# Face Mask Detector

In times like this when there is a global pandemic and a increase in air pollution, the need of face mask has become mandatory for everyone. Due to the global pandemic face mask has become mandatory in all public places like office, schools, etc. Face mask detection refers to detect whether a person is wearing a mask or not. The model detects if a person is wearing a face mask or not with real-time video feed. This model uses MobileNet architecture with a dataset from Kaggle to decide and predict if the person is wearing mask or not. The model has the capability to decide whether the person is wearing mask correctly, if he is wearing mask but incorrectly and not wearing mask at all. 

### Mask
![mask](https://user-images.githubusercontent.com/65342857/147691112-6c3e12f7-b1b5-4d88-85eb-16b1b5e79dbb.jpg)
### No Mask
![No Mask](https://user-images.githubusercontent.com/65342857/147691143-b0c55b68-b88c-4aba-a767-135194964921.jpg)
### Mask Not worn Properly
![Mask not worn Properly](https://user-images.githubusercontent.com/65342857/147691174-3964c98c-167f-4ad9-8814-e84733badefe.jpg)

## Requirements
``
pip install tensorflow
``

``
pip install numpy
``

``
pip install sklearn
``

``
pip install imutils
``

``
pip install opencv-python
``

``
pip install opencv-contrib-python
``

``
pip intall PILLOW
``

Download the file named face_detection from here.

The dataset of the images are taken from https://www.kaggle.com/vijaykumar1799/face-mask-detection.

## Precautions
Few of the above mentioned pacakages few dont work on Python 3.8 so using Python 3.7 is preferable. While working in MacOS we need to do ``pip install tensorflow-macos`` instead and Python 3.8 and 3.7 both work fine in MacOS.
