# Boundary Detection

The aim of a boundary detection algorithm is to achieve image segmentation by identifying boundaries between different objects or regions within an image. Classical edge detection algorithms, including the Canny and Sobel baselines we will compare against, look for intensity discontinuities within an image to detect boundaries.

The more recent pb (**probability of boundary**) boundary detection algorithm significantly outperforms these classical methods by considering texture and color discontinuities in addition to intensity discontinuities. The algorithm leverages the idea that the boundary between two regions is often characterized by a rapid change in image properties, such as color, texture, or intensity.

## Running the package
This package was built using Python 3.7 and OpenCV on Ubuntu 20.04. Follow the instructions on [this](https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html) page to setup OpenCV for Ubuntu. Other packages include `matplotlib`, `scipy` and `scikitlearn`. These are relatively easy to install using `pip install *package_name*`. 

Download the package:
```
git clone git@github.com:latent-pixel/Boundary-Detection.git
```
Then, run the package using the following commands:
```
cd Pb-lite-Boundary-Detection
python3 phase1/code/Wrapper.py
```
The results can then be found in a separate `results` folder in the package.

## Filter Bank Generation
Filter bank is a collection of different filters: Oriented Difference of Gaussian (DOG), Leung-Malik (LM), and Gabor filters. This filter bank helps us capture the texture properties of an image. The filter bank generated is shown in the images below.

DOG Filters             |  Leung-Malik Filters             |  Gabor Filters
:-------------------------:|:-------------------------:|:-------------------------:
![](results/dog_fltrs.png)  |  ![](results/lm_fltrs.png)  |  ![](results/gabor_fltrs.png)

## Sample Output
Texture Map             |  Brightness Map             |  Color Map
:-------------------------:|:-------------------------:|:-------------------------:
![](results/1/image1_texton.png)  |  ![](results/1/image1_brightness.png)  |  ![](results/1/image1_color.png)

Texture Gradients              |  Brightness Gradients             |  Color Gradients
:-------------------------:|:-------------------------:|:-------------------------:
![](results/1/image1_texton_grad.png)  |  ![](results/1/image1_brightness_grad.png)  |  ![](results/1/image1_color_grad.png)

Ouput of the `Pb-lite algorithm` compared with `Canny` and `Sobel` baselines:
Canny Baseline              |  Sobel Baseline             |  Pb-lite Ouput
:-------------------------:|:-------------------------:|:-------------------------:
![](phase1/BSDS500/CannyBaseline/1.png)  |  ![](phase1/BSDS500/SobelBaseline/1.png)  |  ![](results/1/image1_pb_lite.png)