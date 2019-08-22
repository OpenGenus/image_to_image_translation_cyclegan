# image_to_image_translation_cyclegan

***

### Setup

1. Download the Monet2Photos dataset as follows:

```
wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip
```

2. Make a new directory (e.g. 'data'), and make it the current directory

```
mkdir data
cd data
```

3. Unzip the zip file containing the dataset inside the 'data' directory

```
unzip monet2photo.zip
```

4. Install the keras_contrib library as follows:

```
pip install git+https://www.github.com/keras-team/keras-contrib.git
```

5. Create a new directory (e.g. 'results') to store the original, generated and reconstructed images

```
mkdir results
```

***

## About the execution modes

**A) Train mode**

* Vanilla execution mode
* Images will be loaded, and the generator and discriminator networks will be trained
* Generator will generate images from the given input, discriminator will distinguish between real and fake images
* The adversarial model is responsible for reaching an optimum value of the objective function
* After every epoch, the weights are updated to tune both the networks to reach maximum efficiency.

**B) Predict mode**

* The discriminator network isn't involved here - only the weights of the generator network are loaded, and the generator network is trained
* The generator generates images for the input images it is provided
* No networks are optimized, the generator just generates images for the entire batch of inputs.
