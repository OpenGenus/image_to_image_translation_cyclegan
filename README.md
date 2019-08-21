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
