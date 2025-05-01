# Neural-style-Transfer

# COMPANY : CODTECH IT SOLUTION

# INTERN ID :CT08WU25

# DOMAIN : ARTIFICIAL INTELLIGENCE

# DURATION:8 WEEKS

# MENTOR:NEELA SANTOSH

# DESCRIPTION:
This project implements Neural Style Transfer (NST) using TensorFlow and Keras, allowing a user to transform a content image in the style of another image. Inspired by artistic creativity and deep learning, this script blends the visual style of one image with the structural content of another, resulting in a stylized output that resembles artwork.
The core idea behind neural style transfer is to separate and recombine the content and style of images using a convolutional neural network (CNN). This script uses a pre-trained VGG19 network from the ImageNet dataset to extract both content and style representations. Specifically, content features are drawn from a deeper layer (block5_conv2) while style features come from multiple early to mid layers (block1_conv1 through block5_conv1) to capture textures and patterns.
The workflow begins with image preprocessing. Input images (content and style) are resized and normalized using TensorFlow functions to fit within a specified dimension and prepared for feature extraction. The load_img function handles these transformations, while tensor_to_image and imshow assist with displaying and converting images during intermediate steps.
A custom model class called StyleContentModel is built on top of VGG19. It extracts intermediate layer outputs representing content and style. Style features are converted into Gram matrices, which quantify the correlations between different filter responses, effectively capturing the style. These matrices are essential for comparing and minimizing the stylistic differences during training.
The training loop optimizes the content image to adopt the style image's characteristics. This is accomplished by defining a loss function (style_content_loss) that combines both style and content losses, as well as a total variation loss to encourage spatial smoothness and reduce noise. The optimizer used is Adam, a common gradient descent variant effective for image-based tasks.
The train_step function performs a single optimization step using gradient tape to compute gradients of the loss with respect to the image pixels. The image is iteratively updated to reduce the combined loss, resulting in a stylized output over several epochs.
An optional but unique feature of this script is the inclusion of an ASCII preview of the output image, converting the stylized image into grayscale characters for terminal display, adding a creative and accessible visualization layer.
The final stylized image is displayed using matplotlib, and optionally saved to disk if a path is provided. The complete process takes around a few minutes depending on hardware, and is suitable for both experimentation and artistic production.
In summary, this project offers a clean, modular, and effective implementation of neural style transfer, combining computer vision, optimization, and creativity. It showcases how deep learning can be used not only for analytics but also for generating art and new forms of expression.

# OUTPUT:![Image](https://github.com/user-attachments/assets/33249e3c-aaf4-4ccc-a247-4330d24c7a17)
