import tensorflow as tf
import numpy as np
import PIL.Image
import time
import functools
import matplotlib.pyplot as plt

def tensor_to_image(tensor):
    """Converts a tensor to an image"""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img, max_dim=512):
    """Loads and preprocesses images"""
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    
    new_shape = tf.cast(shape * scale, tf.int32)
    
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    """Display an image with an optional title"""
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')

def vgg_layers(layer_names):
    """Creates a VGG model that returns a list of intermediate output values"""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """Calculate Gram Matrix for style representation"""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):
    """Model for extracting style and content features"""
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
        
    def call(self, inputs):
        """Extract features"""
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        
        content_dict = {content_name: value 
                       for content_name, value 
                       in zip(self.content_layers, content_outputs)}
        
        style_dict = {style_name: value
                     for style_name, value
                     in zip(self.style_layers, style_outputs)}
        
        return {'content': content_dict, 'style': style_dict}

def clip_0_1(image):
    """Clips pixel values to [0,1] range"""
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    """Calculate the combined style and content loss"""
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_outputs)
    
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_outputs)
    
    total_loss = style_loss + content_loss
    return total_loss

@tf.function()
def train_step(image, style_targets, content_targets, extractor, optimizer, 
               style_weight, content_weight, total_variation_weight):
    """Update image for one optimization step"""
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets, 
                                  style_weight, content_weight)
        
        # Add total variation loss to reduce high frequency artifacts
        loss += total_variation_weight * tf.image.total_variation(image)
    
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    
    return loss

def neural_style_transfer(content_path, style_path, epochs=10, steps_per_epoch=100,
                         style_weight=1e-2, content_weight=1e4, total_variation_weight=30,
                         save_path=None):
    """Main function to perform neural style transfer"""
    # Load images
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    
    # Display input images
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    imshow(content_image, 'Content Image')
    plt.subplot(1, 2, 2)
    imshow(style_image, 'Style Image')
    plt.show()
    
    # Set up model
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    
    extractor = StyleContentModel(style_layers, content_layers)
    
    # Get style and content targets
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    
    # Initialize image with content image
    image = tf.Variable(content_image)
    
    # Set up optimizer
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    
    # Training loop
    start = time.time()
    step = 0
    
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            loss = train_step(image, style_targets, content_targets, extractor, opt,
                              style_weight, content_weight, total_variation_weight)
            
            if step % 50 == 0:
                print(f"Epoch {n+1}/{epochs}, Step {m+1}/{steps_per_epoch}")
                print(f"Loss: {loss:.4f}")
def image_to_ascii(image, width=80):
grayscale_chars = "@%#*+=-:. "  # 10 characters from dark to light
    image = image.convert("L")  # Convert to grayscale
    # Show ASCII version in terminal
ascii_output = image_to_ascii(result_img)
print("\nStylized Image (ASCII Preview):\n")
print(ascii_output)

    # Resize keeping aspect ratio (height is stretched slightly in terminals)
    aspect_ratio = image.height / image.width
    new_height = int(aspect_ratio * width * 0.55)  # Adjust for terminal font ratio
    image = image.resize((width, new_height))
    
    pixels = np.array(image)
    ascii_image = ""
    for row in pixels:
        ascii_image += "".join(grayscale_chars[pixel // 25] for pixel in row) + "\n"
    
    return ascii_image

    end = time.time()
    print(f"Total time: {end-start:.1f} seconds")
    
    # Create output
    result_img = tensor_to_image(image)
    plt.figure(figsize=(12, 12))
    plt.imshow(result_img)
    plt.axis('off')
    plt.show()
    
    # Save if path provided
    if save_path:
        result_img.save(save_path)
        print(f"Image saved to {save_path}")
    
    return result_img

# Example usage
if __name__ == "__main__":
    # Replace with your actual image paths
    content_path = 'path/to/content_image.jpg'
    style_path = 'path/to/style_image.jpg'
    
    result = neural_style_transfer(
        content_path, 
        style_path,
        epochs=10,
        steps_per_epoch=100,
        save_path='stylized_image.jpg'
    )