from tensorflow.keras.preprocessing.image import ImageDataGenerator


def image_dataset_from_directory(dataset_path, image_size, shuffle=True):
    image_generator = ImageDataGenerator()
    dataset = image_generator.flow_from_directory(dataset_path, target_size=image_size, shuffle=shuffle)
    return dataset
