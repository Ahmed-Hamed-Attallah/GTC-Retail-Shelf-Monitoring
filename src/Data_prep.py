import os
import cv2
import numpy as np



def preprocess_images(input_dir, output_dir, target_size=(640, 640)):

    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in os.listdir(input_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            
            # Resize image
            img = cv2.resize(img, target_size)
            
            # Normalize (convert to float and scale to 0-1)
            img = img.astype(np.float32) / 255.0
            
            # Save processed image
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, img * 255)
    
    print(f"Preprocessed images saved to {output_dir}")
