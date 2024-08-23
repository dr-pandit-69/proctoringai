import os
import cv2
from app.ai_processing import process_frame

class ImageProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def process_images(self):
        for filename in os.listdir(self.input_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_path = os.path.join(self.input_folder, filename)
                output_path = os.path.join(self.output_folder, filename)

     
                image = cv2.imread(input_path)
                
              
                processed_image = process_frame(image)
                
               
                cv2.imwrite(output_path, processed_image)
                print(f'Processed and saved {filename} to {self.output_folder}')
