import os
import cv2
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor

class DigitRecognizer:
    def __init__(self, template_dir):
        self.crop_area = (300, 215, 965, 450)  # (left, top, right, bottom)
        self.template_dir = template_dir
        self.results = []
    
    def load_templates(self, file_path):
        templates = np.load(file_path)
        return templates
    
    def preprocess_image(self, image):
        cropped_image = image.crop(self.crop_area)
        cv_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
        image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(image_gray, (3, 3), 1)
        _, image_threshold = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return image_threshold
    
    def split_digits(self, image_data):
        vertical_projection = np.sum(image_data, axis=0)
        threshold = image_data.shape[0] * 255
        digit_positions = []
        start = None
        for i in range(len(vertical_projection)):
            if vertical_projection[i] < threshold and start is None:
                start = i
            elif vertical_projection[i] >= threshold and start is not None:
                digit_positions.append((start, i))
                start = None
        digit_images = []
        for start, end in digit_positions:
            digit_image = image_data[:, start:end]
            digit_images.append(digit_image)
        return digit_images
    
    def resize_image(self, digits):
        width = 163
        height = 235
        resized_digits = []
        for image in digits:
            original_height, original_width = image.shape[:2]
            scale = min(width/original_width, height/original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            offset_x = (width - new_width) // 2
            offset_y = (height - new_height) // 2
            background = np.ones((height, width), dtype=np.uint8) * 255
            resized_image = cv2.resize(image, (new_width, new_height))
            background[offset_y:offset_y+new_height, offset_x:offset_x+new_width] = resized_image
            resized_digits.append(background)
        return resized_digits
    
    def digit_recognition(self, digit_image, templates):
        best_match = None
        best_score = float('inf')
        for i, template in enumerate(templates):
            diff = cv2.absdiff(digit_image, template)
            score = np.sum(diff)
            if score < best_score:
                best_match = i
                best_score = score
                
        return best_match
    
    def draw_bounding_box(self):
        draw = ImageDraw.Draw(self.image)
        left, top, right, bottom = self.crop_area
        draw.rectangle([(left, top), (right, bottom)], outline="green", width=2)
        draw.text((left, top - 50), f"质量: {self.results}", fill="green", font=ImageFont.truetype("./simhei.ttf", 50))
        return self.image
        #mage.show()
    
    def run_recognize(self, image_path):
        start_time = time.time()
        self.image = Image.open(image_path)
        preprocessed_image = self.preprocess_image(self.image)
        digit_images = self.split_digits(preprocessed_image)
        resized_digits = self.resize_image(digit_images)
        templates = self.load_templates(self.template_dir)
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.digit_recognition, digit_image, templates) for digit_image in resized_digits]
            for future in futures:
                result = future.result()
                self.results.append(result)
                
        self.results=round(float(''.join(str(num) for num in self.results)) / 1000, 3)
        # self.draw_bounding_box(image, round(float(''.join(str(num) for num in results)) / 1000, 3))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("结果：{}，耗时：{}s".format(self.results, elapsed_time))
        return self.draw_bounding_box(), self.results

if __name__ == "__main__":
    recognizer = DigitRecognizer(template_dir='./templates.npy')
    image_path = './test.jpg'
    image, results=recognizer.run_recognize(image_path)
    image.show()
