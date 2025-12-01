import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import os
import random
import pandas as pd
from datetime import datetime
import re

# Configuration
DATASET_ROOTS = ['dataset', 'dataset_variants']
IMAGES_PER_SESSION = 80
RESULTS_DIR = 'human_results'

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Classification Experiment")
        
        # Data loading
        self.all_images = self.load_image_list()
        if len(self.all_images) < IMAGES_PER_SESSION:
            messagebox.showerror("Error", f"Not enough images found! Found {len(self.all_images)}, need {IMAGES_PER_SESSION}")
            self.root.destroy()
            return
            
        self.session_images = random.sample(self.all_images, IMAGES_PER_SESSION)
        self.current_index = 0
        self.results = []
        
        # User ID
        self.user_id = simpledialog.askstring("Input", "Please enter your Name/ID (请输入你的名字或ID):", parent=self.root)
        if not self.user_id:
            self.root.destroy()
            return
            
        # UI Setup
        self.label_info = tk.Label(root, text=f"Image 1/{IMAGES_PER_SESSION}", font=("Arial", 16))
        self.label_info.pack(pady=10)
        
        self.label_instruction = tk.Label(root, text="Left Arrow (←): Persian Cat (波斯猫)  |  Right Arrow (→): Siamese Cat (暹罗猫)", font=("Arial", 12))
        self.label_instruction.pack(pady=5)
        
        self.image_label = tk.Label(root)
        self.image_label.pack(padx=20, pady=20)
        
        # Bindings
        self.root.bind('<Left>', lambda e: self.record_response('Persian cat'))
        self.root.bind('<Right>', lambda e: self.record_response('Siamese cat'))
        
        # Start
        self.show_current_image()
        
    def load_image_list(self):
        image_list = []
        # We look for 'test' folders specifically to match evaluation protocols
        # Structure: root/dataset_name/test/class_name/image.jpg
        
        print("Scanning for images...")
        for root_dir in DATASET_ROOTS:
            if not os.path.exists(root_dir):
                continue
                
            # Walk through directories
            for dirpath, dirnames, filenames in os.walk(root_dir):
                # Filter for 'test' set only
                # Normalize path separators
                path_parts = dirpath.replace('\\', '/').split('/')
                
                if 'test' not in [p.lower() for p in path_parts]:
                    continue
                
                # Determine class based on folder name
                # Assuming folder names contain 'Persian' or 'Siamese' (case insensitive)
                class_label = None
                lower_path = dirpath.lower()
                if 'persian' in lower_path:
                    class_label = 'Persian cat'
                elif 'siamese' in lower_path:
                    class_label = 'Siamese cat'
                
                if not class_label:
                    continue
                    
                # Determine Dataset Type and Level
                # dataset -> Original, Level 0
                # dataset_variants/contrast_level_X -> Contrast, Level X
                # dataset_variants/noise_level_X -> Noise, Level X
                
                dataset_type = 'Original'
                level = 0
                
                # Check for variants in the path
                for part in path_parts:
                    if 'contrast_level_' in part:
                        dataset_type = 'Contrast'
                        match = re.search(r'contrast_level_(\d+)', part)
                        if match:
                            level = int(match.group(1))
                    elif 'noise_level_' in part:
                        dataset_type = 'Noise'
                        match = re.search(r'noise_level_(\d+)', part)
                        if match:
                            level = int(match.group(1))
                
                for f in filenames:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(dirpath, f)
                        image_list.append({
                            'path': full_path,
                            'ground_truth': class_label,
                            'dataset_type': dataset_type,
                            'level': level
                        })
        print(f"Found {len(image_list)} images.")
        return image_list

    def show_current_image(self):
        if self.current_index >= len(self.session_images):
            self.finish_session()
            return
            
        img_data = self.session_images[self.current_index]
        self.label_info.config(text=f"Image {self.current_index + 1}/{IMAGES_PER_SESSION}")
        
        # Load and resize image
        try:
            pil_image = Image.open(img_data['path'])
            # Resize for display if too large, keep aspect ratio
            pil_image.thumbnail((500, 500))
            self.tk_image = ImageTk.PhotoImage(pil_image)
            self.image_label.config(image=self.tk_image)
        except Exception as e:
            print(f"Error loading image: {e}")
            self.current_index += 1
            self.show_current_image()

    def record_response(self, prediction):
        if self.current_index >= len(self.session_images):
            return

        img_data = self.session_images[self.current_index]
        is_correct = (prediction == img_data['ground_truth'])
        
        self.results.append({
            'User_ID': self.user_id,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Image_Path': img_data['path'],
            'Dataset_Type': img_data['dataset_type'],
            'Level': img_data['level'],
            'Ground_Truth': img_data['ground_truth'],
            'User_Prediction': prediction,
            'Is_Correct': is_correct
        })
        
        self.current_index += 1
        self.show_current_image()

    def finish_session(self):
        # Save results
        df = pd.DataFrame(self.results)
        
        # Ensure results directory exists
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            
        # Find unique filename
        base_name = "human_experiment_results"
        ext = ".csv"
        counter = 0
        
        while True:
            if counter == 0:
                filename = f"{base_name}{ext}"
            else:
                filename = f"{base_name}_{counter}{ext}"
            
            full_path = os.path.join(RESULTS_DIR, filename)
            if not os.path.exists(full_path):
                break
            counter += 1
        
        # Save to new file
        df.to_csv(full_path, mode='w', header=True, index=False, encoding='utf-8-sig')
            
        messagebox.showinfo("Finished", f"Session Complete! Results saved to {full_path}")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    # Center window
    window_width = 800
    window_height = 700
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    app = ImageClassifierApp(root)
    root.mainloop()
