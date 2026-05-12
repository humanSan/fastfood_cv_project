import os
from datasets import load_dataset
from PIL import Image
import numpy as np

def download_and_save_foodseg103(output_dir="data/FoodSeg103"):
    """
    Downloads the FoodSeg103 dataset from Hugging Face and saves it locally.
    """
    print(f"Downloading FoodSeg103 from HuggingFace to {output_dir}...")
    
    # Load dataset
    dataset = load_dataset("EduardoPacheco/FoodSeg103")
    
    # Create directories
    for split in dataset.keys():
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "annotations"), exist_ok=True)
        
        print(f"Processing {split} split...")
        for i, item in enumerate(dataset[split]):
            image = item["image"]
            label = item["label"]
            
            # Use original filename if available, otherwise generate one
            filename = f"{i:08d}.jpg"
            mask_filename = f"{i:08d}.png"
            
            image_path = os.path.join(output_dir, split, "images", filename)
            label_path = os.path.join(output_dir, split, "annotations", mask_filename)
            
            # Save image
            if not os.path.exists(image_path):
                image.save(image_path)
                
            # Save label
            if not os.path.exists(label_path):
                label.save(label_path)
                
    print("Download and extraction complete.")

if __name__ == "__main__":
    download_and_save_foodseg103()
