import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.manifold import TSNE
import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple
import torchvision.datasets as datasets
import base64
from io import BytesIO
import requests
from pycocotools.coco import COCO
import random

class CLIPVisualizer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize CLIP model and processor."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def download_image(self, url: str) -> Image.Image:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return None

    def load_coco_data(self, data_dir: str, num_samples: int) -> Tuple[List[Image.Image], List[str], List[Dict]]:
        """Load sample image-text pairs from COCO dataset."""
        # Initialize COCO api for instance annotations
        annFile = os.path.join(data_dir, 'annotations', 'captions_val2014.json')
        coco = COCO(annFile)

        # Get all image ids and randomly sample
        img_ids = list(coco.imgs.keys())
        selected_ids = random.sample(img_ids, num_samples)

        images = []
        texts = []
        image_info = []

        for img_id in selected_ids:
            # Get image info
            img_info = coco.imgs[img_id]
            img_url = f"http://images.cocodataset.org/val2014/{img_info['file_name']}"
            
            # Get caption
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            caption = anns[0]['caption']  # Take first caption
            
            # Download and process image
            img = self.download_image(img_url)
            if img is not None:
                images.append(img)
                texts.append(caption)
                image_info.append({
                    "dataset_index": int(img_id),
                    "label": caption,
                    "base64": self.image_to_base64(img),
                    "url": img_url
                })

        return images, texts, image_info

    def load_flickr_data(self, data_dir: str, num_samples: int) -> Tuple[List[Image.Image], List[str], List[Dict]]:
        """Load sample image-text pairs from Flickr30k dataset."""
        # Load Flickr30k annotations
        with open(os.path.join(data_dir, 'flickr30k', 'annotations.json'), 'r') as f:
            annotations = json.load(f)

        # Randomly sample images
        selected_items = random.sample(list(annotations.items()), num_samples)

        images = []
        texts = []
        image_info = []

        for img_name, captions in selected_items:
            img_path = os.path.join(data_dir, 'flickr30k', 'images', img_name)
            
            try:
                img = Image.open(img_path).convert('RGB')
                caption = captions[0]  # Take first caption
                
                images.append(img)
                texts.append(caption)
                image_info.append({
                    "dataset_index": img_name,
                    "label": caption,
                    "base64": self.image_to_base64(img),
                    "path": img_path
                })
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

        return images, texts, image_info
        
    def load_sample_data(self, dataset_name: str = "cifar100", num_samples: int = 25, data_dir: str = "./data") -> Tuple[List[Image.Image], List[str]]:
        """Load sample image-text pairs from specified dataset."""
        if dataset_name.lower() == "cifar100":
            dataset = datasets.CIFAR100(root=data_dir, train=True, download=True)
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            
            images = []
            texts = []
            image_info = []
            
            for idx in indices:
                img, label = dataset[idx]
                images.append(img)
                text = dataset.classes[label]
                texts.append(text)
                image_info.append({
                    "dataset_index": int(idx),
                    "label": text,
                    "base64": self.image_to_base64(img)
                })

        elif dataset_name.lower() == "cifar10":
            dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            
            images = []
            texts = []
            image_info = []
            
            for idx in indices:
                img, label = dataset[idx]
                images.append(img)
                text = dataset.classes[label]
                texts.append(text)
                image_info.append({
                    "dataset_index": int(idx),
                    "label": text,
                    "base64": self.image_to_base64(img)
                })

        elif dataset_name.lower() == "mnist":
            dataset = datasets.MNIST(root=data_dir, train=True, download=True)
            dataset.transform = lambda x: x.convert('RGB')
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            
            images = []
            texts = []
            image_info = []
            
            for idx in indices:
                img, label = dataset[idx]
                images.append(img)
                text = f"digit {label}"
                texts.append(text)
                image_info.append({
                    "dataset_index": int(idx),
                    "label": text,
                    "base64": self.image_to_base64(img)
                })

        elif dataset_name.lower() == "coco":
            images, texts, image_info = self.load_coco_data(data_dir, num_samples)

        elif dataset_name.lower() == "flickr30k":
            images, texts, image_info = self.load_flickr_data(data_dir, num_samples)

        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from: cifar100, cifar10, mnist, coco, flickr30k")
        
        self.image_info = image_info
        return images, texts
    
    def compute_embeddings(self, images: List[Image.Image], texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CLIP embeddings for images and texts."""
        # Process images
        image_inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        image_embeddings = self.model.get_image_features(**image_inputs)
        
        # Process texts
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        text_embeddings = self.model.get_text_features(**text_inputs)
        
        return image_embeddings.cpu(), text_embeddings.cpu()
    
    def compute_similarity_matrix(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> np.ndarray:
        """Compute cosine similarity matrix between all pairs."""
        # Normalize embeddings
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # Combine embeddings
        all_embeddings = torch.cat([image_embeddings, text_embeddings], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T)
        return similarity_matrix.detach().numpy()
    
    def visualize_embeddings_2d(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, save_path: str):
        """Create 2D visualization of the shared embedding space."""
        combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=0)
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(combined_embeddings.detach())
        
        n_images = image_embeddings.shape[0]
        
        plt.figure(figsize=(10, 10))
        plt.scatter(embeddings_2d[:n_images, 0], embeddings_2d[:n_images, 1], c='blue', label='Images')
        plt.scatter(embeddings_2d[n_images:, 0], embeddings_2d[n_images:, 1], c='red', label='Text')
        plt.legend()
        plt.title('2D CLIP Embeddings Visualization')
        plt.savefig(save_path)
        plt.close()
    
    def visualize_embeddings_3d(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, save_path: str):
        """Create 3D visualization of the shared embedding space."""
        combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=0)
        tsne = TSNE(n_components=3, random_state=42)
        embeddings_3d = tsne.fit_transform(combined_embeddings.detach())
        
        n_images = image_embeddings.shape[0]
        
        # Save 3D coordinates with metadata
        coordinates_data = {
            "images": [],
            "texts": []
        }
        
        # Save image coordinates and metadata
        for i in range(n_images):
            coordinates_data["images"].append({
                "coordinates": embeddings_3d[i].tolist(),
                "metadata": self.image_info[i]
            })
        
        # Save text coordinates and metadata
        for i in range(n_images):
            coordinates_data["texts"].append({
                "coordinates": embeddings_3d[i + n_images].tolist(),
                "text": self.image_info[i]["label"]
            })
        
        # Save coordinates to JSON
        coordinates_path = str(Path(save_path).parent / "embeddings_3d_coordinates.json")
        with open(coordinates_path, "w") as f:
            json.dump(coordinates_data, f, indent=2)
        
        # Create visualization
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(embeddings_3d[:n_images, 0], embeddings_3d[:n_images, 1], embeddings_3d[:n_images, 2], 
                  c='blue', label='Images')
        ax.scatter(embeddings_3d[n_images:, 0], embeddings_3d[n_images:, 1], embeddings_3d[n_images:, 2], 
                  c='red', label='Text')
        
        ax.legend()
        plt.title('3D CLIP Embeddings Visualization')
        plt.savefig(save_path)
        plt.close()
    
    def visualize_similarity_matrix(self, similarity_matrix: np.ndarray, n_images: int, save_path: str):
        """Create heatmap visualization of the similarity matrix."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, cmap='viridis', vmin=-1, vmax=1, center=0)
        
        # Add lines to separate image and text sections
        plt.axhline(y=n_images, color='r', linestyle='-')
        plt.axvline(x=n_images, color='r', linestyle='-')
        
        plt.title('Cosine Similarity Matrix')
        plt.savefig(save_path)
        plt.close()
    
    def save_data(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, 
                  similarity_matrix: np.ndarray, texts: List[str], save_dir: str):
        """Save embeddings and similarity matrix with metadata."""
        data_dir = Path(save_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings with image information
        embeddings_data = {
            "image_embeddings": image_embeddings.detach().numpy().tolist(),
            "text_embeddings": text_embeddings.detach().numpy().tolist(),
            "texts": texts,
            "image_info": self.image_info  # Add image information including base64 and dataset index
        }
        
        with open(data_dir / "embeddings.json", "w") as f:
            json.dump(embeddings_data, f)
        
        # Save similarity matrix with index mapping
        n_images = len(texts)
        index_mapping = {
            "image_indices": list(range(n_images)),
            "text_indices": list(range(n_images, 2*n_images)),
            "texts": texts,
            "image_info": self.image_info  # Add image information here as well
        }
        
        np.save(data_dir / "similarity_matrix.npy", similarity_matrix)
        with open(data_dir / "index_mapping.json", "w") as f:
            json.dump(index_mapping, f)

def main():
    parser = argparse.ArgumentParser(description="CLIP Embeddings Visualizer")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--output", default="./outputs", help="Output directory")
    parser.add_argument("--vis", nargs="+", default=["2d", "3d", "heatmap"], 
                        help="Visualizations to generate (2d, 3d, heatmap)")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "cifar10", "mnist", "coco", "flickr30k"],
                        help="Dataset to use for image-text pairs")
    parser.add_argument("--num-samples", type=int, default=25,
                        help="Number of image-text pairs to sample")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = CLIPVisualizer(args.model)
    
    # Load sample data
    images, texts = visualizer.load_sample_data(args.dataset, args.num_samples)
    
    # Compute embeddings
    image_embeddings, text_embeddings = visualizer.compute_embeddings(images, texts)
    
    # Compute similarity matrix
    similarity_matrix = visualizer.compute_similarity_matrix(image_embeddings, text_embeddings)
    
    # Generate visualizations
    if "2d" in args.vis:
        visualizer.visualize_embeddings_2d(image_embeddings, text_embeddings, 
                                         output_dir / "embeddings_2d.png")
    
    if "3d" in args.vis:
        visualizer.visualize_embeddings_3d(image_embeddings, text_embeddings, 
                                         output_dir / "embeddings_3d.png")
    
    if "heatmap" in args.vis:
        visualizer.visualize_similarity_matrix(similarity_matrix, len(images), 
                                            output_dir / "similarity_heatmap.png")
    
    # Save data
    visualizer.save_data(image_embeddings, text_embeddings, similarity_matrix, texts, args.output)

if __name__ == "__main__":
    main()
