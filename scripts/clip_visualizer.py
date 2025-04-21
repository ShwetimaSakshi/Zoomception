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
import CLIP.clip as clip
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from captum.attr import visualization
import cv2
clip.clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}
start_layer =  -1

start_layer_text =  -1
class CLIPVisualizer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", attention_model_name: str = "ViT-B/32"):
        """Initialize CLIP models - one for embeddings and one for attention visualization."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model for embeddings
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Load model for attention visualization
        self.attention_model_name = attention_model_name
        self.attention_model, self.preprocess = clip.load(attention_model_name, device=self.device, jit=False)
        self.tokenizer = _Tokenizer()
        
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
        
    def load_sample_data(self, dataset_name: str = "cifar100", num_samples: int = 25, data_dir: str = "../data") -> Tuple[List[Image.Image], List[str]]:
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
        
        # Create output directory
        output_dir = Path(save_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
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

    def interpret(self, image, texts, model, device, start_layer=start_layer, start_layer_text=start_layer_text):
        batch_size = texts.shape[0]
        images = image.repeat(batch_size, 1, 1, 1)
        logits_per_image, logits_per_text = model(images, texts)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        index = [i for i in range(batch_size)]
        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)
        model.zero_grad()

        image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

        if start_layer == -1: 
            # calculate index of last layer 
            start_layer = len(image_attn_blocks) - 1
        
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(image_attn_blocks):
            if i < start_layer:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        image_relevance = R[:, 0, 1:]

        text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

        if start_layer_text == -1: 
            # calculate index of last layer 
            start_layer_text = len(text_attn_blocks) - 1

        num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
        R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
        R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(text_attn_blocks):
            if i < start_layer_text:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R_text = R_text + torch.bmm(cam, R_text)
        text_relevance = R_text
   
        return text_relevance, image_relevance

    def show_heatmap_on_text(self, text, text_encoding, R_text):
        CLS_idx = text_encoding.argmax(dim=-1)
        R_text = R_text[CLS_idx, 1:CLS_idx]
        text_scores = R_text / R_text.sum()
        text_scores = text_scores.flatten()
        
        text_tokens = self.tokenizer.encode(text)
        text_tokens_decoded = [self.tokenizer.decode([a]) for a in text_tokens]
        vis_data_records = [visualization.VisualizationDataRecord(
            text_scores,
            0,  # pred_prob
            0,  # pred_class
            0,  # true_class
            0,  # attr_class
            0,  # delta_class
            text_tokens_decoded,
            1   # attr_score
        )]
        
        return visualization.visualize_text(vis_data_records)
        
    def show_image_relevance(self, image_relevance, image, orig_image):
        # Create heatmap from mask on image
        def show_cam_on_image(img, mask):
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)
            return cam

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(orig_image)
        axs[0].axis('off')

        dim = int(image_relevance.numel() ** 0.5)
        image_relevance = image_relevance.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
        image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        image = image[0].permute(1, 2, 0).data.cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        vis = show_cam_on_image(image, image_relevance)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        axs[1].imshow(vis)
        axs[1].axis('off')
        return fig
    
    def visualize_attention_for_embeddings(self, images: List[Image.Image], texts: List[str], output_dir: str = "./outputs"):
        """Generate attention maps for all combinations of images and texts."""
        output_dir = Path(output_dir)
        attention_dir = output_dir / "attention_maps"
        attention_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results dictionary
        results = {
            "attention_maps": []
        }

        # Process each image with every text
        total_combinations = len(images) * len(texts)
        current = 0

        for img_idx, image in enumerate(images):
            # Preprocess image once
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            img_base64 = self.image_to_base64(image)

            for text_idx, text in enumerate(texts):
                try:
                    current += 1
                    print(f"Processing combination {current}/{total_combinations}")

                    # Process text
                    text_tokens = clip.tokenize([text]).to(self.device)
                    
                    # Get attention maps
                    R_text, R_image = self.interpret(img_tensor, text_tokens, self.attention_model, self.device)
                    
                    # Generate attention visualization
                    image_fig = self.show_image_relevance(R_image[0], img_tensor, image)
                    
                    # Save figure to bytes buffer and convert to base64
                    buf = BytesIO()
                    image_fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                    buf.seek(0)
                    attention_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    plt.close('all')

                    # Save to results
                    result_item = {
                        "image_index": img_idx,
                        "text_index": text_idx,
                        "image_base64": img_base64,
                        "text": text,
                        "attention_map_base64": attention_base64
                    }
                    results["attention_maps"].append(result_item)
                    
                except Exception as e:
                    print(f"Error processing image {img_idx} with text {text_idx}: {str(e)}")
                    continue

        # Save results to JSON
        with open(attention_dir / "attention_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nAttention maps saved in: {attention_dir}")
        print(f"Results saved to: {attention_dir}/attention_results.json")

def main():
    parser = argparse.ArgumentParser(description="CLIP Embeddings Visualizer")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--attention_model", default="ViT-B/32", help="CLIP attention model name")
    parser.add_argument("--dataset", choices=["cifar100", "cifar10", "mnist", "coco", "flickr30k"], default="cifar100", help="Dataset to use")
    parser.add_argument("--output", default="./outputs", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to visualize")
    parser.add_argument("--generate_attention", action="store_true", help="Generate attention maps for embeddings")
    
    args = parser.parse_args()
    
    visualizer = CLIPVisualizer(args.model, args.attention_model)
    
    # Generate embeddings and visualizations
    images, texts = visualizer.load_sample_data(args.dataset, args.num_samples)
    image_embeddings, text_embeddings = visualizer.compute_embeddings(images, texts)
    similarity_matrix = visualizer.compute_similarity_matrix(image_embeddings, text_embeddings)
    visualizer.visualize_embeddings_3d(image_embeddings, text_embeddings, args.output + "/embeddings_3d.png")
    visualizer.visualize_similarity_matrix(similarity_matrix, len(images), args.output + "/similarity_heatmap.png")
    visualizer.save_data(image_embeddings, text_embeddings, similarity_matrix, texts, args.output)
    
    # Generate attention maps if requested
    if args.generate_attention:
        visualizer.visualize_attention_for_embeddings(images, texts, args.output)

if __name__ == "__main__":
    main()
