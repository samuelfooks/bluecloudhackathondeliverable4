#!/usr/bin/env python3
"""
PNG Figure Processor Module
===========================

This module handles the processing and integration of PNG figures from the
BlueCloud-Plankton analysis into the visualization pipeline.

Author: BlueCloud Hackathon 2025
"""

import os
import glob
import base64
from PIL import Image
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PNGProcessor:
    def __init__(self, png_dir=None):
        """
        Initialize the PNG Processor

        Args:
            png_dir (str): Directory containing PNG files
        """
        self.png_dir = png_dir or "/home/samwork/Documents/coding/bluecloud-hackathon-2025/BlueCloud-Plankton-master/outputs/figs"
        self.png_files = {}
        self.processed_images = {}
        self.image_metadata = {}

    def discover_png_files(self):
        """Discover PNG files in the directory"""
        print("🖼️ Discovering PNG files...")

        if not os.path.exists(self.png_dir):
            print(f"❌ PNG directory not found: {self.png_dir}")
            return {}

        # Find all PNG files
        png_pattern = os.path.join(self.png_dir, "*.png")
        png_files = glob.glob(png_pattern)

        if not png_files:
            print("⚠️ No PNG files found")
            return {}

        # Categorize PNG files
        for png_file in png_files:
            filename = os.path.basename(png_file)
            name_without_ext = filename.replace('.png', '')

            # Determine category
            if any(species in name_without_ext.lower() for species in
                   ['acartia', 'calanus', 'metridia', 'oithona', 'temora']):
                category = 'plankton_species'
            elif any(env in name_without_ext.lower() for env in
                     ['temperature', 'salinity', 'nitrate', 'phosphate', 'silicate']):
                category = 'environmental'
            else:
                category = 'other'

            self.png_files[name_without_ext] = {
                'path': png_file,
                'category': category,
                'filename': filename
            }

        print(f"✅ Found {len(self.png_files)} PNG files:")
        for name, info in self.png_files.items():
            print(f"  - {name} ({info['category']})")

        return self.png_files

    def process_image(self, image_path: str) -> Dict:
        """Process a single PNG image"""
        try:
            # Load image
            img = Image.open(image_path)

            # Get image properties
            width, height = img.size
            mode = img.mode

            # Convert to base64 for web embedding
            with open(image_path, 'rb') as f:
                img_data = f.read()
                base64_data = base64.b64encode(img_data).decode('utf-8')

            # Create metadata
            metadata = {
                'filename': os.path.basename(image_path),
                'width': width,
                'height': height,
                'mode': mode,
                'size_bytes': len(img_data),
                'base64_data': f"data:image/png;base64,{base64_data}"
            }

            return metadata

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None

    def process_all_images(self):
        """Process all PNG images"""
        print("🖼️ Processing all PNG images...")

        files = self.discover_png_files()
        if not files:
            return {}

        for name, file_info in files.items():
            print(f"📸 Processing: {name}")
            metadata = self.process_image(file_info['path'])

            if metadata:
                self.processed_images[name] = metadata
                self.image_metadata[name] = {
                    **metadata,
                    'category': file_info['category'],
                    'path': file_info['path']
                }

        print(f"✅ Processed {len(self.processed_images)} images")
        return self.processed_images

    def get_plankton_images(self) -> Dict:
        """Get plankton species images"""
        plankton_images = {}
        for name, metadata in self.image_metadata.items():
            if metadata['category'] == 'plankton_species':
                plankton_images[name] = metadata
        return plankton_images

    def get_environmental_images(self) -> Dict:
        """Get environmental variable images"""
        env_images = {}
        for name, metadata in self.image_metadata.items():
            if metadata['category'] == 'environmental':
                env_images[name] = metadata
        return env_images

    def create_image_gallery_data(self) -> Dict:
        """Create data structure for image gallery"""
        gallery_data = {
            'plankton_species': {},
            'environmental': {},
            'all_images': {}
        }

        for name, metadata in self.image_metadata.items():
            category = metadata['category']

            # Create gallery entry
            gallery_entry = {
                'name': name,
                'display_name': name.replace('_', ' ').title(),
                'filename': metadata['filename'],
                'base64_data': metadata['base64_data'],
                'width': metadata['width'],
                'height': metadata['height'],
                'size_bytes': metadata['size_bytes']
            }

            gallery_data[category][name] = gallery_entry
            gallery_data['all_images'][name] = gallery_entry

        return gallery_data

    def create_map_overlay_data(self) -> Dict:
        """Create data structure for map overlays"""
        overlay_data = {
            'plankton_layers': [],
            'environmental_layers': []
        }

        # Plankton species overlays
        plankton_images = self.get_plankton_images()
        for name, metadata in plankton_images.items():
            overlay_data['plankton_layers'].append({
                'name': name,
                'display_name': name.replace('_', ' ').title(),
                'type': 'plankton',
                'base64_data': metadata['base64_data'],
                'filename': metadata['filename']
            })

        # Environmental overlays
        env_images = self.get_environmental_images()
        for name, metadata in env_images.items():
            overlay_data['environmental_layers'].append({
                'name': name,
                'display_name': name.replace('_', ' ').title(),
                'type': 'environmental',
                'base64_data': metadata['base64_data'],
                'filename': metadata['filename']
            })

        return overlay_data

    def create_html_gallery(self, output_path: str = None) -> str:
        """Create HTML gallery of all images"""
        if not output_path:
            output_path = "/home/samwork/Documents/coding/bluecloud-hackathon-2025/deliverable4/data/png_gallery.html"

        gallery_data = self.create_image_gallery_data()

        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BlueCloud Plankton & Environmental Data Gallery</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: #2c3e50;
        }
        .section {
            margin-bottom: 40px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .image-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.3s ease;
        }
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .image-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .image-info {
            padding: 15px;
            background: #f8f9fa;
        }
        .image-info h3 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .image-info p {
            margin: 5px 0;
            color: #7f8c8d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌊 BlueCloud Plankton & Environmental Data Gallery</h1>
        <p>Visualization of plankton species distributions and environmental parameters</p>
    </div>
"""

        # Add plankton species section
        if gallery_data['plankton_species']:
            html_content += """
    <div class="section">
        <h2>🦐 Plankton Species Distributions</h2>
        <div class="image-grid">
"""
            for name, img_data in gallery_data['plankton_species'].items():
                html_content += f"""
            <div class="image-card">
                <img src="{img_data['base64_data']}" alt="{img_data['display_name']}">
                <div class="image-info">
                    <h3>{img_data['display_name']}</h3>
                    <p>File: {img_data['filename']}</p>
                    <p>Size: {img_data['width']} × {img_data['height']} pixels</p>
                </div>
            </div>
"""
            html_content += """
        </div>
    </div>
"""

        # Add environmental section
        if gallery_data['environmental']:
            html_content += """
    <div class="section">
        <h2>🌡️ Environmental Parameters</h2>
        <div class="image-grid">
"""
            for name, img_data in gallery_data['environmental'].items():
                html_content += f"""
            <div class="image-card">
                <img src="{img_data['base64_data']}" alt="{img_data['display_name']}">
                <div class="image-info">
                    <h3>{img_data['display_name']}</h3>
                    <p>File: {img_data['filename']}</p>
                    <p>Size: {img_data['width']} × {img_data['height']} pixels</p>
                </div>
            </div>
"""
            html_content += """
        </div>
    </div>
"""

        html_content += """
</body>
</html>
"""

        # Write HTML file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"✅ HTML gallery created: {output_path}")
        return output_path

    def get_summary(self) -> Dict:
        """Get processing summary"""
        return {
            'total_images': len(self.processed_images),
            'plankton_species': len(self.get_plankton_images()),
            'environmental': len(self.get_environmental_images()),
            'images': self.image_metadata
        }


def main():
    """Test the PNG processor"""
    processor = PNGProcessor()

    # Process all images
    processed_images = processor.process_all_images()

    if processed_images:
        # Create HTML gallery
        gallery_path = processor.create_html_gallery()

        # Get summary
        summary = processor.get_summary()
        print(f"\n📊 PNG Processing Summary:")
        print(f"  Total images: {summary['total_images']}")
        print(f"  Plankton species: {summary['plankton_species']}")
        print(f"  Environmental: {summary['environmental']}")
        print(f"  Gallery created: {gallery_path}")

        print("✅ PNG processor test completed successfully!")
    else:
        print("❌ PNG processor test failed")


if __name__ == "__main__":
    main()
