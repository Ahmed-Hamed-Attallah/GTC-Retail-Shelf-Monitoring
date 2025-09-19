import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


splits = ['train', 'val', 'test']


# Analyze dataset statistics
def analyze_dataset_stats():
    stats = []
    
    for split in splits:
        image_dir = f'data_preprocessed/images/{split}'
        label_dir = f'data_preprocessed/labels/{split}'
        
        image_count = len([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        object_count = 0
        class_counts = {0: 0, 1: 0}  # 0: product, 1: empty_slot
        
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(label_dir, label_file), 'r') as f:
                    lines = f.readlines()
                    object_count += len(lines)
                    for line in lines:
                        cls_id = int(line.split()[0])
                        class_counts[cls_id] += 1
        
        stats.append({
            'split': split,
            'num_images': image_count,
            'num_objects': object_count,
            'avg_objects_per_image': object_count / image_count if image_count > 0 else 0,
            'products_count': class_counts[0],
            'empty_slots_count': class_counts[1]
        })
    
    # Create a DataFrame for easier analysis
    df_stats = pd.DataFrame(stats)
    print("Dataset Statistics:")
    print(df_stats)
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    x = np.arange(len(stats))
    width = 0.35
    
    plt.bar(x - width/2, [s['products_count'] for s in stats], width, label='Products')
    plt.bar(x + width/2, [s['empty_slots_count'] for s in stats], width, label='Empty Slots')
    
    plt.xlabel('Data Split')
    plt.ylabel('Count')
    plt.title('Class Distribution Across Splits')
    plt.xticks(x, [s['split'] for s in stats])
    plt.legend()
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()
    
    return df_stats



# Visualize samples with bounding boxes
def visualize_samples(num_samples=5):
    for split in splits:
        image_dir = f'data_preprocessed/images/{split}'
        label_dir = f'data_preprocessed/labels/{split}'
        
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        sample_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        for img_file in sample_files:
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            
            # Draw bounding boxes
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    cls_id, x_center, y_center, w, h = map(float, line.split())
                    
                    # Convert YOLO format to pixel coordinates
                    x1 = int((x_center - w/2) * width)
                    y1 = int((y_center - h/2) * height)
                    x2 = int((x_center + w/2) * width)
                    y2 = int((y_center + h/2) * height)
                    
                    # Choose color based on class
                    color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
                    label = "Product" if cls_id == 0 else "Empty Slot"
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f"Sample from {split}: {img_file}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'sample_{split}_{img_file}.png')
            plt.show()




# Calculate shelf health metrics
def analyze_shelf_health_metrics():
    health_metrics = []
    
    for split in splits:
        label_dir = f'data_preprocessed/labels/{split}'
        
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(label_dir, label_file), 'r') as f:
                    lines = f.readlines()
                
                products = 0
                empty_slots = 0
                
                for line in lines:
                    cls_id = int(line.split()[0])
                    if cls_id == 0:
                        products += 1
                    else:
                        empty_slots += 1
                
                total = products + empty_slots
                fill_percentage = (products / total) * 100 if total > 0 else 0
                health_score = fill_percentage - (empty_slots * 5)  # Penalize empty slots
                
                health_metrics.append({
                    'image_id': os.path.splitext(label_file)[0],
                    'split': split,
                    'total_objects': total,
                    'products_count': products,
                    'empty_slots_count': empty_slots,
                    'fill_percentage': fill_percentage,
                    'shelf_health_score': health_score
                })
    
    df_health = pd.DataFrame(health_metrics)
    print("Shelf Health Metrics Summary:")
    print(df_health.describe())
    
    # Plot distribution of fill percentage
    plt.figure(figsize=(10, 6))
    plt.hist(df_health['fill_percentage'], bins=20, edgecolor='black')
    plt.title('Distribution of Shelf Fill Percentage')
    plt.xlabel('Fill Percentage')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('fill_percentage_distribution.png')
    plt.show()
    
    return df_health
    