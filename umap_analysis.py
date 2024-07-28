import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet101
import torchvision
import sys
import umap
import math

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 umap_analysis.py <n_components> <image_dir> <n_neighbors>")
        sys.exit(0)
    
    n_components = int(sys.argv[1])
    
    print(f'UMAP Dims: {sys.argv[1]}')

    image_dir = sys.argv[2]
    n_neighbors = 15

    if len(sys.argv) > 3:
        print(f'Setting number of neighbors to {sys.argv[3]}')
        n_neighbors = int(sys.argv[3])
    else:
        print(f'Number of neighbors is not specified. Using {n_neighbors} as default')
    
    image_size = (224, 224)
    class_named_dirs = os.listdir(image_dir)
    images = []
    labels = []

    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    classes_ = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    print(f'Classes: {classes_}')

    use_mps = torch.backends.mps.is_available()

    if use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    resnet = resnet101(weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    resnet.to(device)

    count = 0
    for class_ in classes_:
        class_path = os.path.join(image_dir, class_)

        for image_name in os.listdir(class_path):
            print(f'Processing image: {image_name}')
            image_path = os.path.join(class_path, image_name)
            if image_name == '.DS_Store' or not os.path.isfile(image_path):
                continue

            img = Image.open(image_path).convert('RGB')
            img_tensor = preprocess(img)
            img_tensor = img_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                features = resnet(img_tensor)
            
            features = features.squeeze().cpu().numpy()
            print(f'Feature Shape: {features.shape} \n')
            images.append(features)
            labels.append(classes_.index(class_))
            count += 1

    count_ = 0
    each_ = []
    for root_dir, cur_dir, files in os.walk(image_dir):
        cur_dir.sort()
        each = len(files)
        
        if each != 0:
            each_.append(each)
        
        count_ += each
    
    print(f'File count: {count_}')
    print(f'Each Directory: {each_}')

    import matplotlib.cm as cm
    unique_class = list(set(labels))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_class)))
    color_map = dict(zip(unique_class, colors))
    print(f'Color Map: {color_map}')
    X = np.vstack(images)
    print('X Shape: ', X.shape, '\n')
    labels_ = np.array(labels)
    print('Label Shape: ', labels_.shape, '\n')
    print('Number of instances: ', count, '\n')

    print('Performing UMAP analysis...     \n')
    try:
        print(f'Working with {n_neighbors} neighbors.')
        umap_model = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=n_neighbors)
        X_umap = umap_model.fit_transform(X)

        print('Output of UMAP: ', X_umap)
        print('Output shape of UMAP: ', X_umap.shape)

        def scale(x):
            range = (np.max(x) - np.min(x))
            start_from_0 = x - np.min(x)
            return start_from_0 / range

        tx = X_umap[:, 0]
        ty = X_umap[:, 1]
        tz = X_umap[:, 2] if n_components == 3 else None

        tx = scale(tx)
        ty = scale(ty)
        if tz is not None:
            tz = scale(tz)

        print('TX shape: ', tx.shape)
        print('TY shape: ', ty.shape)

        if tz is not None:
            print('TZ shape: ', tz.shape)

        print('Getting ready to output results... \n')
        try:
            plt.figure(figsize=(20, 15))
            for type_, color in color_map.items():
                indices = [i for i, l in enumerate(labels) if l == type_]
                if type_ < len(classes_):
                    plt.scatter(tx[indices], ty[indices], color=[color], label=classes_[type_])
                else:
                    print(f"Index {type_} is out of range for classes_ list")
            plt.legend()
            plt.xlabel('UMAP Component 1')
            plt.ylabel('UMAP Component 2')
            plt.title('2D UMAP Visualization of Dataset')
            print('Showing... \n')
            if n_components == 2:
                print('SAVING 2D PLOT...')
                plt.savefig(f'./umap_{n_neighbors}_2D.png')
        except Exception as e:
            print('Cannot do 2D UMAP. Skipping...')
            print("An error occurred:", e)
        try:
            if tz is not None and tz.size > 0:
                fig = plt.figure(figsize=(20, 15))
                ax = fig.add_subplot(projection='3d')
                for type_, color in color_map.items():
                    indices = [i for i, l in enumerate(labels) if l == type_]
                    ax.scatter(tx[indices], ty[indices], tz[indices], color=[color], label=classes_[type_])
                ax.legend()
                ax.set_xlabel('UMAP Component 1')
                ax.set_ylabel('UMAP Component 2')
                ax.set_zlabel('UMAP Component 3')
                ax.set_title('3D UMAP Visualization of Dataset')
                print('Showing... \n')
                plt.savefig(f'./umap_{n_neighbors}_3D.png')
            else:
                print("tz is None or empty. Skipping 3D plot...")
        except Exception as e:
            print('Cannot do 3D UMAP. Skipping...')
            print("An error occurred #1:", e)
    except Exception as e:
        print(f'Does not work with {n_neighbors} neighbors.')
        print("An error occurred #2:", e)