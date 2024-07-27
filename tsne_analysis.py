import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet101
import torchvision
import sys
import math

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 tsne_analysis.py <n_iter> <n_components> <image_dir> <perplexity>")
        sys.exit(0)
    
    n_iter = int(sys.argv[1])
    n_components = int(sys.argv[2])
    
    print(f't-SNE Iterations: {sys.argv[1]}')
    print(f't-SNE Dims: {sys.argv[2]}')

    image_dir = sys.argv[3]
    number_of_perplexity = 1

    if len(sys.argv) > 4:
        print(f'Setting Perplexity Number to {sys.argv[4]}')
        number_of_perplexity = int(math.ceil(float(sys.argv[4])))
    else:
        print(f'Perplexity is not specified. Using {number_of_perplexity} as default')
    
    image_size = (224, 224)
    class_named_dirs = os.listdir(image_dir)
    images = []
    labels = []

    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract class names from the train directory structure
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

    x_ = [5]
    for i in range(1, number_of_perplexity):
        m = x_[0] + (5 * i)
        x_.append(m)
    print('Perplexity values to be tested: ', x_)

    for i in x_:
        print('Performing t-SNE analysis...     \n')
        try:
            print(f'Working with perplexity of {i}.')
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=i, n_iter=n_iter, learning_rate='auto', init='random')
            X_tsne = tsne.fit_transform(X)

            print('Output of t-SNE: ', X_tsne)
            print('Output shape of t-SNE: ', X_tsne.shape)

            def scale(x):
                range = (np.max(x) - np.min(x))
                start_from_0 = x - np.min(x)
                return start_from_0 / range

            tx = X_tsne[:, 0]
            ty = X_tsne[:, 1]
            tz = X_tsne[:, 2] if n_components == 3 else None

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
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.title('2D t-SNE Visualization of Dataset')
                print('Showing... \n')
                if n_components == 2:
                    print('SAVING 2D PLOT...')
                    plt.savefig(f'./tsne_P{i}_2D.png')
            except Exception as e:
                print('Cannot do 2D t-SNE. Skipping...')
                print("An error occurred:", e)
            try:
                if tz is not None and tz.size > 0:
                    fig = plt.figure(figsize=(20, 15))
                    ax = fig.add_subplot(projection='3d')
                    for type_, color in color_map.items():
                        indices = [i for i, l in enumerate(labels) if l == type_]
                        ax.scatter(tx[indices], ty[indices], tz[indices], color=[color], label=classes_[type_])
                    ax.legend()
                    ax.set_xlabel('t-SNE Component 1')
                    ax.set_ylabel('t-SNE Component 2')
                    ax.set_zlabel('t-SNE Component 3')
                    ax.set_title('3D t-SNE Visualization of Dataset')
                    print('Showing... \n')
                    plt.savefig(f'./tsne_P{i}_3D.png')
                else:
                    print("tz is None or empty. Skipping 3D plot...")
            except Exception as e:
                print('Cannot do 3D t-SNE. Skipping...')
                print("An error occurred #1:", e)
        except Exception as e:
            print(f'Does not work with perplexity = {i}.')
            print("An error occurred #2:", e)