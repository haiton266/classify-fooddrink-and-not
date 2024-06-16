import os
from prettytable import PrettyTable


def count_images(directory):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            count += 1
    return count


def analyze_dataset(base_path):
    subdirs = ['train/valid', 'train/notvalid', 'val/valid',
               'val/notvalid']
    results = {}

    for subdir in subdirs:
        path = os.path.join(base_path, subdir)
        if os.path.exists(path):
            image_count = count_images(path)
            results[subdir] = image_count
        else:
            results[subdir] = 'Directory not found'

    return results


if __name__ == "__main__":
    base_path = 'data'
    results = analyze_dataset(base_path)

    # Create a PrettyTable
    table = PrettyTable()
    table.field_names = ["Subdirectory", "Image Count"]
    for subdir, count in results.items():
        table.add_row([subdir, count])

    print(table)
