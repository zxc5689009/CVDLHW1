import os

def find_empty_label_files(label_dir):
    empty_files = []
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # 如果文件是空的或只有空行，記錄下來
                    if not lines or all(line.strip() == '' for line in lines):
                        empty_files.append(file_path)
    return empty_files

# 替換為您的標籤目錄路徑
label_directory = '/home/P76121665/AICUP/yolov7/mydata/folds/fold2/valid/labels'
empty_files = find_empty_label_files(label_directory)

if empty_files:
    print("空的標籤文件如下：")
    for file in empty_files:
        print(file)
else:
    print("沒有找到空的標籤文件。")
