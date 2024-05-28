import os

def save_files(df, filename, output_dir_app, output_dir_main):
    output_path_app = os.path.join(output_dir_app, filename)
    output_path_main = os.path.join(output_dir_main, filename)
    df.to_csv(output_path_app, index=False)
    df.to_csv(output_path_main, index=False)

def create_directories():
    upload_dir = 'uploads'
    output_dir_app = 'app/outputs'
    output_dir_main = 'outputs'
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir_app, exist_ok=True)
    os.makedirs(output_dir_main, exist_ok=True)
    return upload_dir, output_dir_app, output_dir_main
