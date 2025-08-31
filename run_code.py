import subprocess

if __name__ == "__main__":
    subprocess.run(["python3", "ddp_fine_1x.py"])
    subprocess.run(["python3", "inference_pseudo.py"])
    
    subprocess.run(["python3", "pseudo_update1x.py", 
        "--config", "configs/labellers/AttnUNet5/pseudo_update1.json",
        "--model_path", "output/Labeller/AttnUNet5-Pass1/model.pth"
    ])
    subprocess.run(["python3", "data_prep/resize_pseudo.py"])


    subprocess.run(["python3", "ddp_fine_small.py"])
    subprocess.run(["python3", "inference.py"])