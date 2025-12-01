import subprocess
import sys

def detect_cuda_version():
    """Засича версията на драйвера от nvidia-smi."""
    try:
        output = subprocess.check_output(["nvidia-smi"], text=True)
        if "CUDA Version:" in output:
            part = output.split("CUDA Version:")[1]
            version = part.split("|")[0].strip()
            clean_version = version.replace(".", "") 
            return clean_version
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    print("⚠️ CUDA not detected. Falling back to CPU.")
    return None

def get_index_url(cuda_version):
    """
    ВАЖНО: Дори драйверът да е 13.0, PyTorch все още няма cu130 wheels.
    Използваме cu126 (или cu124), които са съвместими с по-нови драйвери.
    """
    if not cuda_version:
        return "https://download.pytorch.org/whl/cpu"

    print(f"Detected Driver CUDA: {cuda_version}")

    if cuda_version.startswith("13") or cuda_version == "128" or cuda_version == "126": 
        # Насочваме всичко ново към 12.6 Nightly (най-стабилното ново)
        print("ℹ️ Mapping driver to PyTorch CUDA 12.6 wheels (Best Match)")
        return "https://download.pytorch.org/whl/nightly/cu126"
    
    elif cuda_version == "124":
        return "https://download.pytorch.org/whl/cu124"
    elif cuda_version == "121":
        return "https://download.pytorch.org/whl/cu121"
    elif cuda_version == "118":
        return "https://download.pytorch.org/whl/cu118"
    else:
        print(f"⚠️ Unmapped version {cuda_version}. Using CPU.")
        return "https://download.pytorch.org/whl/cpu"

def install_torch(index_url):
    print(f"⚙️ Force-Installing PyTorch from: {index_url}")
    
    is_nightly = "nightly" in index_url
    flags = ["--pre"] if is_nightly else []

    try:
        # ВАЖНО: --reinstall-package кара uv да изтрие старата CPU версия
        # и да изтегли наново правилната GPU версия.
        cmd = [
            "uv", "pip", "install", 
            "torch", "torchvision", "torchaudio",
            "--index-url", index_url,
            "--reinstall-package", "torch",
            "--reinstall-package", "torchvision",
            "--reinstall-package", "torchaudio"
        ] + flags
        
        print(f"Running command to force GPU install...")
        subprocess.run(cmd, check=True)
        print("✅ PyTorch GPU installation complete.")
        
    except subprocess.CalledProcessError as e:
        print("❌ Installation failed.")
        print(e)

if __name__ == "__main__":
    cuda = detect_cuda_version()
    print(f"🔍 System Check: {cuda if cuda else 'CPU Mode'}")
    
    url = get_index_url(cuda)
    install_torch(url)