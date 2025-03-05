import pkg_resources
import sys
import subprocess

def check_dependencies():
    required_packages = {
        'streamlit': '1.32.0',
        'pandas': '2.2.0',
        'numpy': '1.26.3',
        'scikit-learn': '1.4.0',
        'plotly': '5.18.0',
        'ctgan': '0.7.0',
        'sdv': '1.2.0',
        'networkx': '3.2.1',
        'shap': '0.43.0',
        'tensorflow': '2.15.0',
        'seaborn': '0.13.2',
        'matplotlib': '3.8.2'
    }
    
    missing_packages = []
    outdated_packages = []
    
    print("Checking required packages...")
    print("-" * 50)
    
    for package, required_version in required_packages.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(required_version):
                outdated_packages.append(f"{package} (installed: {installed_version}, required: {required_version})")
            else:
                print(f"✓ {package} {installed_version} (required: {required_version})")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
            print(f"✗ {package} not found")
    
    print("-" * 50)
    
    if missing_packages or outdated_packages:
        print("\nSome packages need to be installed or updated:")
        if missing_packages:
            print("\nMissing packages:")
            for package in missing_packages:
                print(f"- {package}")
        
        if outdated_packages:
            print("\nOutdated packages:")
            for package in outdated_packages:
                print(f"- {package}")
        
        print("\nTo install/update packages, run:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\nAll required packages are installed with correct versions!")
        return True

if __name__ == "__main__":
    check_dependencies() 