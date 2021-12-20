python main.py --dataset="CIFAR10" --data-root="data/CIFAR10/extracted_features/supervised" --use-validation=1 --output-path="results/CIFAR10/supervised" 
python main.py --dataset="CIFAR10" --data-root="data/CIFAR10/extracted_features/simsiam" --use-validation=0 --output-path="results/CIFAR10/simsiam"
python main.py --dataset="CIFAR10" --data-root="data/CIFAR10/extracted_features/simclr" --use-validation=0 --output-path="results/CIFAR10/simclr"

python main.py --dataset="STL10" --data-root="data/STL10/extracted_features/supervised" --use-validation=0 --output-path="results/STL10/supervised"
python main.py --dataset="STL10" --data-root="data/STL10/extracted_features/simsiam" --use-validation=0 --output-path="results/STL10/simsiam"
python main.py --dataset="STL10" --data-root="data/STL10/extracted_features/simclr" --use-validation=0 --output-path="results/STL10/simclr"

