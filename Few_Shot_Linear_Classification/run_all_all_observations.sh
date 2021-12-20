python main.py --dataset="CIFAR10" --data-root="data/CIFAR10/extracted_features/supervised" --use-validation=1 --output-path="results/CIFAR10/all_observations/supervised" --observations-per-class=-1 --repeats=1
python main.py --dataset="CIFAR10" --data-root="data/CIFAR10/extracted_features/simsiam" --use-validation=0 --output-path="results/CIFAR10/all_observations/simsiam" --observations-per-class=-1 --repeats=1
python main.py --dataset="CIFAR10" --data-root="data/CIFAR10/extracted_features/simclr" --use-validation=0 --output-path="results/CIFAR10/all_observations/simclr" --observations-per-class=-1 --repeats=1

python main.py --dataset="STL10" --data-root="data/STL10/extracted_features/supervised" --use-validation=0 --output-path="results/STL10/all_observations/supervised" --observations-per-class=-1 --repeats=1
python main.py --dataset="STL10" --data-root="data/STL10/extracted_features/simsiam" --use-validation=0 --output-path="results/STL10/all_observations/simsiam" --observations-per-class=-1 --repeats=1
python main.py --dataset="STL10" --data-root="data/STL10/extracted_features/simclr" --use-validation=0 --output-path="results/STL10/all_observations/simclr" --observations-per-class=-1 --repeats=1

