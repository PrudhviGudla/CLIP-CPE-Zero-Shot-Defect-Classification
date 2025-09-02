from spec import DefectClassificationSpec
from pathlib import Path
from clip_ac import classify_defects, classify_all_classes_direct, classify_only_defects
import argparse

def main():
    parser = argparse.ArgumentParser(description='CLIP-based defect classification')
    parser.add_argument('--data_dir', type=Path, required=True, default=Path('data'))
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    spec = DefectClassificationSpec()

    # Run complete classification (binary + multi-class in one call)
    y_true, y_pred = classify_defects(spec, data_dir, threshold = 0)

    # Run single stage direct multi class classification
    # uncomment below line if you want to comapre the result of single stage classification from below with 2-stage classification results from above
    # y_true, y_pred = classify_all_classes_direct(spec, data_dir)

    # Run classification only on defective samples
    # uncomment the below line if you want to experiment this
    # y_true, y_pred = classify_only_defects(spec, data_dir)

if __name__ == "__main__":
    main()
