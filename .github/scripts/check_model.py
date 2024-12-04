import re

def check_total_parameters():
    # Assuming the model file is in assignment5/src/model.py
    with open('src/model.py', 'r') as file:
        content = file.read()
        # Check for total parameters
        if 'Total Parameters' in content:
            print("Total Parameter Count Test: PASSED")
        else:
            print("Total Parameter Count Test: FAILED")

def check_batch_normalization():
    with open('src/model.py', 'r') as file:
        content = file.read()
        # Check for use of BatchNorm
        if 'BatchNorm' in content:
            print("Use of Batch Normalization: PASSED")
        else:
            print("Use of Batch Normalization: FAILED")

def check_dropout():
    with open('src/model.py', 'r') as file:
        content = file.read()
        # Check for use of Dropout
        if 'Dropout' in content:
            print("Use of DropOut: PASSED")
        else:
            print("Use of DropOut: FAILED")

def check_gap_or_fc():
    with open('src/model.py', 'r') as file:
        content = file.read()
        # Check for use of GAP or fully connected layer
        if 'AdaptiveAvgPool2d' in content or 'Linear' in content:
            print("Use of Fully Connected Layer or GAP: PASSED")
        else:
            print("Use of Fully Connected Layer or GAP: FAILED")

if __name__ == "__main__":
    check_total_parameters()
    check_batch_normalization()
    check_dropout()
    check_gap_or_fc()
