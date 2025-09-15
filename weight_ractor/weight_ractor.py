import torch
from collections import OrderedDict
# Load the first weight file and extract only the weight of the key name 'conv2d_student'
def load_first_weight_file(file_path):
    state_dict = torch.load(file_path, map_location='cpu')
    weights = modified_weights(state_dict['model_state_dict'], False)
    conv2d_student_weights = {k: v for k, v in weights.items() if 'conv2d_student' in k}
    conv2d_student = {k.replace('conv2d_student', 'conv2d'): v for k, v in conv2d_student_weights.items() if
                              'conv2d_student' in k}
    return conv2d_student

# Load the second weight file and extract all the weights except 'conv2d'
def load_second_weight_file(file_path):
    state_dict = torch.load(file_path, map_location='cpu')
    weights = modified_weights(state_dict['model_state_dict'], False )
    other_weights = {k: v for k, v in weights.items() if 'conv2d' not in k}

    return other_weights

def modified_weights(state_dict, modified=False):
    state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
    if not modified:
        return state_dict
    modified_dict = dict()
    return modified_dict

#
def combine_weights(weights1, weights2):
    combined_weights = {**weights1, **weights2}
    return combined_weights

#
def save_combined_weights(combined_weights, output_file):
    torch.save(combined_weights, output_file)
#
first_weight_file = './Pretraining weights/Semantic_finetuning_2014'
second_weight_file = './CorrNet_weights/CorrNet_dev_18.90_PHOENIX14.pt'
output_file = './Pretraining weights/weight_ractor_00.14_2014.pt'
#
weights1 = load_first_weight_file(first_weight_file)
weights2 = load_second_weight_file(second_weight_file)
#
combined_weights = combine_weights(weights1, weights2)
#
save_combined_weights(combined_weights, output_file)

print("Combined weights saved to:", output_file)
