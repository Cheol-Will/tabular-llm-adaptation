import pickle

file_path = "experiment/results/260318/TFMLLM/data/TFMLLM_r10_BAG_L1/363612/0_0/results.pkl"

with open(file_path, 'rb') as file:
    # Load the data from the file into a Python object
    loaded_object = pickle.load(file)
# for k, v in loaded_object.items():
#     print(k)
#     print(v)

# can we check this during experiment and key the best hp config and validation result?
print(loaded_object["simulation_artifacts"].keys())
print(loaded_object["metric"])
print(loaded_object["metric_error"]) # train 
print(loaded_object["metric_error_val"]) # test

# print(loaded_object["simulation_artifacts"]["pred_proba_dict_test"]) # prediction
print(loaded_object.keys())
