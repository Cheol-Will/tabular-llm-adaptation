import pickle

# file_path = "experiment/results/260318/TFMLLM/data/TFMLLM_r10_BAG_L1/363612/0_0/results.pkl"
file_path = "experiment/results/260320-num_emb/TFMLLM/data/TFMLLM_c1_BAG_L1/363612/0_0/results.pkl"
with open(file_path, 'rb') as file:
    # Load the data from the file into a Python object
    loaded_object = pickle.load(file)
# for k, v in loaded_object.items():
#     print(k)
#     print(v)

# can we check this during experiment and key the best hp config and validation result?
print(loaded_object["simulation_artifacts"].keys())
print(loaded_object["metric"])
print(loaded_object["metric_error_val"]) # validation
print(loaded_object["metric_error"]) # test
print(loaded_object["method_metadata"]["hyperparameters"]) # test

# print(loaded_object["simulation_artifacts"]["pred_proba_dict_test"]) # prediction
print(loaded_object["time_train_s"])
print(loaded_object.keys())


# import pickle
# with open("experiment/results/260320-num_emb/TFMLLM/data/TFMLLM_c1_BAG_L1/363621/0_0/results.pkl", "rb") as f:
#     obj = pickle.load(f)

# print(obj["framework"])
# print(obj["method_metadata"])
# print(type(obj))

from main import load_tid
task_ids = load_tid()
print(task_ids)