import random
import copy
import torch
import os
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")


class ModelFuzzer:
    def __init__(self, model, input_size, error_threshold=1e-1):
        self.model = model
        self.input_size = input_size
        self.error_threshold = error_threshold
        self.bugflag = False
        self.noise_scale = 0.01
        self.noise_scale_min = 0.001
        self.noise_scale_max = 0.1
        self.learning_rate = 0.001
        self.history = []
        self.mutation_methods = [self.add_random_noise, self.multiply_by_factor, self.apply_scaling,
                                 self.random_masking, self.set_uniform_values]
        self.mutation_error_tracking = {method.__name__: [] for method in self.mutation_methods}
        self.test_count = 0
        self.error_counts = {method.__name__: 0 for method in self.mutation_methods}
        self.current_mutation_method = None
        self.current_method_no_error_count = 0
        self.switch_method_threshold = 50

    def add_random_noise(self, input_tensor):
        noise = torch.randn_like(input_tensor)
        return input_tensor + noise

    def multiply_by_factor(self, input_tensor):
        factor = torch.randn(1) * 0.01
        return input_tensor * factor

    def apply_scaling(self, input_tensor):
        scale = torch.randn(1) * 100
        return input_tensor * scale

    def random_masking(self, input_tensor):
        mask = torch.bernoulli(torch.full_like(input_tensor, 0.5))
        return input_tensor * mask

    def set_uniform_values(self, input_tensor):
        uniform_value = torch.randn(1).item()  # 提取标量值
        return torch.full_like(input_tensor, uniform_value)

    def select_mutation_method(self):
        if self.test_count < 20:
            chosen_method = random.choice(self.mutation_methods).__name__
        else:
            if self.current_method_no_error_count >= self.switch_method_threshold:
                self.current_mutation_method = None

            if self.current_mutation_method is None:
                max_error_method = max(self.error_counts, key=self.error_counts.get)
                max_error = max((sum(errors), method) for method, errors in self.mutation_error_tracking.items())[1]

                chosen_method = max_error_method if self.error_counts[max_error_method] > sum(self.mutation_error_tracking[max_error]) else max_error
                self.current_mutation_method = chosen_method
                self.current_method_no_error_count = 0
            else:
                self.current_method_no_error_count += 1
                chosen_method = self.current_mutation_method

        self.test_count += 1
        return getattr(self, chosen_method)

    def update_history(self, mutation_method_name, error, buggy_type=None):
        history_entry = {
            "mutation_method": mutation_method_name,
            "error": error,
            "buggy_type": buggy_type
        }
        self.history.append(history_entry)

        if buggy_type is not None:
            self.error_counts[mutation_method_name] += 1
        self.mutation_error_tracking[mutation_method_name].append(error)


    def calculate_error(self, output_gpu, output_cpu):
        try:
            # 对于tensor类型，计算平均绝对误差
            if torch.is_tensor(output_gpu) and torch.is_tensor(output_cpu):
                return torch.abs(output_gpu - output_cpu).mean().item()

            # 对于int或float类型，计算绝对误差
            elif isinstance(output_gpu, (int, float)) and isinstance(output_cpu, (int, float)):
                return abs(output_gpu - output_cpu)

            # 对于bool类型，比较是否相等
            elif isinstance(output_gpu, bool) and isinstance(output_cpu, bool):
                return 0 if output_gpu == output_cpu else 1

            # 对于字符串或其他类型，转换为字符串并逐字符比较
            else:
                str_gpu = str(output_gpu)
                str_cpu = str(output_cpu)
                return sum(c1 != c2 for c1, c2 in zip(str_gpu, str_cpu)) + abs(len(str_gpu) - len(str_cpu))

        except Exception as e:
            return f"exception: {e}"

    def check_for_nan(self, value):
        try:
            # 对于张量，检查是否包含 NaN
            if torch.is_tensor(value):
                return torch.isnan(value).any()

            # 对于浮点数，检查是否为 NaN
            elif isinstance(value, float):
                return math.isnan(value)

            # 对于整数，整数类型不可能是 NaN，返回 False
            elif isinstance(value, int):
                return False

            # 对于其他类型，返回 False
            else:
                return False

        except Exception as e:
            return f"exception: {e}"


    def perform_fuzzing_and_testing(self, input_seed):
        gpu_device = torch.device("cuda")
        cpu_device = torch.device("cpu")
        for i in range(5000):

            if i < 3000:
                reasonable_error = self.error_threshold
            else:
                reasonable_error = self.error_threshold * 0.1
            try:
                mutation_method = self.select_mutation_method()
                test_input = mutation_method(input_seed)

                # Move model to GPU and ensure test input is also on GPU
                self.model.to(gpu_device)
                test_input_gpu = test_input.to(gpu_device)
                output_gpu, used_apis_list = self.model(test_input_gpu)
                # Check for NaN values or other issues on GPU output
                if self.check_for_nan(output_gpu):
                    print(f"Test {i + 1} - NaN Value on GPU")
                    buggy_type = 'NaN on GPU'
                    self.update_history(mutation_method.__name__, float('inf'), buggy_type)
                    return self.model, used_apis_list, buggy_type, test_input

                # Move model to CPU and ensure test input is also on CPU
                self.model.to(cpu_device)
                test_input_cpu = test_input.to(cpu_device)
                output_cpu, used_apis_list = self.model(test_input_cpu)

                # Check for NaN values or other issues on CPU output
                if self.check_for_nan(output_cpu):
                    print(f"Test {i + 1} - NaN Value on CPU")
                    buggy_type = 'NaN on CPU'
                    self.update_history(mutation_method.__name__, float('inf'), buggy_type)

                    return self.model, used_apis_list, buggy_type, test_input

                # If no NaN values or other issues on both GPU and CPU, then check for inconsistency
                if not torch.allclose(output_gpu.cpu(), output_cpu, atol=reasonable_error):
                    print(f"Test {i + 1} - Inconsistency Found")
                    buggy_model = copy.deepcopy(self.model)
                    buggy_type = 'Inconsistency Found'
                    error = self.calculate_error(output_gpu.cpu(), output_cpu)
                    self.update_history(mutation_method.__name__, error, buggy_type)
                    return buggy_model, used_apis_list, buggy_type, test_input

            except torch.cuda.OutOfMemoryError:
                print(f"Test - Memory Error")
                buggy_type = 'Memory Error'
                buggy_model = copy.deepcopy(self.model)
                return buggy_model, used_apis_list, buggy_type, test_input
            except Exception as e:
                print(f"Test - Exception: {str(e)}")
                buggy_type = 'Exception'
                buggy_model = copy.deepcopy(self.model)
                return buggy_model, None, buggy_type, test_input

            error = self.calculate_error(output_gpu.cpu(), output_cpu)
            self.mutation_error_tracking[mutation_method.__name__].append(error)

        return None, used_apis_list, None, None


'''
import random
import copy
import torch
from scipy.stats import ks_2samp
from dict_comparator import ToleranceDictComparator
import torchvision.transforms.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
class ModelFuzzer:
    def __init__(self, model, input_size, num_tests=5, mutation_prob=0.2, error_threshold=1e-2):
        self.model = model
        self.input_size = input_size
        self.num_tests = num_tests
        self.mutation_prob = mutation_prob
        self.error_threshold = error_threshold
        self.mutation_weights = {
            "swap_layers": 10.0,
            "delete_layer": 10.0,
            "duplicate_layer": 10.0
        }
        self.fuzzing_iterations = 0
        self.best_error = float('inf')

        self.bugflag = False  # Initialize the bugflag to False
        self.bug_tracker = {
            'rotate': 0,
            'hflip': 0,
            'vflip': 0,
            'affine': 0,
            'noise': 0
        }
        # Initialize the noise parameters
        self.noise_scale = 0.01  # 初始噪声水平
        self.noise_scale_min = 0.001  # 最小噪声水平
        self.noise_scale_max = 0.1  # 最大噪声水平
        self.learning_rate = 0.001  # 学习速率
        self.history = []  # Keep a history of whether a bug was found

    def swap_layers(self):
        # Implement layer swap mutation
        layers = list(self.model.children())
        if len(layers) >= 2:
            i, j = random.sample(range(1, len(layers)), 2)
            layers[i], layers[j] = layers[j], layers[i]

    def delete_layer(self):
        # Implement layer deletion mutation
        layers = list(self.model.children())
        if len(layers) >= 3:
            i = random.choice(range(1, len(layers)))
            del layers[i]

    def duplicate_layer(self):
        # Implement layer duplication mutation
        layers = list(self.model.children())
        if len(layers) >= 2:
            i = random.choice(range(1, len(layers)))
            new_layer = copy.deepcopy(layers[i])
            layers.insert(i, new_layer)

    def generate_input(self):
        # Generate random input based on input_size
        input_data = torch.rand(self.input_size)
        return input_data

    def apply_metamorphic_transformation(self, input_data):
        if input_data.dim() != 4:
            raise ValueError(f"Expected input_data to be a 4D tensor with shape (B, C, H, W), but got shape {input_data.shape}")

        B, C, H, W = input_data.shape
        transformed_batch = torch.zeros_like(input_data)

        # Define total bug score and calculate probabilities
        total_bug_score = sum(self.bug_tracker.values()) + 1  # Add 1 to avoid division by zero
        transform_probs = {t: (score + 1) / total_bug_score for t, score in self.bug_tracker.items()}  # Add 1 to ensure non-zero probability

        for b in range(B):
            transformed_input = input_data[b].clone()  # Work with a copy of the input tensor

            # Determine which transformations to apply based on the probabilities
            transformations = {
                'rotate': random.random() < transform_probs['rotate'],
                'hflip': random.random() < transform_probs['hflip'],
                'vflip': random.random() < transform_probs['vflip'],
                'affine': random.random() < transform_probs['affine'],
                'noise': random.random() < transform_probs['noise']
            }

            # Apply transformations
            if transformations['rotate']:
                angle = random.uniform(-30, 30)
                transformed_input = F.rotate(transformed_input, angle, expand=False)

            if transformations['hflip']:
                transformed_input = F.hflip(transformed_input)

            if transformations['vflip']:
                transformed_input = F.vflip(transformed_input)

            if transformations['affine']:
                max_dx, max_dy = W // 4, H // 4
                translations = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
                scale = random.uniform(0.8, 1.2)
                shear = random.uniform(-10, 10)
                transformed_input = F.affine(transformed_input, angle=0, translate=translations, scale=scale, shear=shear, fill=0)
                transformed_input = F.center_crop(transformed_input, [H, W])

            if transformations['noise']:
                noise = torch.randn_like(transformed_input) * 0.05
                transformed_input = transformed_input + noise

            # Store the transformed image in the batch
            transformed_batch[b] = transformed_input

        return transformed_batch

    def report_bug(self, transformation_type):
        # Increment the bug counter for a specific transformation
        if transformation_type in self.bug_tracker:
            self.bug_tracker[transformation_type] += 1
        else:
            raise ValueError(f"Invalid transformation type: {transformation_type}")


    def verify_metamorphic_relationship(self, original_output, transformed_output):
        # Move the tensors to the same device (GPU)
        device = original_output.device
        transformed_output = transformed_output.to(device)

        # Compare the distributions on the GPU
        original_distribution = original_output.view(-1).detach().cpu().numpy()
        transformed_distribution = transformed_output.view(-1).detach().cpu().numpy()

        # Use the KS test from scipy.stats
        ks_statistic, ks_p_value = ks_2samp(original_distribution, transformed_distribution)

        # If p-value is high, the distributions are similar, and the relationship is valid
        if ks_p_value > 0.03:
            return True
        else:
            return False

    def calculate_error(self, original_output, transformed_output):

        loss = torch.abs(original_output-transformed_output).mean()
        return loss

    def mutate_input_seed(self, input_seed):
        # Generate a random noise vector
        noise = torch.randn_like(input_seed)

        # Adjust the noise_scale based on the history of found bugs
        self._adjust_noise_scale()

        # Apply the current noise scale to the input seed
        noisy_input = input_seed + noise * self.noise_scale
        return noisy_input

    def _adjust_noise_scale(self):
        # Use the history of found bugs to adjust the noise_scale
        if len(self.history) > 5:  # Wait until we have enough history to make a decision
            recent_history = self.history[-5:]
            bugs_found = sum(recent_history)
            if bugs_found > 0:
                # If bugs were found, reduce the noise_scale to refine the search
                self.noise_scale *= (1 - self.learning_rate)
            else:
                # If no bugs were found, increase the noise_scale to explore more broadly
                self.noise_scale *= (1 + self.learning_rate)

            # Ensure the noise_scale stays within the specified range
            self.noise_scale = max(min(self.noise_scale, self.noise_scale_max), self.noise_scale_min)

    def update_history(self, found_bug):
        # Update the history of whether a bug was found
        self.history.append(found_bug)

        # Keep the history size manageable
        if len(self.history) > 500:
            self.history.pop(0)

    def perform_fuzzing_and_testing(self, input_seed):
        buggy_model = None
        buggy_input = None
        gpu_device = torch.device("cuda")
        cpu_device = torch.device("cpu")
        success = False
        buggy_type = None

        for i in range(4000):
            print(i)
            if i < 3000:
                reasonable_error = 1e-1
            else:
                reasonable_error = 1e-2
            try:
                # Mutate the input seed if necessary
                test_input = self.mutate_input_seed(input_seed) if success else input_seed

                # Move model to GPU and ensure test input is also on GPU
                self.model.to(gpu_device)
                test_input_gpu = test_input.to(gpu_device)
                output_gpu, used_apis_list, results_gpu = self.model(test_input_gpu)
                api_inputs_gpu = transform_dict(results_gpu, test_input_gpu)
                # Check for NaN values or other issues on GPU output
                if torch.isnan(output_gpu).any():
                    print(f"Test {i + 1} - NaN Value on GPU")
                    buggy_type = 'NaN on GPU'
                    self.update_history(True)
                    buggy_api = check_for_nan(results_gpu)
                    buggy_api_input_gpu = api_inputs_gpu['pre'+buggy_api]
                    return self.model, buggy_api_input_gpu, used_apis_list, buggy_type, buggy_api , None, None

                # Move model to CPU and ensure test input is also on CPU
                self.model.to(cpu_device)
                test_input_cpu = test_input.to(cpu_device)
                output_cpu, used_apis_list, results_cpu = self.model(test_input_cpu)
                api_inputs_cpu = transform_dict(results_cpu, test_input_cpu)

                # Check for NaN values or other issues on CPU output
                if torch.isnan(output_cpu).any():
                    print(f"Test {i + 1} - NaN Value on CPU")
                    buggy_type = 'NaN on CPU'
                    self.update_history(True)
                    buggy_api = check_for_nan(results_cpu)
                    buggy_api_input_cpu = api_inputs_cpu['pre'+buggy_api]
                    return self.model, buggy_api_input_cpu, used_apis_list, buggy_type, buggy_api , None, None

                # If no NaN values or other issues on both GPU and CPU, then check for inconsistency
                if not torch.allclose(output_gpu.cpu(), output_cpu, atol=reasonable_error):
                    print(f"Test {i + 1} - Inconsistency Found")
                    buggy_model = copy.deepcopy(self.model)

                    buggy_type = 'Inconsistency Found'
                    self.update_history(True)
                    comparator = ToleranceDictComparator(results_gpu, results_cpu, tolerance=1e-9)
                    buggy_api, value_a, value_b = comparator.compare()
                    print(buggy_api)
                    buggy_api_input_gpu = api_inputs_gpu['pre'+buggy_api]
                    buggy_api_input_cpu = api_inputs_cpu['pre'+buggy_api]
                    return buggy_model, buggy_api_input_cpu, used_apis_list, buggy_type, buggy_api, buggy_api_input_gpu, None

                #print(f"Test {i + 1} - No Bugs Found")
                success = True

            except torch.cuda.OutOfMemoryError:
                print(f"Test - Memory Error")
                buggy_type = 'Memory Error'
                return self.model, test_input, None, buggy_type, None , None, None
            except Exception as e:
                print(f"Test - Exception: {str(e)}")
                buggy_type = 'Exception'
                self.update_history(True)
                ##################################################
                #if used_apis_list = None:
                #model failed
                ##################################################
                return self.model, test_input, None, buggy_type, None , None , str(e)
        self.update_history(False)
        return buggy_model, buggy_input, used_apis_list, buggy_type, None , None, None

    def should_continue_fuzzing(self, max_iterations=100, min_improvement=1e-5):
        if self.fuzzing_iterations >= max_iterations:
            return False

        if abs(self.best_error - self.error_threshold) < min_improvement:
            return False

        return True

    def select_mutation_method(self):
        total_weight = sum(self.mutation_weights.values())
        r = random.uniform(0, total_weight)
        up_to = 0
        for method, weight in self.mutation_weights.items():
            if up_to + weight >= r:
                return getattr(self, method)
            up_to += weight

    def update_mutation_weight(self, method, success):
        if success:
            self.mutation_weights[method.__name__] *= 0.9
        else:
            self.mutation_weights[method.__name__] *= 1.1

def transform_dict(old_dict, initial_value):
    new_dict = {}
    previous_value = initial_value

    keys = list(old_dict.keys())  # 获取旧字典的键列表
    for key in keys[:-1]:  # 遍历除了最后一个键以外的所有键
        new_dict[key] = previous_value
        previous_value = old_dict[key]

    if keys:  # 如果字典不是空的
        new_dict[keys[-1]] = old_dict[keys[-2]] if len(keys) > 1 else initial_value

    return new_dict
def check_for_nan(dictionary):
    for key, value in dictionary.items():
        if torch.is_tensor(value):  # 确保值是一个张量
            if torch.isnan(value).any():  # 检查张量是否包含NaN
                return key  # 返回包含NaN的第一个元素的名称
    return None  # 如果没有找到NaN，返回None
'''