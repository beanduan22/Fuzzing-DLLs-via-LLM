import tensorflow as tf
import random
import math
import copy
import numpy as np


class ModelFuzzerTF:
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
        noise = tf.random.normal(shape=tf.shape(input_tensor))
        return input_tensor + noise

    def multiply_by_factor(self, input_tensor):
        factor = tf.random.normal([1]) * 0.01
        return input_tensor * factor

    def apply_scaling(self, input_tensor):
        scale = tf.random.normal([1]) * 100
        return input_tensor * scale

    def random_masking(self, input_tensor):
        mask = tf.cast(tf.random.uniform(tf.shape(input_tensor)) > 0.5, input_tensor.dtype)
        return input_tensor * mask

    def set_uniform_values(self, input_tensor):
        uniform_value = tf.random.normal([1])[0].numpy()  # 提取标量值
        return tf.fill(tf.shape(input_tensor), uniform_value)

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
            if tf.is_tensor(output_gpu) and tf.is_tensor(output_cpu):
                return tf.abs(output_gpu - output_cpu).mean().item()

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
            if tf.is_tensor(value):
                return tf.reduce_any(tf.math.is_nan(value))

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
        for i in range(5000):
            if i < 3000:
                reasonable_error = self.error_threshold
            else:
                reasonable_error = self.error_threshold * 0.1
            try:
                mutation_method = self.select_mutation_method()
                test_input = mutation_method(input_seed)

                with tf.device('/GPU:0'):
                    self.model.to_gpu()  # 假设模型有方法来迁移到 GPU
                    test_input_gpu = tf.identity(test_input)
                    output_gpu, used_apis_list = self.model(test_input_gpu)

                if self.check_for_nan(output_gpu):
                    print(f"Test {i + 1} - NaN Value on GPU")
                    buggy_type = 'NaN on GPU'
                    self.update_history(mutation_method.__name__, float('inf'), buggy_type)
                    return self.model, used_apis_list, buggy_type, test_input

                with tf.device('/CPU:0'):
                    self.model.to_cpu()  # 假设模型有方法来迁移到 CPU
                    test_input_cpu = tf.identity(test_input)
                    output_cpu, used_apis_list = self.model(test_input_cpu)

                if self.check_for_nan(output_cpu):
                    print(f"Test {i + 1} - NaN Value on CPU")
                    buggy_type = 'NaN on CPU'
                    self.update_history(mutation_method.__name__, float('inf'), buggy_type)

                    return self.model, used_apis_list, buggy_type, test_input

                if not np.allclose(output_cpu, output_gpu, atol=reasonable_error):
                    print(f"Test {i + 1} - Inconsistency Found")
                    buggy_model = copy.deepcopy(self.model)
                    buggy_type = 'Inconsistency Found'
                    error = self.calculate_error(output_gpu.cpu(), output_cpu)
                    self.update_history(mutation_method.__name__, error, buggy_type)
                    return buggy_model, used_apis_list, buggy_type, test_input


            except tf.errors.ResourceExhaustedError:

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
