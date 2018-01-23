import math
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import Bunch


class Sample:
    def __init__(self, class_name: str, attribute_names: List[str], attributes: List[int]):
        self.classname = class_name
        self.attributes = attributes
        self.attribute_names = attribute_names

    def get_class_name(self) -> str:
        return self.classname

    def get_attributes(self) -> List[int]:
        return self.attributes

    def get_attribute_names(self):
        return self.attribute_names

    def __str__(self):
        result = "%s -" % self.get_class_name()
        attribute_names = self.get_attribute_names()
        attribute_names_count = len(attribute_names)
        for index, attribute in enumerate(self.get_attributes()):
            if index < attribute_names_count:
                result += " %s: %s," % (attribute_names[index], attribute)
            else:
                result += " %s," % attribute
        result = result[:-1]
        return result

    def __repr__(self):
        return self.__str__()


class Samples:

    @classmethod
    def generate_spaced_colors(cls, n: int) -> List[Tuple[int, int, int]]:
        max_value = 255 ** 3
        interval = int(max_value // n)
        colors = [hex(I)[2:].zfill(6) for I in range(0, interval * n, interval)]
        return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

    @classmethod
    def normalize_vector(cls, vector: List[float]) -> float:
        res = 0
        for x in vector:
            res += x ** 2
        return math.sqrt(res)

    @classmethod
    def load_from_bunch(cls, data: Bunch):
        """
        Takes as an input any Bunch from sklearn.datasets

        :param data: Bunch returned from for e.g. load_digits()
        :return: Samples object
        """
        samples_list: List[Sample] = []
        for index, class_index in enumerate(data['target']):
            attribute_names = data.get('feature_names', [])
            attributes = data['data'][index]
            # Not all Bunches have 'target_names' set
            try:
                class_name = data['target_names'][class_index]
            except (IndexError, KeyError):
                class_name = class_index
            samples_list.append(Sample(str(class_name), attribute_names, attributes))

        return Samples(samples_list)

    @classmethod
    def angle_between_vectors(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        norm1 = Samples.normalize_vector(vector1)
        norm2 = Samples.normalize_vector(vector2)

        res = dot_product / (norm1 * norm2)

        if math.fabs(res) > 1:
            res = res / math.fabs(res)
        return math.acos(res)

    def __init__(self, samples: List[Sample]):
        self.samples = {}
        self.class_colors: Dict[str, str] = {}
        if len(samples) < 2:
            raise Exception("Samples parameter has to have at least 3 samples")

        attributes_count = len(samples[0].get_attributes())
        class_names = set()
        for sample in samples:

            if len(sample.get_attributes()) != attributes_count:
                raise Exception("All samples must have the same number of attributes")

            if sample.get_class_name() in self.samples:
                self.samples[sample.get_class_name()].append(sample)
            else:
                self.samples[sample.get_class_name()] = [sample]
                class_names.add(sample.get_class_name())

        generated_colors = Samples.generate_spaced_colors(len(class_names))
        for index, class_name in enumerate(class_names):
            self.class_colors[class_name] = '#%02x%02x%02x' % generated_colors[index]

    def visualize(self):
        visual_data: List[Tuple[float, float, str]] = self.__get_visual_data()
        x = list(zip(*visual_data))[0]
        y = list(zip(*visual_data))[1]
        colors = list(zip(*visual_data))[2]
        plt.scatter(x, y, c=colors)
        plt.show()

    def get_color_for_class(self, class_name: str):
        return self.class_colors.get(class_name, '#000000')

    def get_samples_for_class(self, class_name: str) -> List[Sample]:
        if class_name in self.samples:
            return self.samples.get(class_name)
        raise Exception("Class %s was not found" % class_name)

    def get_all_samples(self) -> List[Sample]:
        return [item for sublist in self.samples.values() for item in sublist]

    def get_classes(self) -> List[str]:
        return list(self.samples.keys())

    def print_details(self):
        print("Has %s classes, with total of %s samples" % (
            len(self.get_classes()), len(self.get_all_samples())))

    def print_class_details(self, class_name: str):
        print("Class %s has %s samples" % (class_name, len(self.get_samples_for_class(class_name))))

    def get_test_data(self, ratio: float = 0.2) -> Tuple[List[Sample], List[Sample]]:
        test_data: List[Sample] = []
        validation_data: List[Sample] = []
        for class_name in self.get_classes():
            class_samples = self.get_samples_for_class(class_name)
            class_samples_count = len(class_samples)
            count_percent = int(class_samples_count * ratio)
            test_data.extend(class_samples[:count_percent])
            validation_data.extend(class_samples[count_percent:])
        return test_data, validation_data

    def __get_visual_data(self) -> List[Tuple[float, float, str]]:
        unit = np.ones(len(self.get_all_samples()[0].get_attributes()))
        result = []
        for sample in self.get_all_samples():
            result.append([
                Samples.normalize_vector(sample.get_attributes()),
                Samples.angle_between_vectors(unit.tolist(), sample.get_attributes()),
                self.get_color_for_class(sample.get_class_name())
            ])
        return result
