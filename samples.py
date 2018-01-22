from typing import List, Tuple

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
                result += " %s," % (attribute)
        result = result[:-1]
        return result

    def __repr__(self):
        return self.__str__()


class Samples:
    def __init__(self, samples: List[Sample]):
        self.samples = {}
        for sample in samples:
            if sample.get_class_name() in self.samples:
                self.samples[sample.get_class_name()].append(sample)
            else:
                self.samples[sample.get_class_name()] = [sample]

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
