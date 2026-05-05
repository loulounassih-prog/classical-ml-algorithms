import numpy as np
from TD.svm import SVM
from TD.kernel import Kernel, KernelType
from TD.dataset import Dataset

class MailSelector:
    def __init__(self, train_file, test_file, kernel_type=KernelType.RBF, gamma=0.5, C=1.0, tol=1e-5):
        self.train_file = train_file
        self.test_file = test_file
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.C = C
        self.tol = tol

    def run(self):
        
        train_data = Dataset(self.train_file)
        test_data = Dataset(self.test_file)
        
        kernel = Kernel(self.kernel_type, gamma=self.gamma)

        svm = SVM(train_data, col_class=0, kernel=kernel)
        svm.train(C=self.C)

        confusion_matrix = svm.test(test_data)

        detection_rate = confusion_matrix.detection_rate()
        false_alarm_rate = confusion_matrix.false_alarm_rate()
        precision = confusion_matrix.precision()
        f_score = confusion_matrix.f_score()

        return detection_rate, false_alarm_rate, precision, f_score

