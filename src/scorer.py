# -*- coding: UTF-8 -*-

class AMRScorer():

    def calculate_score(prediction, labels):
        """Gets
        Args:
            prediction (list of set): the names that are predicted by the model
            labels (list of set): the ground truth for the task

        Return:
            score (float): the score given prediction and names
        """

        for p, l in zip(prediction, labels):
            if p == set() and l == set():
                score += 1
            elif p == set() and l != set():
                score += 0
            elif p != set() and l == set():
                score += 0
            else:
                score += self.f1_score(p, l)

    def f1_score(self, prediction, labels):
        """The F1 score for a data with predicted values and ground truth
        Args:
            prediction (set): The set of names we predicted
            labels (set): The set of names in the ground truth

        Return:
            f1_score (float): The f1 score of the data
        """
        recall = len(prediction & labels)/len(labels)
        precision = len(prediction & labels)/len(prediction)

        return 2 * ((recall * precision) / (recall + precision))