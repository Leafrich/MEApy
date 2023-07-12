# calculate the entropy for a dataset
from math import log2


# # proportion of examples in each class
# class0 = 10/100
# class1 = 90/100
# # calculate entropy
# entropy = -(class0 * log2(class0) + class1 * log2(class1))
#
# print('entropy: %.3f bits' % entropy)

def entropy(class0, class1):
    return -(class0 * log2(class0) + class1 * log2(class1))


class0 = 13 / 20
class1 = 7 / 20
# calculate entropy before the change
s_entropy = entropy(class0, class1)
print('Dataset Entropy: %.3f bits' % s_entropy)

s1_class0 = 7 / 8
s1_class1 = 1 / 8
# calculate the entropy of the first group
s1_entropy = entropy(s1_class0, s1_class1)
print('Group1 Entropy: %.3f bits' % s1_entropy)

# split 2  (split via value2)
s2_class0 = 6 / 12
s2_class1 = 6 / 12
# calculate the entropy of the second group
s2_entropy = entropy(s2_class0, s2_class1)
print('Group2 Entropy: %.3f bits' % s2_entropy)
