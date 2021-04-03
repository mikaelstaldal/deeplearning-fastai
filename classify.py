import sys

from fastai.learner import load_learner

filename = sys.argv[1]

learner = load_learner("export.pkl")

label,index,probs = learner.predict(filename)
prob = probs[index.item()]

print("{} is a {} with probability {}".format(filename, label, prob))
