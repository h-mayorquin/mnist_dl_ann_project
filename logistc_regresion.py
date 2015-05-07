import numpy
import theano
import theano.tensor as T
from data_module.load_files_logistic import training_ims, training_labels
rng = numpy.random

N = 400
feats = 784
x = training_ims[0:N, ...]
# Let's normalize
x = x * 1.0 / numpy.max(x)

y = training_labels[0:N]
print x.shape
print y.shape
# x = rng.randn(N, feats)
# y = rng.randint(size=N, low=0, high=2)
print x.shape
print y.shape

D = (x, y)
training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")

print "Initial model:"
print w.get_value(), b.get_value()

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)  # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()  # The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost

# Compile
train = theano.function(
          inputs=[x, y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    print 'i step =', i
    pred, err = train(D[0], D[1])

print "Final model:"
print w.get_value(), b.get_value()
target_values = D[1]
prediction = predict(D[0])
n_failures = numpy.abs(target_values - prediction).sum()
# Print output
print "target values for D:", target_values
print "prediction on D:", prediction

print 'Accuaracy', 100.0 - n_failures * 100.0 / N
