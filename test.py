import tensorflow as tf
from covid_net import COVIDNet

# TODO: generate tf.data, setting also batch_size (see doc)
training_set = (...).batch(8)
test_set = (...).batch(8)

model = COVIDNet()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'recall'])
model.fit(training_set, epochs=10)
model.evaluate(test_set)
