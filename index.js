// Data
const train_xs = tf.tensor2d([
  [0, 0], [1, 0], [0, 1], [1, 1]  
])

const train_ys = tf.tensor2d([
  [0], [1], [1], [0]
])

// Model
const model = tf.sequential()

// Layers
const hidden = tf.layers.dense({
  inputShape: [2],
  units: 2,
  activation: 'sigmoid'
})

const outputs = tf.layers.dense({
  units: 1,
  activation: 'sigmoid'
})

model.add(hidden)
model.add(outputs)

// Compiling model
const optimizer = tf.train.sgd(0.1)
model.compile({
  loss: 'meanSquaredError',
  optimizer
})

// Train model
const train = () => {
  return model.fit(train_xs, train_ys)
}

// Predict
const data = tf.tensor2d([
  [0, 0], [0, 1], [1, 0], [1, 1]
])
train().then(() => {
  const predict = model.predict(data)
  predict.print()
})