import * as tf from '@tensorflow/tfjs'
class perceptronFromTFJS {
  model: tf.Sequential
  hiddenLayers: number[] = []
  activation: string
  constructor (hiddenLayers: number[], activation: any = 'relu') {
    this.model = tf.sequential()
    this.hiddenLayers = hiddenLayers
    this.activation = activation
  }
  async trainNet (opt: { data: { input: number[], output: number[] }[], callback: (d: Log) => void, epochs: number, batchSize?: number }) {

    const neurons = [opt.data[0].input.length].concat(this.hiddenLayers, [opt.data[0].output.length])
    const layers = neurons.map((count, i) => {
      return { inputShape: [count], units: neurons[i + 1], activation: this.activation as any }
    }).filter(l => l.units)
    console.log({ neurons, layers })
    layers.forEach(l => this.model.add(tf.layers.dense(l)))
    this.model.compile({ optimizer: tf.train.adam(), loss: 'meanSquaredError', metrics: ['accuracy'] })
    console.log(this.model.layers)

    const x = tf.tensor2d(opt.data.map(s => s.input))
    const y = tf.tensor2d(opt.data.map(s => s.output))
    const res = await this.model.fit(x, y, {
      epochs: opt.epochs,
      batchSize: opt.batchSize,
      callbacks: {
        onEpochEnd (epoch, log) {
          console.log(epoch, log)
          // epoch > 3 && opt.callback({ error: log.loss, iterations: epoch })
        }
      }
    })
    x.dispose()
    y.dispose()
    return res
  }
  run (input: number[]): number[] {
    const xs = tf.tensor2d([input])
    const res = (this.model.predict(xs) as any).arraySync()[0]
    xs.dispose()
    return res
  }
  async runAsync (input: number[]) {
    const xs = tf.tensor2d([input])
    const res = (await (this.model.predict(xs) as any).array())[0] as number[]
    xs.dispose()
    return res
  }
  async massPredict (input: number[][]) {
    const xs = tf.tensor2d(input)
    const res = (await (this.model.predict(xs) as any).array()) as number[][]
    xs.dispose()
    return res
  }
}

(async () => {
  const xor = new perceptronFromTFJS([100, 20])
  const data = [
    { input: [0, 0], output: [0] },
    { input: [1, 0], output: [1] },
    { input: [0, 1], output: [1] },
    { input: [0, 0], output: [0] },
  ]
  await xor.trainNet({ data, epochs: 100, callback: console.log })
  data.forEach(d => console.log(xor.run(d.input), d.output))
})()
