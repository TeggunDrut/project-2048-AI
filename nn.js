function randomGaussian() {
  let x1, x2, rad, y1;
  do {
    x1 = 2 * Math.random() - 1;
    x2 = 2 * Math.random() - 1;
    rad = x1 * x1 + x2 * x2;
  } while (rad >= 1 || rad == 0);
  let c = Math.sqrt((-2 * Math.log(rad)) / rad);
  return x1 * c;
}

class ActivationFunction {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

let sigmoid = new ActivationFunction(
  (x) => 1 / (1 + Math.exp(-x)),
  (y) => y * (1 - y)
);

let tanh = new ActivationFunction(
  (x) => Math.tanh(x),
  (y) => 1 - y * y
);

class NeuralNetwork {
  // TODO: document what a, b, c are
  constructor(a, b, c) {
    if (a instanceof NeuralNetwork) {
      this.input_nodes = a.input_nodes;
      this.hidden_nodes = a.hidden_nodes;
      this.output_nodes = a.output_nodes;

      this.weights_ih = a.weights_ih.copy();
      this.weights_ho = a.weights_ho.copy();

      this.bias_h = a.bias_h.copy();
      this.bias_o = a.bias_o.copy();
    } else {
      this.input_nodes = a;
      this.hidden_nodes = b;
      this.output_nodes = c;

      this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
      this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
      this.weights_ih.randomize();
      this.weights_ho.randomize();

      this.bias_h = new Matrix(this.hidden_nodes, 1);
      this.bias_o = new Matrix(this.output_nodes, 1);
      this.bias_h.randomize();
      this.bias_o.randomize();
    }

    // TODO: copy these as well
    this.setLearningRate();
    this.setActivationFunction();
  }

  predict(input_array) {
    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    // activation function!
    hidden.map(this.activation_function.func);

    // Generating the output's output!
    let output = Matrix.multiply(this.weights_ho, hidden);
    output.add(this.bias_o);
    output.map(this.activation_function.func);

    // Sending back to the caller!
    return output.toArray();
  }

  setLearningRate(learning_rate = 0.1) {
    this.learning_rate = learning_rate;
  }

  setActivationFunction(func = sigmoid) {
    this.activation_function = func;
  }

  train(input_array, target_array) {
    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    // activation function!
    hidden.map(this.activation_function.func);

    // Generating the output's output!
    let outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(this.activation_function.func);

    // Convert array to matrix object
    let targets = Matrix.fromArray(target_array);

    // Calculate the error
    // ERROR = TARGETS - OUTPUTS
    let output_errors = Matrix.subtract(targets, outputs);
    // let gradient = outputs * (1 - outputs);
    // Calculate gradient
    let gradients = Matrix.map(outputs, this.activation_function.dfunc);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);

    // Calculate deltas
    let hidden_T = Matrix.transpose(hidden);
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

    // Adjust the weights by deltas
    this.weights_ho.add(weight_ho_deltas);
    // Adjust the bias by its deltas (which is just the gradients)
    this.bias_o.add(gradients);

    // Calculate the hidden layer errors
    let who_t = Matrix.transpose(this.weights_ho);
    let hidden_errors = Matrix.multiply(who_t, output_errors);

    // Calculate hidden gradient
    let hidden_gradient = Matrix.map(hidden, this.activation_function.dfunc);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);

    // Calcuate input->hidden deltas
    let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

    this.weights_ih.add(weight_ih_deltas);
    // Adjust the bias by its deltas (which is just the gradients)
    this.bias_h.add(hidden_gradient);

    // outputs.print();
    // targets.print();
    // error.print();
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data == "string") {
      data = JSON.parse(data);
    }
    let nn = new NeuralNetwork(
      data.input_nodes,
      data.hidden_nodes,
      data.output_nodes
    );
    nn.weights_ih = Matrix.deserialize(data.weights_ih);
    nn.weights_ho = Matrix.deserialize(data.weights_ho);
    nn.bias_h = Matrix.deserialize(data.bias_h);
    nn.bias_o = Matrix.deserialize(data.bias_o);
    nn.learning_rate = data.learning_rate;
    return nn;
  }

  deserialize(data) {
    if (typeof data == "string") {
      data = JSON.parse(data);
    }
    this.input_nodes = data.input_nodes;
    this.hidden_nodes = data.hidden_nodes;
    this.output_nodes = data.output_nodes;
    this.weights_ih = Matrix.deserialize(data.weights_ih);
    this.weights_ho = Matrix.deserialize(data.weights_ho);
    this.bias_h = Matrix.deserialize(data.bias_h);
    this.bias_o = Matrix.deserialize(data.bias_o);
    this.learning_rate = data.learning_rate;
  }

  // Adding function for neuro-evolution
  copy() {
    return new NeuralNetwork(this);
  }

  mutate(rate) {
    function mutate(val) {
      if (Math.random() < rate) {
        // return 2 * Math.random() - 1;
        return val + randomGaussian(0, 0.1);
      } else {
        return val;
      }
    }
    this.weights_ih.map(mutate);
    this.weights_ho.map(mutate);
    this.bias_h.map(mutate);
    this.bias_o.map(mutate);
  }
  crossover(parent) {
    // Create a new child object with the same structure as the parent
    // const child = {
    //   weights_ih: parent.weights_ih.copy(),
    //   weights_ho: parent.weights_ho.copy(),
    //   bias_h: parent.bias_h.copy(),
    //   bias_o: parent.bias_o.copy(),
    // };

    const child = new NeuralNetwork(parent);

    // Randomly pick a crossover point
    const crossoverPoint =
      Math.floor(Math.random() * (parent.weights_ih.rows - 1)) + 1;
    console.log(crossoverPoint);

    // Copy the parent's weights and biases up to the crossover point into the child
    for (let i = 0; i < crossoverPoint; i++) {
      child.weights_ih.data[i] = parent.weights_ih.data[i].slice();
      child.weights_ho.data[i] = parent.weights_ho.data[i].slice();
      child.bias_h.data[i] = parent.bias_h.data[i];
    }

    // Copy the second half of the weights and biases from the current object into the child
    for (let i = crossoverPoint; i < parent.weights_ih.rows; i++) {
      child.weights_ih.data[i] = this.weights_ih.data[i].slice();
      child.weights_ho.data[i] = this.weights_ho.data[i].slice();
      child.bias_h.data[i] = this.bias_h.data[i];
    }

    return child;
  }
}
