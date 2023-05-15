// Links
let nn = "https://cdn.jsdelivr.net/gh/TeggunDrut/JS-NeuralNetwork/nn.js";
let matrix =
  "https://cdn.jsdelivr.net/gh/TeggunDrut/JS-NeuralNetwork/matrix.js";

// Functions
async function injectScripts() {
  let script = document.createElement("script");
  (script.src = nn),
    document.head.appendChild(script),
    console.log("Scripts injected"),
    (script = document.createElement("script")),
    (script.src = matrix),
    document.head.appendChild(script),
    console.log("Scripts injected"),
    (script = document.createElement("script")),
    (script.src = "https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/p5.js"),
    document.head.appendChild(script),
    console.log("Scripts injected");
}
const getGrid = () => {
  return gameHolder.grid;
};
const getScore = () => {
  return gameHolder.score;
};

class AI {
  constructor(nn, maxMoves) {
    if (nn != undefined) this.nn = nn;
    else this.nn = new NeuralNetwork(16, 20, 4);
    this.score = 0;
    this.moveCount = 0;
    this.maxMoves = maxMoves;
    // // 0: up, 1: right, 2: down, 3: left
    this.game = new GameManager(
      4,
      KeyboardInputManager,
      HTMLActuator,
      LocalStorageManager
    );
  }
  move(dir) {
    this.game.inputManager.emit("move", dir);
    this.moveCount++;
  }
  getInputs() {
    let grid = this.game.grid;
    let inputs = [];
    for (let i = 0; i < grid.cells.length; i++) {
      for (let j = 0; j < grid.cells[i].length; j++) {
        let cell = grid.cells[i][j];
        if (cell != null) inputs.push(Math.log2(cell.value));
        else inputs.push(0);
      }
    }
    return inputs;
  }
  // function that runs a loop where the ai keeps playing until it loses
  // and then it trains itself
  run() {
    while (!this.game.isGameTerminated()) {
      let inputs = this.getInputs();
      let outputs = this.nn.predict(inputs);
      let move = outputs.indexOf(Math.max(...outputs));
      this.move(move);
      if (this.moveCount > this.maxMoves) break;
    }
    this.score = this.game.score;
    this.grid = this.game.grid;
    this.game.restart();
  }
  clone() {
    return new AI(this.nn.copy(), this.maxMoves);
  }
  crossover(parent2) {
    // Perform crossover between this AI and parent2
    const parent1 = this;
    const child = new AI(undefined, this.maxMoves);

    // Perform crossover at a random crossover point for weights_ho
    const crossoverPointHO = Math.floor(
      Math.random() * this.nn.weights_ho.length
    );

    // Combine the weights from parent1 and parent2 for weights_ho
    for (let i = 0; i < crossoverPointHO; i++) {
      child.nn.weights_ho[i] = parent1.nn.weights_ho[i].slice();
    }
    for (let i = crossoverPointHO; i < parent2.nn.weights_ho.length; i++) {
      child.nn.weights_ho[i] = parent2.nn.weights_ho[i].slice();
    }

    // Perform crossover at a random crossover point for weights_ih
    const crossoverPointIH = Math.floor(
      Math.random() * this.nn.weights_ih.length
    );

    // Combine the weights from parent1 and parent2 for weights_ih
    for (let i = 0; i < crossoverPointIH; i++) {
      child.nn.weights_ih[i] = parent1.nn.weights_ih[i].slice();
    }
    for (let i = crossoverPointIH; i < parent2.nn.weights_ih.length; i++) {
      child.nn.weights_ih[i] = parent2.nn.weights_ih[i].slice();
    }

    return child;
  }
}
let ais = [];

let bestAi;
let bestAis = [];
let populationSize = 1000;

function init(epochs, count) {
  const initialTournamentSize = 2;
  const maxTournamentSize = 5;

  if (ais.length == 0) {
    for (let i = 0; i < populationSize; i++) {
      ais.push(new AI(undefined, count));
    }
  }
  for (let epoch = 0; epoch < epochs; epoch++) {
    const tournamentSize = Math.min(
      initialTournamentSize + epoch,
      maxTournamentSize
    );
    // sort ais
    ais.sort((a, b) => (a.score > b.score ? -1 : 1));

    let bestAI = ais[0];
    let newAis = [bestAI.clone()]; // Use the best AI from the previous epoch

    for (let i = 0; i < populationSize / 2; i++) {
      let parentA = pickOne(ais, tournamentSize);
      let parentB = pickOne(ais, tournamentSize);
      let child = parentA.crossover(parentB);
      child.nn.mutate(0.1);
      newAis.push(child);
    }

    ais = newAis;
    for (let i = 0; i < populationSize / 2; i++) {
      ais[i].run();
    }

    // sort
    ais.sort((a, b) => (a.score > b.score ? -1 : 1));

    console.log(
      "Best AI in epoch " + (epoch + 1) + ": " + ais[0].score + " points"
    );
    bestAis.push(ais[0]);
  }
}
function pickOne(ais, tournamentSize) {
  let bestAI = null;

  for (let i = 0; i < tournamentSize; i++) {
    const randomIndex = Math.floor(Math.random() * ais.length);
    const randomAI = ais[randomIndex];

    if (bestAI === null || randomAI.score > bestAI.score) {
      bestAI = randomAI;
    }
  }

  return bestAI;
}

injectScripts();

function run(string, times) {
  let ai;
  if (string !== undefined) {
    ai = new NeuralNetwork(16, 20, 4);
    ai.deserialize(string);
  } else {
    bestAis.sort((a, b) => (a.score > b.score ? -1 : 1));
    ai = bestAis[0];
  }
  let currentBoard = gameHolder.grid;
  let inputs = [];
  // sort bestAis
  let bestAi = ai;
  for (let i = 0; i < currentBoard.cells.length; i++) {
    for (let j = 0; j < currentBoard.cells[i].length; j++) {
      let cell = currentBoard.cells[i][j];
      if (cell != null) inputs.push(Math.log2(cell.value));
      else inputs.push(0);
    }
  }
  for (let i = 0; i < times; i++) {
    let outputs = ai.predict(inputs);
    let move = outputs.indexOf(Math.max(...outputs));
    gameHolder.inputManager.emit("move", move);
  }
}
