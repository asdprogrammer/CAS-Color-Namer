let model;
let inputs = [];
let targets = [];
let trainData, trainLabels;
let testData, testLabels;

let colors = ["red", "green", "blue", "yellow", "orange", "purple", "grey", "black", "white", "brown", "pink"];
let r, g, b;
let counter = 0;
let color;


function compileModel() {
  model.compile({
    optimizer: tf.train.sgd(1),
    loss: "meanSquaredError",
    metrics: ["accuracy"],
  });

  model.summary();
}


async function saveModel() {
  await model.save("downloads://" + prompt("Model name:", "mymodel"));
}


async function loadModel() {
  const uploadJSONInput = document.getElementById("upload-json");
  const uploadWeightsInput = document.getElementById("upload-weights");
  model = await tf.loadLayersModel(tf.io.browserFiles(
                      [uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
  compileModel();
}


function newColor() {
  r = Math.random();
  g = Math.random();
  b = Math.random();
  document.querySelector("#colorRGB").style.backgroundColor = "rgb(" + r * 255 + "," + g * 255 + "," + b * 255+ ")";
  document.querySelector("#colorRGB").innerHTML = "RGB: " + Math.floor(r * 255) + " " + Math.floor(g * 255) + " " + Math.floor(b * 255);
}


function submitColor(color) {
  console.log(counter, [Math.floor(r * 255), Math.floor(g * 255), Math.floor(b * 255)], colors[color]);
  target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  target[color] = 1;
  inputs.push([r, g, b]);
  targets.push(target);

  newColor();

  counter++;
}


function onBatchEnd(batch, logs) {
  console.log("Batch", batch);
  console.log(logs);
}


function onEpochEnd(epoch, logs) {
  if (epoch % 100 === 0) {
    console.log("Epoch", epoch);
    console.log(logs);
  }
}


function cleanUp() {
  trainData.dispose();
  trainLabels.dispose();
  testData.dispose();
  testLabels.dispose();
  model.dispose();
  tf.disposeVariables();
  console.log("Cleaned");
  console.table(tf.memory());
}


function train() {
  trainData = tf.tensor(inputs, [inputs.length, 3]);
  trainLabels = tf.tensor(targets, [targets.length, 11]);

  model.fit(trainData, trainLabels, {
    epochs: 1000,
    batchSize: 16,
    shuffle: true,
    callbacks: {/*onBatchEnd, */onEpochEnd}
  }).then(info => {
    console.log("Final accuracy", info.history.acc);
    test();
  });
}


function getMaxValueIndex(arr) {
  return arr.indexOf(Math.max(...arr));
}


function test() {
  let inputs = [];

  for (let i = 0; i < 100; ++i) {
    inputs.push([Math.random(), Math.random(), Math.random()]);
  }

  testData = tf.tensor(inputs, [100, 3]);
  testLabels = model.predict(testData);
  console.log("Test labels");
  testLabels.print();

  let h1 = document.createElement("h1");
  h1.innerHTML = "Tests:";
  document.querySelector("body").appendChild(h1);

  for (let i = 0; i < 100; ++i) {
    const colorName = colors[getMaxValueIndex(testLabels.arraySync()[i])];

    let p = document.createElement("p");
    p.style.backgroundColor = "rgb(" + inputs[i][0] * 255 + "," + inputs[i][1] * 255 + "," + inputs[i][2] * 255 + ")";
    p.innerHTML = "RGB: " + Math.floor(inputs[i][0] * 255) + " " + Math.floor(inputs[i][1] * 255) + " " + Math.floor(inputs[i][2] * 255);
    p.innerHTML += " ";
    p.innerHTML += "Label: " + colorName;
    document.querySelector("body").appendChild(p);
  }
}


const createNewModel = confirm("Create new model?");

if (createNewModel) {
  model = tf.tidy(() => {
    const input = tf.input({shape: [3]});
    const layer1 = tf.layers.dense({units: 10, useBias: true, biasInitializer: "randomNormal", activation: "relu"});
    const layer2 = tf.layers.dense({units: 10, useBias: true, biasInitializer: "randomNormal", activation: "sigmoid"});
    const layer3 = tf.layers.dense({units: 11, useBias: true, biasInitializer: "randomNormal", activation: "tanh"});
    const output = layer3.apply(layer2.apply(layer1.apply(input)));
    return tf.model({inputs: input, outputs: output});
  });

  compileModel();

  document.querySelector("#upload-json-button").style.display = "none";
  document.querySelector("#upload-weights-button").style.display = "none";
  document.querySelector("#load-model").style.display = "none";

  ////////////////////////////////////////

  for (color in data) {
    r = color.split(",")[0] / 255;
    g = color.split(",")[1] / 255;
    b = color.split(",")[2] / 255;
    submitColor(colors.indexOf(data[color]));
  }
}

// newColor();
