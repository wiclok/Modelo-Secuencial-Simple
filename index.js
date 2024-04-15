// Definir los datos de entrenamiento
const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
const ys = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1]);

// Construir el modelo secuencial
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// Compilar el modelo
model.compile({ optimizer: "sgd", loss: "meanSquaredError" });

// Función para entrenar el modelo
function entrenarModelo() {
  // const 400 iteraciones basta para una aproximación bastante acertada 
  model.fit(xs, ys, { epochs: 250 }).then(() => {
    document.getElementById("resultado").innerText =
      "Entrenamiento completado. El modelo está listo para ser utilizado.";
  });
}

// Función para predecir y
function predecirY() {
  const xValue = parseFloat(document.getElementById("valorX").value);
  const x = tf.tensor2d([xValue], [1, 1]);
  const yPred = model.predict(x);
  const prediccionRedondeada = yPred.dataSync()[0].toFixed(1);
  document.getElementById("prediccion").innerText =
    "El valor predicho de Y es: " + yPred.dataSync()[0];
  document.getElementById("prediccion-redondeada").innerText =
    "El valor predicho de Y redondeado es: " + prediccionRedondeada;
}

