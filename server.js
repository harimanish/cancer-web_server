const express = require("express");
const tf = require("@tensorflow/tfjs-node");
const multer = require("multer");
// const upload = multer({ dest: "uploads/" });
const fs = require("fs");
const path = require('path');

const app = express();
const port = 5000;

app.use(express.json());

const TARGET_CLASSES = {
  0: 'akiec, Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
  1: 'bcc, Basal Cell Carcinoma',
  2: 'bkl, Benign Keratosis',
  3: 'df, Dermatofibroma',
  4: 'mel, Melanoma',
  5: 'nv, Melanocytic Nevi',
  6: 'vasc, Vascular skin lesion'

};;

const loadModel = async () => {
  const modelPath = path.join(__dirname, 'model', 'model.json');
  const model = await tf.loadLayersModel(`file://${modelPath}`);
  return model;
};

const preprocessImage = (imageBuffer) => {
  const tensor = tf.node.decodeImage(imageBuffer);
  const resized = tf.image.resizeNearestNeighbor(tensor, [224, 224]);
  const casted = resized.cast("float32");
  const expanded = casted.expandDims();
  const offset = tf.scalar(127.5);
  return expanded.sub(offset).div(offset);
};

const upload = multer({
  storage: multer.memoryStorage(),
});

app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    const tensor = preprocessImage(req.file.buffer);
    const model = await loadModel();
    const predictions = await model.predict(tensor).data();
    const top5 = Array.from(predictions)
      .map((p, i) => ({
        probability: p,
        className: TARGET_CLASSES[i],
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5);
    res.send(top5);
  } catch (err) {
    console.error(err);
    res.status(500).send('Internal Server Error');
  }
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
