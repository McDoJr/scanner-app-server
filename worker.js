const { parentPort, workerData } = require('worker_threads');
const tf = require('@tensorflow/tfjs-node');

parentPort.on('message', async () => {

    const { base64, model, labels } = workerData;

    console.log("model:" + !!model + " labels: " + labels);

    try {
        const result = tf.tidy(() => {
            const imageBuffer = Buffer.from(base64, 'base64');
            const tensor = tf.node.decodeImage(imageBuffer, 3)
                .resizeBilinear([224, 224])
                .div(tf.scalar(255))
                .expandDims(0);

            const prediction = model.predict(tensor);
            const predictionData = prediction.arraySync(); // Synchronous array fetch

            const confidenceThreshold = 0.8;
            const maxIndex = predictionData[0].indexOf(Math.max(...predictionData[0]));
            const confidence = Math.max(...predictionData[0]);

            return confidence > confidenceThreshold
                ? { label: labels[maxIndex], confidence }
                : { label: 'Unknown', confidence };
        });

        parentPort.postMessage(result);
    } catch (err) {
        console.error('Error during prediction:', err);
        parentPort.postMessage({ error: 'Prediction error' });
    }
});
