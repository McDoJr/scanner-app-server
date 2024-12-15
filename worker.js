const { parentPort, workerData } = require('worker_threads');
const tf = require('@tensorflow/tfjs-node');

const fetchLabels = async (url) => {
    try {
        const response = await fetch(url);

        if(!response.ok) {
            console.log("Error fetching labels from url")
            return null;
        }
        const metadata = await response.json();
        return metadata.labels;
    }catch (error) {
        console.log("Error fetching labels");
        return null;
    }
}

let modelLoaded = false;
let datas = {model: undefined, labels: []};

(async () => {
    try {
        const { modelJsonUrl, metaDataUrl } = workerData;
        if(modelJsonUrl && metaDataUrl) {
            const model = await tf.loadLayersModel(modelJsonUrl);
            console.log("Model has been loaded!");
            const labels = await fetchLabels(metaDataUrl);
            console.log("Labels has been loaded");
            datas = {model, labels};
            modelLoaded = true;
        }
    } catch (err) {
        console.error('Error loading models:', err);
    }
})();


parentPort.on('message', async (data) => {

    const { base64, test } = data;

    if(test) {
        console.log("Testing received!");
        return;
    }
    console.log("predicting start ....");

    try {
        const result = tf.tidy(() => {
            const imageBuffer = Buffer.from(base64, 'base64');
            const tensor = tf.node.decodeImage(imageBuffer, 3)
                .resizeBilinear([224, 224])
                .div(tf.scalar(255))
                .expandDims(0);

            const prediction = datas.model.predict(tensor);
            const predictionData = prediction.arraySync(); // Synchronous array fetch

            const confidenceThreshold = 0.8;
            const maxIndex = predictionData[0].indexOf(Math.max(...predictionData[0]));
            const confidence = Math.max(...predictionData[0]);

            return confidence > confidenceThreshold
                ? { label: data.labels[maxIndex], confidence }
                : { label: 'Unknown', confidence };
        });

        parentPort.postMessage(result);
    } catch (err) {
        console.error('Error during prediction:', err);
        parentPort.postMessage({ error: 'Prediction error' });
    }
});
