const { parentPort, workerData } = require('worker_threads');
const tf = require('@tensorflow/tfjs-node');
const {supabase} = require("./supabase");

let datas = {model: undefined, labels: []};
let modelLoaded = false;

const getSignedUrls = async () => {
    try {
        const { data: modelJsonUrl, error: modelJsonError } = supabase
            .storage
            .from('models')
            .getPublicUrl('model2/model.json');

        const { data: weightsUrl, error: weightsError } = supabase
            .storage
            .from('models')
            .getPublicUrl('model2/weights.bin');

        const { data: metaData, error: metaDataError } = supabase
            .storage
            .from('models')
            .getPublicUrl('model2/metadata.json');

        if (modelJsonError || weightsError || metaDataError) {
            console.error('Error generating signed URLs:', modelJsonError || weightsError || metaDataError);
            return null;
        }

        return {
            modelJsonUrl: modelJsonUrl.publicUrl,
            weightsUrl: weightsUrl.publicUrl,
            metaDataUrl: metaData.publicUrl
        };
    } catch (error) {
        console.error('Error during URL fetch:', error);
        return null;
    }
};

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

// Load model and labels once
(async () => {
    try {
        const signUrls = await getSignedUrls();
        if(signUrls) {
            const model = await tf.loadLayersModel(signUrls.modelJsonUrl);
            console.log("v2: Model has been loaded!");
            const labels = await fetchLabels(signUrls.metaDataUrl);
            console.log("v2: Labels has been loaded");
            datas = {model, labels};
            modelLoaded = true;
        }
    } catch (err) {
        console.error('Error loading models:', err);
    }
})();

parentPort.on('message', async () => {

    if (!modelLoaded || !datas || !datas.model) {
        parentPort.postMessage({ error: "Model or labels not loaded" });
        return;
    }

    const { base64 } = workerData;

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
                ? { label: datas.labels[maxIndex], confidence }
                : { label: 'Unknown', confidence };
        });

        parentPort.postMessage(result);
    } catch (err) {
        console.error('Error during prediction:', err);
        parentPort.postMessage({ error: 'Prediction error' });
    }
});
