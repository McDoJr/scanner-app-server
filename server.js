const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');
const os = require('os');
const {supabase} = require("./supabase");
require('dotenv').config();

// Set up Express app
const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

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
            console.error('Error generating signed URLs:', modelJsonError || weightsError);
            return null;
        }


        return {
            modelJsonUrl: modelJsonUrl.publicUrl,
            weightsUrl: weightsUrl.publicUrl,
            metaDataUrl: metaData.publicUrl
        };
    } catch (error) {
        console.error('Error:', error);
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

// Load the model
// let models = [];
// const modelPaths = [
//     'file://./models/model1/model.json',
//     'file://./models/model2/model.json',
// ];
let datas = {model: undefined, labels: []};
(async () => {
    try {
        // for (const path of modelPaths) {
        //     const model = await tf.loadLayersModel(path);
        //     models.push(model);
        //     console.log(`Model loaded from ${path}`);
        // }
        const signUrls = await getSignedUrls();
        if(signUrls) {
            const model = await tf.loadLayersModel(signUrls.modelJsonUrl);
            console.log("Model has been loaded!");
            const labels = await fetchLabels(signUrls.metaDataUrl);
            console.log("Labels has been loaded");
            datas = {model, labels};
        }
    } catch (err) {
        console.error('Error loading models:', err);
    }
})();

// Prediction route
app.post('/predict', async (req, res) => {
    try {

        if(!datas) return res.status(500).json({ error: 'Model and labels not loaded' });

        if (!datas.model) {
            return res.status(500).json({ error: 'Model not loaded' });
        }

        const { base64 } = req.body;

        if (!base64) {
            return res.status(400).json({ error: 'No image data provided' });
        }

        const imageBuffer = Buffer.from(base64, 'base64');


        // Decode image, preprocess, and predict
        const tensor = tf.node.decodeImage(imageBuffer, 3) // Decode to RGB
            .resizeBilinear([224, 224]) // Resize to model input size
            .div(tf.scalar(255)) // Normalize to [0, 1]
            .expandDims(0); // Add batch dimension

        const confidenceThreshold = 0.8; // Minimum confidence for prediction
        let result = { label: 'Unknown', confidence: 0 };

        const labels = datas.labels;

        const prediction = datas.model.predict(tensor);
        const predictionData = await prediction.array();
        const maxIndex = predictionData[0].indexOf(Math.max(...predictionData[0]));
        const confidence = Math.max(...predictionData[0]);
        // Check if confidence meets threshold
        if (confidence > confidenceThreshold) {
            result = { label: labels[maxIndex], confidence };
        }

        res.json(result);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Error during prediction' });
    }
});

const HOST = '0.0.0.0';
const PORT = process.env.PORT;
app.listen(PORT, HOST, () => console.log("Server running on " + (HOST) + ":" + PORT));