// Define the class names corresponding to the model's output indices
const classNames = ['Healthy', 'Powedry', 'Rust'];

// Function to load the model
async function loadModel() {
    try {
        const model = await tf.loadGraphModel('/tfjs_model/model.json');
        return model;
    } catch (err) {
        console.error('Error loading model:', err);
    }
}

// Function to preprocess the image
function preprocessImage(image) {
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();
    return tensor;
}

// Function to handle image upload
document.getElementById('imageUpload').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        const imageElement = document.getElementById('uploadedImage');
        imageElement.src = URL.createObjectURL(file);
        imageElement.onload = async () => {
            const model = await loadModel();
            if (model) {
                const tensor = preprocessImage(imageElement);
                const prediction = model.predict(tensor);
                const predictedClass = (await prediction.argMax(-1).data())[0];
                const predictedClassName = classNames[predictedClass];
                const predictionElement = document.getElementById('prediction');
                predictionElement.innerText = `Predicted class: ${predictedClassName}`;
                predictionElement.style.animation = 'none'; // Reset animation
                setTimeout(() => predictionElement.style.animation = '', 0); // Restart animation
            } else {
                document.getElementById('prediction').innerText = 'Model loading failed.';
            }
        };
    }
});

// Function to handle reset button click
document.getElementById('resetButton').addEventListener('click', () => {
    document.getElementById('imageUpload').value = null;
    document.getElementById('uploadedImage').src = '';
    document.getElementById('prediction').innerText = '';
});
