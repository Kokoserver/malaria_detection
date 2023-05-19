
import os
from anyio import Path
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import torch
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import tempfile


app = FastAPI()



def get_model(file):
    path = os.path.join("yolov7")
    model = torch.hub.load(path, "custom", "best.pt", source='local', verbose=False, trust_repo=True)
    results = model(file)
    return results, model




@app.post("/raw")
async def predict(image: UploadFile):
    image_data = await image.read()
    image = Image.open(BytesIO(image_data))
    results, _ = get_model(image)
    predictions = results.pandas().xyxy[0]
    prediction_list = predictions.values.tolist()
    return {"result": prediction_list}


@app.post("/formatted")
async def predict(image: UploadFile):
    image_data = await image.read()
    image = Image.open(BytesIO(image_data))
    results, _  = get_model(image)

    # Extract the predictions
    predictions = results.pred[0]  # Get the predictions for the first image (assuming batch size of 1)
    labels = predictions[:, -1].tolist()  # Get the class labels
    scores = predictions[:, 4].tolist()  # Get the confidence scores
    boxes = predictions[:, :4].tolist()  # Get the bounding box coordinates

    # Combine the labels, scores, and boxes into a list of dictionaries
    prediction_list = []
    for label, score, box in zip(labels, scores, boxes):
        prediction = {
            "label": label,
            "score": score,
            "box": box
        }
        prediction_list.append(prediction)
    return {"result": prediction_list}





@app.post("/image")
async def predict(image: UploadFile):
    image_data = await image.read()
    image = Image.open(BytesIO(image_data))
    results, model = get_model(image)

    # Extract the predictions
    predictions = results.pred[0]
    labels = predictions[:, -1].tolist()
    scores = predictions[:, 4].tolist()
    boxes = predictions[:, :4].tolist()

    # Visualize the predictions
    image_np = np.array(image)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)

    for label, score, box in zip(labels, scores, boxes):
        class_label = model.names[int(label)]
        confidence = score
        xmin, ymin, xmax, ymax = box

        # Draw bounding box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)

        # Add label and confidence score
        text = f'{class_label}: {confidence:.2f}'
        plt.text(xmin, ymin, text, fontsize=12, color='red')

    plt.axis('off')
    plt.tight_layout()

    # Convert the plot to an image
    plot_image = BytesIO()
    plt.savefig(plot_image, format='PNG')
    plot_image.seek(0)
    # Save the plot to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        plt.savefig(tmp_file.name, format='PNG')
    return FileResponse(tmp_file.name, media_type='image/png')
